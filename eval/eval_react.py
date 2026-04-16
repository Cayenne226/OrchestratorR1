"""
Evaluate ReAct baseline: Qwen2.5-7B-Instruct (zero-shot) + search tool (LLM API).

ReAct uses a single model as the reasoning backbone, with a search() tool that
queries an LLM API (gpt-4o-mini by default). This is a standard tool-use agent
baseline — no training, no multi-agent orchestration.

Supports two inference modes:
  - Local: load model via transformers (needs GPU, ~16-18GB VRAM for BF16)
  - API:   call backbone via OpenAI-compatible API (--api_mode, zero GPU)

Usage:
    # Local mode
    python eval/eval_react.py \
        --model_path models/Qwen2.5-7B-Instruct \
        --data_path data/test_qa.jsonl \
        --api_base YOUR_API_BASE --api_key YOUR_API_KEY \
        --output eval/results/react_7b.json

    # API mode
    python eval/eval_react.py \
        --api_mode \
        --backbone_model Qwen/Qwen2.5-7B-Instruct \
        --backbone_api_base YOUR_BACKBONE_API_BASE \
        --backbone_api_key YOUR_BACKBONE_API_KEY \
        --data_path data/test_qa.jsonl \
        --api_base YOUR_API_BASE --api_key YOUR_API_KEY \
        --output eval/results/react_7b.json
"""

import argparse
import json
import re
from pathlib import Path
from tqdm import tqdm

from orchestrator_r1.agent_pool.agent_registry import AgentRegistry
from eval.metrics import compute_metric


# ── Prompt templates ──────────────────────────────────────────────────────────

REACT_SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions step by step.
You have access to a search tool that queries an LLM for information.

For each step, use this exact format:
Thought: <your reasoning about what to do next>
Action: search("your search query") OR answer("your final answer")

Rules:
- Use search("query") to gather information you need.
- Use answer("your answer") when you have enough information.
- The answer should be a short factual response (a name, number, date, or short phrase).
- You must call answer() within {max_turns} turns."""

REACT_SYSTEM_PROMPT_CODE = """\
You are a helpful assistant that solves coding problems step by step.
You have access to a search tool that queries an LLM for help.

For each step, use this exact format:
Thought: <your reasoning about what to do next>
Action: search("your query about the problem") OR answer("your final code solution")

Rules:
- Use search("query") to ask for hints, algorithms, or partial solutions.
- Use answer("code") when you have the complete solution.
- The answer should contain ONLY the code, no explanation.
- You must call answer() within {max_turns} turns."""


# ── Action parsing ────────────────────────────────────────────────────────────

# Matches search("...") or search('...')
_SEARCH_RE = re.compile(r'search\(\s*["\'](.+?)["\']\s*\)', re.DOTALL)
# Matches answer("...") or answer('...')
_ANSWER_RE = re.compile(r'answer\(\s*["\'](.+?)["\']\s*\)', re.DOTALL)


def parse_action(text: str) -> tuple[str, str]:
    """Parse the action from model output.

    Returns (action_type, content):
      - ("answer", answer_text)
      - ("search", query_text)
      - ("invalid", raw_text)
    """
    # Try answer() first — if model wants to answer, let it
    m = _ANSWER_RE.search(text)
    if m:
        return "answer", m.group(1).strip()

    m = _SEARCH_RE.search(text)
    if m:
        return "search", m.group(1).strip()

    return "invalid", text.strip()


def extract_fallback_answer(trajectory: str) -> str:
    """Extract a fallback answer from the last model output when no answer() was called."""
    # Try to find the last Thought section
    thoughts = re.findall(r'Thought:\s*(.+?)(?=\n(?:Action|Observation|Thought)|$)',
                          trajectory, re.DOTALL)
    if thoughts:
        last = thoughts[-1].strip()
        # If it contains something that looks like an answer, return it
        if len(last) < 200:
            return last

    # Last resort: return last non-empty line
    lines = [l.strip() for l in trajectory.strip().split('\n') if l.strip()]
    return lines[-1] if lines else ""


# ── Backbone: local model ─────────────────────────────────────────────────────

class LocalBackbone:
    """Generate text using a local HuggingFace model."""

    def __init__(self, model_path: str, device: str = "cuda", dtype: str = "bf16"):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch_dtype,
        ).to(device)
        self.model.eval()
        self.device = device

    def generate(self, messages: list[dict], max_new_tokens: int = 512) -> str:
        """Generate a response given chat messages."""
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        import torch
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # Decode only the new tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


class APIBackbone:
    """Generate text using an OpenAI-compatible API."""

    def __init__(self, model_name: str, api_base: str, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.model_name = model_name

    def generate(self, messages: list[dict], max_new_tokens: int = 512) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=max_new_tokens,
        )
        return response.choices[0].message.content or ""


# ── ReAct evaluation loop ────────────────────────────────────────────────────

def eval_react_single(
    question: str,
    record: dict,
    backbone,
    registry: AgentRegistry,
    max_turns: int = 6,
    search_agent_type: str = "executor_cheap",
) -> dict:
    """Run ReAct on a single question. Returns result dict."""
    is_code = record.get("source", "") in ("humaneval", "mbpp", "livecodebench") \
              or record.get("difficulty", "").startswith("code")

    system_prompt = REACT_SYSTEM_PROMPT_CODE if is_code else REACT_SYSTEM_PROMPT
    system_prompt = system_prompt.format(max_turns=max_turns)

    # Build initial messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    trajectory_parts = []
    agent_calls = []
    total_cost = 0.0
    final_answer = None
    turns_used = 0

    for turn in range(max_turns):
        turns_used = turn + 1

        # Generate
        output = backbone.generate(messages, max_new_tokens=512)
        trajectory_parts.append(output)

        # Parse action
        action_type, content = parse_action(output)

        if action_type == "answer":
            final_answer = content
            break

        elif action_type == "search":
            # Call search tool via AgentRegistry
            response, cost = registry.dispatch(search_agent_type, content)
            total_cost += cost
            agent_calls.append({
                "agent_type": search_agent_type,
                "query": content,
                "cost": cost,
            })

            # Append assistant output + observation to messages
            observation = f"Observation: {response}"
            messages.append({"role": "assistant", "content": output})
            messages.append({"role": "user", "content": observation})
            trajectory_parts.append(observation)

        else:
            # Invalid action — give error feedback
            error_msg = (
                "Observation: Invalid action format. "
                "Please use search(\"your query\") or answer(\"your answer\")."
            )
            messages.append({"role": "assistant", "content": output})
            messages.append({"role": "user", "content": error_msg})
            trajectory_parts.append(error_msg)

    # Fallback if no answer() was called
    if final_answer is None:
        trajectory = "\n".join(trajectory_parts)
        final_answer = extract_fallback_answer(trajectory)

    return {
        "pred": final_answer.strip(),
        "n_turns": turns_used,
        "total_cost": total_cost,
        "agent_calls": agent_calls,
        "trajectory": "\n".join(trajectory_parts),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ReAct baseline")

    # Data
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output", type=str, default="eval/results/react_7b.json")
    parser.add_argument("--max_samples", type=int, default=None)

    # ReAct config
    parser.add_argument("--max_turns", type=int, default=6)
    parser.add_argument("--worker_pool", type=str, default="cheap",
                        choices=["cheap", "matched"])

    # Search tool API (for the search() tool that calls an external LLM)
    parser.add_argument("--api_base", type=str, required=True,
                        help="API base URL for the search tool (gpt-4o-mini etc.)")
    parser.add_argument("--api_key", type=str, required=True,
                        help="API key for the search tool")

    # Backbone: local mode (default)
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to local Qwen2.5-7B-Instruct model")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16",
                        choices=["bf16", "fp16", "fp32"])

    # Backbone: API mode
    parser.add_argument("--api_mode", action="store_true",
                        help="Use API for backbone model instead of local loading")
    parser.add_argument("--backbone_model", type=str,
                        default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model name for API mode backbone")
    parser.add_argument("--backbone_api_base", type=str, default=None,
                        help="API base URL for backbone (defaults to --api_base)")
    parser.add_argument("--backbone_api_key", type=str, default=None,
                        help="API key for backbone (defaults to --api_key)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize backbone
    if args.api_mode:
        backbone_api_base = args.backbone_api_base or args.api_base
        backbone_api_key = args.backbone_api_key or args.api_key
        backbone = APIBackbone(args.backbone_model, backbone_api_base, backbone_api_key)
        print(f"Backbone: API mode ({args.backbone_model})")
    else:
        if not args.model_path:
            raise ValueError("--model_path is required in local mode (or use --api_mode)")
        backbone = LocalBackbone(args.model_path, args.device, args.dtype)
        print(f"Backbone: Local mode ({args.model_path})")

    # Initialize search tool registry
    registry = AgentRegistry(
        api_base=args.api_base,
        api_key=args.api_key,
        worker_pool=args.worker_pool,
    )
    print(f"Search tool: {args.worker_pool} pool")

    # Load test data
    records = []
    with open(args.data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    if args.max_samples:
        records = records[:args.max_samples]
    print(f"Loaded {len(records)} samples from {args.data_path}")

    # Evaluate
    results = []
    total_em = total_f1 = total_cost = total_turns = 0.0

    for record in tqdm(records, desc="ReAct eval"):
        output = eval_react_single(
            question=record["input"],
            record=record,
            backbone=backbone,
            registry=registry,
            max_turns=args.max_turns,
        )
        pred = output["pred"]
        metrics = compute_metric(pred, record)
        em = metrics.get("em", 0.0)
        f1 = metrics.get("f1", 0.0)

        total_em += em
        total_f1 += f1
        total_cost += output["total_cost"]
        total_turns += output["n_turns"]

        results.append({
            "input": record["input"],
            "gold": record["answer"],
            "pred": pred,
            "em": em,
            "f1": f1,
            "metrics": metrics,
            "n_turns": output["n_turns"],
            "total_cost": output["total_cost"],
            "agent_calls": output["agent_calls"],
            "source": record.get("source", ""),
            "trajectory": output["trajectory"],
        })

    n = len(results)
    summary = {
        "method": "react_7b",
        "n_samples": n,
        "worker_pool": args.worker_pool,
        "em": total_em / n if n > 0 else 0,
        "f1": total_f1 / n if n > 0 else 0,
        "avg_cost_usd": total_cost / n if n > 0 else 0,
        "avg_turns": total_turns / n if n > 0 else 0,
    }

    print("\n=== ReAct Baseline Results ===")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
