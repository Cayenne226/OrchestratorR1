"""
Auto-generate SFT warmup examples using GPT-4o.

Samples real questions from train_qa.jsonl and train_code.jsonl, sends them
to GPT-4o with path-pattern-specific prompts, and collects properly formatted
orchestration traces covering all 6 agent types.

Usage:
    python data_process/prepare_sft.py \
        --train_qa data/train_qa.jsonl \
        --train_code data/train_code.jsonl \
        --output data/sft_warmup.jsonl \
        --api_base https://api.openai.com/v1 \
        --api_key YOUR_KEY \
        --num_samples 200
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

from openai import OpenAI


# ── The prompt that teaches GPT-4o to generate orchestration traces ──────────

GENERATION_SYSTEM_PROMPT = """You are a training data generator. Given a user question and its gold answer, generate a realistic orchestration trace using EXACTLY the XML tags below.

## Available Agent Types
- executor_cheap: For simple factual questions (single lookup)
- executor_strong: For complex reasoning or analysis
- decomposer: For multi-hop questions that need to be broken into subtasks
- synthesizer: For combining results from multiple agent calls
- critic: For verifying quality of a result
- refiner: For clarifying vague input (rarely needed for QA tasks)

## Required Format
Your output MUST follow this exact structure:

<think>Your reasoning about task complexity and which agents to use</think>
<call type="AGENT_TYPE">Query to the agent</call>
<answer>The final answer (must match or contain the gold answer)</answer>

## Rules
1. ALWAYS start with <think>
2. ALWAYS end with <answer>
3. The <answer> MUST contain the gold answer text
4. Do NOT include <information> tags — those are injected by the system at runtime
5. Keep <think> reasoning concise (1-2 sentences)
6. Keep <answer> concise — just the answer, not a full paragraph"""

# ── Path patterns: different prompts steer GPT-4o toward specific agent combos

PATH_PATTERNS = {
    "simple_direct": {
        "count": 40,
        "difficulty": "simple",
        "task_type": "qa",
        "instruction": "Use ONLY one executor_cheap call. Pattern: think → executor_cheap → answer.",
    },
    "refine_then_exec": {
        "count": 25,
        "difficulty": "simple",
        "task_type": "qa",
        "instruction": "First use refiner to clarify the question, then executor_cheap. Pattern: think → refiner → executor_cheap → answer.",
    },
    "strong_direct": {
        "count": 20,
        "difficulty": "multi_hop",
        "task_type": "qa",
        "instruction": "Use ONLY one executor_strong call for this hard question. Pattern: think → executor_strong → answer.",
    },
    "decompose_exec_synth": {
        "count": 35,
        "difficulty": "multi_hop",
        "task_type": "qa",
        "instruction": "Decompose into subtasks, execute each with executor_cheap, then synthesize. Pattern: think → decomposer → executor_cheap (×2-3) → synthesizer → answer.",
    },
    "decompose_strong_critic": {
        "count": 30,
        "difficulty": "multi_hop",
        "task_type": "qa",
        "instruction": "Decompose, execute with executor_strong, then use critic to verify. Pattern: think → decomposer → executor_strong (×1-2) → critic → answer.",
    },
    "full_pipeline": {
        "count": 20,
        "difficulty": "multi_hop",
        "task_type": "qa",
        "instruction": "Use the full pipeline: decompose, execute with strong, critic finds issue, re-execute, then answer. Pattern: think → decomposer → executor_strong → critic → executor_strong → answer.",
    },
    "code_simple": {
        "count": 15,
        "difficulty": "code",
        "task_type": "code",
        "instruction": "Simple coding task. Use executor_cheap. Pattern: think → executor_cheap → answer. The answer should be the code solution.",
    },
    "code_complex": {
        "count": 15,
        "difficulty": "code",
        "task_type": "code",
        "instruction": "Complex coding task. Decompose, use executor_strong for implementation, critic to verify logic. Pattern: think → decomposer → executor_strong → critic → answer. The answer should be the code solution.",
    },
}

USER_TEMPLATE = """Question: {question}
Gold answer: {answer}

PATH INSTRUCTION: {path_instruction}

Generate the orchestration trace following the exact path pattern above."""


# ── Format validation ────────────────────────────────────────────────────────

VALID_AGENT_TYPES = {"executor_cheap", "executor_strong", "decomposer", "synthesizer", "critic", "refiner"}

def validate_trace(output: str) -> tuple[bool, str]:
    """Check that the generated trace has valid format."""
    if not output.strip():
        return False, "empty output"
    if "<think>" not in output:
        return False, "missing <think>"
    if "</think>" not in output:
        return False, "missing </think>"
    if "<answer>" not in output:
        return False, "missing <answer>"
    if "</answer>" not in output:
        return False, "missing </answer>"

    # Check that all <call> tags have valid agent types
    for match in re.finditer(r'<call\s+type="(\w+)"', output):
        agent_type = match.group(1)
        if agent_type not in VALID_AGENT_TYPES:
            return False, f"invalid agent type: {agent_type}"

    # Must have at least one <call>
    if '<call type="' not in output:
        return False, "no agent calls"

    # <answer> must come after all <call> tags
    last_call = output.rfind("</call>")
    first_answer = output.find("<answer>")
    if last_call > first_answer:
        return False, "<call> after <answer>"

    return True, "ok"


def extract_agent_types(output: str) -> list[str]:
    """Extract all agent types used in a trace."""
    return re.findall(r'<call\s+type="(\w+)"', output)


# ── Main generation logic ────────────────────────────────────────────────────

def generate_trace(client: OpenAI, model: str, question: str, answer: str,
                   path_instruction: str = "", max_retries: int = 3) -> str | None:
    """Call GPT-4o to generate one orchestration trace. Returns None on failure."""
    # Format gold answer for the prompt
    if isinstance(answer, list):
        answer_str = answer[0] if len(answer) == 1 else " / ".join(answer[:3])
    else:
        answer_str = str(answer)

    user_msg = USER_TEMPLATE.format(
        question=question, answer=answer_str, path_instruction=path_instruction
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.7,
                max_tokens=512,
            )
            output = response.choices[0].message.content or ""

            # Strip any markdown code fences GPT might add
            output = output.strip()
            if output.startswith("```"):
                output = re.sub(r'^```\w*\n?', '', output)
                output = re.sub(r'\n?```$', '', output)
                output = output.strip()

            # Remove <information> tags if GPT included them
            output = re.sub(r'<information>.*?</information>\n?', '', output, flags=re.DOTALL)

            is_valid, reason = validate_trace(output)
            if is_valid:
                return output
            if attempt < max_retries - 1:
                time.sleep(1)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [ERROR] API call failed: {e}")
    return None


def load_data_pool(qa_path: str, code_path: str | None) -> dict[str, list[dict]]:
    """Load and categorize training data into pools by difficulty."""
    pools = {"simple": [], "multi_hop": [], "code": []}

    if Path(qa_path).exists():
        with open(qa_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                diff = record.get("difficulty", "simple")
                if diff in pools:
                    pools[diff].append(record)
                else:
                    pools["simple"].append(record)

    if code_path and Path(code_path).exists():
        with open(code_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                pools["code"].append(record)

    for k, v in pools.items():
        print(f"  Pool '{k}': {len(v)} samples")
    return pools


def sample_for_pattern(pools: dict[str, list], pattern_cfg: dict,
                       count: int, rng: random.Random) -> list[dict]:
    """Sample questions matching a path pattern's difficulty requirement."""
    diff = pattern_cfg["difficulty"]
    pool = pools.get(diff, [])
    if not pool:
        # Fallback: use any pool with data
        for fallback_diff in ["simple", "multi_hop", "code"]:
            if pools.get(fallback_diff):
                pool = pools[fallback_diff]
                break
    if not pool:
        return []
    n = min(count, len(pool))
    return rng.sample(pool, n)


def main():
    parser = argparse.ArgumentParser(
        description="Auto-generate SFT warmup data using GPT-4o (200 examples, coverage-aware)"
    )
    parser.add_argument("--train_qa",   type=str, default="data/train_qa.jsonl",
                        help="Path to QA training JSONL")
    parser.add_argument("--train_code", type=str, default="data/train_code.jsonl",
                        help="Path to code training JSONL")
    parser.add_argument("--output",     type=str, default="data/sft_warmup.jsonl")
    parser.add_argument("--api_base",   type=str, default="https://api.openai.com/v1")
    parser.add_argument("--api_key",    type=str, required=True)
    parser.add_argument("--model",      type=str, default="gpt-4o",
                        help="Model to use for trace generation")
    parser.add_argument("--num_samples", type=int, default=200,
                        help="Target sample count (default 200, distributed per coverage matrix)")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # ── Load data pools ───────────────────────────────────────────────────
    print("Loading data pools...")
    pools = load_data_pool(args.train_qa, args.train_code)

    # ── Scale pattern counts to match num_samples ─────────────────────────
    total_default = sum(p["count"] for p in PATH_PATTERNS.values())
    scale = args.num_samples / total_default

    # ── Generate per pattern ──────────────────────────────────────────────
    client = OpenAI(base_url=args.api_base, api_key=args.api_key)
    all_results = []
    agent_type_counts = {}
    pattern_stats = {}  # pattern_name → {generated, failed}
    global_idx = 0

    for pattern_name, pattern_cfg in PATH_PATTERNS.items():
        target = max(1, round(pattern_cfg["count"] * scale))
        questions = sample_for_pattern(pools, pattern_cfg, target, rng)
        instruction = pattern_cfg["instruction"]

        print(f"\n── Pattern: {pattern_name} (target={target}, sampled={len(questions)}) ──")
        generated = 0
        failed = 0

        for i, q in enumerate(questions):
            question = q["input"]
            answer = q["answer"]
            if isinstance(answer, str) and answer.startswith("["):
                try:
                    answer = json.loads(answer)
                except json.JSONDecodeError:
                    pass

            trace = generate_trace(client, args.model, question, answer,
                                   path_instruction=instruction)
            global_idx += 1

            if trace is None:
                failed += 1
                print(f"  [{global_idx}] FAILED: {question[:50]}...")
                continue

            all_results.append({"input": question, "output": trace})
            generated += 1

            for agent_type in extract_agent_types(trace):
                agent_type_counts[agent_type] = agent_type_counts.get(agent_type, 0) + 1

            if generated % 10 == 0:
                print(f"  [{global_idx}] {pattern_name}: {generated}/{target}")

        pattern_stats[pattern_name] = {"generated": generated, "failed": failed, "target": target}
        print(f"  → {pattern_name}: {generated}/{target} generated ({failed} failed)")

    # ── Shuffle and save ──────────────────────────────────────────────────
    rng.shuffle(all_results)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for ex in all_results:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"Generated {len(all_results)} SFT examples → {args.output}")
    print(f"\nPattern coverage:")
    for name, stats in pattern_stats.items():
        print(f"  {name}: {stats['generated']}/{stats['target']} ({stats['failed']} failed)")
    print(f"\nAgent type distribution:")
    for t, c in sorted(agent_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
