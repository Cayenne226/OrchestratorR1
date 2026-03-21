"""
Auto-generate SFT warmup examples using GPT-4o.

Samples real questions from train.jsonl (NQ + HotpotQA), sends them to GPT-4o
with the orchestrator system prompt, and collects properly formatted traces.

Requires: data/train.jsonl (run prepare_data.py first)

Usage:
    python data_process/prepare_sft.py \
        --train_data data/train.jsonl \
        --output data/sft_warmup.jsonl \
        --api_base https://api.openai.com/v1 \
        --api_key YOUR_KEY \
        --num_samples 100
"""

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
3. For simple single-hop factual questions: use ONE executor_cheap call
4. For multi-hop questions (comparing, connecting facts): use decomposer + multiple executor calls + synthesizer
5. The <answer> MUST contain the gold answer text
6. Do NOT include <information> tags — those are injected by the system at runtime
7. Keep <think> reasoning concise (1-2 sentences)
8. Keep <answer> concise — just the answer, not a full paragraph"""

USER_TEMPLATE = """Question: {question}
Gold answer: {answer}

Generate the orchestration trace for this question."""


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
                   max_retries: int = 3) -> str | None:
    """Call GPT-4o to generate one orchestration trace. Returns None on failure."""
    # Format gold answer for the prompt
    if isinstance(answer, list):
        answer_str = answer[0] if len(answer) == 1 else " / ".join(answer[:3])
    else:
        answer_str = str(answer)

    user_msg = USER_TEMPLATE.format(question=question, answer=answer_str)

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


def sample_questions(train_path: str, num_samples: int, seed: int) -> list[dict]:
    """Sample balanced questions from train.jsonl (50% NQ, 50% HotpotQA)."""
    by_source = {}
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            source = record.get("source", "unknown")
            by_source.setdefault(source, []).append(record)

    random.seed(seed)
    per_source = num_samples // len(by_source)
    remainder = num_samples - per_source * len(by_source)

    sampled = []
    for i, (source, records) in enumerate(sorted(by_source.items())):
        n = per_source + (1 if i < remainder else 0)
        n = min(n, len(records))
        sampled.extend(random.sample(records, n))
        print(f"  Sampled {n} from {source} (pool: {len(records)})")

    random.shuffle(sampled)
    return sampled


def main():
    parser = argparse.ArgumentParser(
        description="Auto-generate SFT warmup data using GPT-4o"
    )
    parser.add_argument("--train_data", type=str, default="data/train.jsonl",
                        help="Path to training JSONL (must exist — run prepare_data.py first)")
    parser.add_argument("--output",     type=str, default="data/sft_warmup.jsonl")
    parser.add_argument("--api_base",   type=str, default="https://api.openai.com/v1")
    parser.add_argument("--api_key",    type=str, required=True)
    parser.add_argument("--model",      type=str, default="gpt-4o",
                        help="Model to use for trace generation")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    if not Path(args.train_data).exists():
        print(f"ERROR: {args.train_data} not found. Run prepare_data.py first.")
        return

    # ── Sample questions ──────────────────────────────────────────────────
    print(f"Sampling {args.num_samples} questions from {args.train_data}...")
    questions = sample_questions(args.train_data, args.num_samples, args.seed)
    print(f"  Total sampled: {len(questions)}")

    # ── Generate traces ───────────────────────────────────────────────────
    client = OpenAI(base_url=args.api_base, api_key=args.api_key)
    results = []
    agent_type_counts = {}
    failed = 0

    print(f"\nGenerating traces with {args.model}...")
    for i, q in enumerate(questions):
        question = q["input"]
        answer = q["answer"]
        # Deserialize JSON-encoded list answers
        if isinstance(answer, str) and answer.startswith("["):
            try:
                answer = json.loads(answer)
            except json.JSONDecodeError:
                pass

        trace = generate_trace(client, args.model, question, answer)
        if trace is None:
            failed += 1
            print(f"  [{i+1}/{len(questions)}] FAILED: {question[:60]}...")
            continue

        results.append({"input": question, "output": trace})

        # Track agent type distribution
        for agent_type in extract_agent_types(trace):
            agent_type_counts[agent_type] = agent_type_counts.get(agent_type, 0) + 1

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(questions)}] Generated {len(results)} traces ({failed} failed)")

    # ── Save ──────────────────────────────────────────────────────────────
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for ex in results:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"Generated {len(results)} SFT examples → {args.output}")
    print(f"Failed: {failed}")
    print(f"\nAgent type distribution:")
    for t, c in sorted(agent_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
