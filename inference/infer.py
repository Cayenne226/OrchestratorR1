"""
Local inference for Orchestrator-R1.

Usage:
    python inference/infer.py \
        --model_path models/Qwen2.5-3B-Instruct \
        --api_base YOUR_API_BASE \
        --api_key YOUR_API_KEY \
        --input "用Python写一个带JWT鉴权的REST API"
"""

import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

from orchestrator_r1.agent_pool.agent_registry import AgentRegistry
from orchestrator_r1.orchestrator.generation import OrchestratorGenerationManager, GenerationConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--api_base",   type=str, required=True)
    parser.add_argument("--api_key",    type=str, required=True)
    parser.add_argument("--input",      type=str, required=True)
    parser.add_argument("--max_turns",  type=int, default=6)
    parser.add_argument("--device",     type=str, default="cuda")
    args = parser.parse_args()

    model_path = str(Path(args.model_path).resolve())
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", local_files_only=True,
    ).to(args.device)
    model.eval()

    registry = AgentRegistry(api_base=args.api_base, api_key=args.api_key)
    config = GenerationConfig(max_turns=args.max_turns)
    manager = OrchestratorGenerationManager(model, tokenizer, registry, config)

    print(f"\nInput: {args.input}\n{'='*60}")
    result = manager.rollout(args.input)

    print(result.full_text)
    print("=" * 60)
    print(f"Answer:      {result.answer}")
    print(f"Turns:       {result.n_turns}")
    print(f"Total cost:  ${result.total_cost:.6f}")
    print(f"Agent calls: {len(result.agent_calls)}")
    for c in result.agent_calls:
        print(f"  [{c['turn']}] {c['agent_type']}: ${c['cost']:.6f}")


if __name__ == "__main__":
    main()
