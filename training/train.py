"""
Orchestrator-R1 GRPO Training Script

Supports two modes:
  - FSDP (Linux/WSL2): accelerate launch --config_file training/accelerate_fsdp_4gpu.yaml
  - DDP+Gloo+LoRA (Windows native): accelerate launch --config_file training/accelerate_ddp_4gpu.yaml

Usage:
    accelerate launch --config_file training/accelerate_ddp_4gpu.yaml \
        training/train.py \
        --model_path models/Qwen2.5-3B-Instruct \
        --data_path data/train.jsonl \
        --output_dir checkpoints/orchestrator_r1 \
        --api_base YOUR_API_BASE \
        --api_key YOUR_API_KEY \
        --use_lora
"""

import argparse
import os
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from trl import GRPOConfig, GRPOTrainer

from orchestrator_r1.agent_pool.agent_registry import AgentRegistry
from orchestrator_r1.orchestrator.generation import OrchestratorGenerationManager, GenerationConfig
from orchestrator_r1.orchestrator.reward import compute_reward
from orchestrator_r1.prompts.system_prompt import SYSTEM_PROMPT


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",          type=str, required=True)
    parser.add_argument("--data_path",           type=str, required=True)
    parser.add_argument("--output_dir",          type=str, required=True)
    parser.add_argument("--api_base",            type=str, required=True)
    parser.add_argument("--api_key",             type=str, required=True)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--grad_accum",          type=int, default=8)
    parser.add_argument("--num_generations",     type=int, default=8,
                        help="G in GRPO: rollouts per prompt")
    parser.add_argument("--max_new_tokens",      type=int, default=512)
    parser.add_argument("--max_turns",           type=int, default=6)
    parser.add_argument("--learning_rate",       type=float, default=1e-6)
    parser.add_argument("--num_epochs",          type=int, default=3)
    parser.add_argument("--alpha",               type=float, default=0.3,
                        help="API cost penalty weight")
    parser.add_argument("--beta",                type=float, default=0.1,
                        help="Turn count penalty weight")
    parser.add_argument("--gamma",               type=float, default=0.15,
                        help="Efficiency bonus weight")
    parser.add_argument("--metric",              type=str, default="f1",
                        choices=["em", "f1"])
    parser.add_argument("--max_samples",         type=int, default=None)
    # LoRA options (for DDP+Gloo on Windows)
    parser.add_argument("--use_lora",           action="store_true",
                        help="Use LoRA for parameter-efficient training (required for DDP on Windows)")
    parser.add_argument("--lora_r",             type=int, default=64)
    parser.add_argument("--lora_alpha",         type=int, default=128)
    parser.add_argument("--lora_dropout",       type=float, default=0.05)
    return parser.parse_args()


def load_dataset(data_path: str, max_samples: int | None = None) -> Dataset:
    """Load JSONL file. Each line: {"input": str, "answer": str or list}"""
    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if isinstance(r.get("answer"), list):
                    r["answer"] = json.dumps(r["answer"], ensure_ascii=False)
                records.append(r)
    if max_samples:
        records = records[:max_samples]
    return Dataset.from_list(records)


def build_reward_fn(registry: AgentRegistry, gen_manager_ref: dict, args):
    """
    Build the reward function for GRPOTrainer.

    GRPOTrainer calls reward_fn(prompts, completions, **kwargs)
    where completions are the G sampled outputs for each prompt.
    """
    def reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        gold_answers = kwargs.get("answer", [None] * len(prompts))
        gold_answers = [json.loads(g) if isinstance(g, str) and g.startswith("[") else g
                        for g in gold_answers]
        rewards = []

        for prompt, completion, gold in zip(prompts, completions, gold_answers):
            # Run generation loop starting from existing completion
            # (GRPO already produced the completion — we need to execute agent calls)
            agent_calls = []
            total_cost = 0.0
            n_turns = 0
            full_response = completion

            # Parse all <call> tags in the completion and execute them
            import re
            call_pattern = re.compile(
                r'<call\s+type="(\w+)"[^>]*>(.*?)</call>', re.DOTALL
            )
            for match in call_pattern.finditer(completion):
                agent_type = match.group(1)
                query = match.group(2).strip()
                _, cost = registry.dispatch(agent_type, query)
                total_cost += cost
                agent_calls.append({"agent_type": agent_type, "cost": cost})
                n_turns += 1

            result = compute_reward(
                full_response=full_response,
                gold_answer=gold,
                agent_calls=agent_calls,
                n_turns=max(n_turns, 1),
                metric=args.metric,
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
                max_turns=args.max_turns,
            )
            rewards.append(result["reward"])

        return rewards

    return reward_fn


def main():
    args = parse_args()

    # ── Tokenizer & Model ──────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        trust_remote_code=True,
    )

    # ── LoRA (for DDP+Gloo Windows training) ──────────────────────────────
    if args.use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # ── Agent Registry ─────────────────────────────────────────────────────
    registry = AgentRegistry(api_base=args.api_base, api_key=args.api_key)

    # ── Dataset ────────────────────────────────────────────────────────────
    dataset = load_dataset(args.data_path, args.max_samples)

    # Format prompts using chat template
    def format_prompt(example):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": example["input"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return {"prompt": prompt, "answer": example["answer"]}

    dataset = dataset.map(format_prompt)

    # ── Reward Function ────────────────────────────────────────────────────
    reward_fn = build_reward_fn(registry, {}, args)

    # ── GRPO Config ────────────────────────────────────────────────────────
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,    # G=8 rollouts per prompt
        max_new_tokens=args.max_new_tokens,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        bf16=True,
        dataloader_num_workers=2,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        save_only_model=True,          # skip optimizer states (~28GB saved per checkpoint)
        report_to="wandb",
        run_name=f"orchestrator-r1-alpha{args.alpha}-beta{args.beta}",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        remove_unused_columns=False,
    )

    # ── Trainer ────────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))
    print(f"Model saved to {args.output_dir}/final")


if __name__ == "__main__":
    main()
