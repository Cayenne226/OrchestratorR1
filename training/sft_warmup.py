"""
SFT Warmup: Teach the Orchestrator model the <call>/<answer> tag format.

Uses a small dataset of hand-crafted examples covering all 6 Agent types,
so that GRPO training starts with a model that can already produce valid tags.

Usage (Windows DDP+Gloo+LoRA):
    accelerate launch --config_file training/accelerate_ddp_4gpu.yaml \
        training/sft_warmup.py \
        --model_path models/Qwen2.5-3B-Instruct \
        --data_path data/sft_warmup.jsonl \
        --output_dir checkpoints/sft_warmup \
        --use_lora

Usage (Linux FSDP):
    accelerate launch --config_file training/accelerate_fsdp_4gpu.yaml \
        training/sft_warmup.py \
        --model_path models/Qwen2.5-3B-Instruct \
        --data_path data/sft_warmup.jsonl \
        --output_dir checkpoints/sft_warmup
"""

import argparse
import json
from pathlib import Path
from typing import Optional

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

from orchestrator_r1.prompts.system_prompt import SYSTEM_PROMPT


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",  type=str, required=True)
    parser.add_argument("--data_path",   type=str, required=True)
    parser.add_argument("--output_dir",  type=str, required=True)
    parser.add_argument("--per_device_batch_size", type=int, default=4)
    parser.add_argument("--grad_accum",  type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs",  type=int, default=3)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--use_lora",      action="store_true",
                        help="Use LoRA (required for DDP on Windows)")
    parser.add_argument("--lora_r",        type=int, default=64)
    parser.add_argument("--lora_alpha",    type=int, default=128)
    parser.add_argument("--lora_dropout",  type=float, default=0.05)
    return parser.parse_args()


def load_sft_data(data_path: str) -> Dataset:
    """
    Load SFT warmup data. Each line:
    {
        "input": "user task",
        "output": "<think>...</think>\n<call type=\"...\">...</call>\n<answer>...</answer>"
    }
    """
    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


def main():
    args = parse_args()

    model_path = str(Path(args.model_path).resolve())
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", local_files_only=True,
    )

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

    dataset = load_sft_data(args.data_path)

    def format_chat(example):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    dataset = dataset.map(format_chat)

    from trl import SFTConfig, SFTTrainer

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        max_length=args.max_seq_length,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        report_to="wandb",
        run_name="orchestrator-r1-sft-warmup",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"SFT warmup model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
