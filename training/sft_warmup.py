"""
SFT Warmup: Teach the Orchestrator model the <call>/<answer> tag format.

Uses a small dataset of hand-crafted examples covering all 6 Agent types,
so that GRPO training starts with a model that can already produce valid tags.

Usage (Windows LoRA):      training\\sft_warmup_lora.bat
Usage (Windows Full-param): training\\sft_warmup_full.bat
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

from orchestrator_r1.prompts.system_prompt import SYSTEM_PROMPT

IS_WINDOWS = sys.platform == "win32"


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
                        help="Use LoRA (DDP mode)")
    parser.add_argument("--use_qlora",     action="store_true",
                        help="Use QLoRA (4-bit quantized base + LoRA, DDP mode)")
    parser.add_argument("--lora_r",        type=int, default=64)
    parser.add_argument("--lora_alpha",    type=int, default=128)
    parser.add_argument("--lora_dropout",  type=float, default=0.05)
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing (recommended for full-param)")
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

    import torch

    quantization_config = None
    if args.use_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
    )

    if args.gradient_checkpointing:
        print("Gradient checkpointing will be enabled by SFTConfig")

    if args.use_qlora or args.use_lora:
        if args.use_qlora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=args.gradient_checkpointing
            )
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
    else:
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Full-param training: {trainable:,} / {total_params:,} parameters")

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

    # Windows: dataloader workers must be 0 (spawn-based multiprocessing causes
    # CUDA re-init errors); Linux can use 2 for prefetching.
    num_workers = 0 if IS_WINDOWS else 2

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        max_seq_length=args.max_seq_length,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=num_workers,
        logging_steps=5,
        save_strategy="no",           # disable mid-training saves (7B = 14GB per save)
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
