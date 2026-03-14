#!/bin/bash
# Orchestrator-R1 Training Script (4x RTX 3090)

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=WARNING
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$(dirname "$0")/..

# ── Required: set your API credentials ────────────────────────────────────
API_BASE=${API_BASE:-"YOUR_API_BASE"}
API_KEY=${API_KEY:-"YOUR_API_KEY"}

# ── Paths ──────────────────────────────────────────────────────────────────
MODEL_PATH=${MODEL_PATH:-"models/Qwen2.5-3B-Instruct"}
DATA_PATH=${DATA_PATH:-"data/train.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"checkpoints/orchestrator_r1"}

accelerate launch \
    --config_file training/accelerate_fsdp_4gpu.yaml \
    training/train.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --api_base "$API_BASE" \
    --api_key "$API_KEY" \
    --per_device_batch_size 2 \
    --grad_accum 8 \
    --num_generations 8 \
    --max_new_tokens 512 \
    --max_turns 6 \
    --learning_rate 1e-6 \
    --num_epochs 3 \
    --alpha 0.3 \
    --beta 0.1 \
    --gamma 0.15 \
    --metric f1
