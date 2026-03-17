#!/bin/bash
# Orchestrator-R1 Training Script
# Usage:
#   2x A800 80GB (Linux, 7B full FT):   bash training/train.sh --a800
#   4x RTX 3090 (Linux/WSL2, 3B):       bash training/train.sh
#   4x RTX 3090 (Windows, 3B+LoRA):     bash training/train.sh --windows

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$(dirname "$0")/..

# ── Required: set your API credentials ────────────────────────────────────
API_BASE=${API_BASE:-"YOUR_API_BASE"}
API_KEY=${API_KEY:-"YOUR_API_KEY"}

# ── Parse mode flags ──────────────────────────────────────────────────────
A800_MODE=false
WINDOWS_MODE=false
for arg in "$@"; do
    [ "$arg" = "--a800" ]    && A800_MODE=true
    [ "$arg" = "--windows" ] && WINDOWS_MODE=true
done

# ── Config per mode ───────────────────────────────────────────────────────
if [ "$A800_MODE" = true ]; then
    echo "=== A800 Mode: 2x A800 80GB, FSDP + NCCL, 3B full fine-tuning ==="
    export CUDA_VISIBLE_DEVICES=0,1
    export NCCL_DEBUG=WARNING
    export OMP_NUM_THREADS=8
    ACCEL_CONFIG="training/accelerate_fsdp_2gpu.yaml"
    MODEL_PATH=${MODEL_PATH:-"models/Qwen2.5-3B-Instruct"}
    BATCH_SIZE=4
    GRAD_ACCUM=4
    LEARNING_RATE="1e-6"
    LORA_FLAG=""

elif [ "$WINDOWS_MODE" = true ]; then
    echo "=== Windows Mode: 4x RTX 3090, DDP + Gloo + LoRA, 3B ==="
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export ACCELERATE_TORCH_DISTRIBUTED_BACKEND=gloo
    export OMP_NUM_THREADS=4
    ACCEL_CONFIG="training/accelerate_ddp_4gpu.yaml"
    MODEL_PATH=${MODEL_PATH:-"models/Qwen2.5-3B-Instruct"}
    BATCH_SIZE=2
    GRAD_ACCUM=8
    LEARNING_RATE="5e-5"
    LORA_FLAG="--use_lora"

else
    echo "=== Linux Mode: 4x RTX 3090, FSDP + NCCL, 3B ==="
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export NCCL_DEBUG=WARNING
    export OMP_NUM_THREADS=4
    ACCEL_CONFIG="training/accelerate_fsdp_4gpu.yaml"
    MODEL_PATH=${MODEL_PATH:-"models/Qwen2.5-3B-Instruct"}
    BATCH_SIZE=2
    GRAD_ACCUM=8
    LEARNING_RATE="1e-6"
    LORA_FLAG=""
fi

DATA_PATH=${DATA_PATH:-"data/train.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"checkpoints/orchestrator_r1"}

accelerate launch \
    --config_file "$ACCEL_CONFIG" \
    training/train.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --api_base "$API_BASE" \
    --api_key "$API_KEY" \
    --per_device_batch_size "$BATCH_SIZE" \
    --grad_accum "$GRAD_ACCUM" \
    --num_generations 8 \
    --max_new_tokens 512 \
    --max_turns 6 \
    --learning_rate "$LEARNING_RATE" \
    --num_epochs 3 \
    --alpha 0.3 \
    --beta 0.1 \
    --gamma 0.15 \
    --metric f1 \
    $LORA_FLAG
