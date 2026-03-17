#!/bin/bash
# SFT Warmup Script
# Usage:
#   2x A800 80GB (Linux, 7B):      bash training/sft_warmup.sh --a800
#   4x RTX 3090 (Linux/WSL2, 3B):  bash training/sft_warmup.sh
#   4x RTX 3090 (Windows, 3B):     bash training/sft_warmup.sh --windows

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$(dirname "$0")/..

A800_MODE=false
WINDOWS_MODE=false
for arg in "$@"; do
    [ "$arg" = "--a800" ]    && A800_MODE=true
    [ "$arg" = "--windows" ] && WINDOWS_MODE=true
done

if [ "$A800_MODE" = true ]; then
    echo "=== A800 Mode: 2x A800 80GB, FSDP, 7B full fine-tuning ==="
    export CUDA_VISIBLE_DEVICES=0,1
    export NCCL_DEBUG=WARNING
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    ACCEL_CONFIG="training/accelerate_fsdp_2gpu.yaml"
    MODEL_PATH=${MODEL_PATH:-"models/Qwen2.5-7B-Instruct"}
    BATCH_SIZE=1
    LORA_FLAG=""

elif [ "$WINDOWS_MODE" = true ]; then
    echo "=== Windows Mode: 4x RTX 3090, DDP + Gloo + LoRA ==="
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export ACCELERATE_TORCH_DISTRIBUTED_BACKEND=gloo
    ACCEL_CONFIG="training/accelerate_ddp_4gpu.yaml"
    MODEL_PATH=${MODEL_PATH:-"models/Qwen2.5-3B-Instruct"}
    BATCH_SIZE=4
    LORA_FLAG="--use_lora"

else
    echo "=== Linux Mode: 4x RTX 3090, FSDP, 3B ==="
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export NCCL_DEBUG=WARNING
    ACCEL_CONFIG="training/accelerate_fsdp_4gpu.yaml"
    MODEL_PATH=${MODEL_PATH:-"models/Qwen2.5-3B-Instruct"}
    BATCH_SIZE=4
    LORA_FLAG=""
fi

DATA_PATH=${DATA_PATH:-"data/sft_warmup.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"checkpoints/sft_warmup"}

accelerate launch \
    --config_file "$ACCEL_CONFIG" \
    training/sft_warmup.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --per_device_batch_size "$BATCH_SIZE" \
    --grad_accum 4 \
    --learning_rate 2e-5 \
    --num_epochs 3 \
    $LORA_FLAG
