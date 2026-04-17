#!/bin/bash
# SFT Warmup Script
# Usage:
#   4x RTX 3090, 7B LoRA (recommended): bash training/sft_warmup.sh --lora
#   4x RTX 3090, 3B Full FT:            bash training/sft_warmup.sh
#   2x A800 80GB, 7B Full FT:           bash training/sft_warmup.sh --a800
#   4x RTX 3090, Windows LoRA:          bash training/sft_warmup.sh --windows

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$(dirname "$0")/..

A800_MODE=false
WINDOWS_MODE=false
USE_LORA=false
for arg in "$@"; do
    [ "$arg" = "--a800" ]    && A800_MODE=true
    [ "$arg" = "--windows" ] && WINDOWS_MODE=true
    [ "$arg" = "--lora" ]    && USE_LORA=true
done

if [ "$A800_MODE" = true ]; then
    echo "=== A800 Mode: 2x A800 80GB, FSDP, 7B full fine-tuning ==="
    export CUDA_VISIBLE_DEVICES=0,1
    export NCCL_DEBUG=WARNING
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    ACCEL_CONFIG="training/accelerate_fsdp_2gpu.yaml"
    MODEL_PATH=${MODEL_PATH:-"models/Qwen2.5-7B-Instruct"}
    BATCH_SIZE=4
    GRAD_ACCUM=4
    LORA_FLAG=""
    GC_FLAG=""

elif [ "$WINDOWS_MODE" = true ]; then
    echo "=== Windows Mode: 4x RTX 3090, DDP + Gloo + LoRA ==="
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export ACCELERATE_TORCH_DISTRIBUTED_BACKEND=gloo
    ACCEL_CONFIG="training/accelerate_ddp_4gpu.yaml"
    MODEL_PATH=${MODEL_PATH:-"models/Qwen2.5-3B-Instruct"}
    BATCH_SIZE=4
    GRAD_ACCUM=4
    LORA_FLAG="--use_lora"
    GC_FLAG=""

elif [ "$USE_LORA" = true ]; then
    echo "=== Linux QLoRA Mode: 4x RTX 3090, DDP + 4-bit + LoRA, 7B ==="
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export NCCL_DEBUG=WARNING
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    ACCEL_CONFIG="training/accelerate_ddp_4gpu.yaml"
    MODEL_PATH=${MODEL_PATH:-"models/Qwen2.5-7B-Instruct"}
    BATCH_SIZE=2
    GRAD_ACCUM=8
    LORA_FLAG="--use_qlora --lora_r 32 --lora_alpha 64"
    GC_FLAG="--gradient_checkpointing"

else
    echo "=== Linux Mode: 4x RTX 3090, FSDP, 3B Full FT ==="
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export NCCL_DEBUG=WARNING
    ACCEL_CONFIG="training/accelerate_fsdp_4gpu.yaml"
    MODEL_PATH=${MODEL_PATH:-"models/Qwen2.5-3B-Instruct"}
    BATCH_SIZE=4
    GRAD_ACCUM=4
    LORA_FLAG=""
    GC_FLAG=""
fi

DATA_PATH=${DATA_PATH:-"data/sft_warmup.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"checkpoints/sft_warmup"}

# Pre-flight: check GPU memory availability
echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | while IFS=', ' read -r idx used total; do
    pct=$((used * 100 / total))
    if [ "$pct" -gt 20 ]; then
        echo "WARNING: GPU $idx has ${used}MiB / ${total}MiB used (${pct}%). Kill stale processes first!"
    else
        echo "GPU $idx: ${used}MiB / ${total}MiB (OK)"
    fi
done

echo "Model: $MODEL_PATH"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo ""

accelerate launch \
    --config_file "$ACCEL_CONFIG" \
    training/sft_warmup.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --per_device_batch_size "$BATCH_SIZE" \
    --grad_accum "$GRAD_ACCUM" \
    --learning_rate 2e-5 \
    --num_epochs 3 \
    --max_seq_length 1024 \
    $LORA_FLAG \
    $GC_FLAG
