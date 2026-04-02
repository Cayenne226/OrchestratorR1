#!/bin/bash
# Orchestrator-R1 GRPO Training Script with GPU Selection
# Usage:
#   4x RTX 3090 (7B LoRA, recommended):  bash training/train_flex.sh --lora
#   4x RTX 3090 (3B Full FT):            bash training/train_flex.sh
#   4x A100 80GB (7B LoRA):              bash training/train_flex.sh --gpu a100 --lora
#   4x A100 80GB (7B Full FT):           bash training/train_flex.sh --gpu a100

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$(dirname "$0")/..

# ── API credentials ────────────────────────────────────────────────────────
API_BASE=${API_BASE:}
API_KEY=${API_KEY:}

# ── Parse arguments ────────────────────────────────────────────────────────
GPU_TYPE="3090"  # default
USE_LORA=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            GPU_TYPE="$2"
            shift 2
            ;;
        --lora)
            USE_LORA=true
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# ── Config per GPU type ────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=WARNING
export OMP_NUM_THREADS=4

if [ "$GPU_TYPE" = "a100" ]; then
    echo "=== A100 Mode: 4x A100 80GB ==="
    MODEL_PATH=${MODEL_PATH:-"models/Qwen2.5-7B-Instruct"}
    if [ "$USE_LORA" = true ]; then
        echo "    Training: 7B LoRA"
        ACCEL_CONFIG="training/accelerate_fsdp_4gpu_lora.yaml"
        BATCH_SIZE=4
        GRAD_ACCUM=4
        LEARNING_RATE="5e-5"
        LORA_FLAG="--use_lora --lora_r 64 --lora_alpha 128"
    else
        echo "    Training: 7B Full Fine-Tuning"
        ACCEL_CONFIG="training/accelerate_fsdp_4gpu.yaml"
        BATCH_SIZE=2
        GRAD_ACCUM=8
        LEARNING_RATE="1e-6"
        LORA_FLAG=""
    fi
else
    echo "=== RTX 3090 Mode: 4x RTX 3090 24GB ==="
    if [ "$USE_LORA" = true ]; then
        echo "    Training: 7B LoRA"
        MODEL_PATH=${MODEL_PATH:-"models/Qwen2.5-7B-Instruct"}
        ACCEL_CONFIG="training/accelerate_fsdp_4gpu_lora.yaml"
        BATCH_SIZE=2
        GRAD_ACCUM=8
        LEARNING_RATE="5e-5"
        LORA_FLAG="--use_lora --lora_r 64 --lora_alpha 128"
    else
        echo "    Training: 3B Full Fine-Tuning"
        MODEL_PATH=${MODEL_PATH:-"models/Qwen2.5-3B-Instruct"}
        ACCEL_CONFIG="training/accelerate_fsdp_4gpu.yaml"
        BATCH_SIZE=2
        GRAD_ACCUM=8
        LEARNING_RATE="1e-6"
        LORA_FLAG=""
    fi
fi

DATA_PATH=${DATA_PATH:-"data/train_mixed.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"checkpoints/orchestrator_r1_${GPU_TYPE}_$(date +%m%d)"}

echo "Model: $MODEL_PATH"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "Accelerate config: $ACCEL_CONFIG"
echo ""

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
    --max_completion_length 512 \
    --max_turns 6 \
    --learning_rate "$LEARNING_RATE" \
    --num_epochs 3 \
    --alpha 0.3 \
    --beta 0.1 \
    --gamma 0.15 \
    --metric f1 \
    $LORA_FLAG
