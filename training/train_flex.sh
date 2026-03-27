#!/bin/bash
# Orchestrator-R1 Training Script with GPU Selection
# Usage:
#   4x A100 80GB (LoRA):       bash training/train_flex.sh --gpu a100 --lora
#   4x A100 80GB (Full FT):    bash training/train_flex.sh --gpu a100
#   4x RTX 3090 (LoRA):        bash training/train_flex.sh --gpu 3090 --lora
#   4x RTX 3090 (Full FT):     bash training/train_flex.sh --gpu 3090

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$(dirname "$0")/..

# ── API credentials ────────────────────────────────────────────────────────
API_BASE=${API_BASE:-"http://35.220.164.252:3888/v1/"}
API_KEY=${API_KEY:-"sk-YlG8W7NPhqBSb3WIgsDJl7xekcBoUuAI8YE1kNtF3UY48ITM"}

# ── Parse arguments ────────────────────────────────────────────────────────
GPU_TYPE="3090"  # default
USE_LORA=false
EXTRA_ARGS=""

for arg in "$@"; do
    case "$arg" in
        --gpu)
            shift
            GPU_TYPE="$1"
            shift
            ;;
        --lora)
            USE_LORA=true
            shift
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $arg"
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
    MODEL_PATH=${MODEL_PATH:-"models/Qwen2.5-3B-Instruct"}
    if [ "$USE_LORA" = true ]; then
        echo "    Training: 3B LoRA"
        ACCEL_CONFIG="training/accelerate_fsdp_4gpu_lora.yaml"
        BATCH_SIZE=2
        GRAD_ACCUM=8
        LEARNING_RATE="5e-5"
        LORA_FLAG="--use_lora --lora_r 64 --lora_alpha 128"
    else
        echo "    Training: 3B Full Fine-Tuning"
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
    $LORA_FLAG \
    $EXTRA_ARGS
