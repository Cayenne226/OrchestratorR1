#!/bin/bash
# Orchestrator-R1 Training Script
# Usage:
#   2x H200 141GB (Linux, FSDP, 7B full FT): bash training/train.sh --h200
#   2x A800 80GB (Linux, full FT):           bash training/train.sh --a800
#   4x RTX 3090 (Linux, FSDP full FT):       bash training/train.sh
#   4x RTX 3090 (Windows, LoRA):             training\train_lora.bat
#   4x RTX 3090 (Windows, ZeRO-2 full FT):   training\train_full.bat

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$(dirname "$0")/..

# ── Required: set your API credentials ────────────────────────────────────
API_BASE=${API_BASE:-"YOUR_API_BASE"}
API_KEY=${API_KEY:-"YOUR_API_KEY"}

# ── Parse mode flags, collect extra args for train.py ─────────────────────
A800_MODE=false
H200_MODE=false
WINDOWS_MODE=false
EXTRA_ARGS=""
skip_next=false
for arg in "$@"; do
    if $skip_next; then skip_next=false; EXTRA_ARGS="$EXTRA_ARGS $arg"; continue; fi
    case "$arg" in
        --a800)    A800_MODE=true ;;
        --h200)    H200_MODE=true ;;
        --windows) WINDOWS_MODE=true ;;
        --max_samples|--num_epochs|--alpha|--beta|--gamma|--metric|--max_turns)
                   EXTRA_ARGS="$EXTRA_ARGS $arg"; skip_next=true ;;
        --max_samples=*|--num_epochs=*) EXTRA_ARGS="$EXTRA_ARGS $arg" ;;
        *) ;;
    esac
done

# ── Config per mode ───────────────────────────────────────────────────────
if [ "$H200_MODE" = true ]; then
    echo "=== H200 Mode: 2x H200 141GB, FSDP + NCCL, 7B full fine-tuning ==="
    export CUDA_VISIBLE_DEVICES=0,1
    export NCCL_DEBUG=WARNING
    export OMP_NUM_THREADS=8
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    ACCEL_CONFIG="training/accelerate_fsdp_2gpu.yaml"
    MODEL_PATH=${MODEL_PATH:-"models/Qwen2.5-7B-Instruct"}
    BATCH_SIZE=8
    GRAD_ACCUM=2
    LEARNING_RATE="1e-6"
    NUM_GENERATIONS=16
    MAX_COMPLETION_LENGTH=2048
    MAX_TURNS=6
    LORA_FLAG=""

elif [ "$A800_MODE" = true ]; then
    echo "=== A800 Mode: 2x A800 80GB, DDP + NCCL, 3B full fine-tuning ==="
    export CUDA_VISIBLE_DEVICES=0,1
    export NCCL_DEBUG=WARNING
    export OMP_NUM_THREADS=8
    ACCEL_CONFIG="training/accelerate_ddp_2gpu.yaml"
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

# Defaults (overridden by H200 mode above)
NUM_GENERATIONS=${NUM_GENERATIONS:-8}
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-512}
MAX_TURNS=${MAX_TURNS:-6}

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
    --num_generations "$NUM_GENERATIONS" \
    --max_completion_length "$MAX_COMPLETION_LENGTH" \
    --max_turns "$MAX_TURNS" \
    --learning_rate "$LEARNING_RATE" \
    --num_epochs 3 \
    --alpha 0.3 \
    --beta 0.1 \
    --gamma 0.15 \
    --metric f1 \
    $LORA_FLAG \
    $EXTRA_ARGS
