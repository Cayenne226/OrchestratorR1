@echo off
REM Orchestrator-R1 GRPO Training — Full Parameter Mode (Windows 4x RTX 3090)
REM
REM Strategy: DeepSpeed ZeRO-2 + Gloo + Gradient Checkpointing
REM   - All 3B parameters trainable, ~16.5GB VRAM per GPU
REM   - ZeRO-2 shards optimizer states + gradients across 4 GPUs
REM   - Higher model capacity, but slower and more memory
REM
REM Usage:  training\train_full.bat
REM Extra:  training\train_full.bat --num_epochs 5 --max_samples 500

echo ============================================================
echo   Full-Param Mode: 4x RTX 3090, DeepSpeed ZeRO-2 + Gloo
echo   Qwen2.5-3B full fine-tuning
echo ============================================================

REM ── Environment ─────────────────────────────────────────────────────────────
set TOKENIZERS_PARALLELISM=false
set CUDA_VISIBLE_DEVICES=0,1,2,3
set ACCELERATE_TORCH_DISTRIBUTED_BACKEND=gloo
set OMP_NUM_THREADS=4

pushd %~dp0..
set PYTHONPATH=%CD%
popd

REM ── Defaults (override via env vars) ────────────────────────────────────────
if "%API_BASE%"=="" set API_BASE=YOUR_API_BASE
if "%API_KEY%"=="" set API_KEY=YOUR_API_KEY
if "%MODEL_PATH%"=="" set MODEL_PATH=models\Qwen2.5-3B-Instruct
if "%DATA_PATH%"=="" set DATA_PATH=data\train.jsonl
if "%OUTPUT_DIR%"=="" set OUTPUT_DIR=checkpoints\orchestrator_r1_full

REM ── Full-param training params ──────────────────────────────────────────────
REM Batch size 1 + grad_accum 16 to fit in 24GB with ZeRO-2
set BATCH_SIZE=1
set GRAD_ACCUM=16
set LEARNING_RATE=1e-6
set ACCEL_CONFIG=training\accelerate_ds_4gpu.yaml

accelerate launch ^
    --config_file "%ACCEL_CONFIG%" ^
    training\train.py ^
    --model_path "%MODEL_PATH%" ^
    --data_path "%DATA_PATH%" ^
    --output_dir "%OUTPUT_DIR%" ^
    --api_base "%API_BASE%" ^
    --api_key "%API_KEY%" ^
    --per_device_batch_size %BATCH_SIZE% ^
    --grad_accum %GRAD_ACCUM% ^
    --num_generations 8 ^
    --max_completion_length 512 ^
    --max_turns 6 ^
    --learning_rate %LEARNING_RATE% ^
    --num_epochs 3 ^
    --alpha 0.3 ^
    --beta 0.1 ^
    --gamma 0.15 ^
    --metric f1 ^
    --gradient_checkpointing ^
    %*
