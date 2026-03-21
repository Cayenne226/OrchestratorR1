@echo off
REM Orchestrator-R1 GRPO Training — LoRA Mode (Windows 4x RTX 3090)
REM
REM Strategy: DDP + Gloo + LoRA
REM   - Only ~1%% trainable parameters, ~8GB VRAM per GPU
REM   - Fast training, lower memory, good for quick experiments
REM   - Trades some model capacity for memory efficiency
REM
REM Usage:  training\train_lora.bat
REM Extra:  training\train_lora.bat --num_epochs 5 --max_samples 500

echo ============================================================
echo   LoRA Mode: 4x RTX 3090, DDP + Gloo, Qwen2.5-3B + LoRA
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
if "%OUTPUT_DIR%"=="" set OUTPUT_DIR=checkpoints\orchestrator_r1_lora

REM ── LoRA training params ────────────────────────────────────────────────────
set BATCH_SIZE=2
set GRAD_ACCUM=8
set LEARNING_RATE=5e-5
set ACCEL_CONFIG=training\accelerate_ddp_4gpu.yaml

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
    --use_lora ^
    %*
