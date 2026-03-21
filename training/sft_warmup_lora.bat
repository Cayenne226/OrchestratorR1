@echo off
REM SFT Warmup — LoRA Mode (Windows 4x RTX 3090)
REM Usage:  training\sft_warmup_lora.bat

echo ============================================================
echo   SFT Warmup LoRA: 4x RTX 3090, DDP + Gloo + LoRA
echo ============================================================

set TOKENIZERS_PARALLELISM=false
set CUDA_VISIBLE_DEVICES=0,1,2,3
set ACCELERATE_TORCH_DISTRIBUTED_BACKEND=gloo
set OMP_NUM_THREADS=4

pushd %~dp0..
set PYTHONPATH=%CD%
popd

if "%MODEL_PATH%"=="" set MODEL_PATH=models\Qwen2.5-3B-Instruct
if "%DATA_PATH%"=="" set DATA_PATH=data\sft_warmup.jsonl
if "%OUTPUT_DIR%"=="" set OUTPUT_DIR=checkpoints\sft_warmup_lora

accelerate launch ^
    --config_file training\accelerate_ddp_4gpu.yaml ^
    training\sft_warmup.py ^
    --model_path "%MODEL_PATH%" ^
    --data_path "%DATA_PATH%" ^
    --output_dir "%OUTPUT_DIR%" ^
    --per_device_batch_size 4 ^
    --grad_accum 4 ^
    --learning_rate 2e-5 ^
    --num_epochs 3 ^
    --use_lora ^
    %*
