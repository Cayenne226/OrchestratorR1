@echo off
REM Prepare all data aligned with Router-R1
REM   Step 1: NQ 7k + HotpotQA 7k = 14k train
REM   Step 2: NQ 500 + HotpotQA 500 = 1k test
REM   Step 3: Auto-generate 100 SFT warmup traces via GPT-4o
REM
REM Usage:  data_process\prepare_data.bat
REM
REM Requires: API_KEY environment variable for SFT generation
REM   set API_KEY=sk-xxx
REM   data_process\prepare_data.bat

pushd %~dp0..
set PYTHONPATH=%CD%
popd

if "%HF_ENDPOINT%"=="" set HF_ENDPOINT=https://hf-mirror.com

echo ============================================================
echo   Preparing Router-R1 aligned datasets
echo   Train: NQ 7k + HotpotQA 7k = 14k
echo   Test:  NQ 500 + HotpotQA 500 = 1k
echo   SFT:   100 auto-generated traces via GPT-4o
echo ============================================================

echo.
echo [1/3] Preparing training data (streaming from HuggingFace)...
python data_process/prepare_data.py --preset router_r1_train --output data/train.jsonl
if errorlevel 1 (
    echo FAILED to prepare training data
    exit /b 1
)

echo.
echo [2/3] Preparing test data...
python data_process/prepare_data.py --preset router_r1_test --output data/test.jsonl
if errorlevel 1 (
    echo FAILED to prepare test data
    exit /b 1
)

echo.
echo [3/3] Generating SFT warmup traces via GPT-4o (100 samples)...
if "%API_KEY%"=="" (
    echo ERROR: API_KEY not set. Run: set API_KEY=sk-xxx
    exit /b 1
)
if "%API_BASE%"=="" set API_BASE=https://api.openai.com/v1

python data_process/prepare_sft.py ^
    --train_data data/train.jsonl ^
    --output data/sft_warmup.jsonl ^
    --api_base "%API_BASE%" ^
    --api_key "%API_KEY%" ^
    --num_samples 100
if errorlevel 1 (
    echo FAILED to generate SFT data
    exit /b 1
)

echo.
echo ============================================================
echo   All data prepared successfully!
echo   data\train.jsonl       - GRPO training (14k)
echo   data\test.jsonl        - Evaluation (1k)
echo   data\sft_warmup.jsonl  - SFT warmup (100 auto-generated)
echo ============================================================
