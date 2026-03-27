# OrchestratorR1

Reactive Multi-Agent Orchestration via Reinforcement Learning.

A small LLM (Qwen2.5-3B/7B) learns to **reactively** orchestrate 6 functionally specialized agents through GRPO reinforcement learning. Unlike open-loop approaches (Conductor), the orchestrator observes each agent's response before deciding the next action.

## Quick Start (4x RTX 3090, Linux/WSL2)

### 1. Clone & Enter

```bash
git clone https://github.com/Cayenne226/OrchestratorR1.git
cd OrchestratorR1
```

### 2. Create Environment

```bash
conda create -n orch python=3.10 -y
conda activate orch

# PyTorch 2.4 + CUDA 12.1
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

# Project dependencies
pip install -r requirements.txt

# flash-attn (optional, speeds up training ~20%)
pip install flash-attn --no-build-isolation
```

### 3. Download Models

```bash
# 3B (recommended to start, ~6GB)
huggingface-cli download Qwen/Qwen2.5-3B-Instruct \
    --local-dir models/Qwen2.5-3B-Instruct

# 7B (main model for paper, ~15GB)
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
    --local-dir models/Qwen2.5-7B-Instruct
```

If Hugging Face is slow, use modelscope:

```bash
pip install modelscope
modelscope download --model Qwen/Qwen2.5-3B-Instruct \
    --local_dir models/Qwen2.5-3B-Instruct
```

### 4. Set API Credentials

The agent pool calls external LLM APIs. Set your credentials:

```bash
export API_BASE="YOUR_API_BASE"
export API_KEY="YOUR_API_KEY"
```

### 5. Train

**Stage 0: SFT Warmup** — teach the model `<think>/<call>/<answer>` format:

```bash
# 3B LoRA on 3090x4 (~1-2h)
bash training/train_flex.sh --gpu 3090 --lora \
    MODEL_PATH=models/Qwen2.5-3B-Instruct \
    DATA_PATH=data/sft_warmup.jsonl \
    OUTPUT_DIR=checkpoints/sft_warmup_3b \
    --num_epochs 5

# Or use the dedicated SFT script:
accelerate launch --config_file training/accelerate_fsdp_4gpu.yaml \
    training/sft_warmup.py \
    --model_path models/Qwen2.5-3B-Instruct \
    --data_path data/sft_warmup.jsonl \
    --output_dir checkpoints/sft_warmup_3b \
    --num_epochs 5 --use_lora
```

**Stage 1: GRPO Training** — learn orchestration strategy via RL:

```bash
bash training/train_flex.sh --gpu 3090 --lora \
    MODEL_PATH=checkpoints/sft_warmup_3b \
    DATA_PATH=data/train_mixed.jsonl \
    OUTPUT_DIR=checkpoints/orch_grpo_3b_seed1
```

### 6. Evaluate

```bash
# Cheap worker pool (default)
python eval/eval_orchestrator.py \
    --model_path checkpoints/orch_grpo_3b_seed1 \
    --data_path data/test_qa.jsonl \
    --worker_pool cheap \
    --output eval/results/orch_grpo_3b_cheap_qa.json

# Matched worker pool (frontier models, for Conductor comparison)
python eval/eval_orchestrator.py \
    --model_path checkpoints/orch_grpo_3b_seed1 \
    --data_path data/test_qa.jsonl \
    --worker_pool matched \
    --output eval/results/orch_grpo_3b_matched_qa.json
```

---

## GPU Configurations

```bash
# 4x RTX 3090 + LoRA (3B, ~8GB/GPU)
bash training/train_flex.sh --gpu 3090 --lora

# 4x RTX 3090 + Full FT (3B, ~16GB/GPU)
bash training/train_flex.sh --gpu 3090

# 4x A100 + LoRA (7B, ~20GB/GPU)
bash training/train_flex.sh --gpu a100 --lora

# 4x A100 + Full FT (7B, ~50GB/GPU)
bash training/train_flex.sh --gpu a100

# 2x A800 + Full FT (3B)
bash training/train.sh --a800
```

## Architecture

```
User Query
    │
    ▼
┌──────────────────────┐
│  Orchestrator (7B)   │  ← Trained via SFT + GRPO
│  π_θ(a_t | s_t)     │
└──────┬───────────────┘
       │  <call type="agent_type">query</call>
       ▼
┌──────────────────────────────────────────────┐
│              Agent Pool (fixed, not trained)  │
│                                              │
│  refiner ─── decomposer ─── executor_cheap   │
│  executor_strong ─── critic ─── synthesizer  │
└──────────────────────────────────────────────┘
       │  <information>result</information>
       ▼
  Orchestrator observes result → decides next action
       │
       ▼
  <answer>final answer</answer>
```

### 6 Agent Roles

| Agent | Function | Cheap Pool Model | Matched Pool Model |
|-------|----------|------------------|--------------------|
| refiner | Clarify ambiguous input | gpt-4o-mini | claude-sonnet-4 |
| decomposer | Break complex tasks into subtasks | gpt-4o | gemini-2.5-pro |
| executor_cheap | Fast execution for simple tasks | gpt-4o-mini | qwen3-32b |
| executor_strong | High-quality execution | claude-sonnet-4-6 | gpt-5 |
| critic | Verify and critique results | gemini-2.5-flash | deepseek-r1-32b |
| synthesizer | Combine partial results | gpt-4o-mini | gemma3-27b |

### Dual Worker Pool Strategy

- **Train** with cheap pool only (cost-efficient)
- **Evaluate** on both cheap and matched (frontier) pools
- Core finding: orchestration strategies transfer zero-shot across worker capabilities

## Reward Function

```
R(τ) = R_outcome − α·C_cost − β·C_turns + γ·B_efficiency + R_format
```

| Term | Description | Default Weight |
|------|-------------|----------------|
| R_outcome | EM or F1 score | 1.0 |
| α · C_cost | API cost penalty | 0.3 |
| β · C_turns | Turn count penalty | 0.1 |
| γ · B_efficiency | Efficiency bonus (correct in fewer steps) | 0.15 |
| R_format | Format validity penalty (-1.0 if invalid) | — |

## Data

| File | Size | Description |
|------|------|-------------|
| `data/train_qa.jsonl` | 6,000 | NQ + TriviaQA + PopQA + HotpotQA + 2Wiki + MuSiQue |
| `data/train_code.jsonl` | 374 | MBPP train split |
| `data/train_mixed.jsonl` | 6,374 | QA + Code combined (main training set) |
| `data/sft_warmup.jsonl` | 200 | SFT format examples (6 agent types × 8 patterns) |
| `data/test_qa.jsonl` | 3,000 | QA test (500 per source) |
| `data/test_code.jsonl` | 664 | HumanEval (164) + MBPP (500) |
| `data/test_gpqa.jsonl` | 198 | GPQA Diamond |
| `data/test_livecode.jsonl` | 202 | LiveCodeBench |

## Project Structure

```
OrchestratorR1/
├── orchestrator_r1/           # Core library
│   ├── agent_pool/            # Agent registry, base agent, worker pools
│   ├── orchestrator/          # Generation loop (reactive + open-loop)
│   └── prompts/               # System prompts
├── training/                  # Training scripts & configs
│   ├── train.py               # GRPO training
│   ├── sft_warmup.py          # SFT warmup
│   ├── train_flex.sh          # Flexible launcher (3090/A100, LoRA/Full)
│   └── accelerate_*.yaml      # Accelerate configs
├── eval/                      # Evaluation & baselines
│   ├── eval_orchestrator.py   # Main eval (--worker_pool cheap|matched)
│   ├── run_direct_gpt4o.py    # Direct-GPT-4o baseline
│   ├── run_self_reflection.py # Self-Reflection 5-turn baseline
│   └── metrics.py             # EM, F1, Pass@1, GPQA accuracy
├── analysis/                  # Visualization scripts
│   ├── agent_distribution.py  # Agent call heatmap
│   ├── pareto_curve.py        # Cost vs F1 Pareto frontier
│   └── efficiency_grouping.py # Simple/Multi-hop/Code efficiency
├── data/                      # Datasets (JSONL)
├── data_process/              # Data preparation scripts
├── inference/                 # Inference scripts
├── models/                    # Downloaded model weights
├── checkpoints/               # Training outputs
└── plan/                      # NeurIPS 2026 project plans
```

## Baselines

| Method | Script | Description |
|--------|--------|-------------|
| Direct-GPT-4o | `eval/run_direct_gpt4o.py` | Single strong LLM, no orchestration |
| Self-Reflection | `eval/run_self_reflection.py` | Same LLM, 5 rounds of self-correction |
| ReAct | `eval/baselines.py --method react` | Zero-shot tool-use agent |
| Fixed-Pipeline | `eval/baselines.py --method fixed_pipeline` | All 6 agents in fixed order |
| Non-Reactive | `eval/run_ablation_openloop.py` | Open-loop (all calls planned upfront) |
