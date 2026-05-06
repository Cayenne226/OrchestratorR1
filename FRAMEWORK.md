# OrchestratorR1 — 框架总览

## 系统架构

```
用户输入（简单 / 模糊 / 复杂）
           │
           ▼
  ┌────────────────────────┐
  │    Orchestrator π_θ    │  ← 训练的唯一模型（Qwen2.5-3B/7B）
  │    输出 <call> 序列     │
  │    自适应选择编排路径    │
  └────────────────────────┘
           │ 按需调用（0~6 次，由 RL 涌现决定）
           │ 上下文超预算时触发压缩（保留首尾，摘要中间）
           ▼
  ┌──────────────────────────────────────────────────┐
  │                   Agent Pool P                   │
  │  executor (strong/weak) │ decomposer             │
  │  critic                 │ synthesizer            │
  │  （固定 API 端点，不参与训练）                      │
  │  Worker Pool: Claude Sonnet 4 / Gemini 2.5 Pro   │
  │               / GPT-4o                           │
  └──────────────────────────────────────────────────┘
           │ 返回 <information>
           ▼
  ┌────────────────────────┐
  │     <answer> 输出       │
  └────────────────────────┘
           │
           ▼
  ┌────────────────────────┐
  │    Reward R(τ)         │
  │    = R_outcome         │
  │    − α·C_cost          │
  │    − β·C_turns         │
  │    + γ·B_efficiency    │
  └────────────────────────┘
           │ GRPO 策略梯度更新
           ▼
      Orchestrator π_θ 优化
```

### 设计原则

**角色是人工先验，调用策略由 RL 涌现。** 四个角色（executor / decomposer / critic / synthesizer）编码了不同的认知功能，这是人工设计的。但何时调用每个角色、以什么顺序、用什么子查询、是否调用 critic 进行验证，都完全由 RL 奖励景观涌现。

### 涌现的自适应路径

```
简单事实题:    think → executor → answer                                (1 轮)
多跳推理:      think → decomposer → executor×N → synthesizer → answer   (3~5 轮)
复杂代码:      think → decomposer → executor(strong)×N → critic → answer(4~6 轮)
```

路径不是人工设计的 if-else，而是 RL 训练后模型自发涌现的行为。
---

## 目录结构

```
OrchestratorR1/
│
├── SKILL.md                          # 核心思想、形式化定义、实验设计
├── FRAMEWORK.md                      # 框架总览（本文件）
├── requirements.txt
│
├── orchestrator_r1/                  # 核心代码包
│   │
│   ├── prompts/
│   │   └── system_prompt.py         # Orchestrator 系统提示词和输出格式定义
│   │
│   ├── agent_pool/
│   │   ├── base_agent.py            # Agent 基类：API 调用、重试、计费
│   │   └── agent_registry.py       # 4 种 Agent 注册 + 统一调度接口
│   │
│   └── orchestrator/
│       ├── parser.py                # 解析 <call>/<answer> 标签，格式验证
│       ├── reward.py                # 统一奖励函数 R(τ) + EM/F1 计算
│       ├── context_manager.py      # 上下文压缩：budget 检查 + 中间 information 摘要
│       └── generation.py           # 多轮生成主循环（推理 + 训练 rollout + 上下文压缩）
│
├── training/
│   ├── train.py                     # GRPO 训练入口（trl GRPOTrainer，支持渐进式训练）
│   ├── sft_warmup.py               # SFT 热身：教模型学会 <call> 标签格式
│   ├── train.sh                     # 4×RTX3090 启动脚本
│   └── accelerate_fsdp_4gpu.yaml   # FSDP ZeRO-2 配置（bf16，4卡）
│
├── data_process/
│   ├── prepare_data.py              # 从 HuggingFace 加载 QA 数据集
│   ├── prepare_code.py             # 加载 HumanEval/MBPP 代码数据集
│   └── prepare_sft.py              # 生成 SFT 热身数据（覆盖全部 Agent 类型）
│
├── data/
│   ├── train.jsonl                  # 训练集
│   ├── test.jsonl                   # 测试集
│   └── sft_warmup.jsonl            # SFT 热身示例
│
├── eval/
│   ├── eval_orchestrator.py         # 主评估脚本（EM/F1/成本/轮数 + 分组统计）
│   ├── eval_baselines.py           # 基线评估（Router-R1 / Direct-Strong / ReAct）
│   ├── compare.py                   # 对比多个结果 JSON → 生成表格和图
│   ├── metrics.py                   # EM / F1 / Pass@1 计算
│   └── results/                     # 评估结果 JSON
│
├── inference/
│   └── infer.py                     # 本地单条推理测试
│
├── analysis/
│   ├── agent_distribution.py       # 生成 Agent 调用分布热力图
│   ├── pareto_curve.py             # 生成效率-质量帕累托曲线
│   └── training_dynamics.py        # 训练过程行为变化分析
│
└── test_local.py                    # 无 GPU 本地单元测试
```

---

## 数据流

### 训练数据准备

```
HuggingFace 数据集
│
├─ QA: NQ / TriviaQA / PopQA / HotpotQA / 2WikiMultihop / MuSiQue
│      │ prepare_data.py
│      ▼
│  data/train.jsonl  (每行: {"input": ..., "answer": ..., "source": ...})
│
├─ 代码: HumanEval / MBPP
│      │ prepare_code.py
│      ▼
│  data/train_code.jsonl  (每行: {"input": ..., "answer": ..., "test_cases": ...})
│
└─ SFT 热身
       │ prepare_sft.py
       ▼
   data/sft_warmup.jsonl  (50~100 条覆盖全部 4 种 Agent 类型的示例)
```

### 训练流程

```
阶段 0: SFT 热身
   data/sft_warmup.jsonl → sft_warmup.py → checkpoints/sft_warmup/

阶段 1: 渐进式 GRPO — Stage 1 (max_turns=2)
   data/train_simple_qa.jsonl（简单 QA 子集）
       → train.py --progressive_stage 1 --max_turns 2
       → 学会基本的 call 格式和单步路由
       → checkpoints/orch_grpo_stage1/

阶段 2: 渐进式 GRPO — Stage 2 (max_turns=4)
   data/train_multihop.jsonl（加入多跳推理数据）
       → train.py --progressive_stage 2 --max_turns 4
       → 学会 decompose → execute → synthesize
       → checkpoints/orch_grpo_stage2/

阶段 3: 渐进式 GRPO — Stage 3 (max_turns=6)
   data/train.jsonl + train_code.jsonl（全量混合数据）
       → train.py --progressive_stage 3 --max_turns 6
       → 学会 critic 验证、错误恢复、长链编排
       → checkpoints/orchestrator_r1/final/

每条 prompt 生成 G=8 条轨迹
实际调用 Agent API（strong worker pool）获取 <information>
上下文超预算时触发压缩（保留任务描述+最近交互，摘要中间 information）
compute_reward 计算奖励 → GRPO 策略更新
```

### 评估流程

```
checkpoints/orchestrator_r1/final/
       │ eval_orchestrator.py
       ▼
eval/results/orchestrator_r1_grpo.json
{"summary": {em, f1, avg_cost, avg_turns, per_source: {...}}, "results": [...]}
       │ compare.py
       ▼
对比表格 + 帕累托曲线图
```

---

## 关键模块说明

### `orchestrator_r1/orchestrator/generation.py`

多轮生成主循环，是系统的核心调度器：

1. 构建带系统提示的 prompt（定义 Agent 池和输出格式）
2. 调用 Orchestrator 模型生成下一步输出
3. 解析 `<call type="X">query</call>` 标签
4. 通过 AgentRegistry 调用对应 Agent API
5. 将 Agent 返回以 `<information>` 注入上下文
6. **上下文预算检查**：如果总 token 数超过阈值（max_prompt_length × 80%），触发压缩——保留系统提示 + 原始 query + 最近 1-2 轮完整交互，对中间 `<information>` 块做截断或摘要
7. 循环步骤 2~6 直到出现 `<answer>` 或达到 max_turns
8. 返回完整轨迹 GenerationResult（含 answer、agent_calls、cost、turns）

**训练和推理共用同一主循环**，区别仅在于 GRPO 需要采样 G=8 条不同轨迹。

### `orchestrator_r1/orchestrator/context_manager.py`

上下文压缩模块：

- `check_budget(context, max_tokens)` → 判断是否超预算
- `compress(context, keep_system=True, keep_recent_turns=2)` → 保留首尾，对中间 information 块做截断（默认）或摘要
- 压缩策略：优先截断（快速、确定性），可选摘要（调用 executor_cheap 做一句话总结）

### `orchestrator_r1/agent_pool/agent_registry.py`

Agent 注册表，提供统一的 `dispatch(agent_type, query) → (response, cost)` 接口：

- 4 种 Agent 角色：executor（支持 strong/weak 档位）、decomposer、critic、synthesizer
- 每种 Agent 有独立的系统提示（定义其角色）
- 每种 Agent 映射到 strong worker pool 中的 LLM API（Claude Sonnet 4 / Gemini 2.5 Pro / GPT-4o）
- executor 支持通过 `<call type="executor" tier="strong">` 或 `<call type="executor" tier="weak">` 指定模型档位

### `orchestrator_r1/orchestrator/reward.py`

统一奖励函数 R(τ)：

```
R = R_outcome − α·C_cost − β·C_turns + γ·B_efficiency + R_format
```

关键设计：
- R_format = -1.0 对无效格式直接施加强惩罚，加速格式学习
- B_efficiency 仅在高质量+少轮次时触发，避免简单题也走完整流程
- C_cost 使用固定上限 C_max 归一化，避免成本项主导梯度

### `orchestrator_r1/orchestrator/parser.py`

标签解析和格式验证：
- 解析 `<call type="X">query</call>`，提取 agent_type 和 query
- 解析 `<answer>text</answer>`，提取最终答案
- 验证 agent_type 是否属于合法 Agent 池
- 检测标签配对、缺失等格式错误
- 格式错误信息传递给 reward.py 用于 R_format 计算

---

## 模块间依赖关系

```
training/train.py
  ├── generation.py          ← rollout 生成 G=8 条轨迹
  │     ├── parser.py        ← 解析输出标签
  │     ├── context_manager.py ← 上下文压缩
  │     └── agent_registry.py ← 调用 Agent API
  │           └── base_agent.py
  └── reward.py              ← 计算每条轨迹的 R(τ)

eval/eval_orchestrator.py
  ├── generation.py
  ├── reward.py              ← compute_em / compute_f1
  └── metrics.py             ← Pass@1 (代码任务)

eval/compare.py
  └── 读取多个 results/*.json → 生成对比表格

inference/infer.py
  └── generation.py

analysis/*.py
  └── 读取 results/*.json → 生成论文图表
```

---

## 训练配置（4×RTX 3090）

### FSDP 配置

```yaml
# training/accelerate_fsdp_4gpu.yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_sharding_strategy: SHARD_GRAD_OP   # ZeRO-2：分片梯度+优化器
  fsdp_transformer_layer_cls_to_wrap: Qwen2DecoderLayer
  fsdp_backward_prefetch: BACKWARD_PRE
mixed_precision: bf16
num_processes: 4
```

### 显存估算（单卡 24GB）

| 组件 | Qwen2.5-3B | Qwen2.5-7B |
|---|---|---|
| 模型权重 | ~6GB | ~14GB |
| 优化器（/4 卡）| ~1.5GB | ~3.5GB |
| 梯度（/4 卡）| ~0.75GB | ~1.75GB |
| GRPO rollout (G=8) | ~12GB | ~3GB* |
| 总计 | ~20GB ✓ | ~22GB ✓ |

*7B 模型需减小 G 至 4 或减小 max_new_tokens

> **注意**：渐进式训练的前两阶段（max_turns=2/4）显存压力更小，可以用更大的 batch size 加速。

### 启动命令

```bash
# SFT 热身
accelerate launch --config_file training/accelerate_fsdp_4gpu.yaml \
    training/sft_warmup.py \
    --model_path models/Qwen2.5-3B-Instruct \
    --data_path data/sft_warmup.jsonl \
    --output_dir checkpoints/sft_warmup

# 渐进式 GRPO — Stage 1 (max_turns=2, 简单 QA)
accelerate launch --config_file training/accelerate_fsdp_4gpu.yaml \
    training/train.py \
    --model_path checkpoints/sft_warmup \
    --data_path data/train_simple_qa.jsonl \
    --api_base YOUR_API_BASE \
    --api_key YOUR_API_KEY \
    --output_dir checkpoints/orch_grpo_stage1 \
    --max_turns 2 --progressive_stage 1 \
    --alpha 0.3 --beta 0.1 --gamma 0.15

# 渐进式 GRPO — Stage 2 (max_turns=4, 加入多跳)
accelerate launch --config_file training/accelerate_fsdp_4gpu.yaml \
    training/train.py \
    --model_path checkpoints/orch_grpo_stage1/final \
    --data_path data/train_multihop.jsonl \
    --api_base YOUR_API_BASE \
    --api_key YOUR_API_KEY \
    --output_dir checkpoints/orch_grpo_stage2 \
    --max_turns 4 --progressive_stage 2 \
    --alpha 0.3 --beta 0.1 --gamma 0.15

# 渐进式 GRPO — Stage 3 (max_turns=6, 全量混合)
accelerate launch --config_file training/accelerate_fsdp_4gpu.yaml \
    training/train.py \
    --model_path checkpoints/orch_grpo_stage2/final \
    --data_path data/train.jsonl \
    --api_base YOUR_API_BASE \
    --api_key YOUR_API_KEY \
    --output_dir checkpoints/orchestrator_r1 \
    --max_turns 6 --progressive_stage 3 \
    --alpha 0.3 --beta 0.1 --gamma 0.15
```

---

## 实验对比设计

### 主实验

```
测试数据：
  Track 1 (简单 QA):    NQ-test / TriviaQA-test / PopQA-test
  Track 2 (多跳推理):   HotpotQA-test / 2WikiMultihop-test / MuSiQue-test
  Track 3 (代码):       HumanEval / MBPP

基线:
  Conductor (同 worker pool) → eval/results/conductor.json
  Router-R1 (official)       → eval/results/router_r1.json
  Direct-Strong              → eval/results/direct_strong.json
  ReAct (Qwen2.5-3B)         → eval/results/react.json
  Self-Reflection 5-turn     → eval/results/self_reflection.json
  Fixed-Pipeline             → eval/results/fixed_pipeline.json

本方法:
  Orch-base (未训练)    → eval/results/orch_base.json
  Orch-SFT (仅热身)    → eval/results/orch_sft.json
  Orch-GRPO (完整训练)  → eval/results/orch_grpo.json
```

### 消融实验

```
  w/o reactive          → eval/results/ablation_no_reactive.json    ★★★
  w/o critic            → eval/results/ablation_no_critic.json      ★★ 角色消融组
  w/o decomposer        → eval/results/ablation_no_decomp.json      ★★ 角色消融组
  w/o synthesizer       → eval/results/ablation_no_synth.json       ★★ 角色消融组
  w/ refiner            → eval/results/ablation_add_refiner.json    ★★ 反向消融
  w/o progressive       → eval/results/ablation_no_progressive.json
  w/o context compress  → eval/results/ablation_no_compress.json
  w/o cost penalty      → eval/results/ablation_alpha0.json
  SFT-only              → eval/results/ablation_sft_only.json
  Fixed-Pipeline        → eval/results/ablation_fixed.json
```

### 超参敏感性

```
  alpha = {0, 0.1, 0.3, 0.5, 0.7, 0.9}
  → eval/results/alpha_*.json → 帕累托曲线
```

### 论文核心图表

```
Table 1:  主结果表（3 Track × 基线 × 4 指标）
Table 2:  与 Conductor 同 worker pool controlled comparison（GPQA / LCB）
Table 3:  Agent 角色消融（w/o critic / decomposer / synthesizer / w/ refiner）
Table 4:  其余消融（reactive / progressive / SFT-only / Fixed-Pipeline）
Figure 1: 系统架构图（4 角色）
Figure 2: Agent 调用分布热力图（4 agents × N datasets，证明自适应涌现）
Figure 3: 效率-质量帕累托曲线（α 调节调用频率/轮数）
Figure 4: 训练过程中行为变化（渐进式三阶段 + RL 驱动涌现）
```

---

## 工作推进计划

### 阶段 1（2 周）：核心功能验证

```
  ✓ agent_pool + parser + reward
  ✓ 本地推理测试（infer.py）
  → 收敛 agent pool 为 4 角色（executor / decomposer / critic / synthesizer）
  → 实现 context_manager.py（上下文压缩模块）
  → 补写 training/sft_warmup.py（SFT 热身训练脚本）
  → 补写 data_process/prepare_code.py（HumanEval/MBPP 数据加载）
  → 补写 data_process/prepare_sft.py（SFT 热身示例生成，覆盖 4 种 Agent）
  → 补写 eval/compare.py（多结果对比 → 表格 + 图）
  → 补写 eval/baselines.py（Router-R1 / ReAct / Direct 基线评估）
  → 准备 NQ + HotpotQA + HumanEval 训练/测试数据
  → 跑 base model baseline（3 Track，未训练模型）
```

### 阶段 2（3 周）：训练 + 主实验

```
  → 生成 SFT 热身数据（50~100 条，覆盖 4 种 Agent 类型）
  → SFT 热身训练（1~2h）
  → 渐进式 GRPO 三阶段训练（strong worker pool）
  → Track 1/2/3 三组评估（训练后模型）
  → 基线复现：Router-R1 / ReAct / Self-Reflection / Direct-Strong
  → eval/compare.py 生成主结果对比表
```

### 阶段 3（2 周）：分析 + 消融

```
  → analysis/agent_distribution.py → Agent 调用分布热力图
  → analysis/pareto_curve.py → 效率-质量帕累托曲线
  → analysis/training_dynamics.py → 训练过程行为变化图
  → 消融实验（w/o critic, w/o decomposer, w/o refiner, fixed, SFT-only）
  → α 敏感性分析（α = 0, 0.1, 0.3, 0.5, 0.7, 0.9）
```

### 阶段 4（1 周）：论文撰写

```
  → Table 1: 主结果表（3 Track × 5 基线 × 4 指标）
  → Table 2: 消融实验表
  → Figure 1: 系统架构图
  → Figure 2: Agent 调用分布热力图
  → Figure 3: 效率-质量帕累托曲线
  → Figure 4: 训练过程行为变化
  → Figure 5: α 敏感性分析
  → 形式化定义 + Related Work + Introduction
```

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 本地单元测试（无需 GPU）
python test_local.py --skip_api

# 3. 准备数据
python data_process/prepare_data.py \
    --sources nq,hotpotqa --max_per_source 100 --output data/test_small.jsonl
python data_process/prepare_code.py \
    --source humaneval --output data/test_code.jsonl

# 4. 单条推理测试（需要 GPU + API）
python inference/infer.py \
    --model_path models/Qwen2.5-3B-Instruct \
    --api_base YOUR_API_BASE --api_key YOUR_API_KEY \
    --input "When did World War 2 end?"

# 5. 小规模评估
python eval/eval_orchestrator.py \
    --model_path models/Qwen2.5-3B-Instruct \
    --data_path data/test_small.jsonl \
    --api_base YOUR_API_BASE --api_key YOUR_API_KEY \
    --output eval/results/orch_base.json

# 6. 对比结果
python eval/compare.py \
    --results eval/results/orch_base.json eval/results/router_r1.json \
    --output eval/results/comparison.json

# 7. 训练（4×RTX 3090, Linux）
bash training/train.sh
```
