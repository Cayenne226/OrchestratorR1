# OrchestratorR1 — 核心思想与技术方案

## 1. 解决的问题

现有 LLM 路由工作存在三个核心局限：

1. **路由粒度粗**：Router-R1 仅将整个问题路由到单一模型，无法处理需要多种能力协同的复杂任务
2. **任务范围窄**：仅支持 QA 问答（7 个数据集），无法泛化到代码生成、多跳推理等场景
3. **策略固化**：不区分简单/复杂任务，所有输入走相同处理路径，造成简单任务浪费、复杂任务质量不足

**OrchestratorR1 的核心主张**：一个小型语言模型可以通过强化学习，自适应地编排一组专业 Agent 协作完成任务。单一奖励函数驱动下，模型自发涌现出从简单到复杂的行为谱——这不是多种技术的融合，而是统一 RL 目标下的涌现行为。

---

## 2. 形式化定义

### 2.1 问题建模为 MDP

将 Agent 编排问题建模为马尔可夫决策过程 (S, A, T, R)：

- **状态 S**：当前对话上下文 c_t = (x, a_1, o_1, ..., a_{t-1}, o_{t-1})
  - x：用户原始输入
  - a_i：第 i 步的 Agent 调用动作
  - o_i：Agent 返回的信息

- **动作 A**：Orchestrator 在每步 t 选择一个动作 a_t，属于以下之一：
  - (agent_type, query)：调用指定类型的 Agent，发送 query
  - TERMINATE(answer)：输出最终答案，终止编排

- **转移 T**：确定性转移，由被调用 Agent 的 API 返回决定
  - T(s_t, a_t) = s_t ⊕ o_t，其中 o_t = Agent(a_t.type, a_t.query)

- **奖励 R**：仅在轨迹结束时给出终端奖励（稀疏奖励）
  - R(τ) = R_outcome(y, y*) − α·C_cost(τ) − β·C_turns(τ) + γ·B_eff(τ)

### 2.2 Orchestrator 策略

Orchestrator 是一个参数化策略 π_θ(a_t | s_t)，以 Qwen2.5-3B/7B 为基座。

训练目标：最大化期望奖励

```
J(θ) = E_{x~D} E_{τ~π_θ(·|x)} [R(τ)]
```

使用 GRPO 优化：同一输入 x 采样 G=8 条轨迹 {τ_1, ..., τ_G}，组内相对奖励作为优势估计：

```
A_i = (R(τ_i) - mean(R)) / std(R)
```

梯度更新：

```
∇J ≈ Σ_i A_i · ∇logπ_θ(τ_i | x)
```

### 2.3 Agent Pool（固定，不训练）

Agent 池 P = {p_1, ..., p_K} 是一组固定的 LLM API 端点，每个 Agent p_k 具有：
- 角色系统提示 s_k（决定其功能）
- 底层模型 m_k（决定其能力和成本）
- 单位成本 c_k（$/1M tokens）

Orchestrator 只需学会「何时」调用「哪个」Agent，并构造有效的 query。

---

## 3. 核心创新

### 3.1 与现有工作的本质区别

| 工作 | 方法 | 与本文区别 |
|---|---|---|
| Router-R1 | RL 路由到单一模型 | 本文路由到功能 Agent，支持分解和验证 |
| Prompt-R1 | RL 优化 prompt | 本文 prompt 优化是涌现行为之一 |
| AutoGen/CrewAI | 手工设计多 Agent 流程 | 本文编排策略由 RL 学习，非人工设计 |
| ReAct | 单模型 tool-use | 本文调用专业 Agent，支持多轮协作 |
| Reflexion | 单模型自我反思 | 本文 critic 是独立外部 Agent |
| LLM-Compiler | 静态 DAG 并行执行 | 本文动态自适应路径 |
| ToolFormer | 学习调用工具 | 本文调用 LLM Agent，支持功能路由 |

### 3.2 核心贡献（3 条）

1. **首次将多 Agent 编排建模为可学习的 RL 问题**：统一 MDP 框架，单一奖励函数
2. **证明端到端 RL 可以涌现自适应编排策略**：模型自发学会简单题走捷径、复杂题精雕细琢
3. **在 QA、多跳推理、代码任务上均超越 Router-R1**：更广泛的任务覆盖 + 更低成本

---

## 4. Agent 池设计

```
refiner        → 重写查询以优化检索效果（RAG 风格）  → gpt-4o-mini (低成本)
decomposer     → 将复杂任务拆解为独立子任务       → gpt-4o (中等)
executor_cheap → 执行简单、明确的子任务           → gpt-4o-mini (低成本)
executor_strong→ 执行需要深度推理的复杂子任务     → claude-sonnet (高能力)
critic         → 评估中间结果，指出不足           → gemini-flash (中等)
synthesizer    → 将多个子任务结果整合为最终答案   → gpt-4o-mini (低成本)
```

Agent 池是**固定基础设施**，不参与训练。Orchestrator 通过 RL 学会何时调用哪个 Agent。

---

## 5. 输出格式

模型输出为结构化 XML 标签序列，支持任意组合：

```xml
<think>分析任务复杂度，决定调用哪些 Agent</think>

<!-- 可选，按需出现 -->
<call type="refiner">重写查询以优化检索</call>
<information>优化后的描述（系统注入）</information>

<call type="decomposer">任务描述</call>
<information>分解结果（系统注入）</information>

<call type="executor_cheap|executor_strong">具体查询</call>
<information>执行结果（系统注入）</information>

<call type="critic">需要验证的结果</call>
<information>评估反馈（系统注入）</information>

<call type="synthesizer">整合多个结果</call>
<information>整合结果（系统注入）</information>

<answer>最终答案</answer>
```

**关键设计**：所有 `<call>` 标签均为可选，模型通过 RL 学会哪些该用、哪些该跳过。

### 涌现的行为谱

| 任务复杂度 | 涌现路径 | 预期轮数 |
|---|---|---|
| 简单事实题 | think → executor_cheap → answer | 1 |
| 隐含引用 | think → refiner → executor_cheap → answer | 2 |
| 多跳推理 | think → decomposer → executor×N → synthesizer → answer | 3~5 |
| 复杂代码 | think → decomposer → executor_strong×N → critic → answer | 4~6 |
| 复杂+隐含引用 | think → refiner → decomposer → executor_strong×N → critic → synthesizer → answer | 5~6 |

---

## 6. 奖励函数

```python
R(τ) = R_outcome − α·C_cost − β·C_turns + γ·B_efficiency
```

各项定义：

```python
# 答案质量（主奖励）
R_outcome = compute_f1(pred_answer, gold_answer)   # QA 任务
          # 或 pass_at_1(pred_code, test_cases)     # 代码任务

# API 成本（归一化到 [0,1]）
C_cost = min(Σ_k cost_k / C_max, 1.0)             # C_max = 0.01$

# 轮数惩罚（归一化到 [0,1]）
C_turns = n_turns / max_turns

# 效率奖励（高质量+少轮次时触发）
B_efficiency = 1.0 if (R_outcome > 0.8 and n_turns ≤ 2) else 0.0

# 格式惩罚（未生成合法标签）
R_format = -1.0 if format_invalid else 0.0
```

| 超参 | 推荐值 | 敏感性分析范围 |
|---|---|---|
| α (成本) | 0.3 | {0, 0.1, 0.3, 0.5, 0.7, 0.9} |
| β (轮数) | 0.1 | {0, 0.05, 0.1, 0.2} |
| γ (效率) | 0.15 | {0, 0.1, 0.15, 0.3} |

---

## 7. 训练方案

### 7.1 训练对象

仅训练 Orchestrator 模型（Qwen2.5-3B/7B-Instruct），Agent 池固定不训练。

### 7.2 训练方法：GRPO

| 特性 | PPO (Router-R1) | GRPO (本文) |
|---|---|---|
| 价值函数 | 需要 Critic 网络 | 组内相对奖励替代 |
| 显存占用 | 高（双网络）| 低（单网络）|
| 实现复杂度 | 高 | 低（trl 原生支持）|

### 7.3 训练数据

直接使用原始 QA/代码数据集，无需合成标签：

| 数据源 | 任务类型 | 训练量 |
|---|---|---|
| NQ, TriviaQA, PopQA | 简单 QA | 各 1000 条 |
| HotpotQA, 2WikiMultihop, MuSiQue | 多跳推理 | 各 1000 条 |
| HumanEval, MBPP | 代码 | 各 500 条 |

### 7.4 训练阶段

**阶段 0（SFT 热身，1~2h）**：
- 50~100 条覆盖所有 6 种 Agent 类型的示例
- 确保模型学会 `<call>` 标签格式
- 解决冷启动探索问题

**阶段 1（GRPO 主训练，12~20h）**：
- 混合数据，GRPO G=8
- 奖励：R_outcome − α·C_cost − β·C_turns + γ·B_efficiency

### 7.5 训练成本控制

训练阶段 API 调用量估算：5000 条 × G=8 × 平均 3 次调用 = 120,000 次。

优化策略：
- 训练阶段所有 Agent 统一使用 executor_cheap（gpt-4o-mini，$0.15/1M tokens）
- 推理阶段才切换到完整 Agent 池
- 预计训练 API 成本 < $5

### 7.6 超参数

| 参数 | 值 |
|---|---|
| num_generations (G) | 8 |
| max_turns | 6 |
| per_device_batch_size | 2 |
| gradient_accumulation | 8 |
| learning_rate | 1e-6 |
| bf16 | True |
| warmup_ratio | 0.05 |
| total_steps | ~2000 |

---

## 8. 实验设计

### 8.1 三个评估赛道

| 赛道 | 数据集 | 指标 | 目标 |
|---|---|---|---|
| Track 1: 简单 QA | NQ, TriviaQA, PopQA | EM, F1 | 平手或略优 Router-R1，成本更低 |
| Track 2: 多跳推理 | HotpotQA, 2WikiMultihop, MuSiQue | F1 | 显著优于 Router-R1 |
| Track 3: 代码任务 | HumanEval, MBPP | Pass@1 | 证明泛化（Router-R1 不支持）|

### 8.2 对比基线

| 基线 | 描述 | 来源 |
|---|---|---|
| Router-R1 | 单轮路由到一个模型 | 官方 checkpoint |
| Direct-Strong | 直接调用 GPT-4o | 无需训练 |
| ReAct | 标准 tool-use agent | 基于 Qwen2.5-3B |
| Reflexion | ReAct + 自我反思 | 基于 Qwen2.5-3B |
| Fixed-Pipeline | 强制走完整 6 步 | 本文消融 |

### 8.3 消融实验

| 实验组 | 配置 | 验证目标 |
|---|---|---|
| Full | 完整方案 | 上界 |
| w/o critic | 去掉 critic Agent | 验证能力贡献 |
| w/o decomposer | 去掉 decomposer Agent | 分解能力的贡献 |
| w/o refiner | 去掉 refiner Agent | query rewriting 的贡献 |
| Fixed-Pipeline | 强制走全流程 | 自适应 vs 固定的优势 |
| SFT-only | 仅 SFT，无 GRPO | RL 的贡献 |

### 8.4 超参敏感性分析

- α 从 0 到 0.9：绘制 cost vs F1 帕累托曲线
- β 从 0 到 0.2：轮数 vs F1 权衡

### 8.5 核心分析图（论文关键证据）

**图 1：Agent 调用分布热力图**

横轴 6 种 Agent 类型，纵轴不同数据集，颜色深浅代表调用频率。
直接证明模型学会了根据任务类型调用不同 Agent。

**图 2：效率-质量帕累托曲线**

X 轴 avg cost ($)，Y 轴 F1。
本方法 vs Router-R1 vs Direct-Strong，证明帕累托前沿优势。

**图 3：训练过程中行为变化**

展示训练 steps 推进过程中，简单题/复杂题的平均轮数差异如何扩大。
直接证明"自适应"是 RL 涌现的。

---

## 9. 与 Router-R1 的关系

Router-R1 是本工作的**直接基线和理论出发点**：

```
Router-R1 证明了：「路由」可以被 RL 学习
    ↓ 本文扩展
OrchestratorR1 证明了：「编排」也可以被端到端 RL 学习
```

| 维度 | Router-R1 | OrchestratorR1 |
|---|---|---|
| 路由对象 | 不同能力的 LLM | 不同角色的 Agent |
| 调用目的 | 回答同一个问题 | 完成不同职责 |
| 任务分解 | 无 | 有，RL 驱动按需触发 |
| 中间验证 | 无 | critic 循环 |
| 任务范围 | 7 个 QA 数据集 | QA + 多跳推理 + 代码 |
| 训练方法 | PPO (veRL) | GRPO (trl) |
| 训练组件 | Actor + Critic + vLLM | 仅 Actor |
| 训练数据 | 原始 QA | 原始 QA + 代码（同样无需合成）|
