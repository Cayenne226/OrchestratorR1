# 四大竞品深度对比：Router-R1 / Prompt-R1 / Conductor / AgentConductor

> 更新日期: 2026-03-26
> 目的: 为 OrchestratorR1 定位 NeurIPS 2026 投稿方向
> ⚠️ 2026-03-26 重大更新: 修正 Conductor 实际信息（worker模型、硬件、基线、结果）

---

## 1. 全景对比表

| 维度 | Router-R1 | Prompt-R1 | Conductor | AgentConductor | **OrchestratorR1 (Ours)** |
|------|-----------|-----------|-----------|----------------|---------------------------|
| **发表** | NeurIPS 2025 | arXiv 2511 | ICLR 2026 | arXiv 2602 | NeurIPS 2026 (target) |
| **核心思想** | RL 学路由：选哪个 LLM 回答 | RL 学 prompt：小模型给大模型写提示词 | RL 学编排：一次性生成完整 workflow | SFT+RL 学拓扑：生成 DAG 执行图 | RL 学编排：逐步反应式调度功能 Agent |
| **训练方法** | PPO (veRL) | GRPO | GRPO | SFT + GRPO (veRL) | SFT warmup + GRPO (trl) |
| **基座模型** | Qwen2.5-3B/7B | 小 LLM (具体未公开) | 7B (Qwen2.5-7B-Instruct) | Qwen2.5-7B-Instruct | Qwen2.5-3B/7B-Instruct |
| **被调度对象** | 同质 LLM（不同模型回答同一问题） | 大 LLM（GPT-4o 等） | 7 个前沿 worker LLM（GPT-5/Claude Sonnet 4/Gemini 2.5 Pro/DeepSeek-R1-32B/Qwen3-32B/Gemma3-27B/7B self） | 多个 coding agent | 6 种功能 Agent（异质角色） |
| **决策粒度** | 选模型 + 写 query | 写 prompt | 生成完整 workflow（agent选择+指令+通信拓扑） | 生成分层 DAG（节点+边+并行度） | 逐步选 agent type + 写 query |
| **执行范式** | 反应式多轮 | 多轮交互 | **开环一次性规划** | **开环 DAG 生成**（多轮可更新） | **闭环反应式** |
| **是否有 Critic 网络** | ✅ (PPO 需要) | ❌ | ❌ (GRPO) | ❌ (GRPO) | ❌ (GRPO) |
| **SFT 预热** | ❌ | ❌ | ❌ (纯 GRPO) | ✅ (4500 条) | ✅ (50-100 条) |
| **任务范围** | 7 个 QA 数据集 | 12 个数据集（多类型） | 数学+代码+推理 (MATH, MMLU, LiveCodeBench, GPQA) | 竞赛级代码 (5 个代码数据集) | QA + 多跳推理 + 代码 |
| **成本感知** | cost_coe 惩罚 (默认=0) | ❌ | ❌ | 密度函数隐式控制 | ✅ 显式 α 调节 Pareto |
| **最大轮数** | 4 | 多轮 | 1（一次性输出） | 多轮（DAG 可更新） | 6 |
| **训练数据量** | 7K × 2 = 14K | 未公开 | 960 条（4 领域） | 4500 条 SFT + RL | ~5K-7K |
| **训练硬件** | 4 GPU + vLLM | 未公开 | 2×H100 | veRL + vLLM | 4×A100/V100 80GB |
| **训练成本** | 中等（PPO 双网络 + vLLM） | 未公开 | GRPO 200 iterations, batch 256 | SFT + GRPO | 低（单网络 GRPO，无 vLLM） |

---

## 2. 逐个深度剖析

### 2.1 Router-R1 (NeurIPS 2025) — 直接前作

**核心机制**:
```
用户问题 → Router(Qwen-3B) → <search>ModelName:query</search> → LLM API → <information>回答</information> → 循环最多4轮 → <answer>
```

**技术细节**:
- PPO 训练，Actor + Critic 双网络，都从同一 base model 初始化
- veRL 框架 + vLLM rollout engine（高效但配置复杂）
- **State masking**: 只对模型自己生成的 token 计算 loss，注入的 `<information>` 块被 mask 掉
- GAE 优势估计，KL 系数 0.001
- 训练 225 steps × batch 64 = ~14,400 样本
- cost_coe 默认 = 0（论文实际没怎么强调成本优化）

**优势**:
- 多轮反应式，和你一样
- 已发表，有 official checkpoint 可直接对比
- 代码开源，可复现

**局限**:
- 所有 LLM 做同一件事（回答问题），无功能分工
- 不支持任务分解、验证、综合
- 仅 QA 任务，无代码
- PPO 需要 Critic 网络，显存开销大

**你对它的优势**:
1. 功能角色 vs 同质模型 → 支持分解/验证/综合
2. 跨模态 → QA + 推理 + 代码
3. GRPO vs PPO → 训练更简单、显存更低
4. 显式成本优化 → Pareto 可调

---

### 2.2 Prompt-R1 (arXiv 2511.01016) — 间接竞品

**核心机制**:
```
用户问题 → PromptAgent(小LLM) → <think>分析</think> → 生成 prompt → 大LLM(GPT-4o等) → 返回结果 → 多轮交互 → 最终答案
```

**技术细节**:
- 小 LLM 作为 agent，大 LLM 作为 environment
- 多轮 prompt 交互：小模型思考 → 生成 prompt → 大模型执行 → 返回结果 → 循环
- Dual-constrained reward：同时优化正确性和生成质量
- GRPO 训练
- 12 个数据集，跨多种任务类型

**优势**:
- Plug-and-play：可以接任意大 LLM
- 多轮交互，接近反应式
- 任务覆盖面广

**局限**:
- 只调度**一个**大 LLM（prompt 优化），不是多 agent 编排
- 没有功能分工（refiner/decomposer/critic 都是同一个大 LLM）
- 没有成本优化（因为只调一个模型）

**你对它的优势**:
1. 多 agent 编排 vs 单 agent prompt 优化 → 本质不同
2. 功能分工涌现 → Prompt-R1 只优化 "怎么问"，你还优化 "问谁做什么"
3. 成本可控 → 简单题不需要调大 LLM

**你对它的劣势**:
- Prompt-R1 的 prompt 优化能力可能比你的 refiner agent 更强（它整个训练目标就是 prompt 优化）
- 接入门槛更低（只需一个 API）

---

### 2.3 Conductor (ICLR 2026) — 最强竞品 ⚠️

**核心机制**:
```
用户问题 → Conductor(Qwen2.5-7B) → 生成 Python list:
  [[model_id, subtask_instruction, access_list], ...]
  例如: [["gpt-5", "Solve subproblem A", [1,2]], ["claude-sonnet-4", "Verify...", [0]]]
→ 框架分发到 7 个前沿 worker → 收集输出 → 支持递归 self-invocation → 最终答案
```

**技术细节**:
- 基座: Qwen2.5-7B-Instruct，纯 GRPO，无 SFT 预热
- 硬件: **2×H100**
- 训练数据仅 960 条（4 领域）
- 200 GRPO iterations，batch 256，无 KL 正则化
- **Worker 池（7 个前沿模型）**:
  - GPT-5
  - Claude Sonnet 4
  - Gemini 2.5 Pro
  - DeepSeek-R1-32B
  - Qwen3-32B
  - Gemma3-27B
  - 7B self（Conductor 自身递归调用）
- Workflow 包含三个维度：agent 选择（model_id）、子任务指令（subtask）、通信拓扑（access_list：谁能看到谁的输出）
- **支持递归 self-invocation**: Conductor 可以在 workflow 中调用自己，实现递归分解
- **涌现行为**: 问题分解、prompt 工程、独立尝试 + 辩论轮

**Conductor 论文中的基线**:
| 基线 | 类型 |
|------|------|
| RouterDC | 学习路由（选模型） |
| Smoothie | 权重聚合 |
| MASRouter | 多 agent 路由 |
| MoA (Mixture of Agents) | 多 agent 混合 |
| 5-turn Self-Reflection | 单模型多轮自反思 |
| Single-best worker | 单个最强 worker |

**关键结果**:
- **LiveCodeBench: 83.93%** — 超越所有之前的 LLM（包括 OpenAI o 系列）
- **GPQA-Diamond: 87.5%** — 超越所有单模型基线
- **AIME 2025: 93.3%** — 竞赛级数学推理
- 超越 RouterDC, Smoothie, MASRouter, MoA, Self-Reflection 等全部基线
- 泛化到训练数据之外的任务

**与我们 Direct-GPT-4o baseline 的差距**:
| 指标 | Conductor | 我们的 GPT-4o direct | 差距 |
|------|-----------|---------------------|------|
| GPQA-Diamond | **87.5%** | 51.0% | -36.5% |
| LiveCodeBench | **83.93%** | 41.78% | -42.15% |
| AIME 2025 | **93.3%** | N/A | — |

> ⚠️ **注意**: 差距巨大的根本原因是 Conductor 的 worker 包含 **GPT-5, Claude Sonnet 4, Gemini 2.5 Pro** 等最新前沿模型，而我们的 worker 是 GPT-4o-mini/GPT-4o 级别。这是根本性的资源不对等，不应做绝对分数对比。

**优势**:
- SOTA 结果，极其强
- 训练数据极少（960 条）却泛化很好
- 通信拓扑是独特创新（agent 间可见性）
- 递归 self-invocation 增加了灵活性
- ICLR 2026 已接收，学术认可度高
- 已有多个 strong baseline 对比（RouterDC, MoA, MASRouter 等）

**局限 (你的机会)**:
1. **开环规划**: 一次性生成完整 workflow，执行过程中不能根据中间结果调整
   - 如果 Step 1 的 agent 返回了错误信息，Step 2/3 仍然照原计划执行
   - 无法动态增加/减少步骤
2. **依赖前沿 worker**: 性能严重依赖 GPT-5/Claude Sonnet 4 等顶级模型
   - 总 API 成本极高（每次调 GPT-5 + Claude Sonnet 4 + Gemini 2.5 Pro）
   - 不适合预算有限的实际部署
3. **无显式成本优化**: 每次都生成完整多步 workflow，简单题也走 3+ 步
4. **无显式功能角色**: agent 的功能由 Conductor 的 prompt 决定，不是预定义的
5. **任务范围有限**: 主要在 MATH/GPQA/LiveCodeBench/AIME 等推理+代码任务，没有 QA/多跳推理

**你 vs Conductor 的关键论点（双配置策略）**:

| 维度 | Conductor | OrchestratorR1 | 分析 |
|------|-----------|----------------|------|
| 绝对结果 (matched worker) | 87.5% GPQA, 83.9% LCB | 待验证，预期**接近或超过** | 同等 worker 下 reactive 应该 ≥ open-loop |
| 绝对结果 (cheap worker) | N/A | 待验证 | 成本效率 story |
| Worker 成本 | GPT-5 + Claude Sonnet 4 等（$$$） | cheap 配置用 GPT-4o-mini（$） | ✅ 你：cheap 配置成本低 10-50× |
| 适应性 | 开环，不可调整 | 闭环，逐步调整 | ✅ 你 |
| 错误恢复 | 无法恢复（执行完才看结果） | 可以：critic 反馈 → 重试 | ✅ 你 |
| 简单题处理 | 仍生成多步 workflow | 1 步直接回答 | ✅ 你 |
| 功能清晰度 | 隐式（由 prompt 决定） | 显式（6 种角色） | ✅ 你（更可解释） |
| 通信拓扑 | ✅ agent 间可见性 | ❌ 只有 orchestrator 看到所有 | Conductor |
| 递归分解 | ✅ self-invocation | ❌ | Conductor |
| 训练效率 | 960 条，200 iterations | ~5K 条，~2000 steps | Conductor 数据效率更高 |
| 训练总成本 | 2×H100 + GPT-5 API rollout | 单 GPU + GPT-4o-mini API | ✅ 你（美元成本更低） |
| 并行执行 | ✅ workflow 中可并行 | ❌ 串行 | Conductor |
| QA/多跳推理 | ❌ 未覆盖 | ✅ 7 个 QA 数据集 | ✅ 你 |
| Worker 灵活性 | 固定 7 个前沿模型 | 可切换 matched/cheap 两种配置 | ✅ 你（Pareto 覆盖） |

---

### 2.4 AgentConductor (arXiv 2602) — 代码领域竞品

**核心机制**:
```
用户问题 → Orchestrator(7B) → 生成分层 DAG:
  Layer 0: [Agent_A(task1), Agent_B(task2)]  ← 并行
  Layer 1: [Agent_C(merge)]                   ← 依赖 Layer 0
  Layer 2: [Agent_D(test), Agent_E(review)]   ← 并行
→ 执行 DAG → 收集结果
→ (可选) 根据执行反馈更新 DAG → 再执行
→ 最终答案
```

**技术细节**:
- Qwen2.5-7B-Instruct，SFT (4500 条) + GRPO (veRL + vLLM)
- **分层 DAG 拓扑**: 每层内并行，跨层串行
- **密度函数**: 数学化描述 multi-agent 交互密度，用于控制拓扑复杂度
- **难度感知**: 根据问题难度调整 DAG 密度（简单→稀疏，困难→密集）
- 多轮：可以根据执行反馈更新 DAG（这点和你类似）

**关键结果**:
- 5 个代码数据集（3 个竞赛级 + 2 个基础）
- Pass@1 最高提升 14.6%
- 密度减少 13%，token 成本减少 68%

**优势**:
- DAG 拓扑创新（支持并行 + 依赖关系）
- 密度函数是新颖的形式化贡献
- 代码任务 SOTA
- 多轮反馈可更新拓扑

**局限 (你的机会)**:
1. **仅代码任务**: 没有 QA/推理
2. **DAG 预规划**: 虽然可多轮更新，但每轮仍是"生成完整 DAG → 执行"
3. **功能同质**: agent 都是 coding agent，只是拓扑不同
4. **密度函数设计复杂**: 需要人工设定难度区间和密度上界

**你 vs AgentConductor 的关键论点**:

| 维度 | AgentConductor | OrchestratorR1 | 谁优？ |
|------|----------------|----------------|--------|
| 代码任务 | 竞赛级 SOTA | 基础 HumanEval/MBPP | AgentConductor |
| 任务广度 | 仅代码 | QA + 推理 + 代码 | ✅ 你 |
| 编排粒度 | DAG 拓扑（层+边） | 功能角色调度 | 各有优势 |
| 并行支持 | ✅ 层内并行 | ❌ 串行 | AgentConductor |
| 功能分工 | ❌ 同质 agent | ✅ 6 种角色 | ✅ 你 |
| 反应能力 | 多轮可更新 DAG | 逐步反应 | ✅ 你（更细粒度） |
| 成本控制 | 密度函数隐式控制 | α 显式 Pareto | ✅ 你（更直接） |

---

## 3. 核心定位：OrchestratorR1 在哪里赢？

### 3.0 关键认识：资源层级差异（重新定位基础）

Conductor 的超强结果建立在**前沿 worker 模型**之上（GPT-5, Claude Sonnet 4, Gemini 2.5 Pro），而我们使用 GPT-4o-mini/GPT-4o 级别 worker。这意味着：

1. **绝对分数对比无意义** — Conductor 87.5% GPQA vs 我们的上限约 50-60%，但这是 worker 差距，不是编排能力差距
2. **需要新的对比维度**：
   - 同等 worker 条件下的编排效率对比
   - 成本-质量 Pareto 前沿对比
   - 编排范式（闭环 vs 开环）的结构性优劣
   - 任务覆盖广度
3. **论文定位必须完全避开"数字战"**

### 3.1 唯一性矩阵

| 特性 | Router-R1 | Prompt-R1 | Conductor | AgentConductor | **Ours** |
|------|:---------:|:---------:|:---------:|:--------------:|:--------:|
| 多 agent 编排 | ✅ 多模型 | ❌ 单模型 | ✅ 多 worker | ✅ 多 agent | ✅ |
| 功能异质角色 | ❌ | ❌ | ❌ 隐式 | ❌ | ✅ |
| 闭环反应式 | ✅ | ✅ | ❌ 开环 | 部分（轮级） | ✅ 步级 |
| 跨模态泛化 | ❌ QA only | ✅ 多类型 | ✅ 推理+代码 | ❌ 代码 only | ✅ QA+推理+代码 |
| 显式成本 Pareto | ❌ (coe=0) | ❌ | ❌ | 隐式 | ✅ |
| 自适应路径涌现 | ❌ 固定多轮 | ❌ | ❌ 固定workflow | 密度自适应 | ✅ 1-6步涌现 |

**你的独特组合**: 闭环反应式 × 功能异质角色 × 跨模态 × 显式成本优化 — 这个交叉点没有任何现有工作占据。

### 3.2 建议的论文 Positioning（修订版 — 双配置策略）

**核心策略**: 用**两组实验**同时证明编排范式优势和实用性。

1. **Matched-worker 实验**: 把 Agent Pool 对齐 Conductor（GPT-5/Claude Sonnet 4/Gemini 2.5 Pro 等），正面对标绝对分数 → 证明 **reactive ≥ open-loop**（同等 worker 下编排范式的贡献）
2. **Cheap-worker 实验**: 用 GPT-4o-mini/GPT-4o 作为 worker → 证明**成本效率**和**跨模态泛化**

**论文定位**:

> "Recent work has demonstrated the power of learned multi-agent orchestration, with Conductor (ICLR 2026) achieving remarkable results through plan-then-execute workflows over frontier models.
>
> We propose OrchestratorR1, which learns **reactive sequential orchestration** over **functionally specialized** agents via end-to-end RL. We demonstrate:
> (a) Under matched worker conditions (same frontier models), reactive orchestration matches or exceeds Conductor's open-loop paradigm, with additional error recovery capability;
> (b) The same reactive policy generalizes across QA, reasoning, and code — covering tasks Conductor does not address;
> (c) With cost-efficient workers (GPT-4o-mini), OrchestratorR1 achieves competitive results at 10-50× lower cost, with explicit Pareto optimization."

**关键对比角度**:
1. **同等 worker 下正面对标** — reactive ≥ open-loop，我们有理由赢
2. **跨模态广度** — 一个策略三种任务，Conductor/Router-R1 都做不到
3. **成本 Pareto** — 便宜 worker 配置下的 cost-quality tradeoff
4. **可解释性** — 显式功能角色 vs 隐式 prompt 指令
5. **错误恢复** — 闭环的结构性优势，adversarial 实验量化

### 3.3 新增基线策略（对齐 Conductor 的 baseline 方法）

Conductor 论文中的基线包括 **RouterDC, Smoothie, MASRouter, MoA, 5-turn Self-Reflection**。
我们应该在论文中覆盖这些方法中的关键方法（至少引用数据），以展示对 landscape 的全面了解：

| Conductor 的基线 | 我们是否需要复现 | 策略 |
|-----------------|----------------|------|
| RouterDC | ❌ 不需要 | 引用 Conductor 论文中的数字即可 |
| Smoothie | ❌ 不需要 | 引用数据 |
| MASRouter | ❌ 不需要 | 引用数据 |
| MoA | ★☆ 可选 | 在 QA 任务上复现（Conductor 没有 QA 数据），展示跨模态优势 |
| Self-Reflection | ★★ 建议 | 在我们的数据集上实现 5-turn self-reflection baseline |
| Router-R1 | ★★★ 必须 | 直接前作，必须对比 |
| Direct-GPT-4o | ★★★ 必须 | ✅ 已完成 |
| ReAct | ★★☆ 建议 | 标准 tool-use 基线 |

---

## 4. 实验设计建议（修订版 — 双配置策略）

### 4.0 总体实验策略：双配置 (Matched + Cheap)

**训练**: 用便宜 worker（GPT-4o-mini/GPT-4o）训练 → 降低成本
**测试**: 两组配置分别评估

| 配置 | Worker Pool | 目的 | 对标 |
|------|-------------|------|------|
| **Matched** | GPT-5, Claude Sonnet 4, Gemini 2.5 Pro, DeepSeek-R1-32B, Qwen3-32B, Gemma3-27B | 证明 reactive ≥ open-loop | 正面对标 Conductor |
| **Cheap** | GPT-4o-mini, GPT-4o | 证明成本效率 + 跨模态泛化 | 实用性 story |

> **关键假设**: Orchestrator 学到的编排策略（何时 decompose/critique/refine）在 worker 升级后仍然有效（zero-shot transfer），因为策略是关于"做什么"（认知功能）而非"用哪个模型"。

**四条实验主线**:
1. **主线 A（Matched-worker 正面对标 Conductor）**: GPQA + LiveCodeBench，同等 worker → 证明 reactive ≥ open-loop
2. **主线 B（跨模态泛化 — 独有优势）**: 10 个 benchmark，Conductor/Router-R1 都覆盖不了
3. **主线 C（编排范式消融）**: reactive vs open-loop controlled experiment
4. **主线 D（成本 Pareto — 实用性）**: cheap worker 下的 Pareto 前沿

### 4.1 必须做的实验

**A. Matched-Worker 对标 Conductor（Table 1 — 最有冲击力）**:
```
| Method                | GPQA  | LCB   | Worker Pool              | Paradigm     |
|-----------------------|-------|-------|--------------------------|--------------|
| Conductor†            | 87.5  | 83.93 | GPT-5,Claude,Gemini,...  | Open-loop    |
| RouterDC†             | X     | X     | multiple LLMs            | Routing      |
| MoA†                  | X     | X     | multiple LLMs            | Aggregation  |
| Self-Reflect-5        | TBD   | TBD   | GPT-5                    | Self-reflect |
| **Ours (matched)**    | TBD   | TBD   | GPT-5,Claude,Gemini,...  | **Reactive** |

† Results from Conductor paper
→ 预期: Ours (matched) ≈ Conductor 或略高（闭环优势在困难题上应该体现）
→ 论文叙述: "Under matched worker conditions, reactive orchestration achieves comparable or superior results"
```

**B. 跨模态主结果表（Table 2 — 独有覆盖度）**:
```
| Method            | NQ  | HotpotQA | 2Wiki | MuSiQue | TriviaQA | PopQA | HumanEval | MBPP | GPQA | LCB  |
|-------------------|-----|----------|-------|---------|----------|-------|-----------|------|------|------|
| Router-R1†        | X   | X        | X     | X       | X        | X     | -         | -    | -    | -    |
| Direct-GPT-4o     | X   | X        | X     | X       | X        | X     | X         | X    | X    | X    |
| Self-Reflect-5    | X   | X        | X     | X       | X        | X     | X         | X    | X    | X    |
| Ours (cheap)      | X   | X        | X     | X       | X        | X     | X         | X    | X    | X    |
| Ours (matched)    | -   | -        | -     | -       | -        | -     | -         | -    | X    | X    |

→ 论文叙述: "OrchestratorR1 is the only method covering all 10 benchmarks with a single policy"
```

**C. 闭环 vs 开环 Controlled Experiment（⚠️ 核心消融）**:
```
使用同一个训练好的模型:
- Full (reactive): 正常闭环模式
- w/o reactive (open-loop): 一次性生成全部 call，模拟 Conductor 范式
  → OpenLoopGenerationManager（已实现 ✅）
- w/o reactive + adversarial: 注入噪声 agent 返回

分别在 matched-worker 和 cheap-worker 两种配置下测试
→ 如果两种配置下 reactive 都优于 open-loop，结论非常 convincing
```

**D. Pareto 前沿分析（Figure — 成本优势）**:
```
α = {0, 0.1, 0.3, 0.5, 0.7, 0.9}
两条 Pareto 曲线:
- Ours (cheap worker) → 低成本区域的 Pareto 前沿
- Ours (matched worker) → 高性能区域的点
- Conductor → 固定高成本高性能点
- Direct-GPT-4o → 固定点
- Router-R1 → 固定点
→ 展示: 我们覆盖了从低成本到高性能的完整 spectrum
```

**E. 简单/复杂题效率分析**:
```
按题目难度分组，统计平均轮数/成本
预期: 简单题 1.2 轮 vs 开环 3+ 步
→ matched 配置下更显著（前沿模型更贵，省的更多）
```

### 4.2 修订后的 "可赢" vs "风险" 实验

**有信心赢的实验**:

| 实验 | 为什么你能赢 |
|------|------------|
| Matched-worker GPQA/LCB | 同等 worker 下，reactive 闭环应该 ≥ open-loop；critic→retry 在困难题上有额外收益 |
| 跨模态覆盖度 | Conductor 没有 QA/多跳推理，你覆盖 10 个 benchmark |
| 闭环错误恢复率 | Conductor 开环无法恢复，你的 critic→retry 可以 |
| 简单题成本 | 你 1 步解决 vs Conductor 固定多步 workflow |
| Pareto 可调性 | 你有显式 α 参数，两种 worker 配置覆盖全 spectrum |
| 可解释性 | 显式功能角色 vs 隐式 prompt 指令 |

**有风险的实验（需要验证）**:

| 实验 | 风险点 | 应对 |
|------|--------|------|
| Matched-worker 绝对分数 | 我们训练用 cheap worker，测试换 matched 可能有 gap | 如果 zero-shot transfer 不好，考虑用 matched worker 做少量 fine-tune |
| AIME 2025 | 我们没有数学训练数据 | 不评估，或只作为 out-of-distribution 泛化测试 |
| 训练数据效率 | Conductor 960 条 vs 我们 5K+ | 强调总训练成本（API 费用）我们更低 |

---

## 5. 有 A100/V100 80GB 后的技术路线调整

### 5.1 算力释放的机会

| 升级 | 3090 (24GB) | A100/V100 (80GB) | 影响 |
|------|-------------|-------------------|------|
| 模型规模 | 3B 为主，7B 勉强 | 7B 轻松，14B 可行 | ★★★ |
| batch size | G=8，per_device=2 | G=16，per_device=8 | ★★★ 训练效率翻倍 |
| 上下文长度 | max_prompt=4096 | max_prompt=8192 | ★★ 支持更长编排链 |
| Flash Attention | Windows 不支持 | ✅ 全面支持 | ★★ 推理速度 2-3x |
| 全参数训练 | 3B 全参 / 7B LoRA | 7B 全参 / 14B LoRA | ★★★ |
| vLLM rollout | 不好用 | ✅ 可用 | ★ 可选但非必需 |

### 5.2 推荐配置（4×A100 80GB）

```
基座模型: Qwen2.5-7B-Instruct (主实验)
         Qwen2.5-14B-Instruct (scaling 实验)
         Qwen2.5-3B-Instruct (消融/对比)

GRPO 训练:
  num_generations (G): 16 (A100) vs 8 (3090)
  per_device_batch_size: 4
  gradient_accumulation: 4
  max_prompt_length: 8192
  max_response_length: 2048
  max_obs_length: 1024
  learning_rate: 1e-6
  bf16: True
  flash_attn: True

预计训练时间 (7B, 5K samples):
  SFT 热身: ~30min
  GRPO 主训练: ~8-12h (4×A100)
  × 3 seeds: ~24-36h

预计显存 (单卡):
  7B 模型: ~14GB
  优化器 (FSDP ZeRO-2): ~7GB
  GRPO rollout (G=16): ~40GB
  总计: ~61GB ✓ (80GB 内)
```

### 5.3 有算力后的额外实验

| 实验 | 说明 | 优先级 |
|------|------|--------|
| **7B 全参 GRPO** | 主实验用 7B，直接对标 Conductor | ★★★ |
| **Scaling analysis**: 3B→7B→14B | 展示编排能力随模型规模的变化 | ★★☆ |
| **G=16 vs G=8 消融** | GRPO group size 对训练效果的影响 | ★☆☆ |
| **长上下文编排** | max_turns=8 或 10，测试极复杂任务 | ★☆☆ |
| **多 seed 统计** | 3-5 个 seed 的 mean±std | ★★★ |

---

## 6. 更新后的研究计划（A100 版本）

### Timeline（6 周，假设 4×A100 80GB 可用）

**第 1 周: 数据 + 基础设施**
- [ ] 准备全部数据集（NQ, HotpotQA, 2Wiki, MuSiQue, TriviaQA, PopQA, HumanEval, MBPP）
- [ ] 添加 GPQA 和 LiveCodeBench 数据加载
- [ ] 生成 SFT 热身数据 200 条（覆盖 6 种 agent type × 3 种任务类型 × 多种复杂度）
- [ ] 配置 A100 训练环境（FSDP + flash-attn + bf16）
- [ ] 跑 base model baseline (7B, 未训练)
- [ ] 跑 Direct-GPT-4o baseline

**第 2 周: 训练 + 主实验**
- [ ] SFT 热身（7B, ~30min）
- [ ] GRPO 主训练 × 3 seeds（7B, ~36h 总计）
- [ ] 全部测试集评估（3 seeds × 所有数据集）
- [ ] 启动 3B 训练（消融对比用，可并行）

**第 3 周: 基线 + 消融**
- [ ] 实现 ReAct baseline (Qwen2.5-7B)
- [ ] 实现 "w/o reactive" 消融（⚠️ 最重要）
- [ ] 实现 "adversarial intermediate failure" 实验
- [ ] 其余消融: w/o critic, w/o decomposer, SFT-only, α=0, Fixed-Pipeline
- [ ] α 敏感性 (0, 0.1, 0.3, 0.5, 0.7, 0.9)
- [ ] 复现/引用 Conductor 和 Router-R1 结果

**第 4 周: 分析 + Scaling**
- [ ] Agent 调用分布热力图
- [ ] 训练动态分析
- [ ] Pareto 前沿图
- [ ] Reactive case study × 3（展示中间反馈→策略调整）
- [ ] Scaling: 3B vs 7B (vs 14B 如果有余力)
- [ ] 简单/复杂题效率分组分析

**第 5 周: 论文撰写**
- [ ] Method (3 pages)
- [ ] Experiments + Analysis (4.5 pages)
- [ ] Introduction + Related Work (2.5 pages)
- [ ] Abstract + Conclusion

**第 6 周: 打磨 + 提交**
- [ ] 补实验（reviewer 预判的问题）
- [ ] 图表美化
- [ ] 校对 + 匿名化
- [ ] 5/4 提交 abstract → 5/6 提交 full paper

---

## 7. Reviewer 会问的问题（预判 + 准备回答）— 修订版

### Q1: "Why not just use Conductor? It already achieves SOTA on GPQA/LiveCodeBench."
**A**: Table 1 shows that under **matched worker conditions** (same frontier models), OrchestratorR1 achieves comparable or superior results to Conductor, while additionally supporting QA and multi-hop reasoning tasks that Conductor does not cover. Moreover, our reactive paradigm provides error recovery (Table X: Y% vs 0%) and cost-adaptive early stopping that open-loop planning cannot.

### Q2: "Your numbers on GPQA/LiveCodeBench are much lower than Conductor's."
**A**: Our Table 1 (matched-worker) shows results **under the same worker pool as Conductor** — where we achieve X% GPQA vs Conductor's 87.5%. Table 2 additionally shows results with cost-efficient workers (GPT-4o-mini) at 10-50× lower cost per query, demonstrating the practical applicability of our approach.

### Q3: "Conductor also uses RL (GRPO) and orchestrates multiple agents. What's novel about your approach?"
**A**: 关键区别在编排**范式**：
- Conductor: plan-then-execute（一次性输出完整 workflow，然后执行）
- OrchestratorR1: reactive sequential（每步观察 agent 返回，再决定下一步）
这两种范式有根本的 trade-off: Conductor 支持并行但不能适应中间失败; OrchestratorR1 是串行但能实时调整。我们还引入了显式功能角色（6 种 agent type，而非 model-id 选择）和成本感知奖励。

### Q4: "The agent pool is hand-designed. How do you know 6 agents is optimal?"
**A**: Agent 池是固定基础设施，类似 tool-use 中的 tool 定义。消融实验（Table 2）证明每种 agent 的贡献。未来工作可以探索自动发现 agent 角色。

### Q5: "GRPO without critic — isn't PPO (Router-R1) better?"
**A**: GRPO 训练效率更高（单网络 vs 双网络），在我们的设置下 GRPO 已足够有效（G=16 提供足够的组内方差）。Table X 对比了 3B 下 GRPO vs PPO 的效果。

### Q6: "SFT warmup — doesn't this leak supervision signal？"
**A**: SFT-only 消融（Table 2）准确率仅 X%，GRPO 后提升到 Y%，差异 Z% 全部来自 RL。SFT 仅教格式，不教策略。

### Q7: "Serial execution is a bottleneck. Why not support parallel?"
**A**: 串行是反应式编排的代价——为了获得闭环适应能力。在延迟敏感场景下，我们的 1-2 步路径（简单题）实际比 Conductor 的 3+ 步并行更快（因为我们不需要等待所有 worker 返回）。未来可以引入 conditional parallel execution。

### Q8 (新): "Conductor 用了 RouterDC, MoA, MASRouter 等多个 strong baseline，你的基线覆盖是否充分？"
**A**: 我们的基线包括：(1) Router-R1（直接前作，学习型路由）；(2) Direct-GPT-4o（强模型上限）；(3) Self-Reflection 5-turn（多轮自反思）；(4) ReAct（标准 tool-use）；(5) w/o reactive（模拟 Conductor 的开环范式）。由于任务领域不同（我们覆盖 QA+推理+代码，而 RouterDC/MoA 等主要在推理/代码上有结果），直接复现这些基线在我们的全部 benchmark 上反而会引入不公平对比。我们引用 Conductor 论文中的 RouterDC/MoA 数据作为参考。

### Q9 (新): "Conductor 只需要 960 条训练数据，你需要 5K+，训练效率更低？"
**A**: 虽然我们需要更多训练数据，但关键差异在于**总训练成本**：Conductor 需要 2×H100 + GPT-5 级 API（每次 GRPO rollout 都要调 GPT-5），而我们只需要单 GPU + GPT-4o-mini API。实际美元成本我们更低。此外，我们的训练数据覆盖 3 种模态（QA/推理/代码），Conductor 的 960 条只覆盖推理+代码。
