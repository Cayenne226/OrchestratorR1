# OrchestratorR1 — NeurIPS 2026 研究计划

> **目标会议**: NeurIPS 2026
> **Abstract deadline**: 2026-05-04 AOE
> **Full paper deadline**: 2026-05-06 AOE
> **距截稿**: ~6 周（从 2026-03-24 起算）
> **会议日期**: 2026-12-06 ~ 2026-12-12
> ⚠️ **2026-03-26 重大更新**: 根据 Conductor 实际信息修正竞争分析和实验方案

---

## 0. 竞争格局分析（必读）

### 直接竞争者

| 工作 | 发表 | 方法 | 你的差异点 |
|------|------|------|-----------|
| **Conductor** (Nielsen et al.) | ICLR 2026 | Qwen2.5-7B GRPO，plan-then-execute，调度 GPT-5/Claude Sonnet 4/Gemini 2.5 Pro 等 7 个前沿 worker，生成 Python list (model_id + subtask + access_list)，支持递归 self-invocation。**结果: GPQA 87.5%, LCB 83.93%, AIME 93.3%** | ❶ 你是 reactive（逐步决策），❷ 功能角色而非模型选择，❸ 显式成本优化，❹ 不依赖 GPT-5 级 worker |
| **AgentConductor** (Wang et al.) | arXiv 2602, Feb 2026 | SFT+GRPO，DAG 拓扑演化，密度函数，竞赛级代码生成 | ❶ 你的 agent 是功能异质的（不仅是拓扑差异），❷ 跨模态泛化（QA+推理+代码），❸ reactive 而非 DAG 预规划 |
| **Router-R1** | NeurIPS 2025 | PPO，单轮路由到 LLM | ❶ 你支持多轮编排，❷ 功能角色而非同质模型选择 |
| **Agent-R1** | arXiv 2511 | 端到端 RL 训练 agent（tool-use） | ❶ 你是 orchestrator 而非单 agent，❷ 多 agent 协作 |
| **MAGRPO** | 2025-2026 | Multi-Agent GRPO，多个 agent 共同训练 | ❶ 你只训练 orchestrator，agent pool 固定（更实用） |

### 风险评估

**高风险**：Conductor 已经是 ICLR 2026 级别工作，使用 GPT-5/Claude Sonnet 4/Gemini 2.5 Pro 作为 worker，GPQA 87.5%, LiveCodeBench 83.93%, AIME 93.3%。NeurIPS reviewer 会直接对比。

**关键认识**：我们不可能在绝对分数上与 Conductor 竞争（它的 worker 是 GPT-5 级别），必须在**不同维度**上证明价值。

**关键差异化方向（必须至少选 2 个做透）**：
1. **Reactive 闭环优势**：实验证明中间结果反馈导致策略调整的实例和统计数据
2. **跨模态泛化**：一个策略同时处理 QA/推理/代码（Conductor 只做推理+代码，Router-R1 只做 QA）
3. **成本-质量 Pareto 优化**：显式 α 调节 + Pareto 前沿分析，强调便宜 worker 也能智能编排
4. **实际部署性**：不依赖 GPT-5 等前沿模型，训练成本低（单 GPU + GPT-4o-mini API）

---

## 1. 论文定位与 Story

### 推荐标题
**"OrchestratorR1: Learning Reactive Multi-Agent Orchestration via Reinforcement Learning"**

### One-sentence pitch
> A small LLM learns to reactively orchestrate functionally specialized agents through end-to-end RL, spontaneously developing adaptive strategies that range from single-step execution to multi-round decomposition-verification pipelines, achieving superior cost-quality tradeoffs across QA, reasoning, and code tasks.

### 核心 Story（reviewers 角度）— 修订版

**What's wrong with the world?**
- Router-R1 只能路由到单一模型，无法分解/验证/综合
- Conductor 一次性生成完整 workflow，无法根据中间结果调整策略；且依赖 GPT-5 级前沿 worker，部署成本极高
- AgentConductor 限于代码任务，且 DAG 拓扑是静态的
- AutoGen/CrewAI 需要人工设计流程

**What do you do?**
- 将 multi-agent orchestration 建模为 MDP
- Orchestrator 逐步观察 agent 返回、逐步决策（reactive）
- 使用**廉价、功能化** worker（非前沿模型），通过智能编排最大化性价比
- 单一 GRPO 奖励函数驱动，涌现自适应行为谱
- 跨 QA/推理/代码三个模态泛化

**Why does it work?**
- Reactive 闭环 > 开环规划：可以根据 agent 返回质量临时调整后续动作
- 功能角色 > 模型选择：不同 agent 执行不同认知功能（分解、验证、综合）
- 成本感知奖励 > 忽略成本：α 调节 Pareto 前沿
- 不依赖前沿 worker：用 GPT-4o-mini 也能做智能编排

**What's the evidence?**
- Table 1: 三个 track × 10 benchmarks — 唯一全覆盖方法
- Table 2: 闭环 vs 开环 controlled experiment — reactive 的结构性优势
- Figure: Agent 调用分布热力图（涌现自适应性）
- Figure: 训练过程行为变化（RL 驱动涌现）
- Figure: Pareto 曲线（成本-质量权衡，标注 Conductor 为"高成本高质量"参考点）
- Case study: Reactive 策略调整实例（Conductor 范式无法实现）
- Table (Appendix): 与 Conductor 共有 benchmark 的对比（诚实展示差距，解释 worker 差异）

---

## 2. 实验计划（核心，决定论文成败）

### 2.1 评估基准

| Track | 数据集 | 指标 | 测试量 |
|-------|--------|------|--------|
| Track 1: 简单 QA | NQ, TriviaQA, PopQA | EM, F1 | 各 500 |
| Track 2: 多跳推理 | HotpotQA, 2WikiMultihop, MuSiQue, Bamboogle | F1 | 各 500 |
| Track 3: 代码 | HumanEval, MBPP | Pass@1 | 164 + 427 |

### 2.2 基线（修订版 — 双配置策略）

> **核心策略**: 训练用 cheap worker，测试同时评估 matched（前沿）和 cheap（廉价）两种 worker 配置。

> **Conductor 论文的基线**: RouterDC, Smoothie, MASRouter, MoA, 5-turn Self-Reflection, Single-best worker

| 基线 | 优先级 | 来源 | 说明 |
|------|--------|------|------|
| **Conductor** (matched worker) | ★★★ 最高 | 引用论文数据 | GPQA/LCB 上正面对标（同等 worker 池） |
| **Router-R1** | ★★★ | 官方 checkpoint | 直接前作，QA 任务核心对比 |
| **Direct-GPT-4o** | ★★★ | 直接调 API | ✅ **已完成**。上限参考 |
| **Self-Reflection 5-turn** | ★★★ | 自实现 | Conductor 论文中的重要基线 |
| **ReAct** (Qwen2.5-3B) | ★★☆ | 自实现 | 标准 tool-use 基线 |
| **Ours (matched worker)** | ★★★ | 自评估 | Agent pool 对齐 Conductor（GPT-5/Claude/Gemini 等），正面对标 |
| **Ours (cheap worker)** | ★★★ | 自评估 | Agent pool 用 GPT-4o-mini/GPT-4o，展示成本效率 |
| **AgentConductor** | ★☆☆ | 引用论文数据 | 代码 track 引用 |
| **MoA** | ★☆☆ | 可选实现 | 在 QA 上实现 |
| **Fixed-Pipeline** | ★★☆ | 自实现 | 消融：固定 6 步流水线 |

> **策略**: RouterDC/Smoothie/MASRouter 等 Conductor 基线在 Related Work 中讨论并引用数据，不自己复现。

### 2.3 消融实验

| 实验 | 配置 | 验证目标 |
|------|------|----------|
| Full | 完整方案 | 上界 |
| w/o reactive | 一次性生成全部 call（模拟 Conductor） | reactive 的贡献 |
| w/o critic | 移除 critic agent | 中间验证的价值 |
| w/o decomposer | 移除 decomposer | 任务分解的价值 |
| w/o cost penalty | α=0 | 成本优化的必要性 |
| SFT-only | 仅 SFT 无 GRPO | RL 的贡献 |
| Fixed-Pipeline | 强制 6 步 | 自适应 vs 固定流程 |

> **★★★ 最重要的消融**: "w/o reactive" — 直接回答 "为什么不用 Conductor 的方式？"

### 2.4 分析实验（论文亮点）

| 分析 | 图表 | 说明 |
|------|------|------|
| Agent 调用分布 | 热力图 (6 agents × N datasets) | 证明涌现的 task-dependent 行为 |
| 训练动态 | 折线图 (steps vs avg_turns, by complexity) | 证明 RL 驱动行为变化 |
| Pareto 前沿 | 散点图 (cost vs F1, 多个 α) | 证明成本-质量可调 |
| Reactive 策略调整 | Case study × 3 | 展示中间反馈→策略修正的具体实例 |
| Scaling | 3B vs 7B | 模型规模对编排能力的影响 |

### 2.5 预期结果（双配置策略）

**Matched-worker 配置（Agent pool = GPT-5/Claude Sonnet 4/Gemini 2.5 Pro 等）**:

| Track | vs Conductor (same workers) | 预期 |
|-------|-----------------------------|------|
| GPQA | 正面对标 87.5% | **接近或超过** — reactive 闭环在困难推理题上有 critic→retry 优势 |
| LiveCodeBench | 正面对标 83.93% | **接近** — 代码题可能更依赖 worker 本身能力 |
| QA/多跳推理 | Conductor 无此数据 | **我们独有** — 展示跨模态泛化 |

**Cheap-worker 配置（Agent pool = GPT-4o-mini/GPT-4o）**:

| Track | vs Router-R1 | vs Direct-GPT-4o |
|-------|-------------|-------------------|
| 简单 QA | 平手或 +1~3% F1，成本更低 | 略低，但成本 1/10 |
| 多跳推理 | +5~10% F1（分解+验证优势） | 接近或略低 |
| 代码 | Router-R1 不支持 | 可能略高（编排优势） |
| GPQA | Router-R1 不支持 | 持平或略高 |

**核心叙事**:
> "Under matched worker conditions, OrchestratorR1's reactive paradigm achieves comparable or superior results to Conductor's open-loop approach (Table 1), while additionally covering QA and multi-hop reasoning tasks (Table 2). With cost-efficient workers, OrchestratorR1 provides explicit Pareto optimization (Figure X), enabling deployment at 10-50× lower cost."

**风险点**: matched-worker 下 zero-shot transfer 可能有 gap（训练用 cheap worker，测试用 matched worker）。
**应对**: 如果 gap 明显，用 matched worker 做少量 fine-tune（几百步 GRPO），或直接用 matched worker 训练一个版本。

---

## 3. 技术改进清单（提升论文竞争力）

### 3.1 必做（P0，影响审稿结论）

- [ ] **实现 "w/o reactive" 消融**：让模型一次性输出所有 call（不注入中间 information），对比 reactive 模式 → ✅ OpenLoopGenerationManager 已实现
- [ ] **Matched-worker Agent Pool 配置**：实现可切换的 worker pool（cheap: GPT-4o-mini/GPT-4o, matched: GPT-5/Claude Sonnet 4/Gemini 2.5 Pro 等），支持运行时切换
- [ ] **添加 GPQA 数据集**：✅ 已完成（198 条）
- [ ] **添加 LiveCodeBench**：✅ 已完成（202 条）
- [ ] **Self-Reflection 5-turn 基线**：实现并在全部数据集上评估
- [ ] **Conductor matched-worker 正面对标**：在 GPQA/LiveCodeBench 上用相同 worker pool 对比
- [ ] **统计显著性**：所有结果报告 3 次独立 seed 的 mean ± std

### 3.2 应做（P1，显著提升论文质量）

- [ ] **动态 agent pool**：训练时随机 mask 部分 agent，测试 orchestrator 的鲁棒性（类似 Conductor 的 randomized pool）
- [ ] **7B 模型实验**：至少在 1-2 个数据集上展示 scaling 结果
- [ ] **长上下文处理**：多轮对话后上下文很长，需要 truncation 策略或 sliding window
- [ ] **奖励函数改进**：考虑 process reward（每步给中间奖励）而非仅终端奖励
- [ ] **Curriculum learning**：先训简单 QA → 再训多跳 → 最后混合，可能加速收敛

### 3.3 可选（P2，锦上添花）

- [ ] **多语言评估**：在中文 QA 数据集上测试泛化
- [ ] **agent pool 扩展性**：测试 agent 数从 4→6→8 的效果
- [ ] **与 Conductor 的混合方法**：先 plan-then-execute 生成初始 workflow，再 reactive 调整
- [ ] **理论分析**：reactive MDP 的 regret bound vs open-loop planning

---

## 4. 论文结构

```
Title: OrchestratorR1: Learning Reactive Multi-Agent Orchestration
        via Reinforcement Learning

Abstract (250 words)

1. Introduction (1.5 pages)
   - Multi-agent systems 的局限（手工设计 vs 端到端学习）
   - Router-R1 → Conductor 的进展，及各自的局限（开环规划、依赖前沿 worker）
   - 本文: reactive sequential orchestration + 跨模态 + 成本优化 + 廉价 worker
   - Contributions (3 条)

2. Related Work (1 page)
   - LLM Routing (Router-R1, RouteLLM, FrugalGPT, AutoMix)
   - RL-Learned Orchestration (Conductor, AgentConductor, Agent-R1)
   - Multi-Agent Frameworks (AutoGen, CrewAI, LangGraph)
   - Agentic RL (GRPO, iStar, IGPO)

3. Method (2.5 pages)
   3.1 Problem Formulation (MDP)
   3.2 Orchestrator Policy π_θ
   3.3 Agent Pool Design (6 functional roles)
   3.4 Reactive Generation Loop
   3.5 Reward Function R(τ)
   3.6 Training: SFT Warmup + GRPO

4. Experiments (3 pages)
   4.1 Setup (datasets, baselines, metrics, hyperparameters)
   4.2 Main Results (Table 1: cross-modal, 10 benchmarks × baselines — 不含 Conductor)
   4.3 Comparison with Conductor (Table 2: shared benchmarks, 诚实对比 + worker 差异分析)
   4.4 Ablation Studies (Table 3: 7 ablations, 重点: reactive vs open-loop)
   4.5 Cost-Quality Tradeoff (Pareto curve, 标注 Conductor 为高成本参考点)

5. Analysis (1.5 pages)
   5.1 Emergent Adaptive Behavior (heatmap + training dynamics)
   5.2 Reactive vs Plan-then-Execute (quantitative + case study)
   5.3 Scaling Analysis (3B vs 7B)

6. Conclusion (0.5 page)

Appendix:
   A. Agent System Prompts
   B. Matched-Condition Comparison with Router-R1 (Table A1: same data, architecture-only diff)
   C. Full Evaluation Results per Dataset
   D. Training Details and Hyperparameters
   E. Additional Case Studies
   F. Compute Cost Analysis
```

**页数**: 主文 10 页 + references + appendix（符合 NeurIPS 格式）

---

## 5. 时间线（6 周冲刺计划）

### 第 1 周（3/24 - 3/30）：基础设施 + 数据

| 天 | 任务 | 产出 |
|----|------|------|
| 周一-二 | 准备全部训练/测试数据（NQ, HotpotQA, 2Wiki, MuSiQue, TriviaQA, PopQA, HumanEval, MBPP） | data/*.jsonl |
| 周二-三 | 添加 GPQA 和 LiveCodeBench 数据加载 | prepare_gpqa.py, prepare_livecode.py |
| 周三-四 | 生成 SFT 热身数据（100 条，覆盖所有 agent type 和任务类型） | sft_warmup.jsonl |
| 周五 | 跑 base model baseline（未训练模型在所有数据集上的表现） | eval/results/orch_base.json |
| 周末 | 跑 Direct-GPT-4o 和 Fixed-Pipeline baselines | eval/results/direct_strong.json, fixed_pipeline.json |

### 第 2 周（3/31 - 4/06）：训练

| 天 | 任务 | 产出 |
|----|------|------|
| 周一 | SFT 热身训练（3B, 1-2h） | checkpoints/sft_warmup/ |
| 周一-三 | GRPO 主训练 Seed 1（3B, 12-20h） | checkpoints/orch_grpo_seed1/ |
| 周三-五 | GRPO 主训练 Seed 2, 3（可能需排队） | checkpoints/orch_grpo_seed2,3/ |
| 周末 | 全部测试集评估（3 seeds × 所有数据集） | eval/results/orch_grpo_*.json |

### 第 3 周（4/07 - 4/13）：基线 + 消融

| 天 | 任务 | 产出 |
|----|------|------|
| 周一-二 | 实现 ReAct baseline（Qwen2.5-3B + tool-use） | eval/results/react.json |
| 周二-三 | 实现 "w/o reactive" 消融（关键！） | eval/results/ablation_no_reactive.json |
| 周三-四 | 其余消融：w/o critic, w/o decomposer, SFT-only, α=0 | eval/results/ablation_*.json |
| 周五 | 复现/引用 Conductor 和 Router-R1 结果 | eval/results/conductor.json, router_r1.json |
| 周末 | α 敏感性实验 (α = 0, 0.1, 0.3, 0.5, 0.7, 0.9) | eval/results/alpha_*.json |

### 第 4 周（4/14 - 4/20）：分析 + 7B 实验

| 天 | 任务 | 产出 |
|----|------|------|
| 周一 | Agent 调用分布热力图 | figures/heatmap.pdf |
| 周二 | 训练动态分析（behavior change over steps） | figures/training_dynamics.pdf |
| 周三 | Pareto 曲线 | figures/pareto.pdf |
| 周四 | Reactive case study（3 个典型案例） | case_studies.md |
| 周五-末 | 7B 模型训练 + 评估（2×A800 或缩小数据量在 3090 上跑） | eval/results/orch_7b.json |

### 第 5 周（4/21 - 4/27）：论文撰写

| 天 | 任务 | 产出 |
|----|------|------|
| 周一 | Method section (3.1-3.6) | paper/sections/method.tex |
| 周二 | Experiments section (4.1-4.4) + 所有表格 | paper/sections/experiments.tex |
| 周三 | Analysis section (5.1-5.3) + 所有图 | paper/sections/analysis.tex |
| 周四 | Introduction + Related Work | paper/sections/intro.tex, related.tex |
| 周五 | Abstract + Conclusion + Appendix | paper/main.tex |
| 周末 | 通读 + 统一记号 + 查错 | 完整初稿 |

### 第 6 周（4/28 - 5/04）：打磨 + 提交

| 天 | 任务 | 产出 |
|----|------|------|
| 周一-二 | 补实验（reviewer 可能问的点：额外数据集、更多 seed） | 补充结果 |
| 周三 | 图表美化（matplotlib → 论文级别） | 最终 figures/ |
| 周四 | 论文修改：精简、加强 claim 的证据支撑 | 修改稿 |
| 周五(5/2) | 最终校对 + checklist + 匿名化检查 | 终稿 |
| 周六(5/3) | 准备 OpenReview 提交材料 | supplementary.zip |
| **周日(5/4)** | **提交 Abstract** | ✅ |
| **周二(5/6)** | **提交 Full Paper** | ✅ |

---

## 6. 硬件与资源需求

### 计算资源

| 任务 | 硬件 | 时间 | API 成本 |
|------|------|------|---------|
| SFT 热身 (3B) | 4×RTX 3090 | 1-2h | $0 |
| GRPO 训练 (3B) × 3 seeds | 4×RTX 3090 | 36-60h | ~$15 |
| GRPO 训练 (7B) × 1 seed | 2×A800 或 4×3090 (LoRA) | 20-30h | ~$5 |
| 消融 × 6 | 4×RTX 3090 | 72-120h | ~$30 |
| 评估（全部） | 单 GPU | 10-20h | ~$50 |
| **总计** | | ~150-230h GPU | **~$100** |

### 关键风险与应对

| 风险 | 概率 | 应对 |
|------|------|------|
| GRPO 训练不收敛 | 中 | 增加 SFT 热身数据到 200 条；调低 lr；先用 LoRA 快速验证 |
| Conductor 绝对分数碾压 | **确定** | ⚠️ Conductor 用 GPT-5 级 worker，87.5% GPQA。定位为不同范式+成本效率+跨模态，完全避开数字战 |
| Reviewer 要求 vs Conductor 对比 | 高 | 准备: (1) worker 成本分析 (2) 闭环 vs 开环 controlled experiment (3) 跨模态覆盖度 |
| API 调用预算超支 | 低 | 训练阶段全用 gpt-4o-mini；限制 max_turns=4 |
| 7B 模型显存不够 | 中 | 用 LoRA + gradient checkpointing；或只报 3B 结果 + 附录 7B |
| 6 周写不完 | 中 | 优先保证 Table 1 (主结果) + Figure 2 (热力图) + 1 个消融，其余可精简 |

---

## 7. 最低可发表版本（Plan B）

如果时间/算力不够，以下是**最低可接受的实验集**：

### 必须有（缺一不可）
- [ ] Table 1: 主结果（至少 NQ + HotpotQA + HumanEval，对比 Router-R1 + Direct-Strong + ReAct）
- [ ] Table 2: 至少 3 个消融（Full / SFT-only / Fixed-Pipeline）
- [ ] Figure: Agent 调用分布热力图
- [ ] Figure: Pareto 曲线（至少 3 个 α 值）

### 强烈建议有
- [ ] Conductor 对比（哪怕只引用数据）
- [ ] "w/o reactive" 消融
- [ ] 训练动态图

### 可以不有（Appendix 或 future work）
- [ ] 7B 实验
- [ ] LiveCodeBench
- [ ] 多语言
- [ ] 理论分析

---

## 8. 与 Conductor/AgentConductor 的差异化论述模板

### vs Conductor（最重要的对比）— 双配置策略

> While the Conductor (Nielsen et al., 2026) achieves remarkable results (87.5% GPQA, 83.9% LiveCodeBench)
> by orchestrating frontier models (GPT-5, Claude Sonnet 4, Gemini 2.5 Pro) in a plan-then-execute paradigm,
> OrchestratorR1 adopts a **reactive sequential** paradigm with **functionally specialized** agents.
>
> Under matched worker conditions (Table 1), OrchestratorR1 achieves comparable or superior results,
> demonstrating that the reactive paradigm is at least as effective as open-loop planning when given
> the same worker capabilities. We further show three structural advantages:
> (1) **closed-loop error recovery** — when an agent returns incorrect results, the orchestrator
> dynamically invokes a critic for verification and retries (Table X: Y% recovery rate vs Conductor's 0%);
> (2) **cross-modal generalization** — a single policy handles QA, multi-hop reasoning, and code tasks
> across 10 benchmarks, while Conductor focuses on reasoning and code;
> (3) **cost-adaptive orchestration** — with cost-efficient workers (GPT-4o-mini), OrchestratorR1 provides
> explicit Pareto optimization (Figure X), enabling deployment at 10-50× lower cost.

### vs AgentConductor

> AgentConductor (Wang et al., 2026) trains an orchestrator for DAG topology evolution in code
> generation. While sharing the SFT+RL training paradigm, OrchestratorR1 differs in three key ways:
> (1) our orchestration is **reactive** (sequential decisions conditioned on agent responses) rather
> than upfront DAG generation;
> (2) our agents are **functionally specialized** (refiner, decomposer, critic) rather than
> homogeneous coding agents with topological variations;
> (3) our framework generalizes across **three task modalities** with a single policy, whereas
> AgentConductor is specialized for code generation.

---

## 9. Checklist（提交前检查）

- [ ] 论文匿名化（无作者信息、无 GitHub 链接、无 "our lab"）
- [ ] 所有实验结果可复现（附 seed、超参数表）
- [ ] NeurIPS 格式（neurips_2026.sty）
- [ ] Ethics statement
- [ ] Broader impact statement
- [ ] Reproducibility checklist（NeurIPS 要求）
- [ ] 主文 ≤ 10 页（不含 references 和 appendix）
- [ ] Supplementary materials（代码、完整结果）
- [ ] 所有 claims 有实验证据支撑
- [ ] 所有图表有 error bars 或 std
- [ ] Related work 覆盖 Conductor 和 AgentConductor
