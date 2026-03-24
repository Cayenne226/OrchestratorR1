# OrchestratorR1 — NeurIPS 2026 研究计划

> **目标会议**: NeurIPS 2026
> **Abstract deadline**: 2026-05-04 AOE
> **Full paper deadline**: 2026-05-06 AOE
> **距截稿**: ~6 周（从 2026-03-24 起算）
> **会议日期**: 2026-12-06 ~ 2026-12-12

---

## 0. 竞争格局分析（必读）

### 直接竞争者

| 工作 | 发表 | 方法 | 你的差异点 |
|------|------|------|-----------|
| **Conductor** (Nielsen et al.) | ICLR 2026 | 7B 模型 GRPO，plan-then-execute，一次性生成完整 workflow（agent 选择 + prompt + 通信拓扑） | ❶ 你是 reactive（逐步决策），❷ 功能角色而非模型选择，❸ 显式成本优化 |
| **AgentConductor** (Wang et al.) | arXiv 2602, Feb 2026 | SFT+GRPO，DAG 拓扑演化，密度函数，竞赛级代码生成 | ❶ 你的 agent 是功能异质的（不仅是拓扑差异），❷ 跨模态泛化（QA+推理+代码），❸ reactive 而非 DAG 预规划 |
| **Router-R1** | NeurIPS 2025 | PPO，单轮路由到 LLM | ❶ 你支持多轮编排，❷ 功能角色而非同质模型选择 |
| **Agent-R1** | arXiv 2511 | 端到端 RL 训练 agent（tool-use） | ❶ 你是 orchestrator 而非单 agent，❷ 多 agent 协作 |
| **MAGRPO** | 2025-2026 | Multi-Agent GRPO，多个 agent 共同训练 | ❶ 你只训练 orchestrator，agent pool 固定（更实用） |

### 风险评估

**高风险**：Conductor 已经是 ICLR 2026 oral/spotlight 级别，NeurIPS reviewer 会直接对比。如果你的实验结果不能 convincingly 证明 reactive > plan-then-execute，论文会被打回。

**关键差异化方向（必须至少选 2 个做透）**：
1. **Reactive 闭环优势**：实验证明中间结果反馈导致策略调整的实例和统计数据
2. **跨模态泛化**：一个策略同时处理 QA/推理/代码（Conductor 只做推理，AgentConductor 只做代码）
3. **成本-质量 Pareto 优化**：显式 α 调节 + Pareto 前沿分析
4. **功能角色涌现**：证明 RL 学会了 task-dependent 的角色调度

---

## 1. 论文定位与 Story

### 推荐标题
**"OrchestratorR1: Learning Reactive Multi-Agent Orchestration via Reinforcement Learning"**

### One-sentence pitch
> A small LLM learns to reactively orchestrate functionally specialized agents through end-to-end RL, spontaneously developing adaptive strategies that range from single-step execution to multi-round decomposition-verification pipelines, achieving superior cost-quality tradeoffs across QA, reasoning, and code tasks.

### 核心 Story（reviewers 角度）

**What's wrong with the world?**
- Router-R1 只能路由到单一模型，无法分解/验证/综合
- Conductor 一次性生成完整 workflow，无法根据中间结果调整策略
- AgentConductor 限于代码任务，且 DAG 拓扑是静态的
- AutoGen/CrewAI 需要人工设计流程

**What do you do?**
- 将 multi-agent orchestration 建模为 MDP
- Orchestrator 逐步观察 agent 返回、逐步决策（reactive）
- 单一 GRPO 奖励函数驱动，涌现自适应行为谱
- 跨 QA/推理/代码三个模态泛化

**Why does it work?**
- Reactive 闭环 > 开环规划：可以根据 agent 返回质量临时调整后续动作
- 功能角色 > 模型选择：不同 agent 执行不同认知功能（分解、验证、综合）
- 成本感知奖励 > 忽略成本：α 调节 Pareto 前沿

**What's the evidence?**
- Table 1: 三个 track 超越 Router-R1/Conductor/Direct-Strong/ReAct
- Figure: Agent 调用分布热力图（涌现自适应性）
- Figure: 训练过程行为变化（RL 驱动涌现）
- Figure: Pareto 曲线（成本-质量权衡）
- Case study: Reactive 策略调整实例（Conductor 无法做到）

---

## 2. 实验计划（核心，决定论文成败）

### 2.1 评估基准

| Track | 数据集 | 指标 | 测试量 |
|-------|--------|------|--------|
| Track 1: 简单 QA | NQ, TriviaQA, PopQA | EM, F1 | 各 500 |
| Track 2: 多跳推理 | HotpotQA, 2WikiMultihop, MuSiQue, Bamboogle | F1 | 各 500 |
| Track 3: 代码 | HumanEval, MBPP | Pass@1 | 164 + 427 |

### 2.2 基线（必须覆盖，否则 reviewer 会要求补）

| 基线 | 优先级 | 来源 | 说明 |
|------|--------|------|------|
| **Conductor** | ★★★ 最高 | 复现或用官方 checkpoint | ICLR 2026，必须直接对比 |
| **Router-R1** | ★★★ | 官方 checkpoint | 直接前作 |
| **Direct-GPT-4o** | ★★☆ | 直接调 API | 上限参考 |
| **ReAct** (Qwen2.5-3B) | ★★☆ | 自实现 | 标准 tool-use 基线 |
| **AgentConductor** | ★★☆ | 复现或引用数据 | 代码 track 直接竞品 |
| **Fixed-Pipeline** | ★★☆ | 自实现 | 消融：固定 6 步流水线 |
| **Reflexion** | ★☆☆ | 自实现 | 可选 |

> **关键**：如果无法复现 Conductor，至少要在论文中定性对比（reactive vs plan-then-execute），并在 HotpotQA/GPQA 等 Conductor 报告了结果的数据集上做对比。

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

### 2.5 预期结果（诚实评估）

| Track | vs Router-R1 | vs Conductor | vs Direct-GPT-4o |
|-------|-------------|--------------|-------------------|
| 简单 QA | 平手或 +1~3% F1，成本更低 | 可能略低（Conductor 用 70B workers）| 略低，但成本 1/10 |
| 多跳推理 | +5~10% F1（分解+验证优势） | 需要实验验证 | 接近或略低 |
| 代码 | Router-R1 不支持 | AgentConductor 更强，但你跨模态 | 可能略低 |

**如果 Conductor 全面碾压你怎么办？**
→ 转向强调 (1) 跨模态泛化 (2) 成本效率 (3) 训练效率（单 GPU 可训，$5 API），定位为 "practical, cost-efficient alternative"

---

## 3. 技术改进清单（提升论文竞争力）

### 3.1 必做（P0，影响审稿结论）

- [ ] **实现 "w/o reactive" 消融**：让模型一次性输出所有 call（不注入中间 information），对比 reactive 模式
- [ ] **添加 GPQA 数据集**：Conductor 的核心评估数据集，必须覆盖以便直接对比
- [ ] **添加 LiveCodeBench**：Conductor 和 AgentConductor 都用了，代码 track 的标准基准
- [ ] **Conductor 基线**：至少在 2-3 个数据集上复现或引用其结果
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
   - Router-R1 → Conductor 的进展，但 plan-then-execute 的局限
   - 本文: reactive sequential orchestration + 跨模态 + 成本优化
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
   4.2 Main Results (Table 1: 3 tracks × baselines × 4 metrics)
   4.3 Ablation Studies (Table 2: 7 ablations)
   4.4 Cost-Quality Tradeoff (Pareto curve)

5. Analysis (1.5 pages)
   5.1 Emergent Adaptive Behavior (heatmap + training dynamics)
   5.2 Reactive vs Plan-then-Execute (quantitative + case study)
   5.3 Scaling Analysis (3B vs 7B)

6. Conclusion (0.5 page)

Appendix:
   A. Agent System Prompts
   B. Full Evaluation Results per Dataset
   C. Training Details and Hyperparameters
   D. Additional Case Studies
   E. Compute Cost Analysis
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
| Conductor 全面碾压 | 中高 | 强调跨模态 + 成本效率 + 训练效率（$5 vs $$$）|
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

### vs Conductor（最重要的对比）

> While the Conductor (Nielsen et al., 2026) generates complete workflows in a single forward pass,
> OrchestratorR1 adopts a **reactive sequential** paradigm: the orchestrator observes each agent's
> response before deciding the next action. This closed-loop formulation enables:
> (1) **real-time strategy adaptation** — if an executor returns low-quality results, the orchestrator
> can dynamically invoke a critic for verification and retry;
> (2) **cost-aware early stopping** — simple queries are answered in 1 step without generating
> unnecessary workflow overhead;
> (3) **functional role orchestration** — rather than selecting which model to invoke, our agents
> perform distinct cognitive functions (decomposition, verification, synthesis).
>
> Furthermore, a single OrchestratorR1 policy generalizes across QA, multi-hop reasoning, and code
> generation, whereas Conductor focuses exclusively on reasoning tasks.

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
