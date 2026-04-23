
# OrchestratorR1 相关工作研究线路

> 时间范围：2025 年 1 月 – 2026 年 4 月 23 日
> 目标：围绕 **Router-R1** 与 **Conductor** 两条参照系，梳理"多 LLM agent 协作 / 编排 / 路由"这一方向的研究脉络，形成可供 OrchestratorR1 定位的 roadmap。本文件仅搭骨架（每条工作用 1–2 句话定位），**框架对比、方法细节、有效性分析留待第二轮补全**。
>
> 注意：第一轮（离线知识）+ 第二轮（WebFetch 联网）+ 第三轮（Conductor 锁定）合并稿。带 ✅ 的条目为联网核实；未标记的来自训练知识（截止 2026-01），正式引用前请再次核对。

---

## 0. 两条参照系（锚点）

| 工作 | 定位 | 关键设计 |
|---|---|---|
| **Router-R1** ✅ (NeurIPS 2025, arXiv:2506.09033) | 让一个 LLM 自己当 router，用 RL 学会在多轮中 `<search>Model:query</search>` 调用其它 LLM 并聚合。作者 Haozhen Zhang, Tao Feng, Jiaxuan You (UIUC)；v3 2025-10-24 | 单 planner = 推理者 + 路由器；PPO + state masking；reward = EM/F1 − λ·cost；声称用简单描述符（价格/延迟）就能泛化到未见模型 |
| **Conductor** ✅ (ICLR 2026, arXiv:2512.04388) | "Learning to Orchestrate Agents in Natural Language with the Conductor"。RL 训练的 7B Conductor 协调一池更大的 specialized worker LLM | 学两件事：(1) **通信拓扑设计**（谁跟谁通信）+ (2) **prompt engineering**（给每个 worker 写定制指令）；支持**递归拓扑**（Conductor 可选自己当 worker，做 test-time scaling）；随机化 agent 池训练→对开/闭源都泛化；LiveCodeBench、GPQA SOTA，小 Conductor 反超大 worker |

---

## 1. 研究线路总览（5 条主线）

```
            ┌── L1. 单轮/成本感知 Router ───── RouteLLM, RouterDC, GraphRouter, Eagle
            │
            ├── L2. RL + 工具使用（Router-R1 的方法父类）── Search-R1, ReSearch, R1-Searcher, ToolRL
Orchestrator│
   R1 ──────┤── L3. 多 Agent 编排（Prompted / 图结构）── MoA, DyLAN, GPTSwarm, AgentPrune
            │
            ├── L4. RL 训练的编排者 / Planner（最相关）── FlowReasoner, MALT, MAPoRL, ArCHer, Agent-R
            │
            └── L5. 模型即工具 + 验证者闭环 ─────── Symbolic-MoE, Avengers, PRM-in-loop, AutoMix 级联
```

OrchestratorR1 的主战场是 **L4 + L2 的交集**（RL-trained orchestrator that calls other LLMs as tools），并借鉴 L5 的 verifier-in-loop 与 noise-robust aggregation。

---

## 2. 各主线代表性工作清单

### L1. 单轮 / 成本感知 Router（Router-R1 的"旧范式"）

| 工作 | ID / 会议 | 一句话定位 |
|---|---|---|
| **RouteLLM** | arXiv:2406.18665 · ICLR 2025 | 强/弱模型二分类 router，用 Arena 偏好对训练 |
| **RouterDC** | arXiv:2409.19886 · NeurIPS 2024 | 双对比学习的 embedding 路由器 |
| **GraphRouter** | arXiv:2410.03834 · ICLR 2025 | task–query–LLM 异构图 + GNN 预测性能/成本 |
| **Eagle** | arXiv:2409.15518 | ELO 排序式路由 |
| **FrugalGPT / Hybrid-LLM / Cascade-Routing** | TMLR 2024 / ICLR 2024 / arXiv:2410.10347 | 级联：cheap → expensive，打分早退 |
| **RouterBench** | arXiv:2403.12031 | 11-LLM 路由基准，后续工作的共同测评平台 |

**共同局限（留作 L4 的改进动机）**：单轮决策、无法分解任务、不能重试重路由、无聚合。

### L2. RL + 工具使用（方法学父类）

| 工作 | ID | 一句话定位 |
|---|---|---|
| **Search-R1** | arXiv:2503.09516 | GRPO 训练 LLM 发 `<search>` 调用检索器；Router-R1 架构模板 |
| **ReSearch** | arXiv:2503.19470 | 无 SFT 冷启的 RL + 检索推理 |
| **R1-Searcher / ++** | arXiv:2503.05592 / 2505.xx | 两阶段 RL（格式奖励 → 结果奖励）学搜索 |
| **ToolRL / Nemotron-Tool-RL** | arXiv:2504.13958 | 纯 RL 学通用工具调用，schema reward |
| **WebDancer / WebSailor** | Alibaba Tongyi 2025 | 长 horizon（数十轮）web agent RL，证明 4-turn 是下限 |
| **DeepSeek-R1 / Tulu-3 / s1** | 2025 | RLVR 技术栈，Router-R1 的训练基础 |

### L3. Prompted / 图结构多 Agent 编排

| 工作 | ID | 一句话定位 |
|---|---|---|
| **Mixture-of-Agents (MoA)** | arXiv:2406.04692 | 堆叠多层 LLM，每层看前层全部输出再合成 |
| **Self-MoA / Sparse-MoA** | arXiv:2502.00674 | 反击 MoA：单模型自混合也能赢；加层内路由 |
| **DyLAN** | arXiv:2310.02170 · ICLR 2025 扩展 | Agent DAG + importance score 剪边 |
| **GPTSwarm** | arXiv:2402.16823 · ICML 2024 | Agent 协作图 + REINFORCE 学边权（RL 编排的早期雏形）|
| **AgentVerse / AutoGen-v0.4 / MetaGPT-X** | 2024–2025 | GroupChatManager 开始带学习成分 |
| **AgentPrune / EcoAgent** | 2025 | 预算下对 agent 图剪枝 |
| **Chain-of-Agents** | Google NeurIPS 2024 + 2025 扩展 | 长文本任务沿 agent 链传递，带 manager |

### L4. RL 训练的编排者 / Planner（OrchestratorR1 最相关）

| 工作 | ID | 一句话定位 |
|---|---|---|
| **FlowReasoner** | arXiv:2504.15257 | RL meta-agent 一次性生成整个工作流 DAG（与 Router-R1 的交错路由互补）|
| **MALT** | arXiv:2412.01928 | 生成器/验证器/精炼者三角，多 agent 轨迹级信用分配 |
| **MAPoRL** | arXiv:2502.18439 | 多 agent 联合 post-training，共享奖励；与 Router-R1"冻结 worker"相反 |
| **ArCHer** | ICML 2024 + 2025 follow-up | 分层 actor-critic，高层策略选子策略，天然对应 orchestrator-worker |
| **Agent-R / Agent-RFT** | ByteDance 2025 | 反思式 RL 自我纠错 |
| **ADAS / AFlow** | 2024 / 2025 | 工作流 = 路由计划；有 RL 变体 |
| **Symphony / Maestro / Conductor-RL** *(待核实)* | 2025 | 指挥棒式中央 planner + worker pool |

### L5. 模型即工具 + 验证者闭环

| 工作 | ID | 一句话定位 |
|---|---|---|
| **Symbolic-MoE** | arXiv:2503.05641 | 用符号 skill tag 将 query 路由到 expert LLM；Router-R1 的"无 RL、单轮"表亲 |
| **Avengers / Avengers-Pro** | arXiv:2505.19101 | Query 聚类 → 最佳 LLM，开源集成逼近 GPT-4 |
| **AutoMix** | NeurIPS 2024 | 自我验证后再升级，级联思想 |
| **PRM in loop (OpenR, Math-Shepherd)** | 2024–2025 | 过程奖励模型作 verifier，可视为 2-LLM 路由常态调用验证者 |
| **C-3PO / BudgetedMoA** | 2025 | 明确预算约束下决定哪些 agent 触发 |

---

## 3. 与 Router-R1 最相似的一簇（OrchestratorR1 直接对比对象）

按架构相似度排序：

1. **Search-R1 / ReSearch / R1-Searcher** — 完全同构，但工具只有"检索器"一种。
2. **FlowReasoner** — RL 训 orchestrator，但一次性出 DAG，不是交错的 reason-then-route。
3. **Symbolic-MoE / Avengers** — LLM pool 路由，无 RL、无多轮聚合。
4. **MALT / MAPoRL** — 多 LLM，但 worker 一起训；Router-R1 把 worker 当冻结 API。
5. **GPTSwarm** — 图结构 + RL 学边权，早于 Router-R1 的思路先驱。
6. **Conductor** ✅ (arXiv:2512.04388, ICLR 2026) — 中央 Conductor + worker pool + RL 学通信拓扑 & 定制 prompt，支持递归（Conductor 当 worker）。与 Router-R1 **最本质差异**：Conductor 显式建模 agent 间通信拓扑与 per-worker prompt，Router-R1 的"调用"只是 `<search>Model:query</search>` 单跳无拓扑。

---

## 4. 空白与 OrchestratorR1 可切入的缝隙

这几条缝隙后续可逐条对应到实验设计：

- **Horizon 长度**：Router-R1 截到 4 轮，WebSailor 可到数十轮 — 长 horizon 下的路由稳定性未被研究。
- **预算条件化路由**：现有工作把 cost 作为标量 λ 硬编码进 reward，没有做 **用户给定预算/延迟** 的条件化 RL。
- **Verifier-in-loop 的路由**：L5 的 PRM 与 L4 的 RL orchestrator 尚未融合 — orchestrator 能否学会"先路由再让 verifier 复核"？
- **噪声鲁棒聚合**：当 worker LLM 返回不一致或被污染答案时，Router-R1 的聚合机制未做抗噪设计（对应你的 noise injection 实验）。
- **递归路由 / 层次化 orchestrator**：Planner 调用 Planner，目前几乎空白。
- **延迟感知**（latency-aware routing）：成本维度被覆盖，但端到端延迟尚无 RL 形式化（对应你的 latency timing 实验）。
- **Strip-think / 思维链剥离后路由**：Pilot 实验方向，文献几乎无人触碰。

---

## 5. 下一步（第二轮补全计划）

每条主线下选 2–3 篇代表作，展开回答：

1. 它们的编排 / 路由 framework 具体长什么样（图示 + 伪码）；
2. 相比"之前的任务方式"（即 L1 单轮 router 或 L3 prompted 编排）存在什么问题；
3. 这篇工作做了什么具体改变；
4. 为什么这种改变是有效的（理论动机 + 经验证据）；
5. 与 Router-R1 / OrchestratorR1 的差异列表。

建议优先展开顺序：**Router-R1 → Search-R1 → FlowReasoner → MALT → Symbolic-MoE → MoA/DyLAN → RouteLLM → Conductor（待补）**。

---

## 6. 第二轮联网新发现（2025 下半年 – 2026 上半年关键新作）✅

第二轮通过 arXiv WebFetch 直接验证到的"和 Router-R1 同框架"的新工作。这一批应作为 OrchestratorR1 直接对比对象，优先于第一轮基于训练知识的条目。

### 6.1 RL-trained LLM Router / Orchestrator（最直接对比对象）

| 工作 | arXiv ID | 时间 | 一句话定位 | 与 Router-R1 关系 |
|---|---|---|---|---|
| **HierRouter** ✅ | 2511.09873 | 2025-11 | PPO 训练的 RL agent，在多 hop 推理阶段间动态组装"specialized lightweight LLM"管线，把路由建模为 MDP；6 个 benchmark 上响应质量提升至 2.4×，推理成本只略增 | **几乎同框架**：都是 PPO + 多轮路由到 LLM 池。差异：HierRouter 强调 specialized lightweight 模型管线（pipeline 视角），Router-R1 强调价格/延迟描述符的泛化 |
| **DynaSwarm** ✅ | 2507.23261 | 2025-07 | 用 A2C 强化学习优化 multi-agent 图结构 + 动态图选择器逐 sample 自适应选最佳协作拓扑 | 把 Router-R1 的"线性多轮"扩展为"按 sample 选图"。GPTSwarm 思路 + RL + per-sample 自适应 |
| **Maestro** ✅ | 2511.06134 | 2025-11 | 解耦 exploration（并行 Execution Agents）和 synthesis（Central Agent），提出 CLPO（Conditional Listwise Policy Optimization）做信用分配；数学/推理任务 +6% avg / +10% best | 非常像"Conductor"语义。中央 agent + 并行 worker + RL；与 Router-R1 的串行多轮形成 parallel-then-aggregate 互补 |
| **Dr. MAS** ✅ | 2602.08847 *(ID 需复核)* | 2026-02 | 解决 multi-agent LLM RL 的训练不稳定性，per-agent 归一化 advantage（不用 global baseline），数学 +5.6% avg@16 | **训练稳定性补丁**：可直接迁移到 Router-R1 / OrchestratorR1 的训练栈 |
| **Stronger-MAS / AT-GRPO** | （搜索未直接命中）| 2025-10 | AT-GRPO 算法专门为多 agent workflow 设计，long-horizon planning 从 14% → 96–99.5% | 长 horizon 突破；OrchestratorR1 若做 latency / 多步实验需对比 |

### 6.2 训练-free 路由 / 多专家 QA（Router-R1 的"prompted 对照组"）

| 工作 | arXiv ID | 时间 | 一句话定位 |
|---|---|---|---|
| **RIRS (前身 RopMura)** ✅ | 2501.07813 | 2025-01 | 训练-free orchestration framework，每个 agent 用本地语料 embedding，server 路由 query 到最相关 agent；支持 single-hop 与 multi-hop 迭代 | 显示 training-free 路由的能力天花板，是 RL-trained Router-R1 的 prompted baseline |

### 6.3 工具编排与 agent 规划（2026 年新作）

| 工作 | 时间 | 一句话定位 |
|---|---|---|
| **Training LLMs for Multi-Step Tool Orchestration** ✅ | 2026-03 | 把"正确性"分解为 atomic validity + orchestration consistency 两层 graduated reward，提升工具调用性能 |
| **MagicAgent** ✅ | 2026-02 | 多目标 RL 训通用 agent planning 基础模型，Workbench 75.1%，BFCL-v3 86.9% |
| **WideSeek-R1** ✅ | 2026-02 | 探索 width scaling，并行多 agent + RL，WideSearch item F1 40.0% |

### 6.4 自动化多 agent workflow 生成

| 工作 | arXiv ID | 一句话定位 |
|---|---|---|
| **Mimosa** ✅ | 2603.28986 *(待复核)* | meta-orchestrator 自动合成任务专属 multi-agent workflow，并通过实验反馈迭代精炼 |
| **Youtu-Agent** ✅ | 2512.24615 *(待复核)* | Workflow mode + Meta-Agent mode，自动生成工具代码、prompt 与配置 |

### 6.5 应用领域的"hierarchical orchestrator"实例

| 工作 | 一句话定位 |
|---|---|
| **RecGPT-V2** ✅ | 推荐系统中的 Hierarchical Multi-Agent + 约束 RL 协调多目标 |
| **L2M-AID** ✅ | LLM 推理 + multi-agent RL 用于工业 cyber-physical 防御 |
| **LLM-powered multi-vehicle navigation** ✅ | 1.6M roads / 430K intersections 的实证 scalability |
| **RollArt** ✅ | Agentic RL 训练基础设施，硬件感知任务路由，端到端训练时间降 1.35–2.05× |

---

## 7. 整合后的 OrchestratorR1 直接对比对象（最终版）

按"架构相似度 + 时间相近度"排序：

1. **HierRouter** (2025-11) — 最相似，同样 PPO + 多轮 LLM pool。**必比**。
2. **Router-R1** (2025-06, NeurIPS 2025) — 你的 base。
3. **Maestro** (2025-11) — central + parallel workers + CLPO；可作 parallel-aggregation 的对照实验。
4. **DynaSwarm** (2025-07) — 图结构 + A2C，是 GPTSwarm 的 RL 加强版，per-sample 自适应。
5. **Dr. MAS** (2026-02) — 训练稳定性，可作训练 trick 借鉴而非直接对比。
6. **FlowReasoner** (2504.15257) — 一次出整 DAG 的 RL，可对照"交错 vs 一次性"。
7. **Search-R1** (2503.09516) — 单工具父类。
8. **Symbolic-MoE / Avengers** — 无 RL、单轮的"瘦身对照组"。
9. **Conductor** ✅ (arXiv:2512.04388, ICLR 2026) — 中央 RL Conductor + 拓扑设计 + 递归自调用；与 Router-R1 并列为 OrchestratorR1 **必比**参照系。

---

## 8. 待确认 / TODO（更新版）

- [x] 联网核对：Router-R1、**Conductor (arXiv:2512.04388, ICLR 2026)**、HierRouter、DynaSwarm、Maestro、RIRS 已✅
- [ ] 复核 Dr. MAS (2602.08847)、Mimosa (2603.28986)、Youtu-Agent (2512.24615) 的 ID —— WebFetch 给出的"未来日期"ID 可疑，可能是工具幻觉
- [ ] 补充 Stronger-MAS / AT-GRPO / Training LLMs for Multi-Step Tool Orchestration / MagicAgent / WideSeek-R1 的具体 arXiv ID
- [ ] 若要写 related work 章节，优先阅读 HierRouter + Maestro + DynaSwarm 三篇原文
- [ ] 若 OrchestratorR1 强调 latency / noise / budget，需补 latency-aware serving（Tabi、OptLLM、LLMCascade 2025）
