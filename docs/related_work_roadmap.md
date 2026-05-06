
# OrchestratorR1 相关工作研究线路

> 时间范围：2025 年 1 月 – 2026 年 4 月 29 日
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
| **KLong** | 2025 | 训练 LLM agent 做极长 horizon 任务（复现研究论文）；轨迹拆分 SFT 冷启动 + 渐进式 RL（递增超时阈值）+ 保留早期上下文逐步截断中间上下文；OrchestratorR1 借鉴其渐进式训练和上下文压缩策略 |
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

## 2.X 代表性工作详解（精选 17 篇五段式）

> 每篇按 **Framework → 旧范式问题 → 改变 → 为何有效 → 与 Router-R1 差异** 五段展开。✅ = 联网核实；其余沿用训练知识，引用前请二次核对。

### L1. 单轮 / 成本感知 Router

#### RouteLLM ✅ (arXiv:2406.18665, ICLR 2025)
- **Framework**：训练一个轻量 router（BERT/MF/causal-LLM 三种实现），每条 query 二选一调用 strong 或 weak LLM；用 Chatbot Arena 偏好对 + 数据增强训练。
- **旧范式问题**：开发者必须在"全调 GPT-4（贵）"和"全调小模型（弱）"之间做静态二选一，单一阈值无法因 query 而异。
- **改变**：把"模型选择"从全局策略下放到 per-query 决策，用偏好数据学一个判别器。
- **为何有效**：Arena 偏好对天然包含人类对"哪个模型够用"的相对判断；router 学到的不是绝对能力而是"小模型够不够"的边界，因此可在质量持平下省 >2× 成本，且对换模型仍 transferable。
- **与 Router-R1 差异**：单轮、二选一、无聚合、无推理；Router-R1 是 N 轮、N 选、可重试可聚合。RouteLLM 是 OrchestratorR1 必要的 single-call 基线。

#### GraphRouter (arXiv:2410.03834, ICLR 2025)
- **Framework**：把 (task, query, LLM) 建成异构图，GNN 节点特征学性能-成本预测，路由 = 图上 link prediction。
- **旧范式问题**：纯文本 router 把 task 与 LLM 当独立特征，丢失"同 task 不同 LLM、同 LLM 不同 task"的二阶交互。
- **改变**：用图结构把 task–query–LLM 三方关系显式建模，引入 inductive bias。
- **为何有效**：cost-quality 权衡本质是关系预测；GNN 在稀疏 (LLM, task) 矩阵下泛化优于 MLP，并能 zero-shot 加新 LLM 节点。
- **与 Router-R1 差异**：仍是单轮 + 离线打分，没有 reasoning loop。

#### FrugalGPT / Cascade-Routing
- **Framework**：链式级联（cheap → mid → expensive），每段输出过 scorer，达到阈值就早退。
- **旧范式问题**：固定模型调用浪费能力；级联前已是最朴素的"自适应"思路但靠 hand-crafted scorer。
- **改变**：把 routing 拆成"调一次→打分→是否升级"的串行决策。
- **为何有效**：大多数 query 简单，cheap model 已够，scorer 截断省去贵调用；理论上 Pareto 最优需要预言 scorer，实践中近似已够。
- **与 Router-R1 差异**：单向升级、无聚合、scorer 是手工训练的判别器而非 RL planner。Router-R1 的 reward 与 cascade scorer 角色等价但 end-to-end 学习。

---

### L2. RL + 工具使用（方法学父类）

#### Search-R1 ✅ (arXiv:2503.09516)
- **Framework**：base LLM 在推理过程中自主发 `<search>...</search>` 调检索器，retrieved tokens 被 mask 不参与 loss；GRPO/PPO + 单一 outcome reward（EM）。
- **旧范式问题**：把检索"塞"给 prompted LLM 用 ReAct/RAG 是次优的——LLM 不知道何时搜、搜什么、如何用结果。
- **改变**：用 RL 让模型自己学搜索策略，把"工具使用"内化为 token-level 决策。
- **为何有效**：(1) outcome reward 比 trajectory-level supervision 更弱但够稀疏可学；(2) retrieved-token masking 防止模型把检索文本误当成自己的 reasoning 训练；Qwen2.5-7B 比 RAG baseline 提升 41%。
- **与 Router-R1 差异**：**架构完全同构**，但工具池只有一个 retriever；Router-R1 把 retriever 换成 LLM pool 并加成本约束。Search-R1 是 Router-R1 的直接技术祖先。

#### ReSearch / R1-Searcher
- **Framework**：与 Search-R1 同框架，差异在训练 recipe：ReSearch 无 SFT cold start，纯 RL；R1-Searcher 两阶段 RL（format reward → outcome reward）。
- **旧范式问题**：SFT cold start 受限于人工轨迹质量与覆盖度。
- **改变**：要么直接 RL（ReSearch），要么用 format reward 替代 SFT 解决 cold start（R1-Searcher）。
- **为何有效**：DeepSeek-R1-Zero 的发现（RL 可以无 SFT 直接涌现 reasoning）在工具使用域复现；format reward 是稀疏 outcome reward 的 dense surrogate，加速早期收敛。
- **与 Router-R1 差异**：训练 recipe 可直接迁移到 OrchestratorR1，特别是 7B 全参 + 无 SFT 的方案。

#### ToolRL / Nemotron-Tool-RL (arXiv:2504.13958)
- **Framework**：RL 学通用 function-calling，reward = schema 合规 + 参数正确 + 任务成功的复合分。
- **旧范式问题**：tool-use SFT 只教格式，不教"何时调"；prompted tool-use 在分布外失效。
- **改变**：把 schema 验证作为 dense reward 信号，避免只靠 outcome 的稀疏性。
- **为何有效**：schema 错就直接零回报，迫使模型先学合规再学策略；与 RLVR (verifiable reward) 思路一致。
- **与 Router-R1 差异**：Router-R1 的"call LLM X"也是一种 function-call，可以借用复合 reward 设计提升训练稳定性。

#### WebSailor / WebDancer (Alibaba 2025)
- **Framework**：长 horizon web agent（数十轮），引入 trajectory-level credit assignment 与 search-tree rollout。
- **旧范式问题**：4–5 turn 之内的 RL 收敛容易，但任务复杂度上限低；horizon 扩到 20+ 后 reward sparsity 暴增。
- **改变**：tree-search rollout + 阶段式 reward shaping。
- **为何有效**：把长轨迹拆成可解的子段 + 共享前缀以摊销 rollout 成本，使 RL 能在 30+ turn 收敛。
- **与 Router-R1 差异**：Router-R1 截到 4 turn；OrchestratorR1 若想突破 horizon 必须借鉴 WebSailor 的 rollout 与 credit 分配。

---

### L3. Prompted / 图结构多 Agent 编排

#### Mixture-of-Agents (MoA, arXiv:2406.04692)
- **Framework**：N 层堆叠，每层 K 个 LLM 看上一层全部输出，最后一层 aggregator 综合输出。完全 prompted、零训练。
- **旧范式问题**：单 LLM ensemble 只做 majority vote；agent debate 不结构化、回合数不可控。
- **改变**：把 agent 协作形式化为"分层 read-and-synthesize"，类似 transformer 的 layer-wise attention。
- **为何有效**：每层 aggregator 输入是上层所有意见 → 后层 LLM 可显式参考差异并自我修正；3 层 MoA 用开源模型已超过 GPT-4o on AlpacaEval。
- **与 Router-R1 差异**：MoA 调用所有 agent（成本高），Router-R1 学会"按需选择"；MoA 无学习，Router-R1 用 RL 把"选谁"内化。

#### GPTSwarm ✅ (arXiv:2402.16823, ICML 2024)
- **Framework**：把多 agent 系统视为可优化计算图，节点 = LLM 调用 / 数据处理，边 = 信息流；提供 node-level（prompt 优化）+ edge-level（拓扑优化，REINFORCE）两种 optimizer。
- **旧范式问题**：每个 agent 框架（AutoGen / MetaGPT / CAMEL）都是独立代码库，拓扑硬编码、prompt 手写。
- **改变**：用图统一表示所有 agent 系统，并把图作为可优化对象。
- **为何有效**：(1) 统一抽象使跨框架对比成为可能；(2) edge optimizer 用 REINFORCE 学边权 = 隐式学拓扑剪枝，是 Conductor "学通信拓扑" 的早期版本。
- **与 Router-R1 差异**：GPTSwarm 学的是离线静态拓扑（一次性优化好的图），Router-R1 是 per-query 在线决策；但都用 RL 学协作结构。

#### DyLAN (arXiv:2310.02170)
- **Framework**：agent 网络是 DAG，每条边带"importance score"，剪去低分边；动态调整每轮 active agents。
- **旧范式问题**：固定 agent 数 + 固定回合数浪费 token。
- **改变**：importance score 作为剪枝信号，按需扩缩 agent 集合。
- **为何有效**：agent 贡献度高度长尾，剪 50%+ 边几乎不掉点。
- **与 Router-R1 差异**：DyLAN 的 importance score 是 prompted 评估，不学习；Router-R1 用 RL 把"何时不调"作为策略。

#### AgentPrune / EcoAgent (2025)
- **Framework**：在 agent 协作图上加预算约束，按贡献度剪除冗余 agent。
- **旧范式问题**：MoA / DyLAN 没把 cost 显式建模。
- **改变**：cost-aware pruning，给定预算内最大化 utility。
- **为何有效**：与 RouteLLM 的 cost trade-off 同源思想，但作用在多 agent 而非单调用上。
- **与 Router-R1 差异**：剪枝是离线 pre-processing；Router-R1 把成本融入 reward 在线学习。

---

### L4. RL 训练的编排者 / Planner（最相关）

#### FlowReasoner ✅ (arXiv:2504.15257)
- **Framework**：query-level meta-agent，给定一条 query 输出一份**专为它设计的 multi-agent system**（DAG）；先用 DeepSeek-R1 蒸馏 reasoning，再用 RL + 外部执行反馈微调，多目标 reward。
- **旧范式问题**：现有 multi-agent 系统是 task-level（一份系统跑全 task），无法因 query 调整复杂度。
- **改变**：把"系统设计"本身作为 meta-agent 的输出，per-query 生成 workflow。
- **为何有效**：(1) reasoning 蒸馏让 meta-agent 学会"先想再设计"；(2) 执行反馈 + 多目标 reward 捕获正确性、效率、稳定性；超 o1-mini 10.52%。
- **与 Router-R1 差异**：FlowReasoner **一次性出整张 DAG**（开环），Router-R1 是 **交错式 reason-then-route**（闭环）。OrchestratorR1 的 reactive loop 论点正好打在这个差异上。

#### MALT ✅ (arXiv:2412.01928)
- **Framework**：固定三角色——Generator / Verifier / Refiner，构建 multi-agent search tree，ground-truth 评分 + value iteration 给每个 agent 单独优化；off-policy 学好坏轨迹。
- **旧范式问题**：单链 CoT 不能 self-correct；多 agent debate 不收敛、无信用分配。
- **改变**：把 generation–verification–refinement 解耦为可独立优化的角色；用 search tree 做 trajectory 级 credit assignment。
- **为何有效**：三角色对应 reasoning 的三个 distinct skill，分开训避免"既要又要"的优化冲突；search tree 提供失败轨迹用于 off-policy 学习。MATH +15.66%、GSM8K +7.42%。
- **与 Router-R1 差异**：MALT **co-train 多个 LLM**（角色分工但都更新），Router-R1 **冻结 worker，只训 router**。这是核心立场分歧——是把成本压在训练上还是 inference API 上。

#### MAPoRL (arXiv:2502.18439)
- **Framework**：多 LLM 联合 post-training，共享 reward 通过 multi-agent RL 一起更新。
- **旧范式问题**：多 agent 系统中 worker 是冻结的，能力上限被卡住。
- **改变**：所有 worker 都可训练，共享 task reward。
- **为何有效**：worker 间策略可以协同进化，避免"router 学得很好但 worker 不会配合"的失配；类似 MARL 的 self-play。
- **与 Router-R1 差异**：与 MALT 同立场，进一步走向 full-MARL；适合内部模型场景，不适合 Router-R1 的"调用付费 API"场景。

#### Conductor ✅ (arXiv:2512.04388, ICLR 2026)
- **Framework**：7B Conductor + 一池更大的 specialized worker LLM；RL 端到端学两件事——**通信拓扑设计**（谁与谁通信）+ **per-worker prompt engineering**（给每个 worker 写定制指令）；支持**递归**（Conductor 选自己当 worker 做 test-time scaling）；训练时随机化 agent 池增强泛化。
- **旧范式问题**：现有 multi-agent 协作要么靠手工 prompt（MetaGPT/AutoGen）要么靠固定拓扑（MoA），没人系统地学协调策略。
- **改变**：把"拓扑 + prompt"双重决策都交给 RL 端到端学。
- **为何有效**：(1) 通信拓扑决定信息聚合方式，per-worker prompt 决定单 agent 表现，二者联合优化；(2) 递归拓扑天然支持 test-time scaling；(3) 随机 agent pool 训练 → zero-shot 泛化到新模型。LiveCodeBench、GPQA SOTA，7B Conductor 反超大 worker。
- **与 Router-R1 差异**：**Router-R1 没有"agent 间通信"概念**，所有 worker 输出只回到 router；Conductor 让 worker 之间也能通信。这是 OrchestratorR1 必须做出选择的关键架构岔路。

#### HierRouter ✅ (arXiv:2511.09873)
- **Framework**：PPO RL agent，把多 hop 推理建模为 MDP，每步选一个 specialized lightweight LLM 加进 inference pipeline。
- **旧范式问题**：单 LLM 不擅长所有子任务；调用大 LLM 又贵。
- **改变**：动态组装"小专家 LLM 流水线"，按推理阶段路由。
- **为何有效**：每个子任务交给最擅长的小模型 → 质量 ↑2.4× 而成本只略增；与 cascade 不同的是用 RL 学"调谁"而非"何时升级"。
- **与 Router-R1 差异**：**几乎同框架**——都是 PPO + 多轮 + LLM 池路由。HierRouter 偏管线视角（worker 顺序、特化）；Router-R1 偏推理-工具视角（推理 + 调用穿插）。OrchestratorR1 必须 head-to-head 对比。

#### DynaSwarm ✅ (arXiv:2507.23261)
- **Framework**：A2C RL 优化 multi-agent 协作图结构；动态图选择器逐 sample 选最优拓扑。
- **旧范式问题**：固定拓扑（MoA、AgentVerse）不能因 query 调整。
- **改变**：per-sample 选图，比 GPTSwarm 的离线优化更细粒度。
- **为何有效**：query 之间最优拓扑差异大，per-sample 自适应捕获这种差异；A2C 比 REINFORCE 方差小、训练更稳。
- **与 Router-R1 差异**：DynaSwarm 在"多种预设拓扑里选"，Router-R1 在"调谁的 token 序列上学"；前者拓扑可枚举，后者更接近自由生成。

---

### L5. 模型即工具 + 验证者闭环

#### Symbolic-MoE ✅ (arXiv:2503.05641)
- **Framework**：每个 expert LLM 标注 skill tag（如 algebra、molecular biology）；每条 query 抽取 skill 关键词，召回相关 experts，并行生成后由 aggregator 综合；批量加载使 16 experts 跑在 1 GPU 上。
- **旧范式问题**：task-level 路由（"数学题→Llama-Math"）粒度太粗，同一个 task 内不同 instance 可能需要不同 skill。
- **改变**：把 routing 粒度从 task 降到 instance × skill。
- **为何有效**：(1) skill 是比 task 更细的能力轴；(2) 显式 tag 让选择可解释；(3) batch loading 解决多 expert 显存问题。MMLU-Pro / GPQA / AIME / MedMCQA 平均 +8.15%。
- **与 Router-R1 差异**：Symbolic-MoE **无 RL、无多轮**——是 single-shot 多专家投票；Router-R1 是 sequential decision。Symbolic-MoE 是 OrchestratorR1 必要的"non-RL routing"基线。

#### Avengers (arXiv:2505.19101)
- **Framework**：query embedding 聚类，每个 cluster 离线建立 cluster→best-LLM 的映射。
- **旧范式问题**：训练 router 需要标注；Avengers 想纯 unsupervised。
- **改变**：用 cluster 作为 query 类别的代理，把 routing 退化为查表。
- **为何有效**：query 在 embedding 空间天然成簇，每簇内最佳 LLM 的方差小；开源模型集成可逼近 GPT-4。
- **与 Router-R1 差异**：完全 training-free、单轮；适合作为"零训练 baseline"凸显 RL 的增量价值。

#### AutoMix (NeurIPS 2024)
- **Framework**：cheap LLM 先答 → self-verify → 不通过则升级到 expensive LLM。
- **旧范式问题**：FrugalGPT 的 scorer 需要外部训练；AutoMix 想用 LLM 自己当 verifier。
- **改变**：自我验证替代独立 scorer。
- **为何有效**：LLM 自评在可验证任务上与外部 scorer 接近；省去额外训练。
- **与 Router-R1 差异**：AutoMix 是 2-LLM 级联 + self-verify，Router-R1 是 N-LLM 路由 + 学习决策。两者可组合：Router-R1 内部嵌 self-verify。

#### PRM-in-loop (OpenR / Math-Shepherd, 2024–2025)
- **Framework**：planner LLM 每步生成 → 过程奖励模型（PRM）打分 → beam search 选最优。
- **旧范式问题**：outcome reward 太稀疏；single-shot CoT 错了无法纠。
- **改变**：用专门 verifier LLM 提供 step-level reward。
- **为何有效**：step-level 信号让 planner 在中间步骤纠错，等价于 verifier-guided search。
- **与 Router-R1 差异**：PRM 是常驻 verifier 而非 router 选择对象；OrchestratorR1 的"verifier-in-loop"缝隙正是把 PRM 当成 router 池里的一类特殊 worker。

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

## 8. 2026 年新工作补充（第四轮联网，2026-01 至 2026-04）✅

> 以下工作均通过 arXiv 搜索直接核实，按与 OrchestratorR1 的相关度分组。

### 8.1 RL 训练的多 Agent 编排 / 路由（L4 直接对比）

| 工作 | arXiv ID | 时间 | 一句话定位 | 与 OrchestratorR1 关系 |
|---|---|---|---|---|
| **AgentConductor** ✅ | 2602.17100 | 2026-02 | RL 动态生成 multi-agent 交互拓扑，用于竞赛级代码生成；反馈驱动的拓扑演化 | 与 Conductor 同名但不同工作；**拓扑是 RL 生成的而非预设**，比 DynaSwarm 更自由。OrchestratorR1 若做动态拓扑必比 |
| **FlowSteer** ✅ | 2602.01664 | 2026-02 | 端到端 RL 自动化 workflow 编排，policy model 与可执行 canvas 交互 | 把 workflow 编排建模为 RL policy 与环境交互，与 Router-R1 的 reason-then-route 互补；强调 interactive 而非 one-shot |
| **EvoRoute** ✅ | 2601.02695 | 2026-01 | 自演化路由范式，逐步选 Pareto 最优 LLM backbone，成本降 80%、延迟降 70% | **直接竞品**：同样做 per-step LLM 路由 + 成本/延迟感知，但用经验驱动的自演化而非 PPO。OrchestratorR1 必比 |
| **DyTopo** ✅ | 2602.06039 | 2026-02 | Manager 引导的动态拓扑路由，每轮用语义匹配重建 agent 通信图 | 与 Conductor 的拓扑学习类似但 training-free（语义匹配），可作 prompted baseline |
| **ORCH** ✅ | 2602.01797 | 2026-02 | 确定性多 agent 编排器 + EMA 引导路由，面向可复现的离散选择推理 | 强调确定性与可复现性，与 RL 路由的随机性形成对照 |
| **CORAL (Beyond Rule-Based Workflows)** ✅ | 2601.09883 | 2026-01 | 信息流编排范式，agent 间自然语言通信替代预定义 workflow | 无 RL 但提出 agent-to-agent 通信的新范式，与 Conductor 的拓扑通信思路可对照 |

### 8.2 Agentic RL 训练方法论（L2 方法学，可迁移到 OrchestratorR1 训练栈）

| 工作 | arXiv ID | 时间 | 一句话定位 | 可迁移价值 |
|---|---|---|---|---|
| **StepPO** ✅ | 2604.18401 | 2026-04 | Step-level 而非 token-level 的 policy optimization，专为多轮 agent 任务设计 | 直接适用于 OrchestratorR1 的多轮路由优化，比 PPO token-level 更自然 |
| **CM2** ✅ | 2602.12268 | 2026-02 | Checklist reward 替代 verifiable outcome reward，用于多轮多步 agent RL | 可为 OrchestratorR1 提供比 EM/F1 更细粒度的 reward 设计 |
| **SLEA-RL** ✅ | 2603.18079 | 2026-03 | Step-level experience augmented RL，每步检索相关经验增强多轮 agent 训练 | 长 horizon credit assignment 方案，可缓解 Router-R1 的 4-turn 稀疏 reward 问题 |
| **Reasoning vs Tool-use Interference** ✅ | 2602.00994 | 2026-01 | 量化 reasoning 与 tool-use 在联合训练中的干扰，提出解耦参数更新 | **核心发现**：reasoning 和 tool-use 会互相干扰。OrchestratorR1 同时学推理+路由，此问题高度相关 |
| **ASTER** ✅ | 2602.01204 | 2026-02 | 解决 tool-integrated reasoning 中的 interaction collapse，用交互密集冷启动轨迹 + RL | 直接解决 Router-R1 可能遇到的"模型不愿调用工具"问题 |
| **Tool-R0** ✅ | 2602.21320 | 2026-02 | 从零数据出发，self-play RL 训练 tool-calling agent | 无 SFT 冷启的极端方案，与 ReSearch 思路一致但更彻底 |
| **PEARL** ✅ | 2601.20439 | 2026-01 | Plan exploration + adaptive RL 用于多跳工具调用，离线探索 + 在线 RL | 多跳工具调用的 RL 训练 recipe，与 Router-R1 的多轮路由直接对应 |
| **Demystifying RL for Long-Horizon Agents** ✅ | 2603.21972 | 2026-03 | 系统性实证研究：reward shaping、模型 scaling、算法选择的最佳实践 | **实验设计参考**：OrchestratorR1 的 ablation 可对标此文的实验框架 |
| **Error-Localized PO (ELPO)** ✅ | 2602.09598 | 2026-02 | 定位长 horizon 轨迹中的不可恢复错误，细粒度 credit assignment | 解决 Router-R1 "哪一轮路由出了错"的归因问题 |

### 8.3 多 Agent 协作框架与鲁棒性（L3/L5 扩展）

| 工作 | arXiv ID | 时间 | 一句话定位 | 与 OrchestratorR1 关系 |
|---|---|---|---|---|
| **MARCH** ✅ | 2603.24579 | 2026-03 | 多 agent RL 自检管线，信息不对称下协作验证减少 RAG 幻觉 | 与 OrchestratorR1 的 verifier-in-loop 方向直接相关 |
| **Brain-Inspired Graph MAS** ✅ | 2603.16397 | 2026-03 | 脑启发的图结构多 agent，问题自适应拓扑 + 全局编排 | 图结构 + 自适应拓扑的新视角，可与 DynaSwarm/AgentConductor 对照 |
| **AgentCollab** ✅ | 2603.26034 | 2026-03 | 自评估驱动的协作范式，按 agent 自反馈信号动态升级到更强模型 | 类似 AutoMix 的级联思想但用 agent 自评估触发，training-free |
| **Training-Free Agentic AI (Thompson Sampling)** ✅ | 2603.13256 | 2026-02 | Thompson sampling + 校准 judge 做递归多 agent 委派的路由 | Training-free 路由的新 SOTA baseline，用概率控制替代 RL |
| **MAC (Multi-Agent Constitution)** ✅ | 2603.15968 | 2026-03 | 用 agent 网络优化结构化规则集，无参数更新 | 无训练的 agent 行为优化，可作 prompted 对照 |
| **MAS-FIRE** ✅ | 2602.19843 | 2026-02 | 15 种故障注入评估多 agent 鲁棒性，闭环设计可中和 40%+ 故障 | OrchestratorR1 的 noise injection 实验可参考此评估框架 |
| **Agent Drift** ✅ | 2601.04170 | 2026-01 | 量化多 agent 系统在长交互中的行为退化 | OrchestratorR1 若做长 horizon 实验需关注此问题 |

### 8.4 Agentic RL 基础设施

| 工作 | arXiv ID | 时间 | 一句话定位 |
|---|---|---|---|
| **ProRL Agent** ✅ | 2603.18815 | 2026-03 | Rollout-as-a-Service，解耦 rollout 编排与训练，支持多轮 agent RL |
| **Heddle** ✅ | 2603.28101 | 2026-03 | 分布式 agentic RL rollout 编排，轨迹中心调度 |
| **Agent World Model** ✅ | 2602.10090 | 2026-02 | 合成 1000 个可执行环境 + 丰富工具集，用于 agent RL 训练 |
| **ASTRA** ✅ | 2601.21558 | 2026-01 | 自动合成 agent 轨迹 + 可验证 RL 环境 |

### 8.5 RL 能力边界与理论分析

| 工作 | arXiv ID | 时间 | 一句话定位 |
|---|---|---|---|
| **Does RL Expand Agent Capability? (PASS@k,T)** ✅ | 2604.14877 | 2026-04 | 证明 RL 真正扩展了 agent 在组合工具使用任务上的能力边界，而非仅提升效率 |
| **AgentV-RL** ✅ | 2604.16004 | 2026-04 | 把 reward modeling 变成多轮 tool-augmented 过程，双向验证 agent |
| **Three Roles, One Model** ✅ | 2604.11465 | 2026-04 | 推理时用单个冻结模型扮演三角色（planner/executor/critic），性能翻倍 |
| **Plan-RewardBench** ✅ | 2604.08178 | 2026-04 | 工具使用 agent 轨迹级 reward model 的评估基准 |
| **Timely Machine** ✅ | 2601.16486 | 2026-01 | 把 test-time scaling 重定义为挂钟时间，RL 优化时间预算下的 agent 策略 |

---

## 9. 更新后的 OrchestratorR1 直接对比对象（2026-04-29 版）

按"架构相似度 + 时间相近度"排序，**粗体**为 2026 年新增：

1. **HierRouter** (2025-11) — PPO + 多轮 LLM pool，最相似。**必比**。
2. **Router-R1** (2025-06, NeurIPS 2025) — base。
3. **EvoRoute** ✅ **(2026-01)** — 自演化 per-step LLM 路由，成本降 80%。**必比**。
4. **AgentConductor** ✅ **(2026-02)** — RL 生成交互拓扑，比 DynaSwarm 更自由。**必比**。
5. **Conductor** (2025-12, ICLR 2026) — 中央 RL + 拓扑 + 递归。**必比**。
6. **FlowSteer** ✅ **(2026-02)** — 端到端 RL workflow 编排。
7. **Maestro** (2025-11) — central + parallel workers + CLPO。
8. **DynaSwarm** (2025-07) — 图结构 + A2C。
9. **FlowReasoner** (2025-04) — 一次出整 DAG。
10. **DyTopo** ✅ **(2026-02)** — 语义匹配动态拓扑（training-free baseline）。
11. **Search-R1** (2025-03) — 单工具父类。
12. **Symbolic-MoE / Avengers** — 无 RL 单轮对照组。

### 训练方法论必读（2026 新增）

- **StepPO** (2026-04) — step-level PO，替代 token-level PPO
- **CM2** (2026-02) — checklist reward，替代 EM/F1
- **Reasoning vs Tool-use Interference** (2026-01) — 推理与工具使用的训练干扰量化
- **ASTER** (2026-02) — interaction collapse 解决方案
- **ELPO** (2026-02) — 长 horizon 错误定位与 credit assignment
- **Demystifying RL for Long-Horizon Agents** (2026-03) — 系统性 ablation 参考

---

## 10. 更新后的空白与缝隙（2026-04-29 版）

原有缝隙在 2026 年的填补情况：

| 缝隙 | 2026 年进展 | OrchestratorR1 还能做什么 |
|---|---|---|
| **Horizon 长度** | Demystifying RL (2603.21972) 给出长 horizon recipe；SLEA-RL 做 step-level 经验增强；KLong 提出渐进式 RL + 轨迹拆分 | OrchestratorR1 借鉴 KLong 的渐进式训练（max_turns 2→4→6）和上下文压缩（保留首尾、摘要中间），在 **LLM-as-tool 路由** 场景下支持 >4 turn 编排 |
| **预算条件化路由** | EvoRoute 做了 cost/latency Pareto；Timely Machine 做了时间预算 RL | 但没有 **用户给定预算作为 RL 条件输入** 的工作 |
| **Verifier-in-loop** | MARCH 做了多 agent 自检；AgentV-RL 做了 verifier agent | 但没有把 verifier 作为 **router 池中的一类 worker** 来路由 |
| **噪声鲁棒聚合** | MAS-FIRE 提供了故障注入评估框架 | 但没有在 **RL 训练中** 加入 noise-robust reward |
| **递归路由** | AgentConductor 做了拓扑演化；Three Roles 做了单模型三角色 | 但 **planner 调用 planner** 的递归 RL 仍空白 |
| **Reasoning vs Tool-use 干扰** | 2602.00994 首次量化了此问题 | OrchestratorR1 可以是 **首个在 LLM 路由场景验证解耦训练** 的工作 |
| **Strip-think / 思维链剥离** | 仍无人触碰 | OrchestratorR1 独占缝隙 |

---

## 11. 待确认 / TODO（2026-04-29 更新）

- [x] 联网核对：Router-R1、Conductor、HierRouter、DynaSwarm、Maestro、RIRS 已✅
- [x] Dr. MAS (2602.08847) ✅ 已确认
- [x] WideSeek-R1 (2602.04634) ✅ 已确认
- [x] MagicAgent (2602.17100) ✅ 已确认
- [x] Training LLMs for Multi-Step Tool Orchestration (2603.24709) ✅ 已确认
- [ ] 复核 Mimosa (2603.28986)、Youtu-Agent (2512.24615) 的 ID
- [ ] 若要写 related work 章节，优先精读：EvoRoute、AgentConductor、StepPO、Reasoning vs Tool-use Interference
- [ ] 若 OrchestratorR1 强调 latency / noise / budget，需补 latency-aware serving（Tabi、OptLLM、LLMCascade 2025）
- [ ] 补充 Stronger-MAS / AT-GRPO 的具体 arXiv ID
