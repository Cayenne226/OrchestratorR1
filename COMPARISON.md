# 四大竞品深度对比：Router-R1 / Prompt-R1 / Conductor / AgentConductor

> 更新日期: 2026-03-24
> 目的: 为 OrchestratorR1 定位 NeurIPS 2026 投稿方向

---

## 1. 全景对比表

| 维度 | Router-R1 | Prompt-R1 | Conductor | AgentConductor | **OrchestratorR1 (Ours)** |
|------|-----------|-----------|-----------|----------------|---------------------------|
| **发表** | NeurIPS 2025 | arXiv 2511 | ICLR 2026 | arXiv 2602 | NeurIPS 2026 (target) |
| **核心思想** | RL 学路由：选哪个 LLM 回答 | RL 学 prompt：小模型给大模型写提示词 | RL 学编排：一次性生成完整 workflow | SFT+RL 学拓扑：生成 DAG 执行图 | RL 学编排：逐步反应式调度功能 Agent |
| **训练方法** | PPO (veRL) | GRPO | GRPO | SFT + GRPO (veRL) | SFT warmup + GRPO (trl) |
| **基座模型** | Qwen2.5-3B/7B | 小 LLM (具体未公开) | 7B (Qwen2.5-7B-Instruct) | Qwen2.5-7B-Instruct | Qwen2.5-3B/7B-Instruct |
| **被调度对象** | 同质 LLM（不同模型回答同一问题） | 大 LLM（GPT-4o 等） | 多个 worker LLM | 多个 coding agent | 6 种功能 Agent（异质角色） |
| **决策粒度** | 选模型 + 写 query | 写 prompt | 生成完整 workflow（agent选择+指令+通信拓扑） | 生成分层 DAG（节点+边+并行度） | 逐步选 agent type + 写 query |
| **执行范式** | 反应式多轮 | 多轮交互 | **开环一次性规划** | **开环 DAG 生成**（多轮可更新） | **闭环反应式** |
| **是否有 Critic 网络** | ✅ (PPO 需要) | ❌ | ❌ (GRPO) | ❌ (GRPO) | ❌ (GRPO) |
| **SFT 预热** | ❌ | ❌ | ❌ (纯 GRPO) | ✅ (4500 条) | ✅ (50-100 条) |
| **任务范围** | 7 个 QA 数据集 | 12 个数据集（多类型） | 数学+代码+推理 (MATH, MMLU, LiveCodeBench, GPQA) | 竞赛级代码 (5 个代码数据集) | QA + 多跳推理 + 代码 |
| **成本感知** | cost_coe 惩罚 (默认=0) | ❌ | ❌ | 密度函数隐式控制 | ✅ 显式 α 调节 Pareto |
| **最大轮数** | 4 | 多轮 | 1（一次性输出） | 多轮（DAG 可更新） | 6 |
| **训练数据量** | 7K × 2 = 14K | 未公开 | 960 条（4 领域） | 4500 条 SFT + RL | ~5K-7K |
| **训练硬件** | 4 GPU + vLLM | 未公开 | 未公开 (推测 8×A100) | veRL + vLLM | 4×A100/V100 80GB |
| **训练成本** | 中等（PPO 双网络 + vLLM） | 未公开 | 200 GRPO iterations | SFT + GRPO | 低（单网络 GRPO，无 vLLM） |

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
用户问题 → Conductor(7B) → 一次性输出完整 workflow:
  [Step 1: Agent=Claude, Instruction="...", Visibility={Agent2}]
  [Step 2: Agent=GPT-4o, Instruction="...", Visibility={Agent1, Agent3}]
  [Step 3: Agent=Gemini, Instruction="...", Visibility={ALL}]
→ 框架执行 workflow → 收集所有 agent 输出 → 返回最终答案
```

**技术细节**:
- 7B 模型（Qwen2.5-7B-Instruct），纯 GRPO，无 SFT 预热
- 训练数据仅 960 条（4 领域：MATH, MMLU, RLPR, LiveCodeBench）
- 200 GRPO iterations，batch 256，无 KL 正则化
- AdamW 优化器
- Workflow 包含三个维度：agent 选择、指令 prompt、通信拓扑（谁能看到谁的输出）
- **涌现行为**: 问题分解、prompt 工程、独立尝试 + 辩论轮

**关键结果**:
- LiveCodeBench: 超越所有之前的 LLM（包括 OpenAI O 系列）
- GPQA Diamond: +3% 左右（等于一整代模型的提升）
- 超越传统 self-reflection 和 多 agent 基线
- 泛化到训练数据之外的任务

**优势**:
- SOTA 结果，极其强
- 训练数据极少（960 条）却泛化很好
- 通信拓扑是独特创新（agent 间可见性）
- ICLR 2026 已接收，学术认可度高

**局限 (你的机会)**:
1. **开环规划**: 一次性生成完整 workflow，执行过程中不能根据中间结果调整
   - 如果 Step 1 的 agent 返回了错误信息，Step 2/3 仍然照原计划执行
   - 无法动态增加/减少步骤
2. **无成本优化**: 每次都生成完整多步 workflow，简单题也走 3 步
3. **无显式功能角色**: agent 的功能由 Conductor 的 prompt 决定，不是预定义的
   - 这既是优势（更灵活）也是劣势（学习负担重）
4. **仅推理任务**: MATH/MMLU/GPQA/LiveCodeBench，没有 QA/多跳推理

**你 vs Conductor 的关键论点**:

| 维度 | Conductor | OrchestratorR1 | 谁优？ |
|------|-----------|----------------|--------|
| 结果质量 | SOTA (LiveCodeBench, GPQA) | 待验证 | Conductor 大概率更强 |
| 适应性 | 开环，不可调整 | 闭环，逐步调整 | ✅ 你 |
| 错误恢复 | 无法恢复（执行完才看结果） | 可以：critic 反馈 → 重试 | ✅ 你 |
| 成本效率 | 固定多步 | 按需 1-6 步 | ✅ 你 |
| 简单题处理 | 仍生成多步 workflow | 1 步直接回答 | ✅ 你 |
| 功能清晰度 | 隐式（由 prompt 决定） | 显式（6 种角色） | ✅ 你（更可解释） |
| 通信拓扑 | ✅ agent 间可见性 | ❌ 只有 orchestrator 看到所有 | Conductor |
| 训练效率 | 960 条，200 iterations | ~5K 条，~2000 steps | Conductor 更高效 |
| 并行执行 | ✅ workflow 中可并行 | ❌ 串行 | Conductor |

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

### 3.2 建议的论文 Positioning

不要试图在 LiveCodeBench/GPQA 上打败 Conductor（大概率打不过，因为它用的 worker 模型更强、通信拓扑更灵活）。

**正确定位**:

> "Conductor 证明了开环规划的强大，但现实世界的编排需要闭环适应。
> OrchestratorR1 是第一个证明：反应式编排策略可以通过端到端 RL 从单一奖励函数涌现，
> 并且一个策略可以跨 QA、推理、代码三种模态泛化，同时提供显式的成本-质量权衡。"

关键对比角度不是"谁的数字更高"，而是：
1. **闭环 vs 开环** — 你能做 Conductor 做不到的事（中间失败恢复）
2. **成本效率** — 简单题 1 步解决 vs Conductor 总是多步
3. **跨模态** — 一个策略三种任务 vs 专门化
4. **实用性** — 训练简单（trl + 无 vLLM），成本低（$5-10）

---

## 4. 实验设计建议（基于对比分析）

### 4.1 必须做的实验

**A. 直接对比 Conductor 的开环 vs 你的闭环**:
```
实验: Adversarial Intermediate Failure
设计: 注入 "噪声 agent 返回"（如 executor 返回错误信息）
指标: 恢复率（recovery rate）
预期: 你 > Conductor（Conductor 无法恢复）
```

**B. 简单/复杂题效率分析**:
```
实验: 按题目难度分组统计
指标: 平均轮数 / 平均成本
预期: 简单题你 1.2 轮 vs Conductor 3+ 步
```

**C. 跨模态泛化**:
```
实验: 同一个训练好的模型评估 3 个 track
指标: 每个 track 的 F1/EM/Pass@1
对比: Router-R1 (只有 QA) / Conductor (推理+代码) / AgentConductor (只有代码)
预期: 你是唯一一个三个 track 都有的
```

**D. Pareto 前沿**:
```
实验: α = {0, 0.1, 0.3, 0.5, 0.7, 0.9}
图: cost vs F1 散点图，每个 α 一个点
对比: Router-R1 (单点, coe=0) / Direct-GPT-4o (单点) / Conductor (单点)
预期: 你覆盖一条 Pareto 前沿，其他方法是固定点
```

### 4.2 可以 convincingly 赢 Conductor 的实验

| 实验 | 为什么你能赢 |
|------|------------|
| 错误恢复率 | Conductor 开环无法恢复，你的 critic→retry 可以 |
| 简单题成本 | Conductor 总是多步，你 1 步搞定 |
| 训练效率 | 你无需 vLLM，单网络 GRPO，$5-10 API |
| 跨模态广度 | Conductor 没有 QA/多跳推理数据集 |
| 可解释性 | 你的 agent 角色是显式的（refiner/critic/...），Conductor 的是隐式的 |

### 4.3 不要尝试赢 Conductor 的实验

| 实验 | 为什么你大概率输 |
|------|----------------|
| LiveCodeBench 绝对 Pass@1 | Conductor 用 Claude/GPT-4o 等强 worker，通信拓扑更灵活 |
| GPQA Diamond 绝对分数 | 同上 |
| 并行执行效率 | Conductor/AgentConductor 支持并行，你是串行 |

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

## 7. Reviewer 会问的问题（预判 + 准备回答）

### Q1: "Why not just use Conductor? It already achieves SOTA."
**A**: Conductor 是开环规划，无法根据中间结果调整策略。我们的 Table X 展示了在 adversarial intermediate failure 场景下，Conductor 的准确率下降 Y%，而我们通过闭环调整仅下降 Z%。此外，Conductor 对简单题仍生成多步 workflow（Table X: 平均 N 步），而我们平均仅 1.2 步，成本低 M 倍。

### Q2: "Your absolute numbers are lower than Conductor on GPQA/LiveCodeBench."
**A**: 我们的贡献不是刷 SOTA，而是证明反应式编排策略可以通过 RL 涌现。我们在成本效率（Pareto 前沿）、跨模态泛化（三个 track）、和错误恢复能力上超越 Conductor。

### Q3: "The agent pool is hand-designed. How do you know 6 agents is optimal?"
**A**: Agent 池是固定基础设施，类似 tool-use 中的 tool 定义。消融实验（Table 2）证明每种 agent 的贡献。未来工作可以探索自动发现 agent 角色。

### Q4: "GRPO without critic — isn't PPO (Router-R1) better?"
**A**: GRPO 训练效率更高（单网络 vs 双网络），在我们的设置下 GRPO 已足够有效（G=16 提供足够的组内方差）。Table X 对比了 3B 下 GRPO vs PPO 的效果。

### Q5: "SFT warmup 是否泄露了 supervision signal？和纯 RL 的贡献如何分离？"
**A**: SFT-only 消融（Table 2）准确率仅 X%，GRPO 后提升到 Y%，差异 Z% 全部来自 RL。SFT 仅教格式，不教策略。

### Q6: "Serial execution is a bottleneck. Why not support parallel?"
**A**: 串行是反应式编排的代价——为了获得闭环适应能力。在延迟敏感场景下，我们的 1-2 步路径（简单题）实际比 Conductor 的 3+ 步并行更快。未来可以引入 conditional parallel execution。
