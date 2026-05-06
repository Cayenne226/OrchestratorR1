# OrchestratorR1：基于强化学习的反应式多智能体编排

> **一句话概括**：训练一个小型 LLM（Qwen2.5-3B/7B），通过 GRPO 强化学习使其成为"元控制器"——它反应式地编排 4 个专用外部 Agent（executor 支持 strong/weak 双档位），在每次 Agent 返回结果后观察响应再决定下一步动作，超越固定流水线和开环规划器。

---

## 目录

- [1. 研究动机与核心思想](#1-研究动机与核心思想)
- [2. 系统架构总览](#2-系统架构总览)
- [3. 项目结构](#3-项目结构)
- [4. Agent 池设计](#4-agent-池设计)
- [5. XML 标签通信协议](#5-xml-标签通信协议)
- [6. 核心模块详解](#6-核心模块详解)
  - [6.1 解析器 (`parser.py`)](#61-解析器-parserpy)
  - [6.2 反应式生成循环 (`generation.py`)](#62-反应式生成循环-generationpy)
  - [6.3 开环消融实验 (`generation_openloop.py`)](#63-开环消融实验-generation_openlooppy)
  - [6.4 奖励函数 (`reward.py`)](#64-奖励函数-rewardpy)
  - [6.5 Agent 注册与调度 (`agent_registry.py`)](#65-agent-注册与调度-agent_registrypy)
  - [6.6 基础 Agent (`base_agent.py`)](#66-基础-agent-base_agentpy)
  - [6.7 系统提示词 (`system_prompt.py`)](#67-系统提示词-system_promptpy)
- [7. 训练流程](#7-训练流程)
  - [7.1 阶段 0：SFT 热身](#71-阶段-0sft-热身)
  - [7.2 阶段 1：GRPO 强化学习](#72-阶段-1grpo-强化学习)
  - [7.3 训练中的奖励函数](#73-训练中的奖励函数)
- [8. 数据流程](#8-数据流程)
- [9. 评估系统](#9-评估系统)
- [10. Worker Pool 设计](#10-worker-pool-设计)
- [11. 关键超参数](#11-关键超参数)
- [12. 快速开始](#12-快速开始)

---

## 1. 研究动机与核心思想

### 现有问题

现有多智能体系统主要分两类：

1. **固定流水线** —— 无论任务复杂度如何，始终按相同顺序调用 Agent（例如：decomposer -> executor -> critic -> synthesizer）。对简单任务造成浪费，对新型任务缺乏灵活性。

2. **开环规划器**（如 Conductor）—— 一个强大的 LLM 预先生成完整的执行计划，然后所有 Agent 并行执行。规划器永远看不到中间结果，无法自适应调整。

### 我们的方案：反应式编排

我们训练一个小型、低成本的 LLM 作为**反应式元控制器**。核心洞察：

```
每次 Agent 调用完成后，Agent 的响应会被注入回模型的上下文中。
模型在决定下一步动作之前，先观察上一步的响应。
```

这构成了一个闭环反馈系统：

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  用户问题                                                │
│       │                                                 │
│  [编排器 LLM]  <────────────────────────────┐            │
│       │                                     │            │
│       ├── <think> 分析任务复杂度              │            │
│       ├── <call type="agent_X"> 查询         │            │
│       │        │                             │            │
│       │   [Agent X 执行]                     │            │
│       │        │                             │            │
│       │   <information> 响应 </info>  ───────┘            │
│       │                                                  │
│       │   （模型阅读响应，决定下一步）                       │
│       │                                                  │
│       ├── <call type="agent_Y"> 后续查询                  │
│       │        │                                         │
│       │   [Agent Y 执行]                                 │
│       │        │                                         │
│       │   <information> 响应 </info>  ───────┘            │
│       │                                                  │
│       └── <answer> 最终答案 </answer>                     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**为什么这很重要**：如果 Agent X 返回了错误或不完整的答案，编排器可以调用 critic 进行验证、用更强的模型重试、或者改写查询。开环规划器做不到这一点。

---

## 2. 系统架构总览

```
                          ┌──────────────────────────────┐
                          │    编排器 LLM                  │
                          │   (Qwen2.5-3B, RL 训练)       │ 同时解决：通过该任务training 提升了qwen2.5性能
                          └──────┬───────────────────────┘
                                 │
                    生成 XML 标签: <think>, <call>, <answer>
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
              ┌─────▼─────┐  ┌──▼──┐  ┌──────▼──────┐
              │  解析器     │  │     │  │             │
              │ (parser.py)│  │ ... │  │             │
              └─────┬─────┘  └─────┘  └─────────────┘
                    │
            提取 agent_type + query
                    │
         ┌──────────▼──────────┐
         │   Agent 注册表        │
         │  (agent_registry.py) │
         └──────────┬──────────┘
                    │
    通过 OpenAI 兼容 API 调度到对应 Agent
                    │
    ┌───────┬───────┼───────────┬────────────┐
    │       │       │           │            │
 executor  executor  decomp    critic      synth
 (strong)  (weak)    分解器    评审器      合成器
 强力执行   廉价执行
```

**训练循环** (GRPO)：

```
对每个训练 prompt：
  1. 从模型采样 G=8 条 rollout
  2. 对每条 rollout，解析其中所有 <call> 标签
  3. 执行真实 API 调用（训练时使用真实 API！）
  4. 计算奖励 R(τ) = R_outcome - α·C_cost - β·C_turns + γ·B_efficiency
  5. 使用 GRPO 更新模型（组内奖励优势）
```

---

## 3. 项目结构

```
OrchestratorR1/
├── orchestrator_r1/              # 核心 Python 包
│   ├── agent_pool/
│   │   ├── base_agent.py         # OpenAI 兼容 API 封装（重试 + 成本追踪）
│   │   └── agent_registry.py     # 4-Agent 调度表（executor 支持 strong/weak 双档位）
│   ├── orchestrator/
│   │   ├── parser.py             # XML 标签解析器 + 格式校验器
│   │   ├── reward.py             # 复合奖励 R(τ) 计算
│   │   ├── generation.py         # 反应式多轮生成循环（核心）
│   │   └── generation_openloop.py  # 开环消融（先规划再执行）
│   └── prompts/
│       └── system_prompt.py      # 完整系统提示词（含 Agent 描述）
│
├── training/
│   ├── train.py                  # GRPO 训练入口（trl.GRPOTrainer）
│   ├── sft_warmup.py             # SFT 阶段：教会模型 <call>/<answer> 格式
│   ├── train_lora.bat            # Windows: 4xRTX 3090, LoRA, ~8GB/GPU
│   ├── train_full.bat            # Windows: 4xRTX 3090, ZeRO-2, ~16.5GB/GPU
│   ├── train.sh / train_flex.sh  # Linux: FSDP+NCCL
│   └── accelerate_*.yaml        # 分布式训练配置
│
├── data_process/
│   ├── prepare_data.py           # QA 数据集加载（6 个来源，来自 HuggingFace）
│   ├── prepare_code.py           # 代码数据集加载（HumanEval + MBPP）
│   └── prepare_sft.py            # 通过 GPT-4o 自动生成 SFT 热身数据
│
├── eval/
│   ├── eval_orchestrator.py      # 主评估脚本
│   ├── baselines.py              # Direct-Strong, Direct-Cheap, Fixed-Pipeline 基线
│   ├── run_self_reflection.py    # Self-Reflection 基线（5 轮）
│   ├── run_ablation_openloop.py  # 开环消融评估
│   └── metrics.py                # EM/F1/GPQA-accuracy/Pass@1/LiveCode 指标
│
├── inference/infer.py            # CLI 单条推理
├── analysis/                     # 论文图表生成脚本
└── test_local.py                 # 仅 CPU 的单元测试
```

---

## 4. Agent 池设计

编排器拥有 4 个功能专业化 Agent，每个由不同的 LLM 通过 OpenAI 兼容 API 提供支持。其中 **executor** 支持 strong/weak 双档位，由编排器在调用时通过 `tier` 属性选择：

| Agent | 角色 | 后端模型 | 使用时机 |
|-------|------|---------|----------|
| **executor (strong)** | 高质量、高成本的执行 | Claude Sonnet 4 | 困难推理、复杂代码、深度分析 |
| **executor (weak)** | 快速、低成本的执行 | GPT-4o | 简单事实查询、明确子任务 |
| **decomposer**（分解器） | 将复杂任务拆分为子任务 | Gemini 2.5 Pro | 任务包含多个独立步骤时 |
| **critic**（评审器） | 质量验证 | Claude Sonnet 4 | 对结果质量不确定或要求严格时 |
| **synthesizer**（合成器） | 合并多个部分结果 | GPT-4o | 执行多个子任务之后 |

> **设计说明**：早期版本曾包含独立的 `refiner`（查询改写）Agent，但实验表明编排器在 `<think>` 标签内的内部推理已能完成 query rewriting，独立 refiner 与 executor 输入存在功能重叠，因此被移除。每个 Agent 都有专属的系统提示词来定义其行为（在 `agent_registry.py` 中定义），例如：

```python
AGENT_SYSTEM_PROMPTS = {
    "executor": (
        "You are an expert task executor. Complete the given task accurately. "
        "For straightforward queries, be direct and concise. For complex queries, "
        "provide thorough, well-structured output. Focus on correctness."
    ),
    "decomposer": (
        "You are a task planning expert. Break down the given complex task into a numbered list "
        "of independent, executable subtasks. Each subtask should be self-contained and actionable. "
        "Output only the numbered list, nothing else."
    ),
    "critic": (
        "You are a strict quality reviewer. Evaluate the given result for correctness, "
        "completeness, and quality. Identify specific issues or missing parts. "
        "Output a brief assessment and a score from 1-10, then list concrete improvements needed."
    ),
    "synthesizer": (
        "You are an integration expert. Combine the given partial results into a single coherent, "
        "complete, and well-structured final answer. Eliminate redundancy and ensure consistency. "
        "Output only the final combined result."
    ),
}
```

---

## 5. XML 标签通信协议

编排器模型通过结构化 XML 标签与系统通信：

| 标签 | 方向 | 描述 |
|------|------|------|
| `<think>...</think>` | 模型 -> 系统 | 内部思维链推理 |
| `<call type="X">query</call>` | 模型 -> 系统 | 请求调用 Agent X |
| `<information>...</information>` | 系统 -> 模型 | Agent 的响应（由系统注入） |
| `<answer>...</answer>` | 模型 -> 系统 | 最终答案 |

**交互示例：**

```xml
<think>这是一个关于历史的多跳问题。我应该先分解它。</think>

<call type="decomposer">一战结束的条约签署时，美国总统是谁？</call>

<information>子任务 1：一战结束的条约是什么？
子任务 2：这个条约是什么时候签署的？
子任务 3：当时美国总统是谁？</information>

<think>分解器把问题拆成了 3 部分。我用 executor (weak) 来处理这个事实链。</think>

<call type="executor" tier="weak">一战结束的条约是什么？什么时候签署的？</call>

<information>《凡尔赛条约》于 1919 年 6 月 28 日签署。</information>

<call type="executor" tier="weak">1919 年 6 月的美国总统是谁？</call>

<information>1919 年 6 月的美国总统是伍德罗·威尔逊。</information>

<answer>伍德罗·威尔逊 (Woodrow Wilson)</answer>
```

---

## 6. 核心模块详解

### 6.1 解析器 (`parser.py`)

解析器从模型的原始文本输出中提取结构化信息。

**关键数据结构：**

```python
VALID_AGENT_TYPES = {
    "executor", "decomposer", "critic", "synthesizer"
}
VALID_EXECUTOR_TIERS = {"strong", "weak"}

STOP_TOKENS = ["</call>", "</answer>"]

@dataclass
class CallTag:
    agent_type: str   # VALID_AGENT_TYPES 中的一个
    query: str        # 发送给 Agent 的文本
    raw: str          # 完整的匹配 XML 字符串

@dataclass
class ParseResult:
    call: Optional[CallTag] = None      # 提取到的 <call> 标签（如有）
    answer: Optional[str] = None        # 提取到的 <answer> 标签（如有）
    has_think: bool = False             # 是否存在 <think>
```

**核心解析逻辑** —— 使用正则表达式提取 `<call type="X">query</call>` 和 `<answer>`：

```python
def parse_output(text: str) -> ParseResult:
    result = ParseResult()
    result.has_think = bool(re.search(r"<think>", text))

    # 提取 <call type="X">query</call>
    call_match = re.search(
        r'<call\s+type="(\w+)"[^>]*>(.*?)</call>',
        text, re.DOTALL,
    )
    if call_match:
        agent_type = call_match.group(1).strip()
        query = call_match.group(2).strip()
        if agent_type in VALID_AGENT_TYPES:
            result.call = CallTag(agent_type=agent_type, query=query, raw=call_match.group(0))
        # 如果 agent type 无效，视为格式错误（不返回 call）

    # 提取 <answer>...</answer>
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_match:
        result.answer = answer_match.group(1).strip()

    return result
```

**格式校验** —— 被奖励函数用来检测无效输出并施加 -1.0 惩罚：

```python
def validate_format(text: str) -> tuple[bool, str]:
    """返回 (is_valid, reason)。"""
    has_call = bool(re.search(r"<call\s+type=", text))
    has_answer = bool(re.search(r"<answer>", text))

    if not has_call and not has_answer:
        return False, "No <call> or <answer> tag found"

    # 检查开闭标签是否匹配
    open_calls = len(re.findall(r"<call\s", text))
    close_calls = len(re.findall(r"</call>", text))
    if open_calls != close_calls:
        return False, f"Mismatched <call> tags: {open_calls} open, {close_calls} close"

    # 验证所有 call 标签中的 agent type 是否合法
    call_types = re.findall(r'<call\s+type="(\w+)"', text)
    for t in call_types:
        if t not in VALID_AGENT_TYPES:
            return False, f"Invalid agent type: {t}"

    return True, "ok"
```

---

### 6.2 反应式生成循环 (`generation.py`)

这是**整个项目的核心**。`OrchestratorGenerationManager` 运行一个多轮循环：模型生成文本、调用 Agent、观察响应、决定下一步动作。

**配置：**

```python
@dataclass
class GenerationConfig:
    max_turns: int = 6             # 每个查询的最大 Agent 调用次数
    max_new_tokens: int = 512      # 每步生成的最大 token 数
    temperature: float = 0.7
    top_p: float = 0.9
    max_obs_length: int = 800      # 每轮注入的 Agent 响应最大字符数

@dataclass
class RolloutResult:
    full_text: str                 # 完整上下文（prompt + 所有轮次）
    answer: Optional[str]          # 提取到的最终答案
    agent_calls: List[dict]        # 列表 [{agent_type, query, cost, turn}, ...]
    n_turns: int = 0               # 进行了多少次 Agent 调用
    total_cost: float = 0.0        # 总 API 成本（美元）
    token_ids: List[int]           # 用于 RL 训练的 token ID
```

**Prompt 构建** —— 使用 HuggingFace 的 chat template：

```python
def _build_prompt(self, user_input: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_input},
    ]
    return self.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
```

**反应式 rollout 循环** —— 项目中最重要的方法：

```python
def rollout(self, user_input: str) -> RolloutResult:
    """对单个输入运行一次完整的编排 rollout。"""
    context = self._build_prompt(user_input)    # 通过 chat template 构建系统提示 + 用户消息
    agent_calls = []
    total_cost = 0.0

    for turn in range(self.config.max_turns):   # 最多 6 轮
        # 第 1 步：生成下一段文本（直到 </call> 或 </answer>）
        generated = self._generate_step(context)
        context += generated

        # 第 2 步：解析模型输出
        parsed = parse_output(generated)

        # 第 3a 步：如果模型想调用 Agent -> 调度、注入响应、继续循环
        if parsed.call is not None:
            call = parsed.call
            response, cost = self.registry.dispatch(call.agent_type, call.query)
            total_cost += cost
            agent_calls.append({
                "agent_type": call.agent_type,
                "query": call.query,
                "cost": cost,
                "turn": turn,
            })
            # 截断过长的响应以防上下文溢出
            if len(response) > self.config.max_obs_length:
                response = response[:self.config.max_obs_length] + "..."
            # 关键步骤：将 Agent 响应注入回上下文，供模型阅读
            context += f"\n<information>{response}</information>\n"
            continue   # <-- 回到循环顶部：模型将看到响应并决定下一步动作

        # 第 3b 步：如果模型输出了最终答案 -> 返回
        if parsed.answer is not None:
            return RolloutResult(
                full_text=context,
                answer=parsed.answer,
                agent_calls=agent_calls,
                n_turns=turn + 1,
                total_cost=total_cost,
            )

    # 达到最大轮次仍未输出 <answer> -> 从上下文中提取现有答案
    answer = extract_answer(context)
    return RolloutResult(
        full_text=context, answer=answer,
        agent_calls=agent_calls,
        n_turns=self.config.max_turns,
        total_cost=total_cost,
    )
```

**这个设计为什么重要：**

- 执行 `context += f"\n<information>{response}</information>\n"` 之后，模型的下一次 `_generate_step()` 调用会在上下文窗口中看到 Agent 的响应
- 这使得模型能够**反应式地**应对意外结果：如果答案看起来有误就调用 critic，如果任务复杂就调用 decomposer 拆解，如果结果令人满意就直接输出 `<answer>`
- `continue` 语句意味着循环回到顶部，生成可以包含另一个 `<call>` 或 `<answer>` 的新段落
- 模型通过 RL 学习哪些反应模式能带来更高的奖励

**生成步骤** —— 生成 token 直到遇到停止标记或达到最大长度：

```python
@torch.no_grad()
def _generate_step(self, context: str) -> str:
    """生成直到 </call> 或 </answer> 停止标记。"""
    inputs = self.tokenizer(
        context, return_tensors="pt", truncation=True, max_length=4096,
    ).to(self.model.device)

    output_ids = self.model.generate(
        **inputs,
        max_new_tokens=self.config.max_new_tokens,
        temperature=self.config.temperature,
        top_p=self.config.top_p,
        do_sample=True,
        pad_token_id=self.tokenizer.eos_token_id,
    )

    # 只解码新生成的 token（不包括 prompt）
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return self.tokenizer.decode(new_ids, skip_special_tokens=True)
```

---

### 6.3 开环消融实验 (`generation_openloop.py`)

这是**最重要的消融实验**。它通过移除反馈循环来直接验证"反应式 vs. 开环"的假设。

**与反应式循环的区别：**

```
反应式（我们的方法）：            开环（消融实验）：
  think -> call -> info ->        think -> call_1 -> call_2 -> call_3 ->
  think -> call -> info ->        [并行执行所有调用] ->
  think -> answer                  info_1 + info_2 + info_3 ->
                                   answer
```

**4 阶段流程：**

```python
def rollout(self, user_input: str) -> RolloutResult:
    context = self._build_prompt(user_input)

    # 阶段 1：一次性生成所有调用（2 倍 token 预算，无中间反馈）
    plan_text = self._generate_step(context, max_new_tokens=self.config.max_new_tokens * 2)
    context += plan_text

    # 检查模型是否已经给出答案（简单情况）
    answer = extract_answer(plan_text)
    if answer is not None:
        return RolloutResult(full_text=context, answer=answer, agent_calls=[], n_turns=1, ...)

    # 阶段 2：从生成的计划中提取所有 <call> 标签
    planned_calls = self._extract_all_calls(plan_text)  # regex findall

    # 阶段 3：通过 dispatch_batch 并行执行所有调用（ThreadPoolExecutor）
    results_list = self.registry.dispatch_batch(planned_calls)

    # 将所有结果一次性注入为一个整体块
    all_info_blocks = []
    for call, (response, cost) in zip(planned_calls, results_list):
        total_cost += cost
        if len(response) > self.config.max_obs_length:
            response = response[:self.config.max_obs_length] + "..."
        all_info_blocks.append(
            f'<information source="{call["agent_type"]}">{response}</information>'
        )

    context += "\n\n" + "\n".join(all_info_blocks) + "\n\n"
    context += "Based on all the information above, provide your final answer.\n"

    # 阶段 4：基于所有结果生成最终 <answer>
    answer_text = self._generate_step(context)
    context += answer_text
    answer = extract_answer(answer_text) or extract_answer(context)

    return RolloutResult(
        full_text=context, answer=answer,
        agent_calls=agent_calls,
        n_turns=1,  # 开环 = 1 "轮" 规划
        total_cost=total_cost,
    )
```

**这个实验验证什么**：如果反应式循环只是一种更昂贵的等价方案，那么开环消融应该能达到相似的准确率。如果我们的假设正确，开环版本的表现应该明显更差，因为它无法对中间结果进行自适应调整。

---

### 6.4 奖励函数 (`reward.py`)

奖励函数驱动 RL 训练，平衡四个目标：

```
R(τ) = R_outcome - α * C_cost - β * C_turns + γ * B_efficiency
```

| 分量 | 公式 | 描述 |
|------|------|------|
| R_outcome | F1 或 EM vs. 标准答案 | 答案质量（0 到 1） |
| C_cost | min(总API成本 / $0.01, 1.0) | API 成本，归一化到 [0,1] |
| C_turns | n_turns / max_turns | 轮次惩罚 |
| B_efficiency | 若 (R_outcome >= 0.8 且 n_turns <= 2) 则 1.0，否则 0.0 | 效率奖励 |

**默认权重：** alpha=0.3, beta=0.1, gamma=0.15

**硬格式惩罚：** 如果模型输出未通过 `validate_format()` 检验，整个奖励直接为 **-1.0**。这强烈抑制了格式错误的输出。

**完整实现：**

```python
PUNISH_FORMAT = -1.0      # 无效标签格式的硬惩罚
COST_NORM_BASE = 0.01     # 每次查询 $0.01 = 最大预期成本

def compute_reward(
    full_response: str,
    gold_answer: Union[str, list],
    agent_calls: List[dict],
    n_turns: int,
    metric: str = "f1",
    alpha: float = 0.3,     # API 成本惩罚权重
    beta: float = 0.1,      # 轮次惩罚权重
    gamma: float = 0.15,    # 效率奖励权重
    max_turns: int = 6,
) -> dict:
    # 1. 格式检查 -- 无效标签立即返回 -1.0
    is_valid, reason = validate_format(full_response)
    if not is_valid:
        return {"reward": PUNISH_FORMAT, "R_outcome": 0.0, "format_error": reason, ...}

    # 2. 提取预测答案
    pred = extract_answer(full_response) or ""

    # 3. 答案质量（F1 或 EM）
    if metric == "em":
        R_outcome = compute_em(pred, gold_answer)
    else:
        R_outcome = compute_f1(pred, gold_answer)

    # 4. API 成本惩罚（归一化：$0.01 映射到 1.0，上限为 1.0）
    total_cost = sum(c.get("cost", 0.0) for c in agent_calls)
    C_cost = min(total_cost / COST_NORM_BASE, 1.0)

    # 5. 轮次惩罚（6 轮 -> 1.0，1 轮 -> 0.167）
    C_turns = n_turns / max_turns

    # 6. 效率奖励：用较少调用得到正确答案
    B_efficiency = 1.0 if (R_outcome >= 0.8 and n_turns <= 2) else 0.0

    # 7. 最终复合奖励
    reward = R_outcome - alpha * C_cost - beta * C_turns + gamma * B_efficiency

    return {
        "reward": reward,
        "R_outcome": R_outcome,
        "C_cost": C_cost,
        "C_turns": C_turns,
        "B_efficiency": B_efficiency,
        "total_api_cost_usd": total_cost,
        "n_turns": n_turns,
        "pred_answer": pred,
    }
```

**答案归一化**（对 EM/F1 公平性至关重要）：

```python
def normalize_answer(s: str) -> str:
    """去除冠词、标点、多余空格。'The Beatles' -> 'beatles'"""
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)        # 去除冠词
    s = ''.join(ch for ch in s if ch not in string.punctuation)  # 去除标点
    s = ' '.join(s.split())                        # 折叠空格
    return s
```

**F1 计算**（token 级别的精确率-召回率）：

```python
def compute_f1(pred: str, gold: Union[str, list]) -> float:
    if isinstance(gold, list):
        return max(compute_f1(pred, g) for g in gold)  # 在所有可接受答案中取最佳匹配
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = sum(pred_tokens.count(t) for t in common) / len(pred_tokens)
    recall    = sum(gold_tokens.count(t) for t in common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)
```

---

### 6.5 Agent 注册与调度 (`agent_registry.py`)

注册表将 Agent 类型名映射到具体的 API 模型。**executor 支持 strong/weak 双档位**，由编排器在 `<call>` 标签的 `tier` 属性中指定，从而在同一类型下统一调度高/低成本模型。

**主 Worker Pool**（训练 + 评估时使用，与 Conductor 共有 Worker 做控制变量对比）：

```python
AGENT_MODEL_CONFIG = {
    "executor": {
        "strong": ("claude-sonnet-4",  3.00),   # 每 1M token 的成本（美元）
        "weak":   ("gpt-4o",           2.50),
    },
    "decomposer":  ("gemini-2.5-pro",  1.25),
    "critic":      ("claude-sonnet-4", 3.00),
    "synthesizer": ("gpt-4o",          2.50),
}
```

> **附录配置**：另有一个廉价 Worker Pool（GPT-4o-mini / Gemini Flash 替换主 Pool）用于零样本迁移实验，验证编排策略不依赖具体的 Worker 模型。仅在附录展示，不作为主实验配置。

**调度机制：**

```python
class AgentRegistry:
    def __init__(self, api_base: str, api_key: str, worker_pool: str = "cheap"):
        pool = WORKER_POOLS.get(worker_pool, AGENT_MODEL_CONFIG)
        self.agents: dict[str, BaseAgent] = {}
        for agent_type, (model_name, cost) in pool.items():
            self.agents[agent_type] = BaseAgent(
                model_name=model_name,
                cost_per_1m=cost,
                system_prompt=AGENT_SYSTEM_PROMPTS[agent_type],
                api_base=api_base,
                api_key=api_key,
            )

    def dispatch(self, agent_type: str, query: str) -> tuple[str, float]:
        """将查询调度到指定 Agent。返回 (response_text, cost_usd)。"""
        if agent_type not in self.agents:
            return f"[Unknown agent type: {agent_type}]", 0.0
        return self.agents[agent_type].call(query)

    def dispatch_batch(self, calls: list[dict]) -> list[tuple[str, float]]:
        """并行调度多个 Agent 调用（最多 10 个并发线程）。"""
        results = [None] * len(calls)
        with ThreadPoolExecutor(max_workers=min(len(calls), 10)) as executor:
            futures = {
                executor.submit(self.dispatch, c["agent_type"], c["query"]): i
                for i, c in enumerate(calls)
            }
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
        return results
```

---

### 6.6 基础 Agent (`base_agent.py`)

每个 Agent 封装一个 OpenAI 兼容的 API 调用，带有重试逻辑和成本追踪：

```python
class BaseAgent:
    def __init__(self, model_name, cost_per_1m, system_prompt, api_base, api_key, timeout=60):
        self.model_name = model_name
        self.cost_per_1m = cost_per_1m       # 每 1M token 的成本（用于奖励计算）
        self.system_prompt = system_prompt
        self._client = None                  # 懒加载的 OpenAI 客户端

    def call(self, query: str, max_retries: int = 3) -> tuple[str, float]:
        """调用 Agent API。返回 (response_text, cost_usd)。"""
        client = self._get_client()
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user",   "content": query},
                    ],
                    temperature=0.7,
                    max_tokens=1024,
                    timeout=self.timeout,
                )
                text = response.choices[0].message.content or ""
                total_tokens = response.usage.total_tokens if response.usage else 0
                cost = total_tokens * self.cost_per_1m / 1_000_000
                return text, cost
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"[Agent error: {str(e)}]", 0.0
                time.sleep(2 ** attempt)    # 指数退避：1s, 2s, 4s
        return "[Agent error: max retries exceeded]", 0.0
```

**设计决策：**
- **懒加载客户端** (`_get_client`)：避免创建可能永远不会使用的 OpenAI 客户端
- **成本追踪**：使用 `total_tokens * cost_per_1m / 1M` 计算每次调用的实际美元成本
- **指数退避**：API 失败时以 1s, 2s, 4s 间隔重试
- **优雅降级**：返回错误字符串（而非抛出异常），使编排器能够继续运行

---

### 6.7 系统提示词 (`system_prompt.py`)

系统提示词教会编排器模型其角色定位、可用 Agent、输出格式和效率规则。它包含两个完整示例（简单任务和复杂任务）来演示正确的使用模式。

提示词中的关键规则：
1. 始终以 `<think>` 开始，分析任务复杂度
2. 简单任务：直接使用 `executor`（默认 `tier="weak"`）
3. 复杂任务：考虑先使用 `decomposer`
4. 困难推理 / 复杂代码 / 深度分析：使用 `executor tier="strong"`
5. 仅在结果质量至关重要时使用 `critic`
6. 需要合并多个执行器结果时使用 `synthesizer`
7. 每次响应都以 `<answer>...</answer>` 结尾
8. **保持高效：不要进行不必要的 Agent 调用**

**规则 2 和 8 对 RL 的意义**：它们创建了一个偏向效率的软先验。在 GRPO 训练过程中，模型会发现遵循这些规则能获得更高的奖励（因为效率奖励 gamma*B_efficiency 会奖励在 <= 2 轮内得到正确答案的行为）。

---

## 7. 训练流程

### 7.1 阶段 0：SFT 热身

**问题**：预训练 LLM 不知道 `<think>/<call>/<answer>` XML 格式。如果直接开始 GRPO，模型会输出乱码并获得 -1.0 奖励（格式惩罚），什么也学不到。

**解决方案**：在约 200 条自动生成的 trace 上进行监督微调，覆盖全部 6 种 Agent 类型。

**SFT 数据如何生成** (`prepare_sft.py`)：

GPT-4o 使用 8 种精心设计的路径模式生成编排 trace：

| 模式 | 数量 | Agent 路径 |
|------|------|------------|
| `simple_direct` | 40 | think -> executor (weak) -> answer |
| `strong_direct` | 25 | think -> executor (strong) -> answer |
| `decompose_exec_synth` | 35 | think -> decomposer -> executor (weak) x2-3 -> synthesizer -> answer |
| `decompose_strong_critic` | 30 | think -> decomposer -> executor (strong) -> critic -> answer |
| `full_pipeline` | 20 | think -> decomposer -> executor (strong) -> critic -> executor (strong) -> answer |
| `code_simple` | 15 | think -> executor (weak) -> answer（代码任务） |
| `code_complex` | 15 | think -> decomposer -> executor (strong) -> critic -> answer（代码任务） |

每条生成的 trace 都会被校验：
- 必须包含 `<think>`、至少一个 `<call>` 和 `<answer>` 标签
- 不能包含 `<information>` 块（那些在运行时注入）
- 所有 agent type 必须合法
- `<answer>` 必须出现在所有 `<call>` 标签之后

**SFT 训练配置：**

```python
sft_config = SFTConfig(
    learning_rate=2e-5,
    num_train_epochs=3,
    max_length=512,
    bf16=True,
    save_strategy="no",        # 禁用训练中保存（7B 模型每次保存约 14GB）
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
)
```

---

### 7.2 阶段 1：GRPO 强化学习

GRPO（Group Relative Policy Optimization，组相对策略优化）是 PPO 的一个变体，不需要单独的 value 网络。它通过计算同一 prompt 下其他采样结果的相对优势来更新策略。

**GRPO 训练配置：**

```python
grpo_config = GRPOConfig(
    num_generations=8,               # G=8：每个 prompt 采样 8 条补全
    max_completion_length=512,
    learning_rate=1e-6,              # 比 SFT 低 20 倍
    num_train_epochs=3,
    bf16=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,   # 有效 batch = 2 * 8 * 8 = 128 rollouts / 更新
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    save_only_model=True,            # 跳过优化器状态（每个 checkpoint 节省约 28GB）
    report_to="wandb",
)
```

---

### 7.3 训练中的奖励函数

一个关键设计决策：**训练时进行真实 API 调用**。每条 GRPO 采样的补全都会被解析出 `<call>` 标签，并真正执行对应的 Agent 调用。

```python
def build_reward_fn(registry, gen_manager_ref, args):
    """为 GRPOTrainer 构建奖励函数。"""

    def reward_fn(prompts, completions, **kwargs) -> list[float]:
        gold_answers = kwargs.get("answer", [...])
        rewards = []

        for prompt, completion, gold in zip(prompts, completions, gold_answers):
            agent_calls = []
            total_cost = 0.0
            n_turns = 0

            # 解析 GRPO 采样补全中的所有 <call> 标签
            call_pattern = re.compile(
                r'<call\s+type="(\w+)"[^>]*>(.*?)</call>', re.DOTALL
            )
            for match in call_pattern.finditer(completion):
                agent_type = match.group(1)
                query = match.group(2).strip()
                _, cost = registry.dispatch(agent_type, query)  # <-- 真实 API 调用
                total_cost += cost
                agent_calls.append({"agent_type": agent_type, "cost": cost})
                n_turns += 1

            result = compute_reward(
                full_response=completion,
                gold_answer=gold,
                agent_calls=agent_calls,
                n_turns=max(n_turns, 1),
                metric=args.metric,
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
            )
            rewards.append(result["reward"])

        return rewards

    return reward_fn
```

**为什么训练时要用真实 API 调用**：成本信号必须是真实的。如果我们模拟成本，模型会学会利用模拟的漏洞，而不是学会真正地最小化 API 开支。模型必须体验真实的权衡：调用 `executor` 的 `strong` 档位（Claude Sonnet 4, $3.00/1M）能得到更好的答案，但会招致比 `weak` 档位（GPT-4o, $2.50/1M）更重的成本惩罚。

---

## 8. 数据流程

### QA 数据集 (`prepare_data.py`)

6 个来源，覆盖简单事实问答和多跳推理，全部从 HuggingFace 加载：

```python
DATASET_CONFIGS = {
    # 赛道 1：简单 QA
    "nq":            {"hf_name": "google-research-datasets/nq_open", ...},   # Natural Questions
    "triviaqa":      {"hf_name": "trivia_qa", "hf_config": "rc.nocontext"},  # 无上下文变体
    "popqa":         {"hf_name": "akariasai/PopQA", ...},                     # 按流行度分层

    # 赛道 2：多跳推理
    "hotpotqa":      {"hf_name": "hotpot_qa", "hf_config": "distractor"},    # 2 跳
    "2wikimultihop": {"hf_name": "ohjoonhee/2WikiMultihopQA", ...},           # 基于维基百科的 2 跳
    "musique":       {"hf_name": "bdsaglam/musique", ...},                    # 2-4 跳
}
```

**可复现实验的预设配置：**

| 预设 | 数据源 | 每源样本数 | 总计 |
|------|--------|-----------|------|
| `orch_r1_train` | 全部 6 个 | 1,000 | 6,000 |
| `orch_r1_test` | 全部 6 个 | 500 | 3,000 |

每条记录归一化为统一格式：
```json
{"input": "问题文本", "answer": "标准答案", "source": "nq", "difficulty": "simple"}
```

答案可以是字符串或列表（多个可接受答案，如 NQ 有别名）。

### 代码数据集 (`prepare_code.py`)

HumanEval（164 题）和 MBPP（374 训练 / 500 测试），附带测试用例用于 Pass@1 评估。

### SFT 热身数据 (`prepare_sft.py`)

自动生成约 200 条编排 trace，流程如下：
1. 从 QA/代码训练池中采样真实问题
2. 使用路径模式特定的指令调用 GPT-4o（8 种模式，如 7.1 所述）
3. 校验每条 trace 的格式正确性
4. 按比例分配覆盖全部 8 种路径模式

---

## 9. 评估系统

### 多赛道指标 (`metrics.py`)

系统根据数据来源分派到不同的评估指标：

```python
def compute_metric(pred: str, record: dict) -> dict:
    source = record.get("source", "")

    if source == "gpqa_diamond":
        # 多选题：从各种格式中提取 A/B/C/D 字母
        # 处理："The answer is D"、"\\boxed{D}"、"(D)"、独立的 "D"
        return {"accuracy": compute_gpqa_accuracy(pred, gold)}

    if source in ("humaneval", "mbpp"):
        # 代码：执行预测 + 测试用例，5 秒超时
        # Windows 使用 threading.Thread；Linux 使用 signal.SIGALRM
        return {"pass_at_1": compute_pass_at_1(pred, test_cases, entry_point, prompt)}

    if source == "livecodebench":
        # 代码：子进程 I/O 比较，10 秒超时
        return {"pass_rate": compute_livecode_pass(pred, test_cases)}

    # 默认：QA 指标（EM + F1）
    return {"em": compute_em(pred, gold), "f1": compute_f1(pred, gold)}
```

### 基线方法 (`baselines.py`)

| 基线 | 描述 | Agent 调用次数 |
|------|------|---------------|
| **Direct-Strong** | 直接发送查询到 executor (strong)，即 Claude Sonnet 4 | 1 |
| **Direct-Weak** | 直接发送查询到 executor (weak)，即 GPT-4o | 1 |
| **Fixed-Pipeline** | 无论复杂度如何，始终运行完整 4 步流水线 | 4-5 |
| **Self-Reflection** | 同一模型（GPT-4o）5 轮自我纠正 | 5 |
| **Open-Loop** | OrchestratorR1 模型但采用先规划再执行模式（无反应式反馈） | 不定 |

固定流水线始终运行：decomposer -> executor (strong) -> critic -> （如果 critic 发现问题则重试）-> synthesizer。该基线用于测试静态的"什么都做"方案能否匹配学习到的自适应编排。

---

## 10. Worker Pool 设计

我们采用**单一强 Worker Pool** 作为主实验配置，与 Conductor 在共有基准（GPQA、LiveCodeBench）上做控制变量对比——**相同 Worker，不同编排范式**（反应式 vs 开环），从而将编排范式的效果与 Worker 能力隔离。

**主 Worker Pool**（训练 + 评估）：

| Agent | 后端模型 | 价格 ($/1M tokens) |
|-------|---------|---------------------|
| executor (strong) | Claude Sonnet 4 | 3.00 |
| executor (weak) | GPT-4o | 2.50 |
| decomposer | Gemini 2.5 Pro | 1.25 |
| critic | Claude Sonnet 4 | 3.00 |
| synthesizer | GPT-4o | 2.50 |

**核心论点**：当 Worker 能力相同时，反应式编排 ≥ 开环规划——优势来自闭环错误恢复、自适应重路由和提前终止，而非 Worker 本身。

**成本控制**：成本系数 α 通过调节调用频率（特别是抑制不必要的 strong-tier 调用和重复 critic 验证）来控制总开销。reward 中的 `min(α·C_cost, 0.3)` 上限避免成本项主导梯度。

**附录配置**（廉价 Worker Pool）：作为零样本迁移实验放在附录，用 GPT-4o-mini / Gemini Flash 替换主 Pool，验证编排策略不依赖于具体的 Worker 模型。

---

## 11. 关键超参数

### 训练

| 参数 | SFT | GRPO | 描述 |
|------|-----|------|------|
| 学习率 | 2e-5 | 1e-6 | GRPO 使用 20 倍更低的 LR |
| Epochs | 3 | 3 | |
| Batch size（有效） | 16 | 128 | GRPO: 2 x 8 x 8 |
| max_seq_length | 512 | 512 | |
| LoRA r / alpha | 64 / 128 | 64 / 128 | 两阶段使用相同 LoRA 配置 |
| num_generations (G) | -- | 8 | GRPO 组大小（每 prompt 采样数） |
| LoRA 目标模块 | -- | -- | q,k,v,o_proj + gate,up,down_proj（全部注意力 + MLP） |

### 奖励

| 参数 | 默认值 | 效果 |
|------|--------|------|
| alpha（成本惩罚） | 0.3 | 越高 -> 越偏好廉价 Agent |
| beta（轮次惩罚） | 0.1 | 越高 -> 越偏好更少轮次 |
| gamma（效率奖励） | 0.15 | 在 <= 2 轮内正确回答的奖励 |
| COST_NORM_BASE | $0.01 | 成本归一化锚点 |
| PUNISH_FORMAT | -1.0 | 无效 XML 标签的硬惩罚 |

### 生成

| 参数 | 默认值 | 效果 |
|------|--------|------|
| max_turns | 6 | 每个查询的最大 Agent 调用次数 |
| max_new_tokens | 512 | 每步生成的最大 token 数 |
| max_obs_length | 800 | 每轮注入的 Agent 响应最大字符数 |
| temperature | 0.7 | 采样温度 |

### 硬件配置

| 模式 | 启动命令 | GPU | 模型 | 显存/GPU | 策略 |
|------|---------|-----|------|----------|------|
| LoRA (Windows) | `train_lora.bat` | 4×RTX 3090 | 3B | ~8GB | DDP + Gloo + LoRA |
| 全参数 (Windows) | `train_full.bat` | 4×RTX 3090 | 3B | ~16.5GB | ZeRO-2 + Gloo |
| FSDP (Linux) | `bash train.sh` | 4×RTX 3090 | 3B | ~20GB | FSDP + NCCL |
| A800 (Linux) | `bash train.sh --a800` | 2×A800 80GB | 3B | ~40GB | DDP + NCCL |
| **H200 (Linux,主力推荐)** | **`bash train.sh --h200`** | **2×H200 141GB** | **7B** | **~90GB** | **FSDP 全参 FT** |

**H200 模式关键超参**(对标 Conductor 的配置):
- `per_device_batch_size=8`,`grad_accum=2`(有效 batch = 32)
- `num_generations=16`(GRPO rollout group 翻倍,优势估计更准)
- `max_completion_length=2048`(支持更长推理轨迹)
- `max_turns=6`(reactive 叙事卖点)
- 单 seed 训练时长 **~18–24h**,3 seeds 约 **3 天**

---

## 12. 快速开始

### 安装

```bash
conda create -n orch python=3.10 -y
conda activate orch
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install flash-attn --no-build-isolation  # 可选，训练加速约 20%
```

### 下载模型

```bash
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir models/Qwen2.5-3B-Instruct
```

如果 Hugging Face 较慢，可使用 modelscope：

```bash
pip install modelscope
modelscope download --model Qwen/Qwen2.5-3B-Instruct --local_dir models/Qwen2.5-3B-Instruct
```

### 设置 API 凭据

```bash
export API_BASE="YOUR_API_BASE"
export API_KEY="YOUR_API_KEY"
```

### 准备数据

```bash
# QA 数据集（6k 训练样本）
python data_process/prepare_data.py --preset orch_r1_train --output data/train_qa.jsonl
python data_process/prepare_data.py --preset orch_r1_test  --output data/test.jsonl

# 代码数据集
python data_process/prepare_code.py --output_train data/train_code.jsonl --output_test data/test_code.jsonl

# SFT 热身数据（通过 GPT-4o 生成 200 条样本）
python data_process/prepare_sft.py \
    --train_qa data/train_qa.jsonl \
    --train_code data/train_code.jsonl \
    --output data/sft_warmup.jsonl \
    --api_base $API_BASE --api_key $API_KEY
```

### 训练

```bash
# 阶段 0:SFT 热身(教会 XML 标签格式)
# 4×3090 LoRA(~1-2h):
bash training/sft_warmup.sh --lora
# 2×H200 全参 7B(~2-3h):
bash training/sft_warmup.sh --a800      # 或手动切 H200 模式

# 阶段 1:GRPO 强化学习训练
# 2×H200 141GB(7B 全参,单 seed ~18-24h,推荐):
bash training/train.sh --h200 MODEL_PATH=checkpoints/sft_warmup_7b
# 2×A800 80GB(3B 全参):
bash training/train.sh --a800
# 4×RTX 3090(3B LoRA,Windows):
training\train_lora.bat
```

### 评估

```bash
# OrchestratorR1（廉价 Worker Pool）
python eval/eval_orchestrator.py \
    --model_path checkpoints/orch_grpo_3b_seed1/final \
    --data_path data/test.jsonl \
    --api_base $API_BASE --api_key $API_KEY \
    --worker_pool cheap \
    --output eval/results/orch_r1_cheap.json

# 基线方法
python eval/baselines.py --method direct_strong --data_path data/test.jsonl --api_base $API_BASE --api_key $API_KEY --output eval/results/direct_strong.json
python eval/baselines.py --method fixed_pipeline --data_path data/test.jsonl --api_base $API_BASE --api_key $API_KEY --output eval/results/fixed_pipeline.json
```

### 推理

```bash
python inference/infer.py \
    --model_path checkpoints/orch_grpo_3b_seed1/final \
    --api_base $API_BASE --api_key $API_KEY \
    --input "一战结束的条约签署时美国总统是谁？"
```

---

---

## 13. 后续规划(Roadmap)

> 目标会议:**AAAI 2026 / NeurIPS 2026**(投稿窗口紧,需差异化于 Conductor ICLR 2026)
> 当前进度:数据准备 + 基线评估完成,核心训练待启动

### 阶段 1:训练打通(Week 1,~3 天)

- [ ] **SFT 热身**:在 H200 上跑通 7B 全参 SFT(~3h),验证 `<call>/<answer>` 格式产出率 > 95%
- [ ] **GRPO 单 seed 试跑**:300 steps 小规模训练,确认 reward 上升、KL 不爆、agent 调用分布不塌陷
- [ ] **关键监控指标**:`reward_mean`, `format_error_rate`, `agent_distribution_entropy`, `avg_n_turns`

### 阶段 2:主结果(Week 2,~5 天)

- [ ] **GRPO × 3 seeds 完整训练**(2×H200,~3 天):产出 `orch_grpo_7b_seed{1,2,3}`
- [ ] **全测试集评估**:10 个 benchmark × 3 seeds × 2 worker pools(cheap + matched)
- [ ] **Self-Reflection 5-turn 基线**:Conductor 论文中的核心基线,需自己实现并跑全部数据集
- [ ] **ReAct 基线**:Qwen2.5-7B + tool-use,与我们方法 backbone 对齐

### 阶段 3:消融与差异化证据(Week 3,~5 天)

按 reviewer 关注度排序:

1. ★★★ **w/o reactive** — 用 [generation_openloop.py](OrchestratorR1/orchestrator_r1/orchestrator/generation_openloop.py) 直接对比开环范式,**回答"为什么不用 Conductor 方式"**
2. ★★★ **matched-worker 配置** — GPQA / LiveCodeBench 上用 Claude-Sonnet-4.6 / Gemini-2.5-Pro 做 worker,与 Conductor 正面对标
3. ★★☆ **w/o critic / w/o decomposer** — 验证功能化 agent 角色的必要性
4. ★★☆ **α 敏感性**(0, 0.1, 0.3, 0.5, 0.7) — Pareto 曲线核心数据
5. ★☆☆ **SFT-only / Fixed-Pipeline** — 证明 RL + 自适应的贡献

### 阶段 4:分析图表(Week 4,~4 天)

- [ ] **Agent 调用热力图**(6 agents × 10 datasets) — 涌现的 task-dependent 行为
- [ ] **训练动态曲线**(steps × avg_turns,按任务复杂度分层) — RL 驱动行为变化
- [ ] **Pareto 前沿**(cost vs F1,标注 Conductor 为高成本参考点) — 成本-质量可调
- [ ] **Reactive case study × 3** — 中间反馈触发策略修正的具体实例(论文叙事关键)
- [ ] **Scaling 小实验**:3B vs 7B(可选 14B) — 范式 scale 良好的证据

### 阶段 5:论文撰写(Week 5,~7 天)

- [ ] Method(2.5 页):MDP 形式化 + 6-agent pool + reactive loop + 复合奖励
- [ ] Experiments(3 页):Table 1 主结果 + Table 2 vs Conductor + Table 3 消融 + Pareto 曲线
- [ ] Analysis(1.5 页):热力图 + 训练动态 + case study + scaling
- [ ] Intro + Related Work + Abstract + Appendix

### 阶段 6:打磨与提交(Week 6,~5 天)

- [ ] 补做 reviewer 高概率追问的实验(额外 seed、长上下文 ablation)
- [ ] 图表美化、统一记号、匿名化检查、reproducibility checklist
- [ ] Supplementary materials(完整代码 + 全部 raw results + agent prompts)

### 关键风险与对策

| 风险 | 概率 | 对策 |
|------|------|------|
| GRPO 不收敛 / reward 塌陷成单 agent | 中 | 增大 SFT 数据(200→500),调高 entropy_coef,降低 lr |
| matched-worker zero-shot 迁移效果差 | 中 | 备一个 matched-worker 直接训练的 seed 作为 fallback |
| Conductor 绝对分数远超我们 | **确定** | 不打数字战,主打 reactive/跨模态/成本三个差异化轴 |
| API 预算超支 | 低 | 训练限 cheap pool,matched-worker 只用于评估 |
| H200 时间被调试占用 | 中 | 严格执行"单 seed 试跑→确认信号→放大"的流程,禁止盲目并行 |

### 算力预算(2×H200)

| 阶段 | wall clock | API 成本 |
|------|-----------|---------|
| SFT 热身 | 3h | $5 |
| GRPO × 3 seeds | 3 天 | ~$30 |
| 消融 × 6 | 2-3 天 | ~$20 |
| 全部评估(cheap + matched) | 1 天 | ~$300 |
| **总计** | **~7 天 wall clock** | **~$355** |

---

## 技术栈

- **trl** (GRPOTrainer, SFTTrainer) -- RL 和 SFT 训练框架
- **transformers** -- Qwen2.5 模型加载 + chat template
- **peft** -- LoRA 参数高效微调
- **accelerate** -- FSDP / DDP 分布式训练
- **deepspeed** -- ZeRO-2（Windows 全参数模式）
- **openai** SDK -- 所有 Agent API 调用（通过兼容的 base_url 支持非 OpenAI 模型）
- **wandb** -- 实验追踪

## 许可

本项目用于研究目的。
