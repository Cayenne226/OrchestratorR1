# OrchestratorR1: Reactive Multi-Agent Orchestration via Reinforcement Learning

> **One-sentence summary**: A small LLM (Qwen2.5-3B/7B) is trained with progressive GRPO reinforcement learning to become a "meta-controller" that reactively orchestrates 4 functionally specialized agents (executor / decomposer / critic / synthesizer) — where role definitions are human priors but invocation timing, ordering, and sub-query formulation are all emergent from the RL reward landscape.

---

## Table of Contents

- [1. Motivation & Core Idea](#1-motivation--core-idea)
- [2. Architecture Overview](#2-architecture-overview)
- [3. Project Structure](#3-project-structure)
- [4. Agent Pool Design](#4-agent-pool-design)
- [5. XML Tag Protocol](#5-xml-tag-protocol)
- [6. Core Modules Deep-Dive](#6-core-modules-deep-dive)
  - [6.1 Parser (`parser.py`)](#61-parser-parserpy)
  - [6.2 Reactive Generation Loop (`generation.py`)](#62-reactive-generation-loop-generationpy)
  - [6.3 Open-Loop Ablation (`generation_openloop.py`)](#63-open-loop-ablation-generation_openlooppy)
  - [6.4 Reward Function (`reward.py`)](#64-reward-function-rewardpy)
  - [6.5 Agent Registry & Dispatch (`agent_registry.py`)](#65-agent-registry--dispatch-agent_registrypy)
  - [6.6 Base Agent (`base_agent.py`)](#66-base-agent-base_agentpy)
  - [6.7 System Prompt (`system_prompt.py`)](#67-system-prompt-system_promptpy)
- [7. Training Pipeline](#7-training-pipeline)
  - [7.1 Stage 0: SFT Warmup](#71-stage-0-sft-warmup)
  - [7.2 Stages 1-3: Progressive GRPO Reinforcement Learning](#72-stages-1-3-progressive-grpo-reinforcement-learning)
  - [7.3 The Reward Function in Training](#73-the-reward-function-in-training)
- [8. Data Pipeline](#8-data-pipeline)
- [9. Evaluation System](#9-evaluation-system)
- [10. Worker Pool Configuration](#10-worker-pool-configuration)
- [11. Key Hyperparameters](#11-key-hyperparameters)
- [12. Quick Start](#12-quick-start)

---

## 1. Motivation & Core Idea

### The Problem

Existing multi-agent systems fall into two categories:

1. **Fixed pipelines** — always run the same sequence of agents regardless of task complexity (e.g., always: decomposer -> executor -> critic -> synthesizer). This is wasteful for simple tasks and inflexible for novel ones.

2. **Open-loop planners** (e.g., Conductor) — a strong LLM generates a complete execution plan upfront, then all agents execute in parallel. The planner never sees intermediate results and cannot adapt.

### Our Solution: Reactive Orchestration

We train a small, cheap LLM to be a **reactive meta-controller**. The key insight:

```
After every agent call, the agent's response is injected back into the model's context.
The model observes the response BEFORE deciding what to do next.
```

This creates a closed-loop feedback system:

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│  User Question                                      │
│       │                                             │
│  [Orchestrator LLM]  <──────────────────────┐       │
│       │                                     │       │
│       ├── <think> reason about complexity    │       │
│       ├── <call type="agent_X"> query        │       │
│       │        │                             │       │
│       │   [Agent X executes]                 │       │
│       │        │                             │       │
│       │   <information> response </info>  ───┘       │
│       │                                             │
│       │   (model reads response, decides next step) │
│       │                                             │
│       ├── <call type="agent_Y"> follow-up query     │
│       │        │                                    │
│       │   [Agent Y executes]                        │
│       │        │                                    │
│       │   <information> response </info>  ───┘      │
│       │                                             │
│       └── <answer> final answer </answer>           │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**Why this matters**: If Agent X returns a wrong or incomplete answer, the orchestrator can call a critic, retry with a stronger model, or rephrase the query. An open-loop planner cannot do this.

---

## 2. Architecture Overview

```
                          ┌─────────────────────────────┐
                          │   Orchestrator LLM           │
                          │   (Qwen2.5-3B, RL-trained)  │
                          └──────┬──────────────────────┘
                                 │
                    Generates XML tags: <think>, <call>, <answer>
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
              ┌─────▼─────┐  ┌──▼──┐  ┌──────▼──────┐
              │ Parser     │  │     │  │             │
              │ (parser.py)│  │ ... │  │             │
              └─────┬─────┘  └─────┘  └─────────────┘
                    │
            Extracts agent_type + query
                    │
         ┌──────────▼──────────┐
         │   Agent Registry     │
         │  (agent_registry.py) │
         └──────────┬──────────┘
                    │
    Dispatches to the correct agent via OpenAI-compatible API
                    │
    ┌──────────┬───────┼───────┬────────────┐
    │          │       │       │            │
 executor   decomp  critic  synth
 (tier:     oser            esizer
  strong|weak)
```

**The training loop** (GRPO):

```
For each training prompt:
  1. Sample G=8 rollouts from the model
  2. For each rollout, parse all <call> tags
  3. Execute the API calls (real API calls during training!)
  4. Compute reward R(τ) = R_outcome - α·C_cost - β·C_turns + γ·B_efficiency
  5. Use GRPO to update the model (reward advantage within the group)
```

---

## 3. Project Structure

```
OrchestratorR1/
├── orchestrator_r1/              # Core Python package
│   ├── agent_pool/
│   │   ├── base_agent.py         # OpenAI-compatible API wrapper (retry + cost tracking)
│   │   └── agent_registry.py     # 4-agent dispatch table, strong worker pool
│   ├── orchestrator/
│   │   ├── parser.py             # XML tag parser + format validator
│   │   ├── reward.py             # Composite reward R(τ) computation
│   │   ├── generation.py         # Reactive multi-turn generation loop (CORE)
│   │   ├── generation_openloop.py  # Open-loop ablation (plan-then-execute)
│   │   └── context_manager.py    # Context compression (budget check + middle-turn summarization)
│   └── prompts/
│       └── system_prompt.py      # Full system prompt with agent descriptions
│
├── training/
│   ├── train.py                  # GRPO training entry point (trl.GRPOTrainer)
│   ├── sft_warmup.py             # SFT stage: teaches <call>/<answer> format
│   ├── train_lora.bat            # Windows: 4xRTX 3090, LoRA, ~8GB/GPU
│   ├── train_full.bat            # Windows: 4xRTX 3090, ZeRO-2, ~16.5GB/GPU
│   ├── train.sh / train_flex.sh  # Linux: FSDP+NCCL
│   └── accelerate_*.yaml        # Distributed training configs
│
├── data_process/
│   ├── prepare_data.py           # QA dataset loading (6 sources from HuggingFace)
│   ├── prepare_code.py           # Code dataset loading (HumanEval + MBPP)
│   └── prepare_sft.py            # Auto-generate SFT warmup traces via GPT-4o
│
├── eval/
│   ├── eval_orchestrator.py      # Main evaluation harness
│   ├── baselines.py              # Direct-Strong, Direct-Cheap, Fixed-Pipeline
│   ├── run_self_reflection.py    # Self-Reflection baseline (5-round)
│   ├── run_ablation_openloop.py  # Open-loop ablation
│   └── metrics.py                # EM/F1/GPQA-accuracy/Pass@1/LiveCode
│
├── inference/infer.py            # CLI single-query inference
├── analysis/                     # Paper figure generation scripts
└── test_local.py                 # CPU-only unit tests
```

---

## 4. Agent Pool Design

The orchestrator has 4 specialized agents, each backed by a strong LLM via OpenAI-compatible API. Role definitions are human priors, but invocation timing, ordering, sub-query formulation, and verification decisions are all emergent from RL.

| Agent | Role | When to Use |
|-------|------|-------------|
| **executor** | Task execution (supports `tier="strong"` and `tier="weak"`) | Factual queries, reasoning, code — tier selected by the orchestrator based on perceived difficulty |
| **decomposer** | Breaks complex tasks into subtasks | When the task has multiple independent steps |
| **critic** | Quality verification | When result quality is uncertain or critical |
| **synthesizer** | Merges multiple partial results | After executing multiple subtasks |

The orchestrator handles query rewriting in its own `<think>` reasoning (no separate refiner agent). The executor supports a `tier` attribute: `<call type="executor" tier="strong">` routes to a frontier model, while `<call type="executor" tier="weak">` (or no tier) routes to a cost-efficient model.

Each agent has a dedicated system prompt that defines its behavior (defined in `agent_registry.py`). For example:

```python
AGENT_SYSTEM_PROMPTS = {
    "executor": (
        "You are a versatile task execution agent. Answer the given question or complete the "
        "given task accurately and concisely. Provide only the answer or result, nothing else."
    ),
    "decomposer": (
        "You are a task decomposition expert. Break the given complex question or task into "
        "smaller, independent subtasks that can be solved individually. Output a numbered list "
        "of subtasks, nothing else."
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

## 5. XML Tag Protocol

The orchestrator model communicates with the system via structured XML tags:

| Tag | Direction | Description |
|-----|-----------|-------------|
| `<think>...</think>` | Model -> System | Internal chain-of-thought reasoning |
| `<call type="X">query</call>` | Model -> System | Request to invoke agent X |
| `<information>...</information>` | System -> Model | Agent's response (injected by the system) |
| `<answer>...</answer>` | Model -> System | Final answer to the user |

**Example interaction:**

```xml
<think>This is a multi-hop question about history. I should decompose it first.</think>

<call type="decomposer">Who was the president when the treaty that ended WWI was signed?</call>

<information>Subtask 1: What treaty ended WWI?
Subtask 2: When was this treaty signed?
Subtask 3: Who was the US president at that time?</information>

<think>The decomposer broke it into 3 parts. This is a straightforward factual chain, so a weak executor should suffice.</think>

<call type="executor" tier="weak">What treaty ended World War I and when was it signed?</call>

<information>The Treaty of Versailles was signed on June 28, 1919.</information>

<call type="executor" tier="weak">Who was the US president in June 1919?</call>

<information>Woodrow Wilson was the US president in June 1919.</information>

<answer>Woodrow Wilson</answer>
```

---

## 6. Core Modules Deep-Dive

### 6.1 Parser (`parser.py`)

The parser extracts structured information from the model's raw text output.

**Key data structures:**

```python
VALID_AGENT_TYPES = {
    "executor", "decomposer", "critic", "synthesizer"
}

STOP_TOKENS = ["</call>", "</answer>"]

@dataclass
class CallTag:
    agent_type: str   # one of VALID_AGENT_TYPES
    query: str        # the text to send to the agent
    raw: str          # full matched XML string

@dataclass
class ParseResult:
    call: Optional[CallTag] = None      # extracted <call> tag, if any
    answer: Optional[str] = None        # extracted <answer> tag, if any
    has_think: bool = False             # whether <think> was present
```

**Core parsing logic** — uses regex to extract `<call type="X">query</call>` and `<answer>`:

```python
def parse_output(text: str) -> ParseResult:
    result = ParseResult()
    result.has_think = bool(re.search(r"<think>", text))

    # Extract <call type="X">query</call>
    call_match = re.search(
        r'<call\s+type="(\w+)"[^>]*>(.*?)</call>',
        text, re.DOTALL,
    )
    if call_match:
        agent_type = call_match.group(1).strip()
        query = call_match.group(2).strip()
        if agent_type in VALID_AGENT_TYPES:
            result.call = CallTag(agent_type=agent_type, query=query, raw=call_match.group(0))
        # If invalid agent type, treat as format error (no call returned)

    # Extract <answer>...</answer>
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_match:
        result.answer = answer_match.group(1).strip()

    return result
```

**Format validation** — used by the reward function to detect invalid outputs and assign -1.0 penalty:

```python
def validate_format(text: str) -> tuple[bool, str]:
    """Returns (is_valid, reason)."""
    has_call = bool(re.search(r"<call\s+type=", text))
    has_answer = bool(re.search(r"<answer>", text))

    if not has_call and not has_answer:
        return False, "No <call> or <answer> tag found"

    # Check matching open/close tags
    open_calls = len(re.findall(r"<call\s", text))
    close_calls = len(re.findall(r"</call>", text))
    if open_calls != close_calls:
        return False, f"Mismatched <call> tags: {open_calls} open, {close_calls} close"

    # Validate all agent types in call tags
    call_types = re.findall(r'<call\s+type="(\w+)"', text)
    for t in call_types:
        if t not in VALID_AGENT_TYPES:
            return False, f"Invalid agent type: {t}"

    return True, "ok"
```

---

### 6.2 Reactive Generation Loop (`generation.py`)

This is the **core of the entire project**. The `OrchestratorGenerationManager` runs a multi-turn loop where the model generates text, calls agents, observes their responses, and decides what to do next.

**Configuration:**

```python
@dataclass
class GenerationConfig:
    max_turns: int = 6             # maximum number of agent calls per query
    max_new_tokens: int = 512      # tokens per generation step
    temperature: float = 0.7
    top_p: float = 0.9
    max_obs_length: int = 800      # max chars of agent response injected per turn

@dataclass
class RolloutResult:
    full_text: str                 # complete context (prompt + all turns)
    answer: Optional[str]          # extracted final answer
    agent_calls: List[dict]        # list of {agent_type, query, cost, turn}
    n_turns: int = 0               # how many agent calls were made
    total_cost: float = 0.0        # total API cost in USD
    token_ids: List[int]           # token IDs for RL training
```

**Prompt construction** — uses HuggingFace's chat template:

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

**The reactive rollout loop** — the single most important method in the project:

```python
def rollout(self, user_input: str) -> RolloutResult:
    """Run one complete orchestration rollout for a single input."""
    context = self._build_prompt(user_input)    # system prompt + user message via chat template
    agent_calls = []
    total_cost = 0.0

    for turn in range(self.config.max_turns):   # max 6 turns
        # Step 1: Generate next segment (until </call> or </answer>)
        generated = self._generate_step(context)
        context += generated

        # Step 2: Parse what the model produced
        parsed = parse_output(generated)

        # Step 3a: If model wants to call an agent -> dispatch, inject response, CONTINUE LOOP
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
            # Truncate long responses to prevent context overflow
            if len(response) > self.config.max_obs_length:
                response = response[:self.config.max_obs_length] + "..."
            # KEY STEP: inject agent response back into context for the model to read
            context += f"\n<information>{response}</information>\n"
            # Context compression: if context exceeds budget, preserve system prompt +
            # original query + recent turns, summarize/truncate middle information blocks
            context = self._compress_if_needed(context)
            continue   # <-- loop back: model will see the response and decide next action

        # Step 3b: If model produced a final answer -> return
        if parsed.answer is not None:
            return RolloutResult(
                full_text=context,
                answer=parsed.answer,
                agent_calls=agent_calls,
                n_turns=turn + 1,
                total_cost=total_cost,
            )

    # Max turns reached without <answer> -> extract whatever answer exists
    answer = extract_answer(context)
    return RolloutResult(
        full_text=context, answer=answer,
        agent_calls=agent_calls,
        n_turns=self.config.max_turns,
        total_cost=total_cost,
    )
```

**Why this design matters:**

- After `context += f"\n<information>{response}</information>\n"`, the model's next `_generate_step()` call sees the agent's response in its context window
- This allows the model to **react** to unexpected results: call a critic if the answer looks wrong, rephrase the query in its own reasoning, or simply output `<answer>` if the result is satisfactory
- The `continue` statement means the loop goes back to the top, generating a new segment that can contain another `<call>` or an `<answer>`
- The model learns through RL which reaction patterns yield higher rewards

**Generation step** — generates tokens until a stop token or max length:

```python
@torch.no_grad()
def _generate_step(self, context: str) -> str:
    """Generate until </call> or </answer> stop token."""
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

    # Decode only the NEW tokens (not the prompt)
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return self.tokenizer.decode(new_ids, skip_special_tokens=True)
```

---

### 6.3 Open-Loop Ablation (`generation_openloop.py`)

This is the **most important ablation experiment**. It directly tests the "reactive vs. open-loop" hypothesis by removing the feedback loop.

**How it differs from the reactive loop:**

```
Reactive (our method):         Open-Loop (ablation):
  think -> call -> info ->       think -> call_1 -> call_2 -> call_3 ->
  think -> call -> info ->       [execute ALL in parallel] ->
  think -> answer                 info_1 + info_2 + info_3 ->
                                  answer
```

**The 4-phase process:**

```python
def rollout(self, user_input: str) -> RolloutResult:
    context = self._build_prompt(user_input)

    # Phase 1: Generate ALL calls at once (2x token budget, NO intermediate feedback)
    plan_text = self._generate_step(context, max_new_tokens=self.config.max_new_tokens * 2)
    context += plan_text

    # Check if model already gave an answer (trivial case)
    answer = extract_answer(plan_text)
    if answer is not None:
        return RolloutResult(full_text=context, answer=answer, agent_calls=[], n_turns=1, ...)

    # Phase 2: Extract ALL <call> tags from the generated plan
    planned_calls = self._extract_all_calls(plan_text)  # regex findall

    # Phase 3: Execute ALL calls in parallel via dispatch_batch (ThreadPoolExecutor)
    results_list = self.registry.dispatch_batch(planned_calls)

    # Inject ALL results at once as a single block
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

    # Phase 4: Generate final <answer> from all results simultaneously
    answer_text = self._generate_step(context)
    context += answer_text
    answer = extract_answer(answer_text) or extract_answer(context)

    return RolloutResult(
        full_text=context, answer=answer,
        agent_calls=agent_calls,
        n_turns=1,  # open-loop = 1 "turn" of planning
        total_cost=total_cost,
    )
```

**What this tests**: If the reactive loop is simply a more expensive way to do the same thing, the open-loop ablation would achieve similar accuracy. If our hypothesis is correct, the open-loop version should perform significantly worse because it cannot adapt to intermediate results.

---

### 6.4 Reward Function (`reward.py`)

The reward drives the RL training. It balances four objectives:

```
R(τ) = R_outcome - α * C_cost - β * C_turns + γ * B_efficiency
```

| Component | Formula | Description |
|-----------|---------|-------------|
| R_outcome | F1 or EM vs. gold answer | Answer quality (0 to 1) |
| C_cost | min(total_api_cost / $0.01, 1.0) | API cost, normalized to [0,1] |
| C_turns | n_turns / max_turns | Turn count penalty |
| B_efficiency | 1.0 if (R_outcome >= 0.8 AND n_turns <= 2) else 0.0 | Efficiency bonus |

**Default weights:** alpha=0.3, beta=0.1, gamma=0.15

**Hard format penalty:** If the model's output fails `validate_format()`, the entire reward is **-1.0**. This strongly discourages malformed outputs.

**Full implementation:**

```python
PUNISH_FORMAT = -1.0      # Hard penalty for invalid tag format
COST_NORM_BASE = 0.01     # $0.01 per query = max expected cost

def compute_reward(
    full_response: str,
    gold_answer: Union[str, list],
    agent_calls: List[dict],
    n_turns: int,
    metric: str = "f1",
    alpha: float = 0.3,     # API cost penalty weight
    beta: float = 0.1,      # Turn count penalty weight
    gamma: float = 0.15,    # Efficiency bonus weight
    max_turns: int = 6,
) -> dict:
    # 1. Format check -- immediate -1.0 if invalid tags
    is_valid, reason = validate_format(full_response)
    if not is_valid:
        return {"reward": PUNISH_FORMAT, "R_outcome": 0.0, "format_error": reason, ...}

    # 2. Extract predicted answer
    pred = extract_answer(full_response) or ""

    # 3. Answer quality (F1 or EM)
    if metric == "em":
        R_outcome = compute_em(pred, gold_answer)
    else:
        R_outcome = compute_f1(pred, gold_answer)

    # 4. API cost penalty (normalized: $0.01 maps to 1.0, capped at 1.0)
    total_cost = sum(c.get("cost", 0.0) for c in agent_calls)
    C_cost = min(total_cost / COST_NORM_BASE, 1.0)

    # 5. Turn penalty (6 turns -> 1.0, 1 turn -> 0.167)
    C_turns = n_turns / max_turns

    # 6. Efficiency bonus: correct answer with few calls
    B_efficiency = 1.0 if (R_outcome >= 0.8 and n_turns <= 2) else 0.0

    # 7. Final composite reward
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

**Answer normalization** (critical for EM/F1 fairness):

```python
def normalize_answer(s: str) -> str:
    """Strip articles, punctuation, extra whitespace. 'The Beatles' -> 'beatles'"""
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)        # remove articles
    s = ''.join(ch for ch in s if ch not in string.punctuation)  # remove punctuation
    s = ' '.join(s.split())                        # collapse whitespace
    return s
```

**F1 computation** (token-level precision-recall):

```python
def compute_f1(pred: str, gold: Union[str, list]) -> float:
    if isinstance(gold, list):
        return max(compute_f1(pred, g) for g in gold)  # best match among all acceptable answers
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

### 6.5 Agent Registry & Dispatch (`agent_registry.py`)

The registry maps agent type names to concrete API models. A single **strong worker pool** is used for both training and evaluation, enabling controlled comparison with Conductor on the same worker quality.

**Strong worker pool** (used for both training and evaluation):

```python
AGENT_MODEL_CONFIG = {
    "executor":    {"strong": ("claude-sonnet-4-6", 3.00),   # cost per 1M tokens (USD)
                    "weak":   ("gpt-4o-mini",       0.15)},
    "decomposer":  ("gemini-2.5-pro",    1.25),
    "critic":      ("gpt-4o",            2.50),
    "synthesizer": ("gpt-4o",            2.50),
}
```

The executor agent supports a `tier` attribute parsed from the `<call>` tag. When `tier="strong"`, it routes to a frontier model (Claude Sonnet 4); when `tier="weak"` or unspecified, it routes to a cost-efficient model (GPT-4o-mini). Cheap worker pool experiments are deferred to appendix/future work.

**Dispatch mechanism:**

```python
class AgentRegistry:
    def __init__(self, api_base: str, api_key: str):
        self.agents: dict[str, BaseAgent] = {}
        for agent_type, config in AGENT_MODEL_CONFIG.items():
            if agent_type == "executor":
                # Executor has tier-based dispatch (strong/weak)
                for tier, (model_name, cost) in config.items():
                    key = f"executor:{tier}"
                    self.agents[key] = BaseAgent(
                        model_name=model_name,
                        cost_per_1m=cost,
                        system_prompt=AGENT_SYSTEM_PROMPTS["executor"],
                        api_base=api_base,
                        api_key=api_key,
                    )
            else:
                model_name, cost = config
                self.agents[agent_type] = BaseAgent(
                    model_name=model_name,
                    cost_per_1m=cost,
                    system_prompt=AGENT_SYSTEM_PROMPTS[agent_type],
                    api_base=api_base,
                    api_key=api_key,
                )

    def dispatch(self, agent_type: str, query: str, tier: str = "weak") -> tuple[str, float]:
        """Dispatch a query to the specified agent. Returns (response_text, cost_usd)."""
        if agent_type == "executor":
            key = f"executor:{tier}"
        else:
            key = agent_type
        if key not in self.agents:
            return f"[Unknown agent type: {key}]", 0.0
        return self.agents[key].call(query)

    def dispatch_batch(self, calls: list[dict]) -> list[tuple[str, float]]:
        """Dispatch multiple agent calls in parallel (up to 10 concurrent threads)."""
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

### 6.6 Base Agent (`base_agent.py`)

Each agent wraps an OpenAI-compatible API call with retry logic and cost tracking:

```python
class BaseAgent:
    def __init__(self, model_name, cost_per_1m, system_prompt, api_base, api_key, timeout=60):
        self.model_name = model_name
        self.cost_per_1m = cost_per_1m       # cost per 1M tokens (for reward computation)
        self.system_prompt = system_prompt
        self._client = None                  # lazy-initialized OpenAI client

    def call(self, query: str, max_retries: int = 3) -> tuple[str, float]:
        """Call the agent API. Returns (response_text, cost_usd)."""
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
                time.sleep(2 ** attempt)    # exponential backoff: 1s, 2s, 4s
        return "[Agent error: max retries exceeded]", 0.0
```

**Design decisions:**
- **Lazy client init** (`_get_client`): avoids creating OpenAI clients that may never be used
- **Cost tracking**: computes real USD cost per call using `total_tokens * cost_per_1m / 1M`
- **Exponential backoff**: 1s, 2s, 4s retries on API failures
- **Graceful degradation**: returns error string (not exception) so the orchestrator can continue

---

### 6.7 System Prompt (`system_prompt.py`)

The system prompt teaches the orchestrator model its role, available agents, output format, and efficiency rules. It includes two worked examples (simple task and complex task) to demonstrate proper usage patterns.

Key rules in the prompt:
1. Always start with `<think>` to reason about complexity
2. For simple tasks: go directly to `executor` (weak tier) or `executor` (strong tier)
3. For complex tasks: consider using `decomposer` first
4. Use `<think>` reasoning to rewrite queries with implicit references (no separate refiner)
5. Use `critic` only when result quality is critical
6. Use `synthesizer` when combining results from multiple executor calls
7. End every response with `<answer>...</answer>`
8. **Be efficient: do not call agents unnecessarily**

**Why rules 2 and 8 matter for RL**: They create a soft prior toward efficiency. During GRPO training, the model discovers that following these rules leads to higher rewards (because the efficiency bonus gamma*B_efficiency rewards correct answers in <= 2 turns).

---

## 7. Training Pipeline

### 7.1 Stage 0: SFT Warmup

**Problem**: A pretrained LLM doesn't know the `<think>/<call>/<answer>` XML format. If we start GRPO directly, the model outputs garbage and gets -1.0 rewards (format penalty), learning nothing useful.

**Solution**: Supervised Fine-Tuning on ~200 auto-generated traces that cover all 4 agent types (executor with both tiers, decomposer, critic, synthesizer).

**How SFT data is generated** (`prepare_sft.py`):

GPT-4o generates orchestration traces using 6 carefully designed path patterns:

| Pattern | Count | Agent Path |
|---------|-------|------------|
| `simple_direct` | 40 | think -> executor(weak) -> answer |
| `strong_direct` | 30 | think -> executor(strong) -> answer |
| `decompose_exec_synth` | 40 | think -> decomposer -> executor(weak) x2-3 -> synthesizer -> answer |
| `decompose_strong_critic` | 35 | think -> decomposer -> executor(strong) -> critic -> answer |
| `full_pipeline` | 25 | think -> decomposer -> executor(strong) -> critic -> executor(strong) -> answer |
| `code_complex` | 30 | think -> decomposer -> executor(strong) -> critic -> answer (code tasks) |

Each generated trace is validated:
- Must have `<think>`, at least one `<call>`, and `<answer>` tags
- Must NOT contain `<information>` blocks (those are injected at runtime)
- All agent types must be valid
- `<answer>` must come after all `<call>` tags

**SFT training configuration:**

```python
sft_config = SFTConfig(
    learning_rate=2e-5,
    num_train_epochs=3,
    max_length=512,
    bf16=True,
    save_strategy="no",        # disable mid-training saves (~14GB per save for 7B model)
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
)
```

---

### 7.2 Stages 1-3: Progressive GRPO Reinforcement Learning

GRPO (Group Relative Policy Optimization) is a variant of PPO that doesn't need a separate value network. Instead, it computes advantages relative to other completions sampled for the same prompt.

Training uses a **3-stage progressive curriculum** that gradually increases task complexity and turn budget:

| Stage | max_turns | Data Mix | Purpose |
|-------|-----------|----------|---------|
| Stage 1 | 2 | Simple QA (NQ, TriviaQA, PopQA) | Learn basic call/answer patterns with 1-2 agent calls |
| Stage 2 | 4 | Multi-hop (HotpotQA, 2Wiki, MuSiQue) | Learn decomposition, critic, and multi-step reasoning |
| Stage 3 | 6 | Full mix (all sources + code) | Learn full orchestration with all agent types and tiers |

Each stage initializes from the previous stage's checkpoint. This progressive approach prevents the model from being overwhelmed by complex multi-turn trajectories before it has mastered basic agent invocation.

**GRPO training configuration** (shared across stages, with per-stage overrides for max_turns and data):

```python
grpo_config = GRPOConfig(
    num_generations=8,               # G=8: sample 8 completions per prompt
    max_completion_length=512,
    learning_rate=1e-6,              # 20x lower than SFT
    num_train_epochs=3,
    bf16=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,   # effective batch = 2 * 8 * 8 = 128 rollouts per update
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    save_only_model=True,            # skip optimizer states (~28GB saved per checkpoint)
    report_to="wandb",
)
```

---

### 7.3 The Reward Function in Training

A critical design choice: **real API calls are made during training**. Each GRPO-sampled completion is parsed for `<call>` tags, and those calls are actually executed against the agent pool.

```python
def build_reward_fn(registry, gen_manager_ref, args):
    """Build the reward function for GRPOTrainer."""

    def reward_fn(prompts, completions, **kwargs) -> list[float]:
        gold_answers = kwargs.get("answer", [...])
        rewards = []

        for prompt, completion, gold in zip(prompts, completions, gold_answers):
            agent_calls = []
            total_cost = 0.0
            n_turns = 0

            # Parse ALL <call> tags in the GRPO-sampled completion
            call_pattern = re.compile(
                r'<call\s+type="(\w+)"[^>]*>(.*?)</call>', re.DOTALL
            )
            for match in call_pattern.finditer(completion):
                agent_type = match.group(1)
                query = match.group(2).strip()
                _, cost = registry.dispatch(agent_type, query)  # <-- REAL API CALL
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

**Why real API calls in training**: The cost signal must be real. If we simulated costs, the model would learn to exploit the simulation rather than learning to minimize actual API spending. The model must experience the actual trade-off: calling `executor` with `tier="strong"` (Claude Sonnet, $3.00/1M) gives better answers but incurs a heavier cost penalty than `executor` with `tier="weak"` (GPT-4o-mini, $0.15/1M).

---

## 8. Data Pipeline

### QA Datasets (`prepare_data.py`)

6 sources covering simple factual QA and multi-hop reasoning, all loaded from HuggingFace:

```python
DATASET_CONFIGS = {
    # Track 1: Simple QA
    "nq":            {"hf_name": "google-research-datasets/nq_open", ...},   # Natural Questions
    "triviaqa":      {"hf_name": "trivia_qa", "hf_config": "rc.nocontext"},  # no-context variant
    "popqa":         {"hf_name": "akariasai/PopQA", ...},                     # popularity-stratified

    # Track 2: Multi-hop Reasoning
    "hotpotqa":      {"hf_name": "hotpot_qa", "hf_config": "distractor"},    # 2-hop
    "2wikimultihop": {"hf_name": "ohjoonhee/2WikiMultihopQA", ...},           # 2-hop from Wikipedia
    "musique":       {"hf_name": "bdsaglam/musique", ...},                    # 2-4 hop
}
```

**Presets for reproducible experiments:**

| Preset | Sources | Samples/Source | Total |
|--------|---------|----------------|-------|
| `orch_r1_train` | All 6 | 1,000 | 6,000 |
| `orch_r1_test` | All 6 | 500 | 3,000 |

Each record is normalized to a unified format:
```json
{"input": "question text", "answer": "gold answer", "source": "nq", "difficulty": "simple"}
```

Answers can be strings or lists (multiple acceptable answers, e.g., NQ has aliases).

### Code Datasets (`prepare_code.py`)

HumanEval (164 problems) and MBPP (374 train / 500 test), stored with test cases for Pass@1 evaluation.

### SFT Warmup Data (`prepare_sft.py`)

Auto-generates ~200 orchestration traces by:
1. Sampling real questions from the QA/code training pools
2. Calling GPT-4o with path-pattern-specific instructions (6 patterns, as described in 7.1)
3. Validating every trace for format correctness
4. Distributing coverage proportionally across all 6 path patterns

---

## 9. Evaluation System

### Multi-Track Metrics (`metrics.py`)

The system dispatches to different metrics based on data source:

```python
def compute_metric(pred: str, record: dict) -> dict:
    source = record.get("source", "")

    if source == "gpqa_diamond":
        # Multiple-choice: extract A/B/C/D letter from various formats
        # Handles: "The answer is D", "\\boxed{D}", "(D)", standalone "D"
        return {"accuracy": compute_gpqa_accuracy(pred, gold)}

    if source in ("humaneval", "mbpp"):
        # Code: execute prediction + test cases with 5s timeout
        # Windows uses threading.Thread; Linux uses signal.SIGALRM
        return {"pass_at_1": compute_pass_at_1(pred, test_cases, entry_point, prompt)}

    if source == "livecodebench":
        # Code: subprocess I/O comparison with 10s timeout
        return {"pass_rate": compute_livecode_pass(pred, test_cases)}

    # Default: QA metrics (EM + F1)
    return {"em": compute_em(pred, gold), "f1": compute_f1(pred, gold)}
```

### Baselines (`baselines.py`)

| Baseline | Description | Agent Calls |
|----------|-------------|-------------|
| **Direct-Strong** | Send query directly to executor (strong tier, e.g., Claude Sonnet) | 1 |
| **Direct-Cheap** | Send query directly to executor (weak tier, e.g., GPT-4o-mini) | 1 |
| **Fixed-Pipeline** | Always run full 4-step pipeline regardless of complexity | 3-4 |
| **Self-Reflection** | 5 rounds of self-critique with same model (GPT-4o) | 5 |
| **Open-Loop** | OrchestratorR1 model but plan-then-execute (no reactive feedback) | varies |

The fixed pipeline always runs: decomposer -> executor(strong) -> critic -> (optional retry if critic found issues) -> synthesizer. This baseline tests whether a static "always do everything" approach can match learned, adaptive orchestration.

---

## 10. Worker Pool Configuration

A single **strong worker pool** is used for both training and evaluation. This enables a controlled comparison with Conductor on the same worker quality, isolating the **reactive vs. open-loop** variable rather than competing on absolute scores.

**Strong worker pool:**
- **executor (strong tier):** Claude Sonnet 4 ($3.00/1M tokens)
- **executor (weak tier):** GPT-4o-mini ($0.15/1M tokens)
- **decomposer:** Gemini 2.5 Pro ($1.25/1M tokens)
- **critic:** GPT-4o ($2.50/1M tokens)
- **synthesizer:** GPT-4o ($2.50/1M tokens)

The orchestrator learns through RL when to use the strong vs. weak executor tier. The cost penalty in the reward function (alpha * C_cost) incentivizes the model to prefer the weak tier for straightforward queries and reserve the strong tier for hard reasoning or critical sub-tasks.

**Cheap worker pool experiments** (all agents backed by GPT-4o-mini or Gemini Flash) are deferred to appendix/future work. The key argument: when worker quality is held constant between OrchestratorR1 and Conductor, reactive orchestration should match or exceed open-loop planning because it can adapt to intermediate results.

---

## 11. Key Hyperparameters

### Training

| Parameter | SFT | GRPO | Description |
|-----------|-----|------|-------------|
| Learning rate | 2e-5 | 1e-6 | GRPO uses 20x lower LR |
| Epochs | 3 | 3 | |
| Batch size (effective) | 16 | 128 | GRPO: 2 x 8 x 8 |
| max_seq_length | 512 | 512 | |
| LoRA r / alpha | 64 / 128 | 64 / 128 | Same LoRA config for both stages |
| num_generations (G) | -- | 8 | GRPO group size (rollouts per prompt) |
| LoRA target modules | -- | -- | q,k,v,o_proj + gate,up,down_proj (all attention + MLP) |

### Reward

| Parameter | Default | Effect |
|-----------|---------|--------|
| alpha (cost penalty) | 0.3 | Higher -> prefer cheaper agents |
| beta (turn penalty) | 0.1 | Higher -> prefer fewer turns |
| gamma (efficiency bonus) | 0.15 | Bonus for correct answer in <= 2 turns |
| COST_NORM_BASE | $0.01 | Cost normalization anchor |
| PUNISH_FORMAT | -1.0 | Hard penalty for invalid XML tags |

### Generation

| Parameter | Default | Effect |
|-----------|---------|--------|
| max_turns | 6 | Max agent calls per query |
| max_new_tokens | 512 | Tokens per generation step |
| max_obs_length | 800 | Max chars of agent response injected per turn |
| temperature | 0.7 | Sampling temperature |

### Hardware Configurations

| Mode | GPU | Model | VRAM/GPU | Strategy |
|------|-----|-------|----------|----------|
| LoRA (Windows) | 4xRTX 3090 | 3B | ~8GB | DDP + Gloo |
| Full-param (Windows) | 4xRTX 3090 | 3B | ~16.5GB | ZeRO-2 + Gloo |
| FSDP (Linux) | 4xRTX 3090 | 3B | ~20GB | FSDP + NCCL |
| LoRA (A100) | 4xA100 | 7B | ~20GB | FSDP + LoRA |
| Full-param (A100) | 4xA100 | 7B | ~50GB | FSDP full |

---

## 12. Quick Start

### Install

```bash
conda create -n orch python=3.10 -y
conda activate orch
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install flash-attn --no-build-isolation  # optional, ~20% faster training
```

### Download Model

```bash
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir models/Qwen2.5-3B-Instruct
```

### Set API Credentials

```bash
export API_BASE="YOUR_API_BASE"
export API_KEY="YOUR_API_KEY"
```

### Prepare Data

```bash
# QA datasets (6k training samples)
python data_process/prepare_data.py --preset orch_r1_train --output data/train_qa.jsonl
python data_process/prepare_data.py --preset orch_r1_test  --output data/test.jsonl

# Code datasets
python data_process/prepare_code.py --output_train data/train_code.jsonl --output_test data/test_code.jsonl

# SFT warmup data (200 examples via GPT-4o)
python data_process/prepare_sft.py \
    --train_qa data/train_qa.jsonl \
    --train_code data/train_code.jsonl \
    --output data/sft_warmup.jsonl \
    --api_base $API_BASE --api_key $API_KEY
```

### Train

```bash
# Stage 0: SFT warmup (teaches XML tag format, ~1-2h on 4x3090)
accelerate launch --config_file training/accelerate_fsdp_4gpu.yaml \
    training/sft_warmup.py \
    --model_path models/Qwen2.5-3B-Instruct \
    --data_path data/sft_warmup.jsonl \
    --output_dir checkpoints/sft_warmup_3b \
    --num_epochs 5 --use_lora

# Stage 1: GRPO (simple QA, max_turns=2)
bash training/train_flex.sh --gpu 3090 --lora \
    MODEL_PATH=checkpoints/sft_warmup_3b \
    DATA_PATH=data/train_simple.jsonl \
    OUTPUT_DIR=checkpoints/orch_grpo_3b_stage1 \
    MAX_TURNS=2

# Stage 2: GRPO (multi-hop, max_turns=4)
bash training/train_flex.sh --gpu 3090 --lora \
    MODEL_PATH=checkpoints/orch_grpo_3b_stage1 \
    DATA_PATH=data/train_multihop.jsonl \
    OUTPUT_DIR=checkpoints/orch_grpo_3b_stage2 \
    MAX_TURNS=4

# Stage 3: GRPO (full mix, max_turns=6)
bash training/train_flex.sh --gpu 3090 --lora \
    MODEL_PATH=checkpoints/orch_grpo_3b_stage2 \
    DATA_PATH=data/train_mixed.jsonl \
    OUTPUT_DIR=checkpoints/orch_grpo_3b_stage3 \
    MAX_TURNS=6
```

### Evaluate

```bash
# OrchestratorR1
python eval/eval_orchestrator.py \
    --model_path checkpoints/orch_grpo_3b_stage3/final \
    --data_path data/test.jsonl \
    --api_base $API_BASE --api_key $API_KEY \
    --output eval/results/orch_r1.json

# Baselines
python eval/baselines.py --method direct_strong --data_path data/test.jsonl --api_base $API_BASE --api_key $API_KEY --output eval/results/direct_strong.json
python eval/baselines.py --method fixed_pipeline --data_path data/test.jsonl --api_base $API_BASE --api_key $API_KEY --output eval/results/fixed_pipeline.json
```

### Inference

```bash
python inference/infer.py \
    --model_path checkpoints/orch_grpo_3b_stage3/final \
    --api_base $API_BASE --api_key $API_KEY \
    --input "Who was the president when the treaty that ended WWI was signed?"
```

---

## Tech Stack

- **trl** (GRPOTrainer, SFTTrainer) -- RL and SFT training framework
- **transformers** -- Qwen2.5 model loading + chat template
- **peft** -- LoRA parameter-efficient fine-tuning
- **accelerate** -- FSDP / DDP distributed training
- **deepspeed** -- ZeRO-2 (Windows full-param mode)
- **openai** SDK -- all agent API calls (via compatible base_url for non-OpenAI models)
- **wandb** -- experiment tracking

## License

This project is for research purposes.
