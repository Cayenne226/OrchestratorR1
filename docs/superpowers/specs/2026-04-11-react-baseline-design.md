# ReAct Baseline Evaluation — Design Spec

> **Date**: 2026-04-11
> **Task**: T3.1 from MASTER_PLAN.md
> **Output**: `eval/results/react_7b.json`

## Goal

Implement a standard ReAct (Reason + Act) agent baseline using Qwen2.5-7B-Instruct as backbone, with a search tool that calls an LLM API. This is a **zero-shot, no-training** baseline for comparison against the trained OrchestratorR1 model.

## Architecture

### New File

`eval/eval_react.py` — standalone script, parallel to `eval/eval_orchestrator.py`.

### Two Inference Modes

1. **Local mode** (default): Load Qwen2.5-7B-Instruct via `transformers`, run `model.generate()` on GPU. ~16-18GB VRAM (BF16).
2. **API mode** (`--api_mode`): Call backbone model via OpenAI-compatible API. Zero GPU. Requires a served endpoint.

### Search Tool

Reuse `AgentRegistry.dispatch("executor_cheap", query)` for the search tool. This calls gpt-4o-mini in cheap pool (or frontier model in matched pool). No new agent types needed.

## ReAct Prompt

```
You are a helpful assistant that answers questions step by step.
You have access to a search tool that queries an LLM for information.

For each step, use this exact format:
Thought: <your reasoning about what to do next>
Action: search("your search query") OR answer("your final answer")
Observation: <will be filled by the system>

Rules:
- Use search() to gather information you need
- Use answer() when you have enough information to answer
- The answer should be a short factual response (name, number, date, or short phrase)
- You must call answer() within {max_turns} turns
```

## Multi-Turn Loop

```
for turn in range(max_turns):
    output = generate(context)
    parse action from output:
      - answer("text") → extract answer, break
      - search("query") → dispatch to executor_cheap, append Observation
      - invalid → append error Observation, continue
if no answer after max_turns → extract fallback from last output
```

## Code Task Handling

For code tasks (HumanEval, MBPP, LiveCodeBench), the prompt is adapted:
- No "short factual answer" instruction
- search() tool described as "query an LLM for help with code"
- answer() expects code output

## Output Format

Identical to other baselines (`eval/baselines.py`):

```json
{
  "summary": {
    "method": "react_7b",
    "n_samples": N,
    "em": float,
    "f1": float,
    "avg_cost_usd": float,
    "avg_turns": float
  },
  "results": [{
    "input": str,
    "gold": str,
    "pred": str,
    "em": float,
    "f1": float,
    "metrics": dict,
    "n_turns": int,
    "total_cost": float,
    "agent_calls": [{"agent_type": str, "query": str, "cost": float}],
    "source": str,
    "trajectory": str
  }]
}
```

## CLI Interface

```bash
# Local mode
python eval/eval_react.py \
    --model_path models/Qwen2.5-7B-Instruct \
    --data_path data/test_qa.jsonl \
    --api_base $API_BASE --api_key $API_KEY \
    --output eval/results/react_7b.json \
    --max_turns 6 --worker_pool cheap

# API mode
python eval/eval_react.py \
    --api_mode \
    --backbone_model Qwen/Qwen2.5-7B-Instruct \
    --backbone_api_base $BACKBONE_API_BASE \
    --backbone_api_key $BACKBONE_API_KEY \
    --data_path data/test_qa.jsonl \
    --api_base $API_BASE --api_key $API_KEY \
    --output eval/results/react_7b.json
```

## Fairness Guarantees vs OrchestratorR1

| Dimension | ReAct | OrchestratorR1 | Match |
|-----------|-------|----------------|-------|
| Backbone | Qwen2.5-7B-Instruct | Same | Yes |
| Max turns | 6 | 6 | Yes |
| Search tool | gpt-4o-mini (cheap pool) | Multiple agents | ReAct uses uniform cheap |
| Training | None (zero-shot) | SFT + GRPO | Intentional difference |
| Eval data | test_qa.jsonl | Same | Yes |
| Metrics | EM / F1 | Same | Yes |

## Dependencies

- `transformers` (already in project)
- `torch` (already in project)
- `openai` (already in project, used by BaseAgent)
- `eval/metrics.py` (existing)
- `orchestrator_r1/agent_pool/agent_registry.py` (existing)
