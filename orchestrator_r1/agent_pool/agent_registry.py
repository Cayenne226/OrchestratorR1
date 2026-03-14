from concurrent.futures import ThreadPoolExecutor, as_completed
from .base_agent import BaseAgent

# Agent type → system prompt
AGENT_SYSTEM_PROMPTS = {
    "refiner": (
        "You are a requirements analyst. Rewrite the given vague or ambiguous task "
        "into a clear, complete, and specific instruction. Preserve the original intent "
        "but make it precise enough for direct execution. Output only the rewritten task, nothing else."
    ),
    "decomposer": (
        "You are a task planning expert. Break down the given complex task into a numbered list "
        "of independent, executable subtasks. Each subtask should be self-contained and actionable. "
        "Output only the numbered list, nothing else."
    ),
    "executor_cheap": (
        "You are an efficient task executor. Complete the given task accurately and concisely. "
        "Focus on correctness and clarity. Be direct and avoid unnecessary explanation."
    ),
    "executor_strong": (
        "You are an expert with deep knowledge across many domains. Complete the given task "
        "with high quality, thoroughness, and accuracy. Provide detailed, correct, and well-structured output."
    ),
    "critic": (
        "You are a strict quality reviewer. Evaluate the given result for correctness, completeness, "
        "and quality. Identify specific issues or missing parts. Output a brief assessment and "
        "a score from 1-10, then list concrete improvements needed."
    ),
    "synthesizer": (
        "You are an integration expert. Combine the given partial results into a single coherent, "
        "complete, and well-structured final answer. Eliminate redundancy and ensure consistency. "
        "Output only the final combined result."
    ),
}

# Agent type → (model_name, cost_per_1m_tokens)
AGENT_MODEL_CONFIG = {
    "refiner":        ("gpt-4o-mini",         0.15),
    "decomposer":     ("gpt-4o",              2.50),
    "executor_cheap": ("gpt-4o-mini",         0.15),
    "executor_strong":("claude-sonnet-4-6",   3.00),
    "critic":         ("gemini-2.5-flash",    0.15),
    "synthesizer":    ("gpt-4o-mini",         0.15),
}


class AgentRegistry:
    def __init__(self, api_base: str, api_key: str):
        self.agents: dict[str, BaseAgent] = {}
        for agent_type, (model_name, cost) in AGENT_MODEL_CONFIG.items():
            self.agents[agent_type] = BaseAgent(
                model_name=model_name,
                cost_per_1m=cost,
                system_prompt=AGENT_SYSTEM_PROMPTS[agent_type],
                api_base=api_base,
                api_key=api_key,
            )

    def dispatch(self, agent_type: str, query: str) -> tuple[str, float]:
        """Dispatch a query to the specified agent. Returns (response, cost)."""
        if agent_type not in self.agents:
            return f"[Unknown agent type: {agent_type}]", 0.0
        return self.agents[agent_type].call(query)

    def dispatch_batch(self, calls: list[dict]) -> list[tuple[str, float]]:
        """Dispatch multiple agent calls in parallel.
        Each call is a dict with keys: agent_type, query.
        Returns list of (response, cost) in same order.
        """
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
