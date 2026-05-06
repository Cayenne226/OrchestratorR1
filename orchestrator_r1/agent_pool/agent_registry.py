from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import time
from .base_agent import BaseAgent

# Agent type → system prompt
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

# Strong worker pool — used for both training and evaluation.
# Aligned with Conductor's frontier worker pool for controlled comparison
# (same workers, different orchestration paradigm: reactive vs open-loop).
# executor supports two tiers: "strong" (high-quality, expensive) and "weak" (fast, cheap).
AGENT_MODEL_CONFIG = {
    "executor": {
        "strong": ("claude-sonnet-4",   3.00),
        "weak":   ("gpt-4o",            2.50),
    },
    "decomposer":  ("gemini-2.5-pro",  1.25),
    "critic":      ("claude-sonnet-4", 3.00),
    "synthesizer": ("gpt-4o",          2.50),
}


class AgentRegistry:
    def __init__(self, api_base: str, api_key: str):
        self.api_base = api_base
        self.api_key = api_key
        self.agents: dict[str, BaseAgent] = {}

        for agent_type, cfg in AGENT_MODEL_CONFIG.items():
            if agent_type == "executor":
                # executor has two tiers — instantiate both as separate BaseAgents
                for tier, (model_name, cost) in cfg.items():
                    key = f"executor:{tier}"
                    self.agents[key] = BaseAgent(
                        model_name=model_name,
                        cost_per_1m=cost,
                        system_prompt=AGENT_SYSTEM_PROMPTS["executor"],
                        api_base=api_base,
                        api_key=api_key,
                    )
            else:
                model_name, cost = cfg
                self.agents[agent_type] = BaseAgent(
                    model_name=model_name,
                    cost_per_1m=cost,
                    system_prompt=AGENT_SYSTEM_PROMPTS[agent_type],
                    api_base=api_base,
                    api_key=api_key,
                )

    def _resolve_key(self, agent_type: str, tier: str | None = None) -> str:
        if agent_type == "executor":
            tier = tier or "weak"
            return f"executor:{tier}"
        return agent_type

    def dispatch(
        self,
        agent_type: str,
        query: str,
        tier: str | None = None,
    ) -> tuple[str, float]:
        """Dispatch a query to the specified agent. Returns (response, cost).

        For agent_type="executor", tier="strong"|"weak" selects the model variant.
        Defaults to "weak" if tier is None or unrecognized.
        """
        key = self._resolve_key(agent_type, tier)
        if key not in self.agents:
            return f"[Unknown agent: {agent_type} (tier={tier})]", 0.0
        return self.agents[key].call(query)

    def dispatch_batch(self, calls: list[dict]) -> list[tuple[str, float]]:
        """Dispatch multiple agent calls in parallel.
        Each call is a dict with keys: agent_type, query, [tier].
        Returns list of (response, cost) in same order.
        """
        results = [None] * len(calls)
        with ThreadPoolExecutor(max_workers=min(len(calls), 10)) as executor:
            futures = {
                executor.submit(
                    self.dispatch,
                    c["agent_type"],
                    c["query"],
                    c.get("tier"),
                ): i
                for i, c in enumerate(calls)
            }
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
        return results

    def dispatch_with_noise(
        self,
        agent_type: str,
        query: str,
        tier: str | None = None,
        noise_type: str = "gaussian",
        latency_ms: float = 0.0,
        timeout_prob: float = 0.0,
        corrupt_prob: float = 0.0,
    ) -> tuple[str, float, dict]:
        """Dispatch with injected noise for robustness ablation.

        Args:
            agent_type: Agent to dispatch to.
            query: The query string.
            tier: For agent_type="executor", "strong" or "weak".
            noise_type: Latency noise distribution — "gaussian", "uniform", or "exponential".
            latency_ms: Mean additional latency in milliseconds.
            timeout_prob: Probability of simulating a timeout (returns empty response).
            corrupt_prob: Probability of truncating the response (simulates degraded output).

        Returns:
            (response, cost, noise_meta) where noise_meta contains details of injected noise.
        """
        noise_meta = {"noise_type": noise_type, "latency_injected_ms": 0.0,
                       "timed_out": False, "corrupted": False}

        if timeout_prob > 0 and random.random() < timeout_prob:
            noise_meta["timed_out"] = True
            return "[TIMEOUT]", 0.0, noise_meta

        if latency_ms > 0:
            if noise_type == "gaussian":
                delay = max(0, random.gauss(latency_ms, latency_ms * 0.3))
            elif noise_type == "exponential":
                delay = random.expovariate(1.0 / latency_ms)
            else:
                delay = random.uniform(0, latency_ms * 2)
            noise_meta["latency_injected_ms"] = round(delay, 1)
            time.sleep(delay / 1000.0)

        key = self._resolve_key(agent_type, tier)
        if key not in self.agents:
            return f"[Unknown agent: {agent_type} (tier={tier})]", 0.0, noise_meta
        response, cost = self.agents[key].call(query)

        if corrupt_prob > 0 and random.random() < corrupt_prob:
            cut = max(1, len(response) // 3)
            response = response[:cut] + "..."
            noise_meta["corrupted"] = True

        return response, cost, noise_meta
