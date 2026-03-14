import re
import string
from typing import Union, List
from .parser import extract_answer, validate_format

# Format violation penalty
PUNISH_FORMAT = -1.0

# Normalization baseline: $0.01 per query = max expected cost
COST_NORM_BASE = 0.01


def normalize_answer(s: str) -> str:
    """Normalize answer string for EM/F1 comparison."""
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    s = ' '.join(s.split())
    return s


def compute_em(pred: str, gold: Union[str, list]) -> float:
    pred_norm = normalize_answer(pred)
    if isinstance(gold, list):
        return float(any(pred_norm == normalize_answer(g) for g in gold))
    return float(pred_norm == normalize_answer(gold))


def compute_f1(pred: str, gold: Union[str, list]) -> float:
    if isinstance(gold, list):
        return max(compute_f1(pred, g) for g in gold)
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


def compute_reward(
    full_response: str,
    gold_answer: Union[str, list],
    agent_calls: List[dict],   # list of {"agent_type": str, "cost": float}
    n_turns: int,
    metric: str = "f1",
    alpha: float = 0.3,        # API cost penalty weight
    beta: float = 0.1,         # Turn count penalty weight
    gamma: float = 0.15,       # Efficiency bonus weight
    max_turns: int = 6,
) -> dict:
    """
    Compute the unified reward for one orchestrator rollout.

    Returns dict with all component scores for logging.
    """
    # 1. Format check
    is_valid, reason = validate_format(full_response)
    if not is_valid:
        return {
            "reward": PUNISH_FORMAT,
            "R_outcome": 0.0,
            "C_cost": 0.0,
            "C_turns": 0.0,
            "B_efficiency": 0.0,
            "format_error": reason,
        }

    # 2. Extract predicted answer
    pred = extract_answer(full_response) or ""

    # 3. Answer quality
    if metric == "em":
        R_outcome = compute_em(pred, gold_answer)
    else:
        R_outcome = compute_f1(pred, gold_answer)

    # 4. API cost (normalized to [0, 1])
    total_cost = sum(c.get("cost", 0.0) for c in agent_calls)
    C_cost = min(total_cost / COST_NORM_BASE, 1.0)

    # 5. Turn penalty (normalized)
    C_turns = n_turns / max_turns

    # 6. Efficiency bonus: good answer with few agent calls
    B_efficiency = 1.0 if (R_outcome >= 0.8 and n_turns <= 2) else 0.0

    # 7. Final reward
    reward = (
        R_outcome
        - alpha * C_cost
        - beta  * C_turns
        + gamma * B_efficiency
    )

    return {
        "reward": reward,
        "R_outcome": R_outcome,
        "C_cost": C_cost,
        "C_turns": C_turns,
        "B_efficiency": B_efficiency,
        "total_api_cost_usd": total_cost,
        "n_turns": n_turns,
        "pred_answer": pred,
        "format_error": None,
    }
