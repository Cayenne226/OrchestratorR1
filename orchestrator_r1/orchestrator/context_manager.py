"""Context compression for multi-turn orchestration.

Inspired by KLong's progressive RL strategy of preserving early/recent context
while truncating middle context when total length exceeds the budget.

Strategy:
  - Always preserve: system prompt + original user query + most recent N turns
  - When over budget: truncate or summarize the middle <information> blocks
  - Default mode is hard truncation (fast, deterministic). Optional summarization
    via a cheap executor agent is supported but not enabled by default to keep
    the rollout loop self-contained.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class CompressionConfig:
    max_context_tokens: int = 4096
    budget_ratio: float = 0.80          # trigger compression when used > ratio * max
    keep_recent_turns: int = 2          # always keep the last N <information> blocks intact
    truncated_marker: str = "[...truncated...]"
    middle_block_max_chars: int = 200   # truncate middle information blocks to this length


def estimate_tokens(text: str, tokenizer=None) -> int:
    """Estimate the token count of a string.

    If a HuggingFace tokenizer is provided, use it for an exact count;
    otherwise fall back to a rough heuristic (~4 chars per token).
    """
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
    return max(1, len(text) // 4)


_INFO_BLOCK_RE = re.compile(
    r"(<information>.*?</information>)", re.DOTALL
)


def _split_info_blocks(context: str) -> list[tuple[str, bool]]:
    """Split context into segments, marking which ones are <information> blocks.

    Returns a list of (segment, is_info_block) tuples in order.
    """
    parts = _INFO_BLOCK_RE.split(context)
    return [(p, bool(_INFO_BLOCK_RE.fullmatch(p))) for p in parts if p]


def compress_context(
    context: str,
    config: Optional[CompressionConfig] = None,
    tokenizer=None,
    summarize_fn: Optional[Callable[[str], str]] = None,
) -> tuple[str, dict]:
    """Compress context if it exceeds the budget.

    Args:
        context: The full context string (prompt + all turns).
        config: Compression configuration. Uses defaults if None.
        tokenizer: Optional HF tokenizer for exact token counting.
        summarize_fn: Optional callable that takes an information block and
            returns a compressed summary. If None, hard truncation is used.

    Returns:
        (compressed_context, meta) where meta contains:
          - "compressed": bool — whether compression was applied
          - "tokens_before": int
          - "tokens_after": int
          - "blocks_compressed": int — count of middle blocks that were touched
    """
    cfg = config or CompressionConfig()
    tokens_before = estimate_tokens(context, tokenizer)
    budget = int(cfg.max_context_tokens * cfg.budget_ratio)

    meta = {
        "compressed": False,
        "tokens_before": tokens_before,
        "tokens_after": tokens_before,
        "blocks_compressed": 0,
    }

    if tokens_before <= budget:
        return context, meta

    segments = _split_info_blocks(context)
    info_indices = [i for i, (_, is_info) in enumerate(segments) if is_info]

    # If we have fewer info blocks than keep_recent_turns + 1, nothing to compress
    if len(info_indices) <= cfg.keep_recent_turns:
        return context, meta

    # The last `keep_recent_turns` info blocks stay intact; everything before
    # the last preserved block (in info-block index space) is "middle"
    middle_info_indices = info_indices[: -cfg.keep_recent_turns]
    blocks_compressed = 0

    for idx in middle_info_indices:
        original_block = segments[idx][0]
        if summarize_fn is not None:
            try:
                compressed_block = (
                    f"<information>{summarize_fn(original_block)}</information>"
                )
            except Exception:
                compressed_block = _truncate_info_block(
                    original_block, cfg.middle_block_max_chars, cfg.truncated_marker
                )
        else:
            compressed_block = _truncate_info_block(
                original_block, cfg.middle_block_max_chars, cfg.truncated_marker
            )
        if compressed_block != original_block:
            segments[idx] = (compressed_block, True)
            blocks_compressed += 1

    compressed_context = "".join(seg for seg, _ in segments)
    tokens_after = estimate_tokens(compressed_context, tokenizer)
    meta.update({
        "compressed": blocks_compressed > 0,
        "tokens_after": tokens_after,
        "blocks_compressed": blocks_compressed,
    })
    return compressed_context, meta


def _truncate_info_block(block: str, max_chars: int, marker: str) -> str:
    """Truncate the inner content of an <information>...</information> block."""
    inner_match = re.match(
        r"<information>(.*?)</information>", block, re.DOTALL
    )
    if not inner_match:
        return block
    inner = inner_match.group(1)
    if len(inner) <= max_chars:
        return block
    truncated = inner[:max_chars].rstrip() + " " + marker
    return f"<information>{truncated}</information>"
