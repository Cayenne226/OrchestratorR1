import re
from dataclasses import dataclass
from typing import Optional


VALID_AGENT_TYPES = {"executor", "decomposer", "critic", "synthesizer"}
VALID_EXECUTOR_TIERS = {"strong", "weak"}
DEFAULT_EXECUTOR_TIER = "weak"

STOP_TOKENS = ["</call>", "</answer>"]


@dataclass
class CallTag:
    agent_type: str
    query: str
    raw: str
    tier: Optional[str] = None  # only meaningful for agent_type="executor"


@dataclass
class ParseResult:
    call: Optional[CallTag] = None
    answer: Optional[str] = None
    has_think: bool = False


def parse_output(text: str) -> ParseResult:
    """Parse orchestrator output to extract <call> or <answer> tags."""
    result = ParseResult()

    result.has_think = bool(re.search(r"<think>", text))

    # Extract <call type="X" [tier="Y"]>query</call>
    call_match = re.search(
        r'<call\s+type="(\w+)"([^>]*)>(.*?)</call>',
        text,
        re.DOTALL,
    )
    if call_match:
        agent_type = call_match.group(1).strip()
        attrs = call_match.group(2)
        query = call_match.group(3).strip()
        if agent_type in VALID_AGENT_TYPES:
            tier = None
            if agent_type == "executor":
                tier_match = re.search(r'tier="(\w+)"', attrs)
                tier = (
                    tier_match.group(1).strip()
                    if tier_match and tier_match.group(1).strip() in VALID_EXECUTOR_TIERS
                    else DEFAULT_EXECUTOR_TIER
                )
            result.call = CallTag(
                agent_type=agent_type,
                query=query,
                raw=call_match.group(0),
                tier=tier,
            )

    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_match:
        result.answer = answer_match.group(1).strip()

    return result


def extract_answer(text: str) -> Optional[str]:
    """Extract just the answer text."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def validate_format(text: str) -> tuple[bool, str]:
    """Validate output format. Returns (is_valid, reason)."""
    has_call = bool(re.search(r"<call\s+type=", text))
    has_answer = bool(re.search(r"<answer>", text))

    if not has_call and not has_answer:
        return False, "No <call> or <answer> tag found"

    open_calls = len(re.findall(r"<call\s", text))
    close_calls = len(re.findall(r"</call>", text))
    if open_calls != close_calls:
        return False, f"Mismatched <call> tags: {open_calls} open, {close_calls} close"

    open_ans = len(re.findall(r"<answer>", text))
    close_ans = len(re.findall(r"</answer>", text))
    if open_ans != close_ans:
        return False, f"Mismatched <answer> tags"

    call_types = re.findall(r'<call\s+type="(\w+)"', text)
    for t in call_types:
        if t not in VALID_AGENT_TYPES:
            return False, f"Invalid agent type: {t}"

    # Validate executor tier when present
    for match in re.finditer(r'<call\s+type="executor"([^>]*)>', text):
        attrs = match.group(1)
        tier_match = re.search(r'tier="(\w+)"', attrs)
        if tier_match and tier_match.group(1) not in VALID_EXECUTOR_TIERS:
            return False, f"Invalid executor tier: {tier_match.group(1)}"

    return True, "ok"
