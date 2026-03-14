import re
from dataclasses import dataclass
from typing import Optional


VALID_AGENT_TYPES = {
    "refiner", "decomposer", "executor_cheap", "executor_strong", "critic", "synthesizer"
}

STOP_TOKENS = ["</call>", "</answer>"]


@dataclass
class CallTag:
    agent_type: str
    query: str
    raw: str  # full matched string


@dataclass
class ParseResult:
    call: Optional[CallTag] = None
    answer: Optional[str] = None
    has_think: bool = False


def parse_output(text: str) -> ParseResult:
    """Parse orchestrator output to extract <call> or <answer> tags."""
    result = ParseResult()

    # Check for <think> block
    result.has_think = bool(re.search(r"<think>", text))

    # Extract <call type="X">query</call>
    call_match = re.search(
        r'<call\s+type="(\w+)"[^>]*>(.*?)</call>',
        text,
        re.DOTALL,
    )
    if call_match:
        agent_type = call_match.group(1).strip()
        query = call_match.group(2).strip()
        if agent_type in VALID_AGENT_TYPES:
            result.call = CallTag(
                agent_type=agent_type,
                query=query,
                raw=call_match.group(0),
            )
        # If invalid agent type, treat as format error (no call returned)

    # Extract <answer>...</answer>
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
    # Must have at least one <call> or <answer>
    has_call = bool(re.search(r"<call\s+type=", text))
    has_answer = bool(re.search(r"<answer>", text))

    if not has_call and not has_answer:
        return False, "No <call> or <answer> tag found"

    # Check matching tags
    open_calls = len(re.findall(r"<call\s", text))
    close_calls = len(re.findall(r"</call>", text))
    if open_calls != close_calls:
        return False, f"Mismatched <call> tags: {open_calls} open, {close_calls} close"

    open_ans = len(re.findall(r"<answer>", text))
    close_ans = len(re.findall(r"</answer>", text))
    if open_ans != close_ans:
        return False, f"Mismatched <answer> tags"

    # Check for valid agent types in call tags
    call_types = re.findall(r'<call\s+type="(\w+)"', text)
    for t in call_types:
        if t not in VALID_AGENT_TYPES:
            return False, f"Invalid agent type: {t}"

    return True, "ok"
