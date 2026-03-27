"""
Unified evaluation metrics for all task types.

Supports:
  - QA: EM (Exact Match) and F1 (token-level)
  - GPQA: Accuracy (multiple choice letter extraction)
  - Code (HumanEval/MBPP): Pass@1 via execution
  - LiveCodeBench: I/O matching via execution
"""

import re
import string
from typing import Union


# ── QA Metrics ─────────────────────────────────────────────────────────────────

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
    recall = sum(gold_tokens.count(t) for t in common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ── GPQA Metrics ───────────────────────────────────────────────────────────────

def extract_choice(text: str) -> str:
    """Extract the multiple-choice answer letter from model output.

    Handles formats like:
      - "The answer is D"
      - "\\boxed{D}"
      - "(D)"
      - Just "D"
    """
    # Try \boxed{X} first
    m = re.search(r'\\boxed\{([A-Da-d])\}', text)
    if m:
        return m.group(1).upper()

    # Try "answer is X" / "answer: X"
    m = re.search(r'(?:answer|choice)\s*(?:is|:)\s*\(?([A-Da-d])\)?', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Try "(X)" standalone
    m = re.search(r'\(([A-Da-d])\)', text)
    if m:
        return m.group(1).upper()

    # Last resort: find last standalone letter A-D
    matches = re.findall(r'\b([A-Da-d])\b', text)
    if matches:
        return matches[-1].upper()

    return ""


def compute_gpqa_accuracy(pred: str, gold: str) -> float:
    """Compute accuracy for GPQA multiple-choice questions."""
    pred_choice = extract_choice(pred)
    gold_choice = gold.strip().upper()
    return float(pred_choice == gold_choice)


# ── Code Metrics ───────────────────────────────────────────────────────────────

def clean_code_output(text: str) -> str:
    """Strip markdown code fences and language tags from model output."""
    text = text.strip()
    # Remove ```python ... ``` or ``` ... ```
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```python or ```)
        lines = lines[1:]
        # Remove last line if it's ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def compute_pass_at_1(pred_code: str, test_cases: str, entry_point: str = "",
                      prompt: str = "", timeout: float = 5.0) -> float:
    """Compute Pass@1 by executing code + test cases.

    For HumanEval: pred_code is the function body, test_cases is the check() function.
    For MBPP: pred_code is the full function, test_cases are assert statements.
    """
    pred_code = clean_code_output(pred_code)

    try:
        # For HumanEval: the prompt contains the function signature,
        # pred_code is the body — we need to combine them
        if prompt and "def " in prompt:
            # Check if pred already has the function def
            if not re.search(r'^\s*def\s+', pred_code, re.MULTILINE):
                # pred_code is just the body — may or may not be indented
                # If it's not indented (no leading whitespace on first line),
                # auto-indent it to fit inside the function
                first_line = pred_code.split('\n')[0] if pred_code else ''
                if first_line and not first_line[0].isspace():
                    # Auto-indent: add 4 spaces to each line
                    indented_lines = []
                    for line in pred_code.split('\n'):
                        if line.strip():
                            indented_lines.append('    ' + line)
                        else:
                            indented_lines.append('')
                    pred_code = '\n'.join(indented_lines)
                exec_code = prompt.rstrip() + "\n" + pred_code
            else:
                exec_code = pred_code
        else:
            exec_code = pred_code

        exec_code = exec_code.strip() + "\n\n" + test_cases.strip()

        # If test_cases contain a check() function, we need to call it
        if "def check(" in test_cases and "check(" not in test_cases.split("def check(")[0]:
            # Find the function name from pred_code
            func_match = re.search(r'def\s+(\w+)\s*\(', pred_code)
            if func_match:
                entry = entry_point or func_match.group(1)
                exec_code += f"\ncheck({entry})"

        # Execute with timeout
        import signal
        import sys

        # Windows doesn't support signal.alarm, use threading
        if sys.platform == "win32":
            import threading
            result = {"passed": False, "error": None}

            def run():
                try:
                    exec_globals = {"__builtins__": __builtins__}
                    exec(exec_code, exec_globals)
                    result["passed"] = True
                except Exception as e:
                    result["error"] = str(e)

            t = threading.Thread(target=run)
            t.daemon = True
            t.start()
            t.join(timeout)
            return float(result["passed"])
        else:
            def timeout_handler(signum, frame):
                raise TimeoutError()

            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout))
            try:
                exec_globals = {"__builtins__": __builtins__}
                exec(exec_code, exec_globals)
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                return 1.0
            except Exception:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                return 0.0
    except Exception:
        return 0.0


# ── LiveCodeBench Metrics ──────────────────────────────────────────────────────

def compute_livecode_pass(pred_code: str, test_cases_str: str,
                          timeout: float = 10.0) -> float:
    """Compute pass rate for LiveCodeBench I/O-based problems.

    test_cases_str format: "Input:\n...\nExpected Output:\n...\n---\n..."
    """
    if not pred_code.strip() or not test_cases_str.strip():
        return 0.0

    pred_code = clean_code_output(pred_code)

    cases = test_cases_str.split("---")
    if not cases:
        return 0.0

    passed = 0
    total = 0

    for case in cases:
        case = case.strip()
        if not case:
            continue

        input_match = re.search(r'Input:\n(.*?)(?:\nExpected Output:)', case, re.DOTALL)
        output_match = re.search(r'Expected Output:\n(.*)', case, re.DOTALL)

        if not input_match or not output_match:
            continue

        expected = output_match.group(1).strip()
        stdin_data = input_match.group(1).strip()
        total += 1

        try:
            import subprocess
            import sys
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False,
                                             encoding='utf-8') as f:
                f.write(pred_code)
                tmp_path = f.name

            try:
                result = subprocess.run(
                    [sys.executable, tmp_path],
                    input=stdin_data,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                actual = result.stdout.strip()
                if actual == expected:
                    passed += 1
            finally:
                os.unlink(tmp_path)
        except Exception:
            continue

    return float(passed / total) if total > 0 else 0.0


# ── Unified metric dispatch ────────────────────────────────────────────────────

def compute_metric(pred: str, record: dict) -> dict:
    """Compute the appropriate metric based on record source/difficulty.

    Returns dict with metric name → value.
    """
    source = record.get("source", "")
    difficulty = record.get("difficulty", "")
    gold = record.get("answer", "")

    if source == "gpqa_diamond" or difficulty == "expert_reasoning":
        acc = compute_gpqa_accuracy(pred, gold)
        return {"accuracy": acc, "em": acc, "f1": acc}

    if source in ("humaneval", "mbpp") or difficulty == "code":
        test_cases = record.get("test_cases", "")
        task_id = record.get("task_id", "")
        prompt = record.get("input", "")
        entry_point = ""
        if "HumanEval" in task_id:
            # Extract entry point from the prompt
            func_match = re.search(r'def\s+(\w+)\s*\(', prompt)
            if func_match:
                entry_point = func_match.group(1)
        p1 = compute_pass_at_1(pred, test_cases, entry_point, prompt=prompt)
        return {"pass_at_1": p1, "em": p1, "f1": p1}

    if source == "livecodebench" or difficulty.startswith("code_"):
        test_cases = record.get("test_cases", "")
        p = compute_livecode_pass(pred, test_cases)
        return {"pass_rate": p, "em": p, "f1": p}

    # Default: QA metrics
    em = compute_em(pred, gold)
    f1 = compute_f1(pred, gold)
    return {"em": em, "f1": f1}
