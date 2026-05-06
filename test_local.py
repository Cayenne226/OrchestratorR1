"""
本地测试 agent_pool 和 parser，不需要 GPU。

Usage:
    python test_local.py --api_base YOUR_API_BASE --api_key YOUR_API_KEY
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from orchestrator_r1.orchestrator.parser import parse_output, validate_format
from orchestrator_r1.agent_pool.agent_registry import AgentRegistry


# ── 1. Parser 测试（纯本地，无需 API）─────────────────────────────────────

def test_parser():
    print("=" * 60)
    print("TEST 1: Parser")
    print("=" * 60)

    cases = [
        # 简单任务格式
        (
            '<think>简单问题直接查</think>\n'
            '<call type="executor" tier="weak">二战什么时候结束</call>',
            "should detect call: executor (tier=weak)"
        ),
        # 复杂任务格式
        (
            '<think>需要分解</think>\n'
            '<call type="decomposer">构建推荐系统</call>',
            "should detect call: decomposer"
        ),
        # 带 answer
        (
            '<think>已有信息</think>\n'
            '<answer>1945年9月2日</answer>',
            "should detect answer"
        ),
        # 无效 agent 类型
        (
            '<call type="unknown_agent">query</call>',
            "should fail validation: invalid agent type"
        ),
        # 无任何标签
        (
            '这是一段普通文本',
            "should fail validation: no tags"
        ),
        # 标签不匹配
        (
            '<call type="executor" tier="weak">query',
            "should fail validation: unclosed tag"
        ),
    ]

    all_pass = True
    for i, (text, description) in enumerate(cases):
        is_valid, reason = validate_format(text)
        parsed = parse_output(text)

        print(f"\nCase {i+1}: {description}")
        print(f"  Input:    {text[:80].replace(chr(10), ' ')}")
        print(f"  Valid:    {is_valid} ({reason})")
        if parsed.call:
            print(f"  Call:     type={parsed.call.agent_type}, query={parsed.call.query[:50]}")
        if parsed.answer:
            print(f"  Answer:   {parsed.answer[:50]}")
        if parsed.has_think:
            print(f"  HasThink: True")

    print("\nParser tests done.")
    return True


# ── 2. Agent Registry 测试（需要 API）─────────────────────────────────────

def test_agent_registry(api_base: str, api_key: str):
    print("\n" + "=" * 60)
    print("TEST 2: Agent Registry (requires API)")
    print("=" * 60)

    registry = AgentRegistry(api_base=api_base, api_key=api_key)

    test_calls = [
        ("executor",   "二战是什么时候结束的？", "weak"),
        ("executor",   "解释拓扑学中的同调群", "strong"),
        ("decomposer", "构建一个带用户认证的博客系统", None),
    ]

    for agent_type, query, tier in test_calls:
        print(f"\nAgent: {agent_type}" + (f" (tier={tier})" if tier else ""))
        print(f"Query: {query}")
        response, cost = registry.dispatch(agent_type, query, tier=tier)
        print(f"Response: {response[:200]}")
        print(f"Cost: ${cost:.6f}")

    print("\nAgent registry tests done.")


# ── 3. Reward 测试（纯本地）──────────────────────────────────────────────

def test_reward():
    print("\n" + "=" * 60)
    print("TEST 3: Reward Function")
    print("=" * 60)

    from orchestrator_r1.orchestrator.reward import compute_reward

    cases = [
        {
            "desc": "完全正确 + 1轮 → 高奖励（含效率bonus）",
            "response": '<think>简单</think><call type="executor" tier="weak">q</call><answer>1945年9月2日</answer>',
            "gold": "1945年9月2日",
            "agent_calls": [{"agent_type": "executor", "tier": "weak", "cost": 0.0001}],
            "n_turns": 1,
        },
        {
            "desc": "答案错误 → 低奖励",
            "response": '<think>.</think><answer>1944年</answer>',
            "gold": "1945年9月2日",
            "agent_calls": [],
            "n_turns": 1,
        },
        {
            "desc": "格式错误（无标签）→ 惩罚",
            "response": "二战结束于1945年",
            "gold": "1945年",
            "agent_calls": [],
            "n_turns": 1,
        },
        {
            "desc": "正确 + 多轮调用 → 中等奖励（轮数惩罚）",
            "response": '<think>复杂</think><call type="decomposer">q</call><call type="executor" tier="strong">q</call><answer>1945年9月2日</answer>',
            "gold": "1945年9月2日",
            "agent_calls": [
                {"agent_type": "decomposer",      "cost": 0.005},
                {"agent_type": "executor", "tier": "strong", "cost": 0.003},
            ],
            "n_turns": 4,
        },
    ]

    for case in cases:
        result = compute_reward(
            full_response=case["response"],
            gold_answer=case["gold"],
            agent_calls=case["agent_calls"],
            n_turns=case["n_turns"],
        )
        print(f"\n{case['desc']}")
        print(f"  reward={result['reward']:.4f}  "
              f"R_outcome={result['R_outcome']:.2f}  "
              f"C_cost={result['C_cost']:.4f}  "
              f"C_turns={result['C_turns']:.2f}  "
              f"B_eff={result['B_efficiency']:.1f}")
        if result.get("format_error"):
            print(f"  format_error: {result['format_error']}")

    print("\nReward tests done.")


# ── 4. strip_think 测试（纯本地）─────────────────────────────────────────

def test_strip_think():
    print("\n" + "=" * 60)
    print("TEST 4: strip_think")
    print("=" * 60)

    from data_process.strip_think import strip_think

    cases = [
        (
            '<think>简单</think>\n<call type="executor" tier="weak">q</call>',
            '<call type="executor" tier="weak">q</call>',
            "should remove think block"
        ),
        (
            '<think>first</think>\n<think>second</think>\n<answer>42</answer>',
            '<answer>42</answer>',
            "should remove multiple think blocks"
        ),
        (
            '<call type="decomposer">plan</call>',
            '<call type="decomposer">plan</call>',
            "no think block — unchanged"
        ),
    ]

    all_pass = True
    for text, expected, desc in cases:
        result = strip_think(text)
        ok = result == expected
        all_pass = all_pass and ok
        print(f"\n  {'[OK]' if ok else '[FAIL]'} {desc}")
        if not ok:
            print(f"    expected: {expected!r}")
            print(f"    got:      {result!r}")

    print(f"\nstrip_think tests {'PASSED' if all_pass else 'FAILED'}.")
    return all_pass


# ── 5. dispatch_with_noise 测试（纯本地）─────────────────────────────────

def test_dispatch_with_noise():
    print("\n" + "=" * 60)
    print("TEST 5: dispatch_with_noise (mock)")
    print("=" * 60)

    # Test with a mock registry — just verify noise mechanics
    import time

    class MockAgent:
        def call(self, query):
            return f"answer to {query}", 0.001

    class MockRegistry(AgentRegistry):
        def __init__(self):
            # Mock executor at weak tier — keys match _resolve_key("executor", "weak")
            self.agents = {"executor:weak": MockAgent()}
            self.api_base = ""
            self.api_key = ""

    reg = MockRegistry()

    # Test timeout
    resp, cost, meta = reg.dispatch_with_noise("executor", "test", tier="weak", timeout_prob=1.0)
    assert meta["timed_out"], "should timeout"
    assert resp == "[TIMEOUT]"
    print("  [OK] timeout simulation")

    # Test corruption
    resp, cost, meta = reg.dispatch_with_noise("executor", "test", tier="weak", corrupt_prob=1.0)
    assert meta["corrupted"], "should corrupt"
    assert resp.endswith("...")
    print("  [OK] corruption simulation")

    # Test latency injection
    t0 = time.time()
    resp, cost, meta = reg.dispatch_with_noise("executor", "test", tier="weak",
                                                latency_ms=200, noise_type="uniform")
    elapsed_ms = (time.time() - t0) * 1000
    assert meta["latency_injected_ms"] > 0, "should inject latency"
    assert elapsed_ms > 50, f"should have delayed, got {elapsed_ms:.0f}ms"
    print(f"  [OK] latency injection ({meta['latency_injected_ms']:.0f}ms injected, {elapsed_ms:.0f}ms elapsed)")

    # Test normal (no noise)
    resp, cost, meta = reg.dispatch_with_noise("executor", "hello", tier="weak")
    assert not meta["timed_out"] and not meta["corrupted"]
    assert "hello" in resp
    print("  [OK] no-noise passthrough")

    print("\ndispatch_with_noise tests PASSED.")
    return True


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument("--api_key",  type=str, default=None)
    parser.add_argument("--skip_api", action="store_true",
                        help="Skip agent API tests (only run parser + reward)")
    args = parser.parse_args()

    test_parser()
    test_reward()
    test_strip_think()
    test_dispatch_with_noise()

    if not args.skip_api:
        if not args.api_base or not args.api_key:
            print("\n[SKIP] Agent registry test: pass --api_base and --api_key to enable")
        else:
            test_agent_registry(args.api_base, args.api_key)

    print("\n[OK] All local tests complete.")
