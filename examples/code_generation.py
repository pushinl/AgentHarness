"""End-to-end example: Code Generation with Test Verification.

Demonstrates:
  1. CodeExecutionEnv with run_code / run_tests tools
  2. A multi-attempt agent that iterates on failing code
  3. Reward combining test-pass + code quality signals
  4. A/B reward comparison with the debugger

Run: python examples/code_generation.py
"""

from __future__ import annotations

import re

import agent_harness as ah
from agent_harness.backends import DummyBackend, TrainingConfig
from agent_harness.debug import RewardDebugger


# ---------------------------------------------------------------------------
# 1. Dataset — coding tasks with test suites
# ---------------------------------------------------------------------------
TASKS = [
    {
        "prompt": "Write a Python function `add(a, b)` that returns the sum of two numbers.",
        "test_code": "assert add(2, 3) == 5\nassert add(-1, 1) == 0\nassert add(0, 0) == 0",
        "answer": "def add(a, b): return a + b",
    },
    {
        "prompt": "Write a Python function `factorial(n)` that returns n! for non-negative integers.",
        "test_code": "assert factorial(0) == 1\nassert factorial(1) == 1\nassert factorial(5) == 120",
        "answer": "def factorial(n):\n    if n <= 1: return 1\n    return n * factorial(n - 1)",
    },
    {
        "prompt": "Write a Python function `is_palindrome(s)` that checks if a string is a palindrome.",
        "test_code": "assert is_palindrome('racecar') == True\nassert is_palindrome('hello') == False\nassert is_palindrome('') == True",
        "answer": "def is_palindrome(s): return s == s[::-1]",
    },
    {
        "prompt": "Write a Python function `fibonacci(n)` that returns the n-th Fibonacci number (0-indexed).",
        "test_code": "assert fibonacci(0) == 0\nassert fibonacci(1) == 1\nassert fibonacci(10) == 55",
        "answer": "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n): a, b = b, a + b\n    return a",
    },
    {
        "prompt": "Write a Python function `flatten(lst)` that flattens a nested list.",
        "test_code": "assert flatten([1, [2, 3], [4, [5]]]) == [1, 2, 3, 4, 5]\nassert flatten([]) == []\nassert flatten([1, 2, 3]) == [1, 2, 3]",
        "answer": "def flatten(lst):\n    result = []\n    for item in lst:\n        if isinstance(item, list): result.extend(flatten(item))\n        else: result.append(item)\n    return result",
    },
]

# Correct solutions for the agent to "generate"
SOLUTIONS = {
    "add": "def add(a, b): return a + b",
    "factorial": "def factorial(n):\n    if n <= 1: return 1\n    return n * factorial(n - 1)",
    "is_palindrome": "def is_palindrome(s): return s == s[::-1]",
    "fibonacci": "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n): a, b = b, a + b\n    return a",
    "flatten": "def flatten(lst):\n    result = []\n    for item in lst:\n        if isinstance(item, list): result.extend(flatten(item))\n        else: result.append(item)\n    return result",
}


# ---------------------------------------------------------------------------
# 2. Agent — simulates a coding agent that writes, tests, and iterates
# ---------------------------------------------------------------------------
class CodeAgent:
    """Stateful agent that tries to write code, tests it, then submits."""

    def __init__(self):
        self._attempt = 0
        self._current_fn = ""
        self._code = ""

    def reset(self):
        self._attempt = 0
        self._current_fn = ""
        self._code = ""

    def __call__(self, observation: str, available_tools: list[str]) -> ah.Action:
        # First turn: figure out the function name and write code
        if self._attempt == 0:
            fn_match = re.search(r"`(\w+)\(", observation)
            self._current_fn = fn_match.group(1) if fn_match else "unknown"
            self._code = SOLUTIONS.get(self._current_fn, "# I don't know")
            self._attempt += 1

            # Try running tests first
            if "run_tests" in available_tools:
                return ah.Action.tool("run_tests", {"code": self._code})
            return ah.Action.finish(self._code)

        # After testing: if tests passed, submit; otherwise just submit anyway
        if "passed" in observation.lower() or self._attempt > 2:
            return ah.Action.finish(self._code)

        self._attempt += 1
        return ah.Action.finish(self._code)


# ---------------------------------------------------------------------------
# 3. Reward composition
# ---------------------------------------------------------------------------
reward_strict = ah.RewardComposer([
    ah.rewards.code_passes_tests(weight=0.7),
    ah.rewards.trajectory_efficiency(max_turns=5, weight=0.2),
    ah.rewards.length_penalty(max_tokens=500, weight=0.1),
])

reward_lenient = ah.RewardComposer([
    ah.rewards.fuzzy_match(weight=0.5),
    ah.rewards.trajectory_efficiency(max_turns=5, weight=0.3),
    ah.rewards.length_penalty(max_tokens=500, weight=0.2),
])


# ---------------------------------------------------------------------------
# 4. Run
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  AgentHarness E2E Example: Code Generation")
    print("=" * 60)

    env = ah.envs.CodeExecutionEnv(dataset=TASKS, timeout=10, max_turns=6)
    agent = CodeAgent()

    harness = ah.Harness(
        env=env,
        reward=reward_strict,
        credit=ah.CreditAssigner("outcome_only"),
        backend=DummyBackend(),
        config=TrainingConfig(model="demo-code"),
    )

    # --- Collect ---
    print("\n[Phase 1] Collecting trajectories...")
    trajectories = []
    for task in TASKS:
        agent.reset()
        traj = harness.collect_trajectory(task, agent)
        trajectories.append(traj)
        status = "✓" if traj.success else "✗"
        print(f"  {status} {task['prompt'][:50]:<52} "
              f"reward={traj.total_reward:.3f}  turns={traj.num_turns}")

    # --- Train ---
    print("\n[Phase 2] Training (3 iterations)...")
    history = harness.train(trajectories=trajectories, iterations=3, batch_size=3)
    for m in history:
        print(f"  Iter {m['iteration']:.0f}: loss={m['loss']:.4f}")

    # --- A/B comparison ---
    print("\n[Phase 3] A/B Reward Comparison: strict vs lenient")
    debugger = RewardDebugger(reward_strict)
    comparison = debugger.compare(
        reward_strict, reward_lenient, trajectories,
        ground_truth=None,  # test-based doesn't need GT
    )
    print(comparison.summary())

    # --- Evaluate ---
    print("[Phase 4] Evaluation")
    agent.reset()
    eval_metrics = harness.evaluate(TASKS[:3], agent)
    print(f"  Success Rate: {eval_metrics['success_rate']:.1%}")
    print(f"  Mean Reward:  {eval_metrics['mean_reward']:.3f}")

    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
