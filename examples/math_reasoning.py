"""End-to-end example: Math Reasoning with Tool-Assisted Agent.

This example demonstrates the full AgentHarness pipeline:
  1. Define a MathReasoningEnv with calculator + python tools
  2. Compose a multi-signal reward function
  3. Build a simple rule-based agent
  4. Collect trajectories via the Harness orchestrator
  5. Run offline training with DummyBackend
  6. Debug rewards with RewardDebugger
  7. Evaluate final agent performance

Run: python examples/math_reasoning.py
"""

from __future__ import annotations

import agent_harness as ah
from agent_harness.backends import DummyBackend, TrainingConfig
from agent_harness.debug import RewardDebugger
from agent_harness.store import TrajectoryStore


# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------
TASKS = [
    {"prompt": "What is 12 * 15?", "answer": "180", "difficulty": "easy"},
    {"prompt": "What is 99 + 101?", "answer": "200", "difficulty": "easy"},
    {"prompt": "What is 256 / 8?", "answer": "32", "difficulty": "easy"},
    {"prompt": "What is 17 * 23?", "answer": "391", "difficulty": "medium"},
    {"prompt": "What is 144 / 12?", "answer": "12", "difficulty": "medium"},
    {"prompt": "What is 2**10?", "answer": "1024", "difficulty": "medium"},
    {"prompt": "What is 37 * 43 - 100?", "answer": "1491", "difficulty": "hard"},
    {"prompt": "What is (99**2 - 1) / 98?", "answer": "100", "difficulty": "hard"},
]


# ---------------------------------------------------------------------------
# 2. Agent — a simple rule-based agent that uses the calculator
# ---------------------------------------------------------------------------
def math_agent(observation: str, available_tools: list[str]) -> ah.Action:
    """Simple agent: extracts a math expression and calls calculator,
    then submits the result."""
    # If we already got a numeric result from the calculator, submit it
    obs = observation.strip()
    try:
        # Try parsing as a number — means calculator already returned
        float(obs)
        return ah.Action.finish(obs)
    except ValueError:
        pass

    # If this looks like a calculator response (e.g. "42.0"), finish
    if obs.replace(".", "").replace("-", "").isdigit():
        return ah.Action.finish(obs)

    # Extract the math expression from the prompt
    expr = _extract_expression(obs)
    if expr and "calculator" in available_tools:
        return ah.Action.tool("calculator", {"expression": expr})

    # Fallback: just say we don't know
    return ah.Action.finish("unknown")


def _extract_expression(text: str) -> str | None:
    """Crude extraction: grab everything after 'What is'."""
    import re
    m = re.search(r"What is (.+)\?", text, re.IGNORECASE)
    if m:
        expr = m.group(1).strip()
        # Clean up for eval: keep math chars
        expr = expr.replace("^", "**")
        return expr
    return None


# ---------------------------------------------------------------------------
# 3. Reward composition
# ---------------------------------------------------------------------------
reward = ah.RewardComposer([
    ah.rewards.exact_match(key="answer", weight=0.6, extract_number=True),
    ah.rewards.tool_call_valid(weight=0.15),
    ah.rewards.trajectory_efficiency(max_turns=6, weight=0.15),
    ah.rewards.length_penalty(max_tokens=200, weight=0.1),
])


# ---------------------------------------------------------------------------
# 4. Environment
# ---------------------------------------------------------------------------
env = ah.envs.MathReasoningEnv(
    dataset=TASKS,
    tools=["calculator", "python_exec"],
    max_turns=6,
)


# ---------------------------------------------------------------------------
# 5. Harness orchestrator
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  AgentHarness E2E Example: Math Reasoning")
    print("=" * 60)

    harness = ah.Harness(
        env=env,
        reward=reward,
        credit=ah.CreditAssigner("hybrid", turn_weight=0.2, trajectory_weight=0.8),
        backend=DummyBackend(),
        config=TrainingConfig(model="demo", algorithm="grpo"),
    )

    # --- Phase 1: Collect trajectories ---
    print("\n[Phase 1] Collecting trajectories...")
    trajectories = harness.collect_batch(TASKS, math_agent)
    for traj in trajectories:
        answer = traj.get_final_answer() or "?"
        gt = traj.task.get("answer", "?")
        status = "✓" if traj.success else "✗"
        print(f"  {status} {traj.task['prompt']:<35} "
              f"agent={answer:<10} gt={gt:<10} reward={traj.total_reward:.3f}")

    # --- Phase 2: Train offline ---
    print("\n[Phase 2] Training (5 iterations, offline)...")
    history = harness.train(trajectories=trajectories, iterations=5, batch_size=4)
    for m in history:
        print(f"  Iter {m['iteration']:.0f}: loss={m['loss']:.4f}, "
              f"avg_reward={m['avg_reward']:.3f}")

    # --- Phase 3: Debug rewards ---
    print("\n[Phase 3] Reward Debug Report")
    debugger = RewardDebugger(reward)
    report = debugger.analyze(trajectories, ground_truth=None)
    # We need per-task ground truth, so compute manually
    all_trajs_with_gt: list[tuple] = []
    for traj in trajectories:
        gt = traj.task.get("answer")
        score = reward(traj, ground_truth=gt)
        all_trajs_with_gt.append((traj, gt, score))

    print(f"  Trajectories: {report.total_trajectories}")
    print(f"  Composite mean: {report.composite_stats.mean:.3f}")
    print(f"  Composite std:  {report.composite_stats.std:.3f}")

    if report.hacking_alerts:
        print("\n  Hacking Alerts:")
        for alert in report.hacking_alerts:
            print(f"    [{alert.risk_level.upper()}] {alert.component}: {alert.reason}")

    # --- Phase 4: Evaluate ---
    print("\n[Phase 4] Evaluation")
    eval_metrics = harness.evaluate(TASKS, math_agent)
    print(f"  Mean Reward:  {eval_metrics['mean_reward']:.3f}")
    print(f"  Success Rate: {eval_metrics['success_rate']:.1%}")
    print(f"  Mean Turns:   {eval_metrics['mean_turns']:.1f}")

    # --- Phase 5: Save trajectories ---
    store = TrajectoryStore("/tmp/math_trajectories")
    store.add_batch(trajectories)
    path = store.save()
    print(f"\n[Phase 5] Saved {len(store)} trajectories to {path}")

    stats = store.statistics()
    print(f"  Reward: mean={stats['reward_mean']:.3f}, "
          f"min={stats['reward_min']:.3f}, max={stats['reward_max']:.3f}")

    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
