"""End-to-end example: Custom Tool-Calling Agent with Curriculum Learning.

Demonstrates:
  1. ToolCallingEnv with custom-registered tools
  2. Curriculum scheduler (easy → medium → hard)
  3. Full training loop with stage progression
  4. Reward debugging after training

Run: python examples/tool_calling_curriculum.py
"""

from __future__ import annotations

import agent_harness as ah
from agent_harness.backends import DummyBackend, TrainingConfig
from agent_harness.debug import RewardDebugger
from agent_harness.envs.tool_call import ToolDef
from agent_harness.store.curriculum import CurriculumScheduler, Stage


# ---------------------------------------------------------------------------
# 1. Custom tools
# ---------------------------------------------------------------------------
def weather_lookup(city: str) -> str:
    """Fake weather API."""
    data = {
        "london": "London: 15°C, cloudy",
        "tokyo": "Tokyo: 22°C, sunny",
        "new york": "New York: 18°C, rainy",
        "paris": "Paris: 16°C, partly cloudy",
    }
    return data.get(city.lower(), f"{city}: data not available")


def unit_convert(value: str, from_unit: str, to_unit: str) -> str:
    """Simple unit converter."""
    v = float(value)
    conversions = {
        ("celsius", "fahrenheit"): lambda x: x * 9 / 5 + 32,
        ("km", "miles"): lambda x: x * 0.621371,
        ("kg", "lbs"): lambda x: x * 2.20462,
    }
    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        result = conversions[key](v)
        return f"{v} {from_unit} = {result:.1f} {to_unit}"
    return f"Cannot convert {from_unit} to {to_unit}"


def database_query(table: str, condition: str) -> str:
    """Fake database query."""
    fake_db = {
        "users": [
            {"name": "Alice", "age": 30, "city": "London"},
            {"name": "Bob", "age": 25, "city": "Tokyo"},
            {"name": "Charlie", "age": 35, "city": "Paris"},
        ],
    }
    rows = fake_db.get(table, [])
    # Very simple filtering
    results = []
    for row in rows:
        for key, val in row.items():
            if condition.lower() in str(val).lower():
                results.append(str(row))
                break
    return f"Found {len(results)} results: " + "; ".join(results) if results else "No results"


# ---------------------------------------------------------------------------
# 2. Tasks at different difficulty levels
# ---------------------------------------------------------------------------
TASKS = [
    # Easy: single tool call
    {"prompt": "What's the weather in London?", "answer": "15", "difficulty": "easy"},
    {"prompt": "What's the weather in Tokyo?", "answer": "22", "difficulty": "easy"},
    {"prompt": "Convert 100 km to miles.", "answer": "62.1", "difficulty": "easy"},
    {"prompt": "Convert 30 celsius to fahrenheit.", "answer": "86", "difficulty": "easy"},
    # Medium: requires two tool calls
    {"prompt": "What's the weather in London? Convert the temperature to fahrenheit.",
     "answer": "59", "difficulty": "medium"},
    {"prompt": "What's the weather in Tokyo? Convert the temperature to fahrenheit.",
     "answer": "71.6", "difficulty": "medium"},
    # Hard: requires reasoning + multiple tools
    {"prompt": "Find users in London from the database and tell me the weather there.",
     "answer": "Alice", "difficulty": "hard"},
    {"prompt": "Find users older than 30 and check the weather in Paris.",
     "answer": "Charlie", "difficulty": "hard"},
]


# ---------------------------------------------------------------------------
# 3. Agent
# ---------------------------------------------------------------------------
def tool_agent(observation: str, available_tools: list[str]) -> ah.Action:
    """Simple rule-based agent for tool-calling tasks."""
    obs_lower = observation.lower()

    # If we got a tool result back, try to answer
    if "°c" in obs_lower or "found" in obs_lower or "=" in obs_lower:
        # Extract a useful answer
        return ah.Action.finish(observation)

    # Decide which tool to call
    if "weather" in obs_lower and "weather_lookup" in available_tools:
        for city in ["london", "tokyo", "new york", "paris"]:
            if city in obs_lower:
                return ah.Action.tool("weather_lookup", {"city": city})

    if "convert" in obs_lower and "unit_convert" in available_tools:
        import re
        m = re.search(r"(\d+)\s+(\w+)\s+to\s+(\w+)", obs_lower)
        if m:
            return ah.Action.tool("unit_convert", {
                "value": m.group(1), "from_unit": m.group(2), "to_unit": m.group(3),
            })

    if ("find" in obs_lower or "database" in obs_lower) and "database_query" in available_tools:
        if "london" in obs_lower:
            return ah.Action.tool("database_query", {"table": "users", "condition": "London"})
        if "older" in obs_lower or "30" in obs_lower:
            return ah.Action.tool("database_query", {"table": "users", "condition": "35"})

    return ah.Action.finish(observation[:100])


# ---------------------------------------------------------------------------
# 4. Run
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  AgentHarness E2E Example: Tool Calling + Curriculum")
    print("=" * 60)

    # Build environment with custom tools
    tools = {
        "weather_lookup": ToolDef(
            spec=ah.ToolSpec(name="weather_lookup", description="Look up weather"),
            fn=weather_lookup,
        ),
        "unit_convert": ToolDef(
            spec=ah.ToolSpec(name="unit_convert", description="Convert units"),
            fn=unit_convert,
        ),
        "database_query": ToolDef(
            spec=ah.ToolSpec(name="database_query", description="Query database"),
            fn=database_query,
        ),
    }
    env = ah.envs.ToolCallingEnv(tools=tools, max_turns=8)

    reward = ah.RewardComposer([
        ah.rewards.contains_match(weight=0.5),
        ah.rewards.tool_call_valid(weight=0.2),
        ah.rewards.trajectory_efficiency(max_turns=8, weight=0.2),
        ah.rewards.tool_usage_rate(weight=0.1, min_tool_calls=1),
    ])

    curriculum = CurriculumScheduler(stages=[
        Stage(name="easy", difficulty="easy", epochs=2, promotion_threshold=0.6),
        Stage(name="medium", difficulty="medium", epochs=2, promotion_threshold=0.5),
        Stage(name="hard", difficulty="hard", epochs=3, promotion_threshold=0.4),
    ])

    harness = ah.Harness(
        env=env,
        reward=reward,
        credit=ah.CreditAssigner("hybrid", turn_weight=0.3, trajectory_weight=0.7),
        curriculum=curriculum,
        backend=DummyBackend(),
        config=TrainingConfig(model="demo-tool"),
    )

    # --- Collect all trajectories ---
    print("\n[Phase 1] Collecting trajectories on all tasks...")
    all_trajectories = harness.collect_batch(TASKS, tool_agent)
    for traj in all_trajectories:
        diff = traj.task.get("difficulty", "?")
        status = "✓" if traj.success else "✗"
        print(f"  {status} [{diff:<6}] {traj.task['prompt'][:45]:<47} "
              f"reward={traj.total_reward:.3f}  turns={traj.num_turns}")

    # --- Train with curriculum ---
    print(f"\n[Phase 2] Training with curriculum ({len(curriculum.stages)} stages)...")
    history = harness.train(trajectories=all_trajectories, iterations=6, batch_size=3)
    for m in history:
        stage = curriculum.current_stage
        stage_name = stage.name if stage and not curriculum.is_complete else "done"
        print(f"  Iter {m['iteration']:.0f}: loss={m['loss']:.4f}, "
              f"avg_reward={m['avg_reward']:.3f}, stage={stage_name}")

    print(f"\n  Curriculum progress: {curriculum.progress:.0%}")
    print(f"  Curriculum history: {len(curriculum.get_history())} updates")

    # --- Debug ---
    print("\n[Phase 3] Reward Debug Report")
    debugger = RewardDebugger(reward)
    report = debugger.analyze(all_trajectories)
    print(report.summary())

    # --- Final stats ---
    stats = harness.get_stats()
    print(f"\n[Summary]")
    print(f"  Total iterations: {stats['iteration']}")
    print(f"  Total trajectories collected: {stats['total_trajectories']}")
    print(f"  Store size: {stats['store_size']}")

    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
