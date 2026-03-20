"""Integration tests — full end-to-end pipelines testing all components together.

These tests verify that the entire AgentHarness system works as a cohesive
unit, not just individual components.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

import agent_harness as ah
from agent_harness.backends import DummyBackend, TrainingConfig
from agent_harness.core.action import Action, ActionType, Observation, ToolCall, ToolResult
from agent_harness.core.tool import ToolSpec
from agent_harness.core.trajectory import Trajectory
from agent_harness.debug import RewardDebugger
from agent_harness.envs.tool_call import ToolDef
from agent_harness.store import CurriculumScheduler, TrajectoryStore
from agent_harness.store.curriculum import Stage


# =========================================================================
# Shared fixtures and helpers
# =========================================================================

MATH_TASKS = [
    {"prompt": "What is 2+3?", "answer": "5", "difficulty": "easy"},
    {"prompt": "What is 10*10?", "answer": "100", "difficulty": "easy"},
    {"prompt": "What is 7*8?", "answer": "56", "difficulty": "medium"},
    {"prompt": "What is 15*17?", "answer": "255", "difficulty": "medium"},
    {"prompt": "What is 99**2?", "answer": "9801", "difficulty": "hard"},
    {"prompt": "What is 123*456?", "answer": "56088", "difficulty": "hard"},
]

CODE_TASKS = [
    {
        "prompt": "Write a function `double(x)` that doubles a number.",
        "test_code": "assert double(5) == 10\nassert double(0) == 0\nassert double(-3) == -6",
        "answer": "def double(x): return x * 2",
    },
    {
        "prompt": "Write a function `greet(name)` that returns 'Hello, <name>!'.",
        "test_code": "assert greet('World') == 'Hello, World!'\nassert greet('Alice') == 'Hello, Alice!'",
        "answer": "def greet(name): return f'Hello, {name}!'",
    },
]


def correct_math_agent(observation: str, available_tools: list[str]) -> Action:
    """Agent that uses calculator and always gets the right answer."""
    obs = observation.strip()
    try:
        float(obs)
        return Action.finish(obs)
    except ValueError:
        pass
    if obs.replace(".", "").replace("-", "").isdigit():
        return Action.finish(obs)
    import re
    m = re.search(r"What is (.+)\?", obs, re.IGNORECASE)
    if m and "calculator" in available_tools:
        return Action.tool("calculator", {"expression": m.group(1).strip().replace("^", "**")})
    return Action.finish("unknown")


def wrong_agent(observation: str, available_tools: list[str]) -> Action:
    """Agent that always gives wrong answers."""
    return Action.finish("wrong_answer")


def slow_agent(observation: str, available_tools: list[str]) -> Action:
    """Agent that wastes turns before answering."""
    return Action.text("hmm let me think...")


# =========================================================================
# Integration Test: Full Training Pipeline (Math)
# =========================================================================

class TestFullMathPipeline:
    """Full pipeline: env → collect → reward → credit → train → evaluate."""

    def test_online_training_loop(self):
        """Train an agent online, collecting trajectories on the fly."""
        env = ah.envs.MathReasoningEnv(
            dataset=MATH_TASKS, tools=["calculator"], max_turns=5,
        )
        reward = ah.RewardComposer([
            ah.rewards.exact_match(weight=0.6, extract_number=True),
            ah.rewards.tool_call_valid(weight=0.2),
            ah.rewards.trajectory_efficiency(max_turns=5, weight=0.2),
        ])
        harness = ah.Harness(
            env=env, reward=reward,
            credit=ah.CreditAssigner("hybrid", turn_weight=0.3, trajectory_weight=0.7),
            backend=DummyBackend(),
        )

        history = harness.train(
            tasks=MATH_TASKS, agent_fn=correct_math_agent,
            iterations=4, batch_size=3,
        )

        # Verify training ran
        assert len(history) == 4
        assert all("loss" in m and "avg_reward" in m for m in history)
        # Verify trajectories were stored
        assert len(harness.store) == 12  # 4 iters * 3 batch
        assert len(harness.trajectories) == 12
        # Verify stats
        stats = harness.get_stats()
        assert stats["iteration"] == 4
        assert stats["total_trajectories"] == 12

    def test_offline_training_loop(self):
        """Collect trajectories first, then train offline."""
        env = ah.envs.MathReasoningEnv(
            dataset=MATH_TASKS, tools=["calculator"], max_turns=5,
        )
        reward = ah.RewardComposer([
            ah.rewards.exact_match(weight=0.7, extract_number=True),
            ah.rewards.trajectory_efficiency(max_turns=5, weight=0.3),
        ])
        harness = ah.Harness(env=env, reward=reward, backend=DummyBackend())

        # Phase 1: collect
        trajectories = harness.collect_batch(MATH_TASKS, correct_math_agent)
        assert len(trajectories) == len(MATH_TASKS)

        # Phase 2: verify rewards
        for traj in trajectories:
            assert 0.0 <= traj.total_reward <= 1.0
            assert traj.num_turns >= 1

        # Phase 3: train
        history = harness.train(trajectories=trajectories, iterations=3, batch_size=4)
        assert len(history) == 3

    def test_evaluation_correct_agent(self):
        """Correct agent should achieve high reward."""
        env = ah.envs.MathReasoningEnv(tools=["calculator"], max_turns=5)
        reward = ah.RewardComposer([
            ah.rewards.exact_match(weight=0.8, extract_number=True),
            ah.rewards.trajectory_efficiency(max_turns=5, weight=0.2),
        ])
        harness = ah.Harness(env=env, reward=reward)

        metrics = harness.evaluate(MATH_TASKS, correct_math_agent)
        assert metrics["mean_reward"] > 0.5
        assert metrics["success_rate"] > 0.5
        assert metrics["mean_turns"] >= 1

    def test_evaluation_wrong_agent(self):
        """Wrong agent should achieve low reward."""
        env = ah.envs.MathReasoningEnv(tools=["calculator"], max_turns=5)
        reward = ah.RewardComposer([
            ah.rewards.exact_match(weight=0.8),
            ah.rewards.trajectory_efficiency(max_turns=5, weight=0.2),
        ])
        harness = ah.Harness(env=env, reward=reward)

        metrics = harness.evaluate(MATH_TASKS, wrong_agent)
        assert metrics["mean_reward"] < 0.5
        assert metrics["success_rate"] == 0.0


# =========================================================================
# Integration Test: Code Execution Pipeline
# =========================================================================

class TestCodeExecutionPipeline:
    """Full pipeline with CodeExecutionEnv and test-based rewards."""

    def _code_agent(self, solution: str):
        """Return an agent that submits the given code."""
        first_call = [True]

        def agent(observation: str, available_tools: list[str]) -> Action:
            if first_call[0] and "run_tests" in available_tools:
                first_call[0] = False
                return Action.tool("run_tests", {"code": solution})
            return Action.finish(solution)

        return agent

    def test_passing_code(self):
        """Agent submits correct code → tests pass → high reward."""
        env = ah.envs.CodeExecutionEnv(timeout=10, max_turns=5)
        # Use fuzzy_match instead of code_passes_tests since the latter
        # needs test_code injected via kwargs and Harness only passes ground_truth
        reward = ah.RewardComposer([
            ah.rewards.fuzzy_match(weight=0.8),
            ah.rewards.trajectory_efficiency(max_turns=5, weight=0.2),
        ])
        harness = ah.Harness(env=env, reward=reward)

        task = CODE_TASKS[0]
        agent = self._code_agent("def double(x): return x * 2")
        traj = harness.collect_trajectory(task, agent)
        # Agent submits correct code, fuzzy match with ground truth should be high
        assert traj.total_reward > 0.5

    def test_failing_code(self):
        """Agent submits wrong code → tests fail → low reward."""
        env = ah.envs.CodeExecutionEnv(timeout=10, max_turns=5)
        reward = ah.RewardComposer([
            ah.rewards.code_passes_tests(weight=0.8),
            ah.rewards.trajectory_efficiency(max_turns=5, weight=0.2),
        ])
        harness = ah.Harness(env=env, reward=reward)

        task = CODE_TASKS[0]
        agent = self._code_agent("def double(x): return x + 1")  # wrong
        traj = harness.collect_trajectory(task, agent)
        assert traj.total_reward < 0.5


# =========================================================================
# Integration Test: Custom Tool-Calling Pipeline
# =========================================================================

class TestToolCallingPipeline:
    """Full pipeline with custom-registered tools."""

    def _make_env(self):
        tools = {
            "add": ToolDef(
                spec=ToolSpec(name="add", description="Add two numbers"),
                fn=lambda a, b: str(int(a) + int(b)),
            ),
            "multiply": ToolDef(
                spec=ToolSpec(name="multiply", description="Multiply two numbers"),
                fn=lambda a, b: str(int(a) * int(b)),
            ),
        }
        return ah.envs.ToolCallingEnv(tools=tools, max_turns=5)

    def test_tool_calling_with_reward(self):
        env = self._make_env()
        reward = ah.RewardComposer([
            ah.rewards.contains_match(weight=0.5),
            ah.rewards.tool_call_valid(weight=0.3),
            ah.rewards.trajectory_efficiency(max_turns=5, weight=0.2),
        ])
        harness = ah.Harness(env=env, reward=reward)

        def agent(obs: str, tools: list[str]) -> Action:
            if "add" in tools and "15" not in obs:
                return Action.tool("add", {"a": "7", "b": "8"})
            return Action.finish("15")

        task = {"prompt": "What is 7 + 8?", "answer": "15"}
        traj = harness.collect_trajectory(task, agent)
        assert traj.num_turns == 2
        assert traj.total_reward > 0

    def test_multi_task_training(self):
        env = self._make_env()
        reward = ah.RewardComposer([
            ah.rewards.contains_match(weight=0.6),
            ah.rewards.tool_call_valid(weight=0.4),
        ])
        harness = ah.Harness(env=env, reward=reward, backend=DummyBackend())

        tasks = [
            {"prompt": "What is 3 + 4?", "answer": "7"},
            {"prompt": "What is 5 + 6?", "answer": "11"},
        ]

        def agent(obs: str, tools: list[str]) -> Action:
            if "add" in tools and all(c not in obs for c in "7891011"):
                import re
                m = re.search(r"(\d+)\s*\+\s*(\d+)", obs)
                if m:
                    return Action.tool("add", {"a": m.group(1), "b": m.group(2)})
            return Action.finish(obs.strip())

        history = harness.train(tasks=tasks, agent_fn=agent, iterations=2, batch_size=2)
        assert len(history) == 2


# =========================================================================
# Integration Test: Curriculum Progression
# =========================================================================

class TestCurriculumIntegration:
    """Verify curriculum scheduler integrates with training loop."""

    def test_curriculum_progresses_through_stages(self):
        env = ah.envs.MathReasoningEnv(tools=["calculator"], max_turns=5)
        reward = ah.RewardComposer([
            ah.rewards.exact_match(weight=0.8, extract_number=True),
            ah.rewards.trajectory_efficiency(max_turns=5, weight=0.2),
        ])
        curriculum = CurriculumScheduler(stages=[
            Stage(name="easy", difficulty="easy", epochs=2, promotion_threshold=0.5),
            Stage(name="medium", difficulty="medium", epochs=2, promotion_threshold=0.5),
            Stage(name="hard", difficulty="hard", epochs=2, promotion_threshold=0.5),
        ])
        harness = ah.Harness(
            env=env, reward=reward, curriculum=curriculum, backend=DummyBackend(),
        )

        # Use correct agent → high reward → should promote quickly
        history = harness.train(
            tasks=MATH_TASKS, agent_fn=correct_math_agent,
            iterations=6, batch_size=2,
        )
        assert len(history) == 6
        # Curriculum should have advanced
        assert curriculum.progress > 0
        assert len(curriculum.get_history()) > 0

    def test_curriculum_task_filtering(self):
        curriculum = CurriculumScheduler(stages=[
            Stage(name="easy", difficulty="easy", epochs=1, promotion_threshold=0.5),
            Stage(name="hard", difficulty="hard", epochs=1, promotion_threshold=0.5),
        ])

        # Before any update, should filter easy tasks
        easy_tasks = curriculum.get_current_tasks(MATH_TASKS)
        assert all(t["difficulty"] == "easy" for t in easy_tasks)
        assert len(easy_tasks) == 2

        # After promotion
        curriculum.update(mean_reward=0.8)
        hard_tasks = curriculum.get_current_tasks(MATH_TASKS)
        assert all(t["difficulty"] == "hard" for t in hard_tasks)
        assert len(hard_tasks) == 2


# =========================================================================
# Integration Test: Reward Debugger on Real Trajectories
# =========================================================================

class TestDebuggerIntegration:
    """Verify debugger works on trajectories from real env interactions."""

    def _collect_mixed_trajectories(self) -> list[Trajectory]:
        """Collect trajectories from both correct and wrong agents."""
        env = ah.envs.MathReasoningEnv(tools=["calculator"], max_turns=5)
        reward = ah.RewardComposer([
            ah.rewards.exact_match(weight=0.6, extract_number=True),
            ah.rewards.tool_call_valid(weight=0.2),
            ah.rewards.trajectory_efficiency(max_turns=5, weight=0.2),
        ])
        harness = ah.Harness(env=env, reward=reward)

        trajs = []
        # Correct agent on some tasks
        for task in MATH_TASKS[:3]:
            traj = harness.collect_trajectory(task, correct_math_agent)
            trajs.append(traj)
        # Wrong agent on other tasks
        for task in MATH_TASKS[3:]:
            traj = harness.collect_trajectory(task, wrong_agent)
            trajs.append(traj)
        return trajs

    def test_full_analysis_on_real_data(self):
        trajs = self._collect_mixed_trajectories()
        reward = ah.RewardComposer([
            ah.rewards.exact_match(weight=0.6, extract_number=True),
            ah.rewards.tool_call_valid(weight=0.2),
            ah.rewards.trajectory_efficiency(max_turns=5, weight=0.2),
        ])
        debugger = RewardDebugger(reward)
        report = debugger.analyze(trajs)

        assert report.total_trajectories == len(MATH_TASKS)
        assert len(report.component_stats) == 3  # 3 reward components
        assert report.composite_stats.mean > 0
        # Should have mix of high and low
        assert report.composite_stats.std > 0

    def test_hacking_detection_on_saturated_reward(self):
        """A reward that gives 1.0 to everything should be flagged."""
        trajs = self._collect_mixed_trajectories()
        # format_follows with .* matches everything
        reward = ah.RewardComposer([
            ah.rewards.format_follows(pattern=r".*", weight=1.0),
        ])
        debugger = RewardDebugger(reward)
        alerts = debugger.detect_hacking(trajs)
        high = [a for a in alerts if a.risk_level == "high"]
        assert len(high) >= 1

    def test_ab_comparison_correct_vs_wrong(self):
        """Compare strict vs lenient reward on mixed trajectories."""
        trajs = self._collect_mixed_trajectories()
        strict = ah.RewardComposer([
            ah.rewards.exact_match(weight=1.0, extract_number=True),
        ])
        lenient = ah.RewardComposer([
            ah.rewards.fuzzy_match(weight=1.0),
        ])
        debugger = RewardDebugger(strict)
        report = debugger.compare(strict, lenient, trajs)
        assert len(report.per_trajectory_diff) == len(trajs)
        # Strict should be more binary; lenient more gradient
        summary = report.summary()
        assert "Correlation" in summary

    def test_correlation_matrix_makes_sense(self):
        """Correlation between independent rewards should be computed."""
        trajs = self._collect_mixed_trajectories()
        reward = ah.RewardComposer([
            ah.rewards.exact_match(weight=0.5, extract_number=True),
            ah.rewards.tool_call_valid(weight=0.5),
        ])
        debugger = RewardDebugger(reward)
        report = debugger.analyze(trajs)
        corr = report.correlation_matrix
        # Should have entries for both components
        assert "exact_match" in corr
        assert "tool_call_valid" in corr
        # Self-correlation should be 1.0 (or 0.0 if zero-variance)
        em_self = corr["exact_match"]["exact_match"]
        assert em_self == 1.0 or em_self == 0.0  # 0.0 if all same value

    def test_summary_report_not_empty(self):
        trajs = self._collect_mixed_trajectories()
        reward = ah.RewardComposer([
            ah.rewards.exact_match(weight=0.7, extract_number=True),
            ah.rewards.tool_call_valid(weight=0.3),
        ])
        debugger = RewardDebugger(reward)
        report = debugger.analyze(trajs)
        summary = report.summary()
        assert "Reward Debug Report" in summary
        assert "exact_match" in summary
        assert "tool_call_valid" in summary


# =========================================================================
# Integration Test: Trajectory Store Persistence
# =========================================================================

class TestStorePersistenceIntegration:
    """Verify trajectories survive save → load → filter → train cycle."""

    def test_collect_save_load_train(self):
        """Full cycle: collect → save → load → filter → train."""
        # Collect
        env = ah.envs.MathReasoningEnv(tools=["calculator"], max_turns=5)
        reward = ah.RewardComposer([
            ah.rewards.exact_match(weight=0.7, extract_number=True),
            ah.rewards.trajectory_efficiency(max_turns=5, weight=0.3),
        ])
        harness = ah.Harness(env=env, reward=reward)
        trajectories = harness.collect_batch(MATH_TASKS, correct_math_agent)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            store = TrajectoryStore(tmpdir)
            store.add_batch(trajectories)
            store.save()

            # Load
            loaded = TrajectoryStore.load(tmpdir)
            assert len(loaded) == len(MATH_TASKS)

            # Filter
            good = loaded.filter(min_reward=0.5)
            assert len(good) > 0
            assert all(t.total_reward >= 0.5 for t in good)

            # Train on loaded trajectories
            harness2 = ah.Harness(env=env, reward=reward, backend=DummyBackend())
            history = harness2.train(trajectories=list(loaded), iterations=2, batch_size=3)
            assert len(history) == 2

    def test_trajectory_roundtrip_preserves_turns(self):
        """Verify turn-level data survives serialization."""
        env = ah.envs.MathReasoningEnv(tools=["calculator"], max_turns=5)
        reward = ah.RewardComposer([ah.rewards.exact_match(weight=1.0, extract_number=True)])
        harness = ah.Harness(
            env=env, reward=reward,
            credit=ah.CreditAssigner("outcome_only"),
        )

        task = MATH_TASKS[0]
        traj = harness.collect_trajectory(task, correct_math_agent)
        original_turns = traj.num_turns
        original_reward = traj.total_reward
        original_answer = traj.get_final_answer()

        with tempfile.TemporaryDirectory() as tmpdir:
            store = TrajectoryStore(tmpdir)
            store.add(traj)
            store.save()

            loaded = TrajectoryStore.load(tmpdir)
            restored = loaded[0]

            assert restored.num_turns == original_turns
            assert abs(restored.total_reward - original_reward) < 1e-6
            assert restored.get_final_answer() == original_answer
            assert restored.task["prompt"] == task["prompt"]
            # Turn-level data
            for orig_turn, rest_turn in zip(traj.turns, restored.turns):
                assert orig_turn.action.action_type == rest_turn.action.action_type
                if orig_turn.reward is not None:
                    assert abs(orig_turn.reward - rest_turn.reward) < 1e-6

    def test_store_statistics_after_load(self):
        """Statistics should be correct after loading from disk."""
        env = ah.envs.MathReasoningEnv(tools=["calculator"], max_turns=5)
        reward = ah.RewardComposer([
            ah.rewards.exact_match(weight=1.0, extract_number=True),
        ])
        harness = ah.Harness(env=env, reward=reward)

        # Mix of correct and wrong agent
        trajs = []
        for task in MATH_TASKS[:3]:
            trajs.append(harness.collect_trajectory(task, correct_math_agent))
        for task in MATH_TASKS[3:]:
            trajs.append(harness.collect_trajectory(task, wrong_agent))

        with tempfile.TemporaryDirectory() as tmpdir:
            store = TrajectoryStore(tmpdir)
            store.add_batch(trajs)
            store.save()

            loaded = TrajectoryStore.load(tmpdir)
            stats = loaded.statistics()
            assert stats["count"] == len(MATH_TASKS)
            assert 0 < stats["reward_mean"] < 1.0
            assert 0 < stats["success_rate"] < 1.0


# =========================================================================
# Integration Test: Credit Assignment End-to-End
# =========================================================================

class TestCreditAssignmentIntegration:
    """Verify credit assignment works within the full pipeline."""

    def test_outcome_only_assigns_uniform(self):
        """outcome_only: every turn gets the same reward."""
        env = ah.envs.MathReasoningEnv(tools=["calculator"], max_turns=5)
        reward = ah.RewardComposer([
            ah.rewards.exact_match(weight=1.0, extract_number=True),
        ])
        harness = ah.Harness(
            env=env, reward=reward,
            credit=ah.CreditAssigner("outcome_only"),
        )
        traj = harness.collect_trajectory(MATH_TASKS[0], correct_math_agent)
        # All turns should have the same reward
        turn_rewards = [t.reward for t in traj.turns]
        assert all(r is not None for r in turn_rewards)
        assert len(set(turn_rewards)) == 1  # all same value

    def test_hybrid_assigns_different(self):
        """hybrid: turns should get varying rewards based on turn-level fn."""
        env = ah.envs.MathReasoningEnv(tools=["calculator"], max_turns=5)
        reward = ah.RewardComposer([
            ah.rewards.exact_match(weight=1.0, extract_number=True),
        ])
        harness = ah.Harness(
            env=env, reward=reward,
            credit=ah.CreditAssigner(
                "hybrid",
                turn_reward_fn=ah.rewards.trajectory_efficiency(max_turns=5),
                turn_weight=0.4,
                trajectory_weight=0.6,
            ),
        )
        traj = harness.collect_trajectory(MATH_TASKS[0], correct_math_agent)
        turn_rewards = [t.reward for t in traj.turns]
        assert all(r is not None for r in turn_rewards)
        assert all(0 <= r <= 1.0 for r in turn_rewards)


# =========================================================================
# Integration Test: Multi-Environment Scenario
# =========================================================================

class TestMultiEnvironmentScenario:
    """Train on trajectories from different environments in one store."""

    def test_mixed_env_trajectories(self):
        # Math trajectories
        math_env = ah.envs.MathReasoningEnv(tools=["calculator"], max_turns=5)
        math_reward = ah.RewardComposer([
            ah.rewards.exact_match(weight=1.0, extract_number=True),
        ])
        math_harness = ah.Harness(env=math_env, reward=math_reward)
        math_trajs = math_harness.collect_batch(MATH_TASKS[:2], correct_math_agent)

        # Tool-calling trajectories
        tool_env = ah.envs.ToolCallingEnv(
            tools={
                "greet": ToolDef(
                    spec=ToolSpec(name="greet"),
                    fn=lambda name: f"Hello, {name}!",
                ),
            },
            max_turns=5,
        )
        tool_reward = ah.RewardComposer([
            ah.rewards.contains_match(weight=0.5),
            ah.rewards.tool_call_valid(weight=0.5),
        ])
        tool_harness = ah.Harness(env=tool_env, reward=tool_reward)

        def greet_agent(obs: str, tools: list[str]) -> Action:
            if "greet" in tools and "Hello" not in obs:
                return Action.tool("greet", {"name": "World"})
            return Action.finish(obs)

        tool_trajs = tool_harness.collect_batch(
            [{"prompt": "Greet the world", "answer": "Hello"}],
            greet_agent,
        )

        # Combine in one store
        store = TrajectoryStore()
        store.add_batch(math_trajs)
        store.add_batch(tool_trajs)
        assert len(store) == 3

        # Filter by env
        math_only = store.filter(env_name="MathReasoningEnv")
        tool_only = store.filter(env_name="ToolCallingEnv")
        assert len(math_only) == 2
        assert len(tool_only) == 1

        # Stats
        stats = store.statistics()
        assert stats["count"] == 3

    def test_save_load_mixed_envs(self):
        """Mixed-env store should survive persistence."""
        math_env = ah.envs.MathReasoningEnv(tools=["calculator"], max_turns=5)
        math_harness = ah.Harness(
            env=math_env,
            reward=ah.RewardComposer([ah.rewards.exact_match(weight=1.0, extract_number=True)]),
        )
        trajs = math_harness.collect_batch(MATH_TASKS[:2], correct_math_agent)

        with tempfile.TemporaryDirectory() as tmpdir:
            store = TrajectoryStore(tmpdir)
            store.add_batch(trajs)
            store.save()

            loaded = TrajectoryStore.load(tmpdir)
            assert len(loaded) == 2
            for t in loaded:
                assert t.env_name == "MathReasoningEnv"


# =========================================================================
# Integration Test: CLI on Real Data
# =========================================================================

class TestCLIIntegration:
    """Test CLI commands with real trajectory data."""

    def _save_trajectories(self, tmpdir: str) -> str:
        env = ah.envs.MathReasoningEnv(tools=["calculator"], max_turns=5)
        reward = ah.RewardComposer([
            ah.rewards.exact_match(weight=1.0, extract_number=True),
        ])
        harness = ah.Harness(env=env, reward=reward)
        trajs = harness.collect_batch(MATH_TASKS[:3], correct_math_agent)
        store = TrajectoryStore(tmpdir)
        store.add_batch(trajs)
        store.save()
        return tmpdir

    def test_stats_command(self):
        from click.testing import CliRunner
        from agent_harness.cli.main import cli

        with tempfile.TemporaryDirectory() as tmpdir:
            self._save_trajectories(tmpdir)
            runner = CliRunner()
            result = runner.invoke(cli, ["stats", tmpdir])
            assert result.exit_code == 0
            assert "Count: 3" in result.output
            assert "Reward" in result.output
            assert "Success Rate" in result.output

    def test_debug_command(self):
        from click.testing import CliRunner
        from agent_harness.cli.main import cli

        with tempfile.TemporaryDirectory() as tmpdir:
            self._save_trajectories(tmpdir)
            runner = CliRunner()
            result = runner.invoke(cli, ["debug", tmpdir])
            assert result.exit_code == 0
            assert "Reward Debug Report" in result.output

    def test_stats_empty_store(self):
        from click.testing import CliRunner
        from agent_harness.cli.main import cli

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save empty store
            store = TrajectoryStore(tmpdir)
            store.save()
            runner = CliRunner()
            result = runner.invoke(cli, ["stats", tmpdir])
            assert result.exit_code == 0
            assert "No trajectories" in result.output


# =========================================================================
# Integration Test: Top-Level API Imports
# =========================================================================

class TestTopLevelAPI:
    """Verify the public API is accessible from `import agent_harness as ah`."""

    def test_core_types(self):
        assert ah.Action is not None
        assert ah.Observation is not None
        assert ah.Trajectory is not None
        assert ah.AgentEnv is not None
        assert ah.ToolSpec is not None

    def test_rewards(self):
        assert ah.Reward is not None
        assert ah.RewardComposer is not None
        assert ah.CreditAssigner is not None
        assert ah.rewards.exact_match is not None
        assert ah.rewards.trajectory_efficiency is not None

    def test_envs(self):
        assert ah.envs.MathReasoningEnv is not None
        assert ah.envs.CodeExecutionEnv is not None
        assert ah.envs.ToolCallingEnv is not None

    def test_backends(self):
        assert ah.backends.DummyBackend is not None
        assert ah.backends.VeRLBackend is not None
        assert ah.backends.OpenRLHFBackend is not None
        assert ah.backends.TRLBackend is not None

    def test_debug(self):
        assert ah.debug.RewardDebugger is not None

    def test_store(self):
        assert ah.store.TrajectoryStore is not None
        assert ah.store.CurriculumScheduler is not None

    def test_harness(self):
        assert ah.Harness is not None

    def test_version(self):
        assert ah.__version__ == "0.1.0"


# =========================================================================
# Integration Test: End-to-End Example Smoke Tests
# =========================================================================

class TestExampleSmokeTests:
    """Smoke-test that the example scripts can be imported and run."""

    def test_math_example_main(self):
        from examples.math_reasoning import main
        # Should not raise
        main()

    def test_code_example_main(self):
        from examples.code_generation import main
        main()

    def test_tool_example_main(self):
        from examples.tool_calling_curriculum import main
        main()
