"""Tests for the Harness runtime and training backends."""

from agent_harness.backends import DummyBackend, TrainingConfig, VeRLBackend
from agent_harness.core.action import Action
from agent_harness.envs.math import MathReasoningEnv
from agent_harness.harness import Harness
from agent_harness.rewards import (
    CreditAssigner,
    RewardComposer,
    exact_match,
    trajectory_efficiency,
)
from agent_harness.store.curriculum import CurriculumScheduler, Stage


# Simple agent that always answers directly
def simple_agent(observation: str, available_tools: list[str]) -> Action:
    return Action.finish("42")


# Agent that uses a tool then answers
def tool_agent(observation: str, available_tools: list[str]) -> Action:
    if "calculator" in available_tools and "42" not in observation:
        return Action.tool("calculator", {"expression": "6*7"})
    return Action.finish("42")


class TestDummyBackend:
    def test_setup(self):
        backend = DummyBackend()
        config = TrainingConfig(model="test", algorithm="grpo")
        backend.setup(config)
        assert backend.config is not None

    def test_train_step(self):
        backend = DummyBackend()
        backend.setup(TrainingConfig())
        from agent_harness.core.trajectory import Trajectory
        traj = Trajectory(task={"prompt": "test"}, total_reward=0.8)
        metrics = backend.train_step([traj], [0.8])
        assert "loss" in metrics
        assert "avg_reward" in metrics
        assert metrics["avg_reward"] == 0.8

    def test_step_count(self):
        backend = DummyBackend()
        backend.setup(TrainingConfig())
        from agent_harness.core.trajectory import Trajectory
        traj = Trajectory()
        backend.train_step([traj], [0.5])
        backend.train_step([traj], [0.5])
        assert backend.step_count == 2


class TestVeRLBackend:
    def test_not_initialized(self):
        backend = VeRLBackend()
        from agent_harness.core.trajectory import Trajectory
        import pytest
        with pytest.raises(RuntimeError):
            backend.train_step([Trajectory()], [0.5])

    def test_setup_and_step(self):
        backend = VeRLBackend(model="test")
        backend.setup(TrainingConfig(model="test"))
        from agent_harness.core.trajectory import Trajectory
        metrics = backend.train_step([Trajectory()], [0.5])
        assert "avg_reward" in metrics


class TestHarness:
    def _make_harness(self, **kwargs):
        env = MathReasoningEnv(dataset=[
            {"prompt": "What is 6*7?", "answer": "42"},
            {"prompt": "What is 3+4?", "answer": "7"},
            {"prompt": "What is 10-5?", "answer": "5"},
        ])
        reward = RewardComposer([
            exact_match(weight=0.7),
            trajectory_efficiency(max_turns=10, weight=0.3),
        ])
        return Harness(env=env, reward=reward, **kwargs)

    def test_collect_trajectory(self):
        harness = self._make_harness()
        task = {"prompt": "What is 6*7?", "answer": "42"}
        traj = harness.collect_trajectory(task, simple_agent)
        assert traj.num_turns == 1  # Direct finish
        assert traj.total_reward > 0.5  # Should get exact match

    def test_collect_trajectory_with_tools(self):
        harness = self._make_harness()
        task = {"prompt": "What is 6*7?", "answer": "42"}
        traj = harness.collect_trajectory(task, tool_agent)
        assert traj.num_turns >= 1

    def test_collect_batch(self):
        harness = self._make_harness()
        tasks = [
            {"prompt": "What is 6*7?", "answer": "42"},
            {"prompt": "What is 3+4?", "answer": "7"},
        ]
        batch = harness.collect_batch(tasks, simple_agent)
        assert len(batch) == 2

    def test_train_offline(self):
        harness = self._make_harness()
        # Collect some trajectories first
        tasks = [
            {"prompt": "What is 6*7?", "answer": "42"},
            {"prompt": "What is 3+4?", "answer": "7"},
        ]
        trajectories = harness.collect_batch(tasks, simple_agent)

        # Train offline
        history = harness.train(trajectories=trajectories, iterations=3, batch_size=2)
        assert len(history) == 3
        assert all("loss" in m for m in history)

    def test_train_online(self):
        harness = self._make_harness()
        tasks = [
            {"prompt": "What is 6*7?", "answer": "42"},
            {"prompt": "What is 3+4?", "answer": "7"},
        ]
        history = harness.train(tasks=tasks, agent_fn=simple_agent, iterations=2, batch_size=2)
        assert len(history) == 2

    def test_evaluate(self):
        harness = self._make_harness()
        tasks = [
            {"prompt": "What is 6*7?", "answer": "42"},
            {"prompt": "What is 3+4?", "answer": "7"},
        ]
        metrics = harness.evaluate(tasks, simple_agent)
        assert "mean_reward" in metrics
        assert "success_rate" in metrics
        assert metrics["num_tasks"] == 2.0
        # simple_agent always answers "42", so only first task is correct
        assert metrics["success_rate"] == 0.5

    def test_with_curriculum(self):
        curriculum = CurriculumScheduler(stages=[
            Stage(name="easy", difficulty="easy", epochs=2, promotion_threshold=0.5),
            Stage(name="hard", difficulty="hard", epochs=2, promotion_threshold=0.8),
        ])
        harness = self._make_harness(curriculum=curriculum)
        tasks = [{"prompt": "What is 6*7?", "answer": "42"}]
        harness.train(tasks=tasks, agent_fn=simple_agent, iterations=3, batch_size=1)
        assert harness.curriculum is not None

    def test_credit_assignment(self):
        credit = CreditAssigner(strategy="outcome_only")
        harness = self._make_harness(credit=credit)
        task = {"prompt": "What is 6*7?", "answer": "42"}
        traj = harness.collect_trajectory(task, simple_agent)
        # Credit should be assigned to all turns
        assert all(t.reward is not None for t in traj.turns)

    def test_get_stats(self):
        harness = self._make_harness()
        stats = harness.get_stats()
        assert stats["iteration"] == 0
        assert stats["total_trajectories"] == 0

    def test_store_accumulation(self):
        harness = self._make_harness()
        tasks = [{"prompt": "What is 6*7?", "answer": "42"}]
        harness.train(tasks=tasks, agent_fn=simple_agent, iterations=3, batch_size=1)
        assert len(harness.store) == 3


class TestCLI:
    def test_cli_info(self):
        from click.testing import CliRunner
        from agent_harness.cli.main import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["info"])
        assert result.exit_code == 0
        assert "AgentHarness" in result.output
        assert "MathReasoningEnv" in result.output
        assert "exact_match" in result.output

    def test_cli_version(self):
        from click.testing import CliRunner
        from agent_harness.cli.main import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output
