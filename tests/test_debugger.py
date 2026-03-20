"""Tests for the Reward Debugger."""

from agent_harness.core.action import Action, Observation
from agent_harness.core.trajectory import Trajectory
from agent_harness.debug.debugger import RewardDebugger, ComparisonReport
from agent_harness.rewards import (
    RewardComposer,
    exact_match,
    trajectory_efficiency,
    format_follows,
    length_penalty,
)


def _make_trajectory(answer: str, n_turns: int = 2, task_prompt: str = "Q?") -> Trajectory:
    traj = Trajectory(task={"prompt": task_prompt})
    for i in range(n_turns - 1):
        traj.add_turn(
            action=Action.tool("calc", {"expr": "1+1"}),
            observation=Observation.simple("2"),
        )
    traj.add_turn(action=Action.finish(answer), observation=Observation.simple(""))
    return traj


def _make_trajectories(answers: list[str], ground_truth: str = "42") -> list[Trajectory]:
    return [_make_trajectory(a) for a in answers]


class TestRewardDebugger:
    def test_analyze_basic(self):
        reward = RewardComposer([
            exact_match(weight=0.7),
            trajectory_efficiency(max_turns=10, weight=0.3),
        ])
        trajectories = _make_trajectories(["42", "43", "42", "wrong", "42"])
        debugger = RewardDebugger(reward)
        report = debugger.analyze(trajectories, ground_truth="42")

        assert report.total_trajectories == 5
        assert "exact_match" in report.component_stats
        assert "trajectory_efficiency" in report.component_stats
        assert report.composite_stats.mean > 0

    def test_analyze_empty(self):
        reward = exact_match()
        debugger = RewardDebugger(reward)
        report = debugger.analyze([])
        assert report.total_trajectories == 0

    def test_component_stats(self):
        reward = RewardComposer([exact_match(weight=1.0)])
        trajectories = _make_trajectories(["42", "42", "42", "42", "42"])
        debugger = RewardDebugger(reward)
        report = debugger.analyze(trajectories, ground_truth="42")

        stats = report.component_stats["exact_match"]
        assert stats.mean == 1.0
        assert stats.min == 1.0
        assert stats.max == 1.0

    def test_summary_output(self):
        reward = RewardComposer([
            exact_match(weight=0.6),
            trajectory_efficiency(max_turns=10, weight=0.4),
        ])
        trajectories = _make_trajectories(["42", "43", "42"])
        debugger = RewardDebugger(reward)
        report = debugger.analyze(trajectories, ground_truth="42")

        summary = report.summary()
        assert "Reward Debug Report" in summary
        assert "exact_match" in summary
        assert "trajectory_efficiency" in summary


class TestHackingDetection:
    def test_detect_saturation(self):
        """All trajectories scoring near 1.0 should trigger saturation alert."""
        reward = RewardComposer([format_follows(pattern=r".*", weight=1.0)])
        # Pattern ".*" matches everything → all will score 1.0
        trajectories = _make_trajectories(["anything", "something", "else", "text", "more"])
        debugger = RewardDebugger(reward)
        alerts = debugger.detect_hacking(trajectories, ground_truth="42")

        high_alerts = [a for a in alerts if a.risk_level == "high"]
        assert len(high_alerts) >= 1
        assert "saturated" in high_alerts[0].reason.lower()

    def test_no_hacking_healthy(self):
        """Mixed scores should not trigger high alerts."""
        reward = RewardComposer([exact_match(weight=1.0)])
        # Mix of correct and incorrect
        trajectories = _make_trajectories(["42", "43", "42", "wrong", "42"])
        debugger = RewardDebugger(reward)
        alerts = debugger.detect_hacking(trajectories, ground_truth="42")

        high_alerts = [a for a in alerts if a.risk_level == "high"]
        assert len(high_alerts) == 0

    def test_detect_floor(self):
        """All trajectories scoring near 0 should trigger floor alert."""
        reward = RewardComposer([exact_match(weight=1.0)])
        trajectories = _make_trajectories(["wrong1", "wrong2", "wrong3", "wrong4", "wrong5"])
        debugger = RewardDebugger(reward)
        alerts = debugger.detect_hacking(trajectories, ground_truth="42")

        floor_alerts = [a for a in alerts if a.risk_level == "medium" and "floor" in a.reason.lower()]
        assert len(floor_alerts) >= 1


class TestComparison:
    def test_compare_same_reward(self):
        reward = exact_match(weight=1.0)
        trajectories = _make_trajectories(["42", "43", "42"])
        debugger = RewardDebugger(reward)
        report = debugger.compare(reward, reward, trajectories, ground_truth="42")

        assert isinstance(report, ComparisonReport)
        assert report.correlation == 1.0 or len(trajectories) < 3
        assert report.agreement_rate == 1.0

    def test_compare_different_rewards(self):
        reward_a = exact_match(weight=1.0)
        reward_b = trajectory_efficiency(max_turns=10, weight=1.0)
        trajectories = [
            _make_trajectory("42", n_turns=2),
            _make_trajectory("42", n_turns=5),
            _make_trajectory("wrong", n_turns=1),
        ]
        debugger = RewardDebugger(reward_a)
        report = debugger.compare(reward_a, reward_b, trajectories, ground_truth="42")

        assert isinstance(report, ComparisonReport)
        assert len(report.per_trajectory_diff) == 3

    def test_comparison_summary(self):
        reward_a = exact_match(weight=1.0)
        reward_b = trajectory_efficiency(max_turns=10, weight=1.0)
        trajectories = _make_trajectories(["42", "43", "42"])
        debugger = RewardDebugger(reward_a)
        report = debugger.compare(reward_a, reward_b, trajectories, ground_truth="42")

        summary = report.summary()
        assert "Comparison" in summary
        assert "Correlation" in summary


class TestCorrelations:
    def test_correlation_matrix(self):
        reward = RewardComposer([
            exact_match(weight=0.5),
            trajectory_efficiency(max_turns=10, weight=0.5),
        ])
        trajectories = _make_trajectories(["42", "43", "42", "wrong", "42"])
        debugger = RewardDebugger(reward)
        report = debugger.analyze(trajectories, ground_truth="42")

        corr = report.correlation_matrix
        assert "exact_match" in corr
        assert "trajectory_efficiency" in corr
        # Self-correlation should be 1.0
        assert abs(corr["exact_match"]["exact_match"] - 1.0) < 1e-6

    def test_single_reward_no_matrix(self):
        reward = exact_match()
        trajectories = _make_trajectories(["42", "43"])
        debugger = RewardDebugger(reward)
        report = debugger.analyze(trajectories, ground_truth="42")

        # Single reward: 1x1 matrix
        assert len(report.correlation_matrix) == 1
