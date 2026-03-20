"""Reward Debugger — visualization, hacking detection, and A/B comparison.

This is AgentHarness's killer feature: no other project provides tools for
debugging and diagnosing reward functions in agentic RL.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from agent_harness.core.trajectory import Trajectory
from agent_harness.rewards.base import Reward, RewardComposer


@dataclass
class ComponentStats:
    """Statistics for a single reward component."""

    name: str
    scores: list[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return statistics.mean(self.scores) if self.scores else 0.0

    @property
    def std(self) -> float:
        return statistics.stdev(self.scores) if len(self.scores) > 1 else 0.0

    @property
    def min(self) -> float:
        return min(self.scores) if self.scores else 0.0

    @property
    def max(self) -> float:
        return max(self.scores) if self.scores else 0.0

    @property
    def median(self) -> float:
        return statistics.median(self.scores) if self.scores else 0.0


@dataclass
class HackingAlert:
    """A detected reward hacking risk."""

    component: str
    risk_level: str  # "low", "medium", "high"
    reason: str
    suggestion: str


@dataclass
class DebugReport:
    """Complete reward debug report."""

    total_trajectories: int
    component_stats: dict[str, ComponentStats]
    composite_stats: ComponentStats
    hacking_alerts: list[HackingAlert]
    correlation_matrix: dict[str, dict[str, float]]

    def summary(self) -> str:
        """Generate a text summary of the report."""
        lines = []
        lines.append("=" * 60)
        lines.append("              Reward Debug Report")
        lines.append("=" * 60)
        lines.append(f"  Trajectories analyzed: {self.total_trajectories}")
        lines.append(f"  Composite score: {self.composite_stats.mean:.3f} "
                      f"(std={self.composite_stats.std:.3f})")
        lines.append("")
        lines.append(f"  {'Component':<22} {'Mean':>6} {'Std':>6} {'Min':>6} {'Max':>6} {'Risk'}")
        lines.append("  " + "-" * 56)

        for name, stats in self.component_stats.items():
            # Find hacking risk for this component
            risk = "Low"
            for alert in self.hacking_alerts:
                if alert.component == name:
                    risk = alert.risk_level.capitalize()
                    break
            lines.append(
                f"  {name:<22} {stats.mean:>6.3f} {stats.std:>6.3f} "
                f"{stats.min:>6.3f} {stats.max:>6.3f} {risk}"
            )

        if self.hacking_alerts:
            lines.append("")
            lines.append("  Alerts:")
            for alert in self.hacking_alerts:
                icon = {"low": "OK", "medium": "WARN", "high": "ALERT"}[alert.risk_level]
                lines.append(f"  [{icon}] {alert.component}: {alert.reason}")
                lines.append(f"        -> {alert.suggestion}")

        lines.append("=" * 60)
        return "\n".join(lines)


class RewardDebugger:
    """Debug and diagnose reward functions.

    Provides:
    - Statistical analysis of reward components
    - Reward hacking detection
    - A/B comparison of reward functions
    - Correlation analysis between components

    Example:
        debugger = RewardDebugger(reward_fn=my_reward)
        report = debugger.analyze(trajectories, ground_truth="42")
        print(report.summary())
        debugger.detect_hacking(trajectories)
    """

    # Thresholds for hacking detection
    SATURATION_THRESHOLD = 0.9  # Mean > this → near-saturated
    SATURATION_RATIO = 0.85     # % of trajectories scoring > 0.9
    LOW_VARIANCE_THRESHOLD = 0.05
    FLOOR_THRESHOLD = 0.1       # Mean < this → near-floor

    def __init__(self, reward_fn: Reward | RewardComposer):
        self.reward_fn = reward_fn

    def analyze(
        self,
        trajectories: list[Trajectory],
        **kwargs: Any,
    ) -> DebugReport:
        """Run full analysis on a set of trajectories.

        Returns a DebugReport with component stats, hacking alerts, and correlations.
        """
        if not trajectories:
            return DebugReport(
                total_trajectories=0,
                component_stats={},
                composite_stats=ComponentStats(name="composite"),
                hacking_alerts=[],
                correlation_matrix={},
            )

        # Compute composite scores
        composite_scores = [self.reward_fn(t, **kwargs) for t in trajectories]
        composite_stats = ComponentStats(name="composite", scores=composite_scores)

        # Compute per-component scores if it's a composer
        component_stats: dict[str, ComponentStats] = {}
        component_scores_map: dict[str, list[float]] = {}

        if isinstance(self.reward_fn, RewardComposer):
            for r in self.reward_fn.rewards:
                scores = [r(t, **kwargs) for t in trajectories]
                component_stats[r.name] = ComponentStats(name=r.name, scores=scores)
                component_scores_map[r.name] = scores
        else:
            component_stats[self.reward_fn.name] = ComponentStats(
                name=self.reward_fn.name, scores=composite_scores
            )
            component_scores_map[self.reward_fn.name] = composite_scores

        # Detect hacking
        alerts = self._detect_hacking(component_stats)

        # Compute correlation matrix
        corr = self._compute_correlations(component_scores_map)

        return DebugReport(
            total_trajectories=len(trajectories),
            component_stats=component_stats,
            composite_stats=composite_stats,
            hacking_alerts=alerts,
            correlation_matrix=corr,
        )

    def detect_hacking(
        self,
        trajectories: list[Trajectory],
        **kwargs: Any,
    ) -> list[HackingAlert]:
        """Run hacking detection only."""
        report = self.analyze(trajectories, **kwargs)
        return report.hacking_alerts

    def _detect_hacking(self, stats: dict[str, ComponentStats]) -> list[HackingAlert]:
        """Detect potential reward hacking risks."""
        alerts: list[HackingAlert] = []

        for name, cs in stats.items():
            if not cs.scores:
                continue

            # Check 1: Near-saturation
            high_ratio = sum(1 for s in cs.scores if s > 0.9) / len(cs.scores)
            if cs.mean > self.SATURATION_THRESHOLD and high_ratio > self.SATURATION_RATIO:
                alerts.append(HackingAlert(
                    component=name,
                    risk_level="high",
                    reason=f"Near-saturated: {high_ratio:.0%} of trajectories score > 0.9 "
                           f"(mean={cs.mean:.3f})",
                    suggestion="Increase difficulty, tighten criteria, or reduce weight",
                ))
            # Check 2: Low variance (suspicious uniformity)
            elif cs.std < self.LOW_VARIANCE_THRESHOLD and cs.mean > 0.5:
                alerts.append(HackingAlert(
                    component=name,
                    risk_level="medium",
                    reason=f"Suspiciously low variance (std={cs.std:.4f}, mean={cs.mean:.3f})",
                    suggestion="Check if reward is too easy to game or lacks discrimination",
                ))
            # Check 3: Floor (reward is too hard / broken)
            elif cs.mean < self.FLOOR_THRESHOLD:
                alerts.append(HackingAlert(
                    component=name,
                    risk_level="medium",
                    reason=f"Near-floor: mean={cs.mean:.3f}, most trajectories score very low",
                    suggestion="The reward may be too strict or misconfigured",
                ))
            else:
                alerts.append(HackingAlert(
                    component=name,
                    risk_level="low",
                    reason="Healthy distribution",
                    suggestion="No action needed",
                ))

        return alerts

    def compare(
        self,
        reward_a: Reward,
        reward_b: Reward,
        trajectories: list[Trajectory],
        **kwargs: Any,
    ) -> ComparisonReport:
        """A/B compare two reward functions on the same trajectories."""
        scores_a = [reward_a(t, **kwargs) for t in trajectories]
        scores_b = [reward_b(t, **kwargs) for t in trajectories]

        stats_a = ComponentStats(name=reward_a.name, scores=scores_a)
        stats_b = ComponentStats(name=reward_b.name, scores=scores_b)

        # Compute agreement (% of trajectories where both agree on ranking)
        agreements = 0
        disagreements = 0
        for i in range(len(trajectories)):
            for j in range(i + 1, len(trajectories)):
                a_rank = scores_a[i] > scores_a[j]
                b_rank = scores_b[i] > scores_b[j]
                if a_rank == b_rank:
                    agreements += 1
                else:
                    disagreements += 1

        total_pairs = agreements + disagreements
        agreement_rate = agreements / total_pairs if total_pairs > 0 else 1.0

        # Correlation
        corr = self._pearson(scores_a, scores_b)

        return ComparisonReport(
            reward_a_stats=stats_a,
            reward_b_stats=stats_b,
            correlation=corr,
            agreement_rate=agreement_rate,
            per_trajectory_diff=[a - b for a, b in zip(scores_a, scores_b)],
        )

    def _compute_correlations(
        self, scores_map: dict[str, list[float]]
    ) -> dict[str, dict[str, float]]:
        """Compute pairwise Pearson correlations between components."""
        names = list(scores_map.keys())
        corr: dict[str, dict[str, float]] = {}
        for a in names:
            corr[a] = {}
            for b in names:
                corr[a][b] = self._pearson(scores_map[a], scores_map[b])
        return corr

    @staticmethod
    def _pearson(x: list[float], y: list[float]) -> float:
        """Compute Pearson correlation coefficient."""
        n = len(x)
        if n < 2:
            return 0.0

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        var_x = sum((xi - mean_x) ** 2 for xi in x)
        var_y = sum((yi - mean_y) ** 2 for yi in y)

        denom = (var_x * var_y) ** 0.5
        if denom == 0:
            return 0.0
        return cov / denom


@dataclass
class ComparisonReport:
    """Report comparing two reward functions."""

    reward_a_stats: ComponentStats
    reward_b_stats: ComponentStats
    correlation: float
    agreement_rate: float
    per_trajectory_diff: list[float]

    def summary(self) -> str:
        lines = []
        lines.append("=" * 50)
        lines.append("         Reward A/B Comparison")
        lines.append("=" * 50)
        lines.append(
            f"  A ({self.reward_a_stats.name}): "
            f"mean={self.reward_a_stats.mean:.3f}, std={self.reward_a_stats.std:.3f}"
        )
        lines.append(
            f"  B ({self.reward_b_stats.name}): "
            f"mean={self.reward_b_stats.mean:.3f}, std={self.reward_b_stats.std:.3f}"
        )
        lines.append(f"  Correlation: {self.correlation:.3f}")
        lines.append(f"  Ranking Agreement: {self.agreement_rate:.1%}")
        if self.per_trajectory_diff:
            mean_diff = statistics.mean(self.per_trajectory_diff)
            lines.append(f"  Mean Score Diff (A-B): {mean_diff:+.3f}")
        lines.append("=" * 50)
        return "\n".join(lines)
