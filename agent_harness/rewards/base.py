"""Reward base class and RewardComposer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agent_harness.core.trajectory import Trajectory


class Reward(ABC):
    """Base class for all reward functions.

    A reward function takes a trajectory and returns a scalar reward.
    All rewards should be normalized to [0, 1] range.
    """

    def __init__(self, weight: float = 1.0, name: str | None = None):
        self.weight = weight
        self._name = name or self.__class__.__name__

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def compute(self, trajectory: Trajectory, **kwargs: Any) -> float:
        """Compute the reward for a trajectory.

        Args:
            trajectory: The complete agent trajectory.
            **kwargs: Additional context (e.g., ground_truth, env_state).

        Returns:
            A float in [0, 1].
        """
        ...

    def __call__(self, trajectory: Trajectory, **kwargs: Any) -> float:
        return self.compute(trajectory, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(weight={self.weight})"


class RewardComposer(Reward):
    """Compose multiple reward functions into a weighted sum.

    Example:
        reward = RewardComposer([
            exact_match(key="answer", weight=0.7),
            tool_call_valid(weight=0.15),
            trajectory_efficiency(max_turns=8, weight=0.15),
        ])
        score = reward(trajectory, ground_truth="42")
    """

    def __init__(self, rewards: list[Reward], normalize: bool = True):
        """Initialize the composer.

        Args:
            rewards: List of reward functions to compose.
            normalize: If True, normalize weights to sum to 1.
        """
        super().__init__(weight=1.0, name="RewardComposer")
        self.rewards = rewards
        self._normalize = normalize

    @property
    def total_weight(self) -> float:
        return sum(r.weight for r in self.rewards)

    def compute(self, trajectory: Trajectory, **kwargs: Any) -> float:
        """Compute weighted sum of all component rewards."""
        if not self.rewards:
            return 0.0

        total_weight = self.total_weight
        if total_weight == 0:
            return 0.0

        score = 0.0
        for r in self.rewards:
            component_score = r.compute(trajectory, **kwargs)
            if self._normalize:
                score += r.weight / total_weight * component_score
            else:
                score += r.weight * component_score

        return max(0.0, min(1.0, score))

    def compute_breakdown(self, trajectory: Trajectory, **kwargs: Any) -> dict[str, float]:
        """Compute each component's contribution separately.

        Returns:
            Dict mapping reward name to its individual score (before weighting).
        """
        breakdown: dict[str, float] = {}
        for r in self.rewards:
            breakdown[r.name] = r.compute(trajectory, **kwargs)
        return breakdown

    def compute_weighted_breakdown(self, trajectory: Trajectory, **kwargs: Any) -> dict[str, float]:
        """Compute each component's weighted contribution.

        Returns:
            Dict mapping reward name to its weighted contribution.
        """
        total_weight = self.total_weight
        if total_weight == 0:
            return {}

        breakdown: dict[str, float] = {}
        for r in self.rewards:
            raw = r.compute(trajectory, **kwargs)
            if self._normalize:
                breakdown[r.name] = r.weight / total_weight * raw
            else:
                breakdown[r.name] = r.weight * raw
        return breakdown

    def __repr__(self) -> str:
        components = ", ".join(f"{r.name}(w={r.weight})" for r in self.rewards)
        return f"RewardComposer([{components}])"
