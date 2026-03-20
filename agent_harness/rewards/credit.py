"""Credit assignment strategies for multi-turn trajectories."""

from __future__ import annotations

from enum import Enum
from typing import Any

from agent_harness.core.trajectory import Trajectory
from agent_harness.rewards.base import Reward


class CreditStrategy(str, Enum):
    OUTCOME_ONLY = "outcome_only"
    TURN_LEVEL = "turn_level"
    HYBRID = "hybrid"


class CreditAssigner:
    """Assign credit (per-turn rewards) to turns in a trajectory.

    Supports three strategies:
    - outcome_only: All turns get the same trajectory-level reward.
    - turn_level: Each turn gets its own reward from turn-level reward fns.
    - hybrid: Weighted combination of outcome and turn-level.
    """

    def __init__(
        self,
        strategy: str | CreditStrategy = CreditStrategy.OUTCOME_ONLY,
        turn_reward_fn: Reward | None = None,
        trajectory_reward_fn: Reward | None = None,
        turn_weight: float = 0.3,
        trajectory_weight: float = 0.7,
    ):
        self.strategy = CreditStrategy(strategy)
        self.turn_reward_fn = turn_reward_fn
        self.trajectory_reward_fn = trajectory_reward_fn
        self.turn_weight = turn_weight
        self.trajectory_weight = trajectory_weight

    def assign(self, trajectory: Trajectory, **kwargs: Any) -> list[float]:
        """Assign per-turn rewards to a trajectory.

        Returns:
            List of per-turn reward values, one for each turn.
        """
        n = trajectory.num_turns
        if n == 0:
            return []

        if self.strategy == CreditStrategy.OUTCOME_ONLY:
            return self._outcome_only(trajectory, **kwargs)
        elif self.strategy == CreditStrategy.TURN_LEVEL:
            return self._turn_level(trajectory, **kwargs)
        else:
            return self._hybrid(trajectory, **kwargs)

    def _outcome_only(self, trajectory: Trajectory, **kwargs: Any) -> list[float]:
        """All turns get the same trajectory-level reward."""
        if self.trajectory_reward_fn:
            reward = self.trajectory_reward_fn(trajectory, **kwargs)
        else:
            reward = trajectory.total_reward
        return [reward] * trajectory.num_turns

    def _turn_level(self, trajectory: Trajectory, **kwargs: Any) -> list[float]:
        """Each turn gets individually computed reward."""
        if not self.turn_reward_fn:
            return [0.0] * trajectory.num_turns

        rewards = []
        for i, turn in enumerate(trajectory.turns):
            # Create a sub-trajectory up to and including this turn
            sub_traj = Trajectory(
                task=trajectory.task,
                turns=trajectory.turns[: i + 1],
                env_name=trajectory.env_name,
            )
            rewards.append(self.turn_reward_fn(sub_traj, **kwargs))
        return rewards

    def _hybrid(self, trajectory: Trajectory, **kwargs: Any) -> list[float]:
        """Weighted combination of outcome and turn-level rewards."""
        outcome = self._outcome_only(trajectory, **kwargs)
        turn = self._turn_level(trajectory, **kwargs)
        return [
            self.trajectory_weight * o + self.turn_weight * t
            for o, t in zip(outcome, turn)
        ]

    def apply(self, trajectory: Trajectory, **kwargs: Any) -> Trajectory:
        """Assign rewards and write them into the trajectory turns (in-place).

        Returns the same trajectory for chaining.
        """
        rewards = self.assign(trajectory, **kwargs)
        for turn, r in zip(trajectory.turns, rewards):
            turn.reward = r
        return trajectory
