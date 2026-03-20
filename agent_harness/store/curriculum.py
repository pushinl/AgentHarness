"""Curriculum Scheduler — progressive difficulty for agent training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_harness.core.trajectory import Trajectory


@dataclass
class Stage:
    """A single curriculum stage."""

    name: str
    difficulty: str  # e.g., "easy", "medium", "hard"
    epochs: int = 1
    task_filter: dict[str, Any] = field(default_factory=dict)
    promotion_threshold: float = 0.7


class CurriculumScheduler:
    """Schedule training progression through difficulty stages.

    Implements curriculum learning: start with easy tasks, and promote
    to harder tasks when the agent achieves the promotion threshold.

    Example:
        curriculum = CurriculumScheduler(stages=[
            Stage(name="easy", difficulty="easy", epochs=2, promotion_threshold=0.7),
            Stage(name="medium", difficulty="medium", epochs=3, promotion_threshold=0.8),
            Stage(name="hard", difficulty="hard", epochs=5),
        ])

        while not curriculum.is_complete:
            tasks = curriculum.get_current_tasks(all_tasks)
            # ... train on tasks ...
            curriculum.update(mean_reward=0.75)
    """

    def __init__(
        self,
        stages: list[Stage] | list[dict[str, Any]] | None = None,
    ):
        if stages is None:
            stages = []

        self.stages: list[Stage] = []
        for s in stages:
            if isinstance(s, dict):
                self.stages.append(Stage(
                    name=s.get("name", s.get("difficulty", "unknown")),
                    difficulty=s.get("difficulty", "unknown"),
                    epochs=s.get("epochs", 1),
                    task_filter=s.get("task_filter", {}),
                    promotion_threshold=s.get("promotion_threshold", 0.7),
                ))
            else:
                self.stages.append(s)

        self._current_stage_idx = 0
        self._current_epoch = 0
        self._stage_history: list[dict[str, Any]] = []

    @property
    def current_stage(self) -> Stage | None:
        if self._current_stage_idx < len(self.stages):
            return self.stages[self._current_stage_idx]
        return None

    @property
    def current_stage_index(self) -> int:
        return self._current_stage_idx

    @property
    def current_epoch(self) -> int:
        return self._current_epoch

    @property
    def is_complete(self) -> bool:
        return self._current_stage_idx >= len(self.stages)

    @property
    def progress(self) -> float:
        """Overall progress as a fraction [0, 1]."""
        if not self.stages:
            return 1.0
        total_epochs = sum(s.epochs for s in self.stages)
        completed = sum(s.epochs for s in self.stages[: self._current_stage_idx])
        completed += self._current_epoch
        return min(1.0, completed / total_epochs) if total_epochs > 0 else 1.0

    def update(self, mean_reward: float) -> bool:
        """Update the scheduler with the latest training reward.

        Args:
            mean_reward: Average reward from the current epoch.

        Returns:
            True if promoted to next stage, False otherwise.
        """
        if self.is_complete:
            return False

        stage = self.current_stage
        assert stage is not None

        self._stage_history.append({
            "stage": stage.name,
            "epoch": self._current_epoch,
            "reward": mean_reward,
        })

        self._current_epoch += 1

        # Check promotion conditions
        promoted = False
        if mean_reward >= stage.promotion_threshold:
            promoted = self._promote()
        elif self._current_epoch >= stage.epochs:
            # Finished all epochs for this stage, promote anyway
            promoted = self._promote()

        return promoted

    def _promote(self) -> bool:
        """Advance to the next stage."""
        self._current_stage_idx += 1
        self._current_epoch = 0
        return not self.is_complete

    def get_current_tasks(
        self,
        all_tasks: list[dict[str, Any]],
        difficulty_key: str = "difficulty",
    ) -> list[dict[str, Any]]:
        """Filter tasks matching the current stage's difficulty.

        Args:
            all_tasks: All available tasks with metadata.
            difficulty_key: Key in task dict for difficulty level.

        Returns:
            Tasks matching current stage difficulty.
        """
        if self.is_complete:
            return all_tasks  # Return all if curriculum is done

        stage = self.current_stage
        assert stage is not None

        return [
            t for t in all_tasks
            if t.get(difficulty_key) == stage.difficulty
        ]

    def get_history(self) -> list[dict[str, Any]]:
        """Get the full training history."""
        return list(self._stage_history)

    def reset(self) -> None:
        """Reset the scheduler to the beginning."""
        self._current_stage_idx = 0
        self._current_epoch = 0
        self._stage_history.clear()
