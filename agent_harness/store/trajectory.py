"""Trajectory Store — collect, filter, save/load trajectories."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from agent_harness.core.trajectory import Trajectory


class TrajectoryStore:
    """Persistent storage for agent trajectories.

    Supports collecting, filtering, saving and loading trajectories.
    Stores as JSONL for simplicity and portability.

    Example:
        store = TrajectoryStore("./trajectories")
        store.add(trajectory)
        store.save()

        # Later...
        store = TrajectoryStore.load("./trajectories")
        good = store.filter(min_reward=0.5, max_turns=10)
    """

    def __init__(self, path: str | Path = "./trajectories"):
        self.path = Path(path)
        self.trajectories: list[Trajectory] = []

    def add(self, trajectory: Trajectory) -> None:
        """Add a single trajectory to the store."""
        self.trajectories.append(trajectory)

    def add_batch(self, trajectories: list[Trajectory]) -> None:
        """Add multiple trajectories at once."""
        self.trajectories.extend(trajectories)

    def filter(
        self,
        min_reward: float | None = None,
        max_reward: float | None = None,
        min_turns: int | None = None,
        max_turns: int | None = None,
        env_name: str | None = None,
        success_only: bool = False,
        custom_fn: Callable[[Trajectory], bool] | None = None,
    ) -> list[Trajectory]:
        """Filter trajectories by criteria.

        Returns a new list (does not modify the store).
        """
        results = list(self.trajectories)

        if min_reward is not None:
            results = [t for t in results if t.total_reward >= min_reward]
        if max_reward is not None:
            results = [t for t in results if t.total_reward <= max_reward]
        if min_turns is not None:
            results = [t for t in results if t.num_turns >= min_turns]
        if max_turns is not None:
            results = [t for t in results if t.num_turns <= max_turns]
        if env_name is not None:
            results = [t for t in results if t.env_name == env_name]
        if success_only:
            results = [t for t in results if t.success]
        if custom_fn is not None:
            results = [t for t in results if custom_fn(t)]

        return results

    def sample(self, n: int, seed: int | None = None) -> list[Trajectory]:
        """Randomly sample n trajectories."""
        import random

        rng = random.Random(seed)
        n = min(n, len(self.trajectories))
        return rng.sample(self.trajectories, n)

    def sort_by_reward(self, descending: bool = True) -> list[Trajectory]:
        """Return trajectories sorted by total reward."""
        return sorted(self.trajectories, key=lambda t: t.total_reward, reverse=descending)

    def statistics(self) -> dict[str, Any]:
        """Compute basic statistics about the stored trajectories."""
        if not self.trajectories:
            return {"count": 0}

        rewards = [t.total_reward for t in self.trajectories]
        turns = [t.num_turns for t in self.trajectories]
        success_count = sum(1 for t in self.trajectories if t.success)

        return {
            "count": len(self.trajectories),
            "reward_mean": sum(rewards) / len(rewards),
            "reward_min": min(rewards),
            "reward_max": max(rewards),
            "turns_mean": sum(turns) / len(turns),
            "turns_min": min(turns),
            "turns_max": max(turns),
            "success_rate": success_count / len(self.trajectories),
        }

    def save(self, filename: str = "trajectories.jsonl") -> Path:
        """Save trajectories to a JSONL file."""
        self.path.mkdir(parents=True, exist_ok=True)
        filepath = self.path / filename
        with open(filepath, "w") as f:
            for traj in self.trajectories:
                f.write(json.dumps(traj.to_dict()) + "\n")
        return filepath

    @classmethod
    def load(cls, path: str | Path, filename: str = "trajectories.jsonl") -> TrajectoryStore:
        """Load trajectories from a JSONL file."""
        store = cls(path=path)
        filepath = Path(path) / filename
        if filepath.exists():
            with open(filepath) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        store.trajectories.append(Trajectory.from_dict(data))
        return store

    def clear(self) -> None:
        """Clear all trajectories from the store."""
        self.trajectories.clear()

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Trajectory:
        return self.trajectories[idx]

    def __iter__(self):
        return iter(self.trajectories)
