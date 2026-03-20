"""Training backend adapter interface and implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from agent_harness.core.trajectory import Trajectory


@dataclass
class TrainingConfig:
    """Common training configuration."""

    model: str = ""
    algorithm: str = "grpo"
    num_gpus: int = 1
    learning_rate: float = 1e-6
    batch_size: int = 8
    max_steps: int = 100
    save_steps: int = 50
    output_dir: str = "./output"
    extra: dict[str, Any] = field(default_factory=dict)


class TrainingBackend(ABC):
    """Abstract interface for RL training backends.

    Implementations adapt AgentHarness trajectories to specific
    training frameworks (veRL, OpenRLHF, TRL, etc.).
    """

    @abstractmethod
    def setup(self, config: TrainingConfig) -> None:
        """Initialize the training backend."""
        ...

    @abstractmethod
    def train_step(
        self,
        trajectories: list[Trajectory],
        rewards: list[float],
    ) -> dict[str, float]:
        """Run one training step with the given trajectories and rewards.

        Args:
            trajectories: Batch of trajectories to train on.
            rewards: Corresponding reward for each trajectory.

        Returns:
            Dict of training metrics (loss, etc.).
        """
        ...

    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """Save a model checkpoint."""
        ...

    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """Load a model checkpoint."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name."""
        ...


class DummyBackend(TrainingBackend):
    """A dummy backend for testing that simulates training."""

    def __init__(self, **kwargs: Any):
        self.config: TrainingConfig | None = None
        self.step_count = 0
        self.history: list[dict[str, float]] = []

    def setup(self, config: TrainingConfig) -> None:
        self.config = config
        self.step_count = 0

    def train_step(
        self,
        trajectories: list[Trajectory],
        rewards: list[float],
    ) -> dict[str, float]:
        self.step_count += 1
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        metrics = {
            "step": float(self.step_count),
            "loss": max(0.01, 1.0 - avg_reward * 0.1 * self.step_count),
            "avg_reward": avg_reward,
            "num_trajectories": float(len(trajectories)),
        }
        self.history.append(metrics)
        return metrics

    def save_checkpoint(self, path: str) -> None:
        pass  # No-op for dummy

    def load_checkpoint(self, path: str) -> None:
        pass  # No-op for dummy

    @property
    def name(self) -> str:
        return "DummyBackend"


class VeRLBackend(TrainingBackend):
    """Adapter for the veRL training framework.

    This is a stub that shows the integration pattern.
    Full implementation requires veRL to be installed.
    """

    def __init__(
        self,
        model: str = "",
        algorithm: str = "grpo",
        num_gpus: int = 1,
        **kwargs: Any,
    ):
        self._model = model
        self._algorithm = algorithm
        self._num_gpus = num_gpus
        self._kwargs = kwargs
        self._initialized = False

    def setup(self, config: TrainingConfig) -> None:
        self._model = config.model or self._model
        self._algorithm = config.algorithm or self._algorithm
        self._num_gpus = config.num_gpus or self._num_gpus
        self._initialized = True
        # In real impl: initialize veRL trainer here

    def train_step(
        self,
        trajectories: list[Trajectory],
        rewards: list[float],
    ) -> dict[str, float]:
        if not self._initialized:
            raise RuntimeError("Backend not initialized. Call setup() first.")

        # Convert trajectories to veRL format
        batch = self._to_verl_batch(trajectories, rewards)

        # In real impl: call veRL trainer
        # metrics = self.trainer.step(batch)
        return {"loss": 0.0, "avg_reward": sum(rewards) / len(rewards) if rewards else 0.0}

    def _to_verl_batch(
        self,
        trajectories: list[Trajectory],
        rewards: list[float],
    ) -> list[dict[str, Any]]:
        """Convert AgentHarness trajectories to veRL training format."""
        batch = []
        for traj, reward in zip(trajectories, rewards):
            messages = traj.to_messages()
            batch.append({
                "messages": messages,
                "reward": reward,
                "metadata": traj.metadata,
            })
        return batch

    def save_checkpoint(self, path: str) -> None:
        pass  # Stub

    def load_checkpoint(self, path: str) -> None:
        pass  # Stub

    @property
    def name(self) -> str:
        return "VeRLBackend"


class OpenRLHFBackend(TrainingBackend):
    """Adapter for the OpenRLHF training framework (stub)."""

    def __init__(self, model: str = "", **kwargs: Any):
        self._model = model
        self._kwargs = kwargs

    def setup(self, config: TrainingConfig) -> None:
        pass

    def train_step(
        self,
        trajectories: list[Trajectory],
        rewards: list[float],
    ) -> dict[str, float]:
        return {"loss": 0.0, "avg_reward": sum(rewards) / len(rewards) if rewards else 0.0}

    def save_checkpoint(self, path: str) -> None:
        pass

    def load_checkpoint(self, path: str) -> None:
        pass

    @property
    def name(self) -> str:
        return "OpenRLHFBackend"


class TRLBackend(TrainingBackend):
    """Adapter for the TRL (Transformer Reinforcement Learning) framework (stub)."""

    def __init__(self, model: str = "", **kwargs: Any):
        self._model = model
        self._kwargs = kwargs

    def setup(self, config: TrainingConfig) -> None:
        pass

    def train_step(
        self,
        trajectories: list[Trajectory],
        rewards: list[float],
    ) -> dict[str, float]:
        return {"loss": 0.0, "avg_reward": sum(rewards) / len(rewards) if rewards else 0.0}

    def save_checkpoint(self, path: str) -> None:
        pass

    def load_checkpoint(self, path: str) -> None:
        pass

    @property
    def name(self) -> str:
        return "TRLBackend"
