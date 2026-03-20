"""AgentHarness — the main orchestrator that ties everything together."""

from __future__ import annotations

import logging
from typing import Any, Callable

from agent_harness.backends.adapter import DummyBackend, TrainingBackend, TrainingConfig
from agent_harness.core.action import Action
from agent_harness.core.env import AgentEnv
from agent_harness.core.trajectory import Trajectory
from agent_harness.rewards.base import Reward, RewardComposer
from agent_harness.rewards.credit import CreditAssigner
from agent_harness.store.curriculum import CurriculumScheduler
from agent_harness.store.trajectory import TrajectoryStore

logger = logging.getLogger(__name__)


AgentFn = Callable[[str, list[str]], Action]
"""Type for agent functions: (observation_text, available_tools) -> Action"""


class Harness:
    """The main AgentHarness orchestrator.

    Ties together environment, reward, credit assignment, trajectory store,
    curriculum, and training backend into a unified training loop.

    Example:
        harness = Harness(
            env=MathReasoningEnv(dataset=my_tasks),
            reward=RewardComposer([...]),
            backend=VeRLBackend(model="Qwen/Qwen2.5-7B"),
        )
        harness.train(iterations=50)
    """

    def __init__(
        self,
        env: AgentEnv,
        reward: Reward | RewardComposer,
        credit: CreditAssigner | None = None,
        store: TrajectoryStore | None = None,
        curriculum: CurriculumScheduler | None = None,
        backend: TrainingBackend | None = None,
        config: TrainingConfig | None = None,
    ):
        self.env = env
        self.reward = reward
        self.credit = credit or CreditAssigner(strategy="outcome_only")
        self.store = store or TrajectoryStore()
        self.curriculum = curriculum
        self.backend = backend or DummyBackend()
        self.config = config or TrainingConfig()

        self.trajectories: list[Trajectory] = []
        self.training_history: list[dict[str, Any]] = []
        self._iteration = 0

    def collect_trajectory(
        self,
        task: dict[str, Any],
        agent_fn: AgentFn,
        max_turns: int | None = None,
    ) -> Trajectory:
        """Run one episode: agent interacts with env until done.

        Args:
            task: Task definition dict.
            agent_fn: Function that takes (observation, available_tools) and returns an Action.
            max_turns: Override max turns (uses env default if None).

        Returns:
            The collected trajectory with computed rewards.
        """
        obs = self.env.reset(task)
        trajectory = Trajectory(task=task, env_name=self.env.name)
        turns_limit = max_turns or self.env.max_turns or 50

        for _ in range(turns_limit):
            action = agent_fn(obs.content, obs.available_tools)
            next_obs, done = self.env.step(action)
            trajectory.add_turn(action=action, observation=next_obs)

            if done:
                break
            obs = next_obs

        # Compute reward
        ground_truth = self.env.get_ground_truth()
        reward_kwargs: dict[str, Any] = {}
        if ground_truth is not None:
            reward_kwargs["ground_truth"] = ground_truth

        trajectory.total_reward = self.reward(trajectory, **reward_kwargs)
        trajectory.success = trajectory.total_reward > 0.5

        # Credit assignment
        self.credit.apply(trajectory, **reward_kwargs)

        return trajectory

    def collect_batch(
        self,
        tasks: list[dict[str, Any]],
        agent_fn: AgentFn,
        max_turns: int | None = None,
    ) -> list[Trajectory]:
        """Collect trajectories for a batch of tasks."""
        batch = []
        for task in tasks:
            traj = self.collect_trajectory(task, agent_fn, max_turns)
            batch.append(traj)
        return batch

    def train(
        self,
        tasks: list[dict[str, Any]] | None = None,
        agent_fn: AgentFn | None = None,
        iterations: int = 10,
        batch_size: int = 8,
        trajectories: list[Trajectory] | None = None,
    ) -> list[dict[str, float]]:
        """Run the training loop.

        Can operate in two modes:
        1. Online: Provide tasks + agent_fn, collect trajectories on the fly.
        2. Offline: Provide pre-collected trajectories directly.

        Args:
            tasks: List of task dicts (for online mode).
            agent_fn: Agent function (for online mode).
            iterations: Number of training iterations.
            trajectories: Pre-collected trajectories (for offline mode).

        Returns:
            List of training metrics per iteration.
        """
        self.backend.setup(self.config)

        if trajectories:
            # Offline mode
            self.store.add_batch(trajectories)
            self.trajectories.extend(trajectories)

        history: list[dict[str, float]] = []

        for i in range(iterations):
            self._iteration = i + 1
            logger.info(f"Training iteration {self._iteration}/{iterations}")

            if trajectories:
                # Offline: sample from store
                batch = self.store.sample(batch_size)
            elif tasks and agent_fn:
                # Online: collect fresh trajectories
                import random
                batch_tasks = random.sample(tasks, min(batch_size, len(tasks)))
                batch = self.collect_batch(batch_tasks, agent_fn)
                self.store.add_batch(batch)
                self.trajectories.extend(batch)
            else:
                raise ValueError("Must provide either (tasks + agent_fn) or trajectories")

            # Extract rewards
            rewards = [t.total_reward for t in batch]

            # Training step
            metrics = self.backend.train_step(batch, rewards)
            metrics["iteration"] = float(self._iteration)
            history.append(metrics)
            self.training_history.append(metrics)

            # Curriculum update
            if self.curriculum and rewards:
                mean_reward = sum(rewards) / len(rewards)
                self.curriculum.update(mean_reward)

            logger.info(f"  Iteration {self._iteration}: {metrics}")

        return history

    def evaluate(
        self,
        tasks: list[dict[str, Any]],
        agent_fn: AgentFn,
    ) -> dict[str, float]:
        """Evaluate agent on a set of tasks without training.

        Returns aggregate metrics.
        """
        trajectories = self.collect_batch(tasks, agent_fn)
        rewards = [t.total_reward for t in trajectories]
        success_count = sum(1 for t in trajectories if t.success)

        return {
            "num_tasks": float(len(tasks)),
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "success_rate": success_count / len(tasks) if tasks else 0.0,
            "mean_turns": sum(t.num_turns for t in trajectories) / len(trajectories)
            if trajectories
            else 0.0,
        }

    def get_stats(self) -> dict[str, Any]:
        """Get current training statistics."""
        return {
            "iteration": self._iteration,
            "total_trajectories": len(self.trajectories),
            "store_size": len(self.store),
            "curriculum_progress": self.curriculum.progress if self.curriculum else None,
            "training_history": self.training_history,
        }
