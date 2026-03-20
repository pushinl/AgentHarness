"""AgentEnv protocol — the core abstraction for task environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agent_harness.core.action import Action, Observation
from agent_harness.core.tool import ToolSpec


class AgentEnv(ABC):
    """The universal environment protocol for agentic RL.

    Any task benchmark can be plugged into AgentHarness by implementing
    these 5 methods. This is the "Gymnasium for Agentic RL".

    Lifecycle:
        1. reset(task) -> initial observation
        2. step(action) -> (observation, done)  [repeat until done]
        3. get_ground_truth() -> answer for reward computation
        4. get_available_tools() -> tools the agent can use
        5. get_state_snapshot() -> env state for debugging / reward
    """

    @abstractmethod
    def reset(self, task: dict[str, Any]) -> Observation:
        """Initialize the environment with a task and return the initial observation.

        Args:
            task: A dictionary describing the task. Must contain at least
                  a "prompt" or "question" key.

        Returns:
            The initial observation the agent sees.
        """
        ...

    @abstractmethod
    def step(self, action: Action) -> tuple[Observation, bool]:
        """Execute an agent action and return the next observation.

        Args:
            action: The action taken by the agent.

        Returns:
            A tuple of (observation, done) where done indicates
            whether the episode has ended.
        """
        ...

    def get_ground_truth(self) -> Any:
        """Return the ground truth answer for the current task.

        Used for verifiable reward computation. Override this if your
        environment has a known correct answer.

        Returns:
            The ground truth, or None if not available.
        """
        return None

    def get_available_tools(self) -> list[ToolSpec]:
        """Return the list of tools available in this environment.

        Override to expose tools the agent can call.

        Returns:
            List of ToolSpec describing available tools.
        """
        return []

    def get_state_snapshot(self) -> dict[str, Any]:
        """Return a snapshot of the current environment state.

        Used for reward computation, debugging, and trajectory logging.
        Should be cheap to compute and JSON-serializable.

        Returns:
            A dictionary representing the current state.
        """
        return {}

    @property
    def name(self) -> str:
        """Human-readable environment name."""
        return self.__class__.__name__

    @property
    def max_turns(self) -> int | None:
        """Maximum number of turns allowed, or None for unlimited."""
        return None
