"""Trajectory data structures for recording agent-environment interactions."""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field

from agent_harness.core.action import Action, Observation


class Turn(BaseModel):
    """A single turn in an agent-environment interaction.

    A turn consists of one agent action and one environment observation.
    """

    turn_number: int = 0
    action: Action
    observation: Observation
    timestamp: float = Field(default_factory=time.time)
    reward: float | None = None  # per-turn reward (if credit assignment is used)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Step(BaseModel):
    """A single step — alias for Turn for compatibility.

    In multi-step reasoning, a 'step' often refers to one thought+action pair.
    """

    thought: str = ""
    action: Action
    observation: Observation
    reward: float | None = None


class Trajectory(BaseModel):
    """A complete trajectory (episode) of agent-environment interaction.

    Contains the full sequence of turns, the task definition,
    final outcome, and aggregate reward.
    """

    task: dict[str, Any] = Field(default_factory=dict)
    turns: list[Turn] = Field(default_factory=list)
    total_reward: float = 0.0
    success: bool = False
    env_name: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    @property
    def actions(self) -> list[Action]:
        return [t.action for t in self.turns]

    @property
    def observations(self) -> list[Observation]:
        return [t.observation for t in self.turns]

    def add_turn(self, action: Action, observation: Observation, reward: float | None = None) -> Turn:
        """Append a turn to this trajectory."""
        turn = Turn(
            turn_number=len(self.turns),
            action=action,
            observation=observation,
            reward=reward,
        )
        self.turns.append(turn)
        return turn

    def get_final_answer(self) -> str | None:
        """Extract the final answer from the last action."""
        if not self.turns:
            return None
        last = self.turns[-1].action
        if last.content:
            return last.content
        return None

    def to_messages(self) -> list[dict[str, str]]:
        """Convert trajectory to a chat message format.

        Useful for feeding into training pipelines.
        """
        messages: list[dict[str, str]] = []
        if "prompt" in self.task:
            messages.append({"role": "user", "content": self.task["prompt"]})

        for turn in self.turns:
            if turn.action.content:
                messages.append({"role": "assistant", "content": turn.action.content})
            if turn.observation.content:
                messages.append({"role": "user", "content": turn.observation.content})

        return messages

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary (for storage)."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trajectory:
        """Deserialize from a plain dictionary."""
        return cls.model_validate(data)
