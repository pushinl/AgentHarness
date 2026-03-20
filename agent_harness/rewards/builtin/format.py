"""Format-based reward functions."""

from __future__ import annotations

import re
from typing import Any

from agent_harness.core.trajectory import Trajectory
from agent_harness.rewards.base import Reward


class FormatFollows(Reward):
    """Reward for following a specified output format (regex pattern)."""

    def __init__(self, pattern: str, weight: float = 1.0, flags: int = re.DOTALL):
        super().__init__(weight=weight, name="format_follows")
        self.pattern = pattern
        self.flags = flags
        self._compiled = re.compile(pattern, flags)

    def compute(self, trajectory: Trajectory, **kwargs: Any) -> float:
        answer = trajectory.get_final_answer()
        if not answer:
            return 0.0
        return 1.0 if self._compiled.search(answer) else 0.0


class LengthPenalty(Reward):
    """Penalty for exceeding a maximum token/character length.

    Returns 1.0 if within limit, linearly decreasing to 0.0 at 2x limit.
    """

    def __init__(self, max_tokens: int = 2000, weight: float = 1.0, use_words: bool = False):
        super().__init__(weight=weight, name="length_penalty")
        self.max_tokens = max_tokens
        self.use_words = use_words

    def _count(self, text: str) -> int:
        if self.use_words:
            return len(text.split())
        return len(text)

    def compute(self, trajectory: Trajectory, **kwargs: Any) -> float:
        # Count total output length across all turns
        total = sum(
            self._count(t.action.content) for t in trajectory.turns if t.action.content
        )
        if total <= self.max_tokens:
            return 1.0
        if total >= self.max_tokens * 2:
            return 0.0
        return 1.0 - (total - self.max_tokens) / self.max_tokens


class StructuredOutput(Reward):
    """Reward for producing valid structured output (JSON, XML, etc.)."""

    def __init__(self, format_type: str = "json", weight: float = 1.0):
        super().__init__(weight=weight, name="structured_output")
        self.format_type = format_type

    def compute(self, trajectory: Trajectory, **kwargs: Any) -> float:
        answer = trajectory.get_final_answer()
        if not answer:
            return 0.0

        if self.format_type == "json":
            return self._check_json(answer)
        return 0.0

    def _check_json(self, text: str) -> float:
        import json

        # Try to extract JSON from code blocks
        json_match = re.search(r"```json?\s*\n(.*?)```", text, re.DOTALL)
        candidate = json_match.group(1) if json_match else text

        try:
            json.loads(candidate.strip())
            return 1.0
        except (json.JSONDecodeError, ValueError):
            return 0.0


def format_follows(pattern: str, weight: float = 1.0, **kwargs: Any) -> FormatFollows:
    return FormatFollows(pattern=pattern, weight=weight, **kwargs)


def length_penalty(max_tokens: int = 2000, weight: float = 1.0, **kwargs: Any) -> LengthPenalty:
    return LengthPenalty(max_tokens=max_tokens, weight=weight, **kwargs)


def structured_output(format_type: str = "json", weight: float = 1.0) -> StructuredOutput:
    return StructuredOutput(format_type=format_type, weight=weight)
