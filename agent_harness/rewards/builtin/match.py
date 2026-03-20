"""Match-based reward functions: exact_match, fuzzy_match."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from agent_harness.core.trajectory import Trajectory
from agent_harness.rewards.base import Reward


class ExactMatch(Reward):
    """Reward for exact match between final answer and ground truth.

    Supports optional normalization (strip, lowercase, number extraction).
    """

    def __init__(
        self,
        key: str = "answer",
        weight: float = 1.0,
        normalize: bool = True,
        extract_number: bool = False,
    ):
        super().__init__(weight=weight, name="exact_match")
        self.key = key
        self._normalize = normalize
        self.extract_number = extract_number

    def _normalize_text(self, text: str) -> str:
        text = text.strip()
        if self._normalize:
            text = text.lower()
            text = re.sub(r"\s+", " ", text)
        if self.extract_number:
            numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
            if numbers:
                text = numbers[-1]
        return text

    def compute(self, trajectory: Trajectory, **kwargs: Any) -> float:
        ground_truth = kwargs.get("ground_truth") or kwargs.get(self.key)
        if ground_truth is None:
            return 0.0

        answer = trajectory.get_final_answer()
        if answer is None:
            return 0.0

        norm_answer = self._normalize_text(str(answer))
        norm_truth = self._normalize_text(str(ground_truth))
        return 1.0 if norm_answer == norm_truth else 0.0


class FuzzyMatch(Reward):
    """Reward based on fuzzy string similarity between answer and ground truth."""

    def __init__(
        self,
        key: str = "answer",
        weight: float = 1.0,
        threshold: float = 0.0,
    ):
        super().__init__(weight=weight, name="fuzzy_match")
        self.key = key
        self.threshold = threshold

    def compute(self, trajectory: Trajectory, **kwargs: Any) -> float:
        ground_truth = kwargs.get("ground_truth") or kwargs.get(self.key)
        if ground_truth is None:
            return 0.0

        answer = trajectory.get_final_answer()
        if answer is None:
            return 0.0

        ratio = SequenceMatcher(None, str(answer).lower(), str(ground_truth).lower()).ratio()
        return ratio if ratio >= self.threshold else 0.0


class ContainsMatch(Reward):
    """Reward if the answer contains the ground truth (or vice versa)."""

    def __init__(self, key: str = "answer", weight: float = 1.0):
        super().__init__(weight=weight, name="contains_match")
        self.key = key

    def compute(self, trajectory: Trajectory, **kwargs: Any) -> float:
        ground_truth = kwargs.get("ground_truth") or kwargs.get(self.key)
        if ground_truth is None:
            return 0.0

        answer = trajectory.get_final_answer()
        if answer is None:
            return 0.0

        a = str(answer).lower().strip()
        g = str(ground_truth).lower().strip()
        return 1.0 if g in a or a in g else 0.0


# Factory functions for clean API
def exact_match(key: str = "answer", weight: float = 1.0, **kwargs: Any) -> ExactMatch:
    return ExactMatch(key=key, weight=weight, **kwargs)


def fuzzy_match(key: str = "answer", weight: float = 1.0, **kwargs: Any) -> FuzzyMatch:
    return FuzzyMatch(key=key, weight=weight, **kwargs)


def contains_match(key: str = "answer", weight: float = 1.0) -> ContainsMatch:
    return ContainsMatch(key=key, weight=weight)
