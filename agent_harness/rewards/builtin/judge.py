"""LLM-as-judge reward function."""

from __future__ import annotations

from typing import Any, Callable, Protocol

from agent_harness.core.trajectory import Trajectory
from agent_harness.rewards.base import Reward


class LLMClient(Protocol):
    """Protocol for LLM clients used in judge-based rewards."""

    def complete(self, prompt: str) -> str: ...


class LLMJudge(Reward):
    """Use an LLM as a reward judge.

    The LLM is prompted to rate the trajectory on a scale of 0-10,
    which is then normalized to [0, 1].

    Requires an LLM client to be provided at compute time via kwargs
    or at init time.
    """

    DEFAULT_PROMPT = """You are evaluating an AI agent's performance on a task.

Task: {task}

Agent's trajectory:
{trajectory}

Ground truth (if available): {ground_truth}

Rate the agent's performance on a scale of 0 to 10, where:
- 0: Completely wrong or harmful
- 5: Partially correct
- 10: Perfect execution

Respond with ONLY a number between 0 and 10."""

    def __init__(
        self,
        weight: float = 1.0,
        prompt_template: str | None = None,
        llm_client: LLMClient | None = None,
        parse_fn: Callable[[str], float] | None = None,
    ):
        super().__init__(weight=weight, name="llm_judge")
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT
        self.llm_client = llm_client
        self.parse_fn = parse_fn or self._default_parse

    @staticmethod
    def _default_parse(response: str) -> float:
        """Extract a number from the LLM response and normalize to [0, 1]."""
        import re

        numbers = re.findall(r"\d+\.?\d*", response.strip())
        if numbers:
            score = float(numbers[0])
            return max(0.0, min(1.0, score / 10.0))
        return 0.0

    def compute(self, trajectory: Trajectory, **kwargs: Any) -> float:
        client = kwargs.get("llm_client") or self.llm_client
        if client is None:
            return 0.0

        task_str = str(trajectory.task.get("prompt", trajectory.task))
        traj_str = "\n".join(
            f"Turn {t.turn_number}: {t.action.content or str(t.action.tool_calls)}"
            for t in trajectory.turns
        )
        ground_truth = kwargs.get("ground_truth", "N/A")

        prompt = self.prompt_template.format(
            task=task_str,
            trajectory=traj_str,
            ground_truth=ground_truth,
        )

        response = client.complete(prompt)
        return self.parse_fn(response)


def llm_judge(weight: float = 1.0, **kwargs: Any) -> LLMJudge:
    return LLMJudge(weight=weight, **kwargs)
