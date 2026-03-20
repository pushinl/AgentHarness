"""Tool-use reward functions."""

from __future__ import annotations

from typing import Any

from agent_harness.core.action import ActionType
from agent_harness.core.trajectory import Trajectory
from agent_harness.rewards.base import Reward


class ToolCallValid(Reward):
    """Reward for valid tool calls (correct name, parseable arguments).

    Checks that tool calls reference tools that exist in available_tools
    and that tool results are successful.
    """

    def __init__(self, weight: float = 1.0, available_tools: list[str] | None = None):
        super().__init__(weight=weight, name="tool_call_valid")
        self.available_tools = set(available_tools) if available_tools else None

    def compute(self, trajectory: Trajectory, **kwargs: Any) -> float:
        available = self.available_tools
        if available is None:
            available_list = kwargs.get("available_tools", [])
            available = set(available_list) if available_list else None

        tool_turns = [
            t for t in trajectory.turns if t.action.action_type == ActionType.TOOL_CALL
        ]
        if not tool_turns:
            return 1.0  # No tool calls needed = valid

        valid_count = 0
        total_count = 0

        for turn in tool_turns:
            for tc in turn.action.tool_calls:
                total_count += 1
                # Check tool name validity
                name_valid = available is None or tc.tool_name in available
                # Check if the tool result was successful
                result_ok = True
                for tr in turn.observation.tool_results:
                    if tr.call_id == tc.call_id and not tr.success:
                        result_ok = False
                if name_valid and result_ok:
                    valid_count += 1

        return valid_count / total_count if total_count > 0 else 1.0


class TrajectoryEfficiency(Reward):
    """Reward for completing the task in fewer turns.

    Returns 1.0 if done in 1 turn, linearly decreasing to 0.0 at max_turns.
    """

    def __init__(self, max_turns: int = 10, weight: float = 1.0):
        super().__init__(weight=weight, name="trajectory_efficiency")
        self.max_turns = max(1, max_turns)

    def compute(self, trajectory: Trajectory, **kwargs: Any) -> float:
        n = trajectory.num_turns
        if n <= 1:
            return 1.0
        if n >= self.max_turns:
            return 0.0
        return 1.0 - (n - 1) / (self.max_turns - 1)


class ToolUsageRate(Reward):
    """Reward for actually using tools (penalizes not using tools when available)."""

    def __init__(self, weight: float = 1.0, min_tool_calls: int = 1):
        super().__init__(weight=weight, name="tool_usage_rate")
        self.min_tool_calls = min_tool_calls

    def compute(self, trajectory: Trajectory, **kwargs: Any) -> float:
        tool_calls = sum(
            len(t.action.tool_calls)
            for t in trajectory.turns
            if t.action.action_type == ActionType.TOOL_CALL
        )
        if tool_calls >= self.min_tool_calls:
            return 1.0
        return tool_calls / self.min_tool_calls if self.min_tool_calls > 0 else 0.0


def tool_call_valid(weight: float = 1.0, **kwargs: Any) -> ToolCallValid:
    return ToolCallValid(weight=weight, **kwargs)


def trajectory_efficiency(max_turns: int = 10, weight: float = 1.0) -> TrajectoryEfficiency:
    return TrajectoryEfficiency(max_turns=max_turns, weight=weight)


def tool_usage_rate(weight: float = 1.0, **kwargs: Any) -> ToolUsageRate:
    return ToolUsageRate(weight=weight, **kwargs)
