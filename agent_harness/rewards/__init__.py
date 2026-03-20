"""Reward engine — composable rewards, credit assignment, and builtins."""

from agent_harness.rewards.base import Reward, RewardComposer
from agent_harness.rewards.builtin.code import CodeExecutable, CodePassesTests, code_executable, code_passes_tests
from agent_harness.rewards.builtin.format import (
    FormatFollows,
    LengthPenalty,
    StructuredOutput,
    format_follows,
    length_penalty,
    structured_output,
)
from agent_harness.rewards.builtin.judge import LLMJudge, llm_judge
from agent_harness.rewards.builtin.match import (
    ContainsMatch,
    ExactMatch,
    FuzzyMatch,
    contains_match,
    exact_match,
    fuzzy_match,
)
from agent_harness.rewards.builtin.tool import (
    ToolCallValid,
    ToolUsageRate,
    TrajectoryEfficiency,
    tool_call_valid,
    tool_usage_rate,
    trajectory_efficiency,
)
from agent_harness.rewards.credit import CreditAssigner, CreditStrategy

__all__ = [
    # Base
    "Reward",
    "RewardComposer",
    # Credit
    "CreditAssigner",
    "CreditStrategy",
    # Match
    "ExactMatch",
    "FuzzyMatch",
    "ContainsMatch",
    "exact_match",
    "fuzzy_match",
    "contains_match",
    # Code
    "CodePassesTests",
    "CodeExecutable",
    "code_passes_tests",
    "code_executable",
    # Tool
    "ToolCallValid",
    "TrajectoryEfficiency",
    "ToolUsageRate",
    "tool_call_valid",
    "trajectory_efficiency",
    "tool_usage_rate",
    # Format
    "FormatFollows",
    "LengthPenalty",
    "StructuredOutput",
    "format_follows",
    "length_penalty",
    "structured_output",
    # Judge
    "LLMJudge",
    "llm_judge",
]
