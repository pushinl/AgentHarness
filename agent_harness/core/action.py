"""Action and Observation types for the AgentEnv protocol."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """Types of actions an agent can take."""

    TEXT = "text"
    TOOL_CALL = "tool_call"
    FINISH = "finish"


class ToolCall(BaseModel):
    """A single tool invocation."""

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    call_id: str = ""


class ToolResult(BaseModel):
    """Result returned from a tool execution."""

    call_id: str = ""
    tool_name: str = ""
    output: Any = None
    error: str | None = None
    is_error: bool = False

    @property
    def success(self) -> bool:
        return not self.is_error and self.error is None


class Action(BaseModel):
    """An action taken by the agent.

    Can be:
    - A text response (reasoning, final answer)
    - A tool call
    - A finish signal with final answer
    """

    action_type: ActionType = ActionType.TEXT
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def text(cls, content: str) -> Action:
        return cls(action_type=ActionType.TEXT, content=content)

    @classmethod
    def tool(cls, tool_name: str, arguments: dict[str, Any] | None = None, call_id: str = "") -> Action:
        tc = ToolCall(tool_name=tool_name, arguments=arguments or {}, call_id=call_id)
        return cls(action_type=ActionType.TOOL_CALL, tool_calls=[tc])

    @classmethod
    def finish(cls, answer: str) -> Action:
        return cls(action_type=ActionType.FINISH, content=answer)


class Observation(BaseModel):
    """An observation returned by the environment after a step.

    Contains the textual observation, optional tool results, and metadata.
    """

    content: str = ""
    tool_results: list[ToolResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    available_tools: list[str] = Field(default_factory=list)

    @classmethod
    def simple(cls, content: str) -> Observation:
        return cls(content=content)

    @classmethod
    def from_tool_result(cls, result: ToolResult) -> Observation:
        content = str(result.output) if result.success else f"Error: {result.error}"
        return cls(content=content, tool_results=[result])
