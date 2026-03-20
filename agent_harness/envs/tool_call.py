"""ToolCallingEnv — generic environment for tool-use tasks."""

from __future__ import annotations

from typing import Any, Callable

from agent_harness.core.action import Action, ActionType, Observation, ToolResult
from agent_harness.core.env import AgentEnv
from agent_harness.core.tool import ToolSpec


class ToolCallingEnv(AgentEnv):
    """Generic environment where an agent uses tools to complete tasks.

    Users register tools as callables and the env dispatches calls.

    Example:
        def search(query: str) -> str:
            return f"Results for: {query}"

        env = ToolCallingEnv(tools={
            "search": ToolDef(
                spec=ToolSpec(name="search", description="Search the web"),
                fn=search,
            ),
        })
    """

    def __init__(
        self,
        tools: dict[str, ToolDef] | None = None,
        max_turns: int = 20,
    ):
        self._tools = tools or {}
        self._max_turns = max_turns
        self._current_task: dict[str, Any] = {}
        self._turn_count = 0
        self._done = False

    def reset(self, task: dict[str, Any]) -> Observation:
        self._current_task = task
        self._turn_count = 0
        self._done = False
        prompt = task.get("prompt", "")
        return Observation(
            content=prompt,
            available_tools=list(self._tools.keys()),
        )

    def step(self, action: Action) -> tuple[Observation, bool]:
        self._turn_count += 1

        if self._done:
            return Observation.simple("Episode ended."), True

        if self._turn_count >= self._max_turns:
            self._done = True
            return Observation.simple("Max turns reached."), True

        if action.action_type == ActionType.FINISH:
            self._done = True
            return Observation.simple("Done."), True

        if action.action_type == ActionType.TOOL_CALL:
            results = []
            for tc in action.tool_calls:
                if tc.tool_name in self._tools:
                    try:
                        output = self._tools[tc.tool_name].fn(**tc.arguments)
                        results.append(ToolResult(
                            call_id=tc.call_id,
                            tool_name=tc.tool_name,
                            output=output,
                        ))
                    except Exception as e:
                        results.append(ToolResult(
                            call_id=tc.call_id,
                            tool_name=tc.tool_name,
                            error=str(e),
                            is_error=True,
                        ))
                else:
                    results.append(ToolResult(
                        call_id=tc.call_id,
                        tool_name=tc.tool_name,
                        error=f"Unknown tool: {tc.tool_name}",
                        is_error=True,
                    ))
            content = "\n".join(
                str(r.output) if r.success else f"Error: {r.error}" for r in results
            )
            return Observation(content=content, tool_results=results), False

        return Observation.simple("Continue."), False

    def get_ground_truth(self) -> Any:
        return self._current_task.get("answer")

    def get_available_tools(self) -> list[ToolSpec]:
        return [td.spec for td in self._tools.values()]

    def get_state_snapshot(self) -> dict[str, Any]:
        return {
            "task": self._current_task,
            "turn_count": self._turn_count,
            "done": self._done,
        }

    def register_tool(self, name: str, tool_def: ToolDef) -> None:
        """Register a new tool at runtime."""
        self._tools[name] = tool_def

    @property
    def name(self) -> str:
        return "ToolCallingEnv"

    @property
    def max_turns(self) -> int:
        return self._max_turns


class ToolDef:
    """A tool definition: spec + callable implementation."""

    def __init__(self, spec: ToolSpec, fn: Callable[..., Any]):
        self.spec = spec
        self.fn = fn
