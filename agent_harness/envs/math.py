"""MathReasoningEnv — environment for math problem solving with tools."""

from __future__ import annotations

import re
from typing import Any

from agent_harness.core.action import Action, ActionType, Observation, ToolResult
from agent_harness.core.env import AgentEnv
from agent_harness.core.tool import ParameterType, ToolParameter, ToolSpec


class MathReasoningEnv(AgentEnv):
    """Environment for math reasoning tasks.

    Supports tool-assisted solving with calculator and Python execution.

    Example:
        env = MathReasoningEnv(dataset=[
            {"prompt": "What is 2+2?", "answer": "4"},
        ])
        obs = env.reset(env.tasks[0])
        obs, done = env.step(Action.tool("calculator", {"expression": "2+2"}))
        obs, done = env.step(Action.finish("4"))
    """

    CALCULATOR_SPEC = ToolSpec(
        name="calculator",
        description="Evaluate a mathematical expression",
        parameters=[
            ToolParameter(
                name="expression",
                param_type=ParameterType.STRING,
                description="Mathematical expression to evaluate (e.g., '2+2', '3**2')",
                required=True,
            ),
        ],
    )

    PYTHON_EXEC_SPEC = ToolSpec(
        name="python_exec",
        description="Execute Python code and return the result",
        parameters=[
            ToolParameter(
                name="code",
                param_type=ParameterType.STRING,
                description="Python code to execute",
                required=True,
            ),
        ],
    )

    def __init__(
        self,
        dataset: list[dict[str, Any]] | None = None,
        tools: list[str] | None = None,
        max_turns: int = 10,
    ):
        self.tasks = dataset or []
        self._enabled_tools = set(tools) if tools is not None else {"calculator", "python_exec"}
        self._max_turns = max_turns
        self._current_task: dict[str, Any] = {}
        self._turn_count = 0
        self._done = False

    def reset(self, task: dict[str, Any]) -> Observation:
        self._current_task = task
        self._turn_count = 0
        self._done = False
        prompt = task.get("prompt", task.get("question", ""))
        tools = [t.name for t in self.get_available_tools()]
        return Observation(
            content=prompt,
            available_tools=tools,
        )

    def step(self, action: Action) -> tuple[Observation, bool]:
        self._turn_count += 1

        if self._done:
            return Observation.simple("Episode already ended."), True

        if self._turn_count >= self._max_turns:
            self._done = True
            return Observation.simple("Max turns reached."), True

        if action.action_type == ActionType.FINISH:
            self._done = True
            return Observation.simple("Answer submitted."), True

        if action.action_type == ActionType.TOOL_CALL:
            results = []
            for tc in action.tool_calls:
                result = self._execute_tool(tc.tool_name, tc.arguments)
                results.append(ToolResult(
                    call_id=tc.call_id,
                    tool_name=tc.tool_name,
                    output=result if not isinstance(result, Exception) else None,
                    error=str(result) if isinstance(result, Exception) else None,
                    is_error=isinstance(result, Exception),
                ))
            content = "\n".join(
                str(r.output) if r.success else f"Error: {r.error}" for r in results
            )
            return Observation(content=content, tool_results=results), False

        # Text action — just acknowledge
        return Observation.simple("Continue reasoning."), False

    def _execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if tool_name not in self._enabled_tools:
            return Exception(f"Tool '{tool_name}' is not available")

        if tool_name == "calculator":
            return self._run_calculator(arguments.get("expression", ""))
        elif tool_name == "python_exec":
            return self._run_python(arguments.get("code", ""))
        else:
            return Exception(f"Unknown tool: {tool_name}")

    def _run_calculator(self, expression: str) -> Any:
        """Evaluate a math expression safely."""
        try:
            # Only allow safe math characters
            safe = re.sub(r"[^0-9+\-*/().%** ]", "", expression)
            if not safe:
                return Exception("Empty expression")
            result = eval(safe, {"__builtins__": {}}, {})  # noqa: S307
            return result
        except Exception as e:
            return Exception(f"Calculator error: {e}")

    def _run_python(self, code: str) -> Any:
        """Execute Python code in a restricted namespace."""
        import math

        namespace: dict[str, Any] = {"math": math, "__builtins__": {"print": print, "range": range, "len": len, "sum": sum, "abs": abs, "int": int, "float": float, "str": str, "round": round}}
        try:
            exec(code, namespace)  # noqa: S102
            return namespace.get("result", "Code executed successfully")
        except Exception as e:
            return Exception(f"Python error: {e}")

    def get_ground_truth(self) -> Any:
        return self._current_task.get("answer")

    def get_available_tools(self) -> list[ToolSpec]:
        tools = []
        if "calculator" in self._enabled_tools:
            tools.append(self.CALCULATOR_SPEC)
        if "python_exec" in self._enabled_tools:
            tools.append(self.PYTHON_EXEC_SPEC)
        return tools

    def get_state_snapshot(self) -> dict[str, Any]:
        return {
            "task": self._current_task,
            "turn_count": self._turn_count,
            "done": self._done,
        }

    @property
    def name(self) -> str:
        return "MathReasoningEnv"

    @property
    def max_turns(self) -> int:
        return self._max_turns
