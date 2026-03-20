"""CodeExecutionEnv — environment for code generation and testing."""

from __future__ import annotations

import subprocess
import tempfile
from typing import Any

from agent_harness.core.action import Action, ActionType, Observation, ToolResult
from agent_harness.core.env import AgentEnv
from agent_harness.core.tool import ParameterType, ToolParameter, ToolSpec


class CodeExecutionEnv(AgentEnv):
    """Environment for code generation tasks with test verification.

    The agent generates code, optionally runs it, and submits.
    Supports Python execution in a sandboxed subprocess.

    Example:
        env = CodeExecutionEnv(dataset=[{
            "prompt": "Write a function that adds two numbers",
            "test_code": "assert add(2, 3) == 5",
            "answer": "def add(a, b): return a + b",
        }])
    """

    RUN_CODE_SPEC = ToolSpec(
        name="run_code",
        description="Execute Python code and return stdout/stderr",
        parameters=[
            ToolParameter(
                name="code",
                param_type=ParameterType.STRING,
                description="Python code to execute",
                required=True,
            ),
        ],
    )

    RUN_TESTS_SPEC = ToolSpec(
        name="run_tests",
        description="Run the test suite against submitted code",
        parameters=[
            ToolParameter(
                name="code",
                param_type=ParameterType.STRING,
                description="Code to test",
                required=True,
            ),
        ],
    )

    def __init__(
        self,
        dataset: list[dict[str, Any]] | None = None,
        timeout: int = 30,
        max_turns: int = 15,
    ):
        self.tasks = dataset or []
        self.timeout = timeout
        self._max_turns = max_turns
        self._current_task: dict[str, Any] = {}
        self._turn_count = 0
        self._done = False
        self._last_code: str = ""

    def reset(self, task: dict[str, Any]) -> Observation:
        self._current_task = task
        self._turn_count = 0
        self._done = False
        self._last_code = ""
        prompt = task.get("prompt", "")
        return Observation(
            content=prompt,
            available_tools=["run_code", "run_tests"],
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
            self._last_code = action.content
            return Observation.simple("Code submitted."), True

        if action.action_type == ActionType.TOOL_CALL:
            results = []
            for tc in action.tool_calls:
                result = self._execute_tool(tc.tool_name, tc.arguments)
                results.append(result)
            content = "\n".join(
                str(r.output) if r.success else f"Error: {r.error}" for r in results
            )
            return Observation(content=content, tool_results=results), False

        return Observation.simple("Continue."), False

    def _execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        code = arguments.get("code", "")

        if tool_name == "run_code":
            stdout, stderr, returncode = self._run_python(code)
            output = stdout if returncode == 0 else f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            return ToolResult(
                tool_name=tool_name,
                output=output,
                is_error=returncode != 0,
                error=stderr if returncode != 0 else None,
            )

        elif tool_name == "run_tests":
            test_code = self._current_task.get("test_code", "")
            full_code = f"{code}\n\n{test_code}"
            stdout, stderr, returncode = self._run_python(full_code)
            if returncode == 0:
                return ToolResult(tool_name=tool_name, output="All tests passed!")
            else:
                return ToolResult(
                    tool_name=tool_name,
                    output=f"Tests failed:\n{stderr}",
                    is_error=True,
                    error=stderr,
                )

        return ToolResult(tool_name=tool_name, error=f"Unknown tool: {tool_name}", is_error=True)

    def _run_python(self, code: str) -> tuple[str, str, int]:
        """Run Python code in a subprocess."""
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=True) as f:
                f.write(code)
                f.flush()
                result = subprocess.run(
                    ["python", f.name],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            return "", "Execution timed out", 1
        except Exception as e:
            return "", str(e), 1

    def get_ground_truth(self) -> Any:
        return self._current_task.get("answer")

    def get_available_tools(self) -> list[ToolSpec]:
        return [self.RUN_CODE_SPEC, self.RUN_TESTS_SPEC]

    def get_state_snapshot(self) -> dict[str, Any]:
        return {
            "task": self._current_task,
            "turn_count": self._turn_count,
            "done": self._done,
            "last_code": self._last_code,
        }

    @property
    def name(self) -> str:
        return "CodeExecutionEnv"

    @property
    def max_turns(self) -> int:
        return self._max_turns
