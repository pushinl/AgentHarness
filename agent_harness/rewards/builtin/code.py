"""Code execution reward functions."""

from __future__ import annotations

import subprocess
import tempfile
from typing import Any

from agent_harness.core.trajectory import Trajectory
from agent_harness.rewards.base import Reward


class CodePassesTests(Reward):
    """Reward based on whether generated code passes test cases.

    Extracts code from the trajectory's final answer, writes it along with
    test code to a temp file, and executes it.
    """

    def __init__(
        self,
        weight: float = 1.0,
        timeout: int = 30,
        language: str = "python",
    ):
        super().__init__(weight=weight, name="code_passes_tests")
        self.timeout = timeout
        self.language = language

    def _extract_code(self, text: str) -> str:
        """Extract code from markdown code blocks or plain text."""
        import re

        patterns = [
            rf"```{self.language}\s*\n(.*?)```",
            r"```\s*\n(.*?)```",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                return matches[-1].strip()
        return text.strip()

    def compute(self, trajectory: Trajectory, **kwargs: Any) -> float:
        test_code = kwargs.get("test_code", "")
        if not test_code:
            return 0.0

        answer = trajectory.get_final_answer()
        if not answer:
            return 0.0

        code = self._extract_code(answer)
        full_code = f"{code}\n\n{test_code}"

        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=True) as f:
                f.write(full_code)
                f.flush()
                result = subprocess.run(
                    ["python", f.name],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                return 1.0 if result.returncode == 0 else 0.0
        except (subprocess.TimeoutExpired, Exception):
            return 0.0


class CodeExecutable(Reward):
    """Reward for whether generated code can be executed without errors."""

    def __init__(self, weight: float = 1.0, timeout: int = 10):
        super().__init__(weight=weight, name="code_executable")
        self.timeout = timeout

    def compute(self, trajectory: Trajectory, **kwargs: Any) -> float:
        answer = trajectory.get_final_answer()
        if not answer:
            return 0.0

        # Extract code
        import re

        patterns = [r"```python\s*\n(.*?)```", r"```\s*\n(.*?)```"]
        code = answer
        for pattern in patterns:
            matches = re.findall(pattern, answer, re.DOTALL)
            if matches:
                code = matches[-1].strip()
                break

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
                return 1.0 if result.returncode == 0 else 0.0
        except (subprocess.TimeoutExpired, Exception):
            return 0.0


def code_passes_tests(weight: float = 1.0, **kwargs: Any) -> CodePassesTests:
    return CodePassesTests(weight=weight, **kwargs)


def code_executable(weight: float = 1.0, **kwargs: Any) -> CodeExecutable:
    return CodeExecutable(weight=weight, **kwargs)
