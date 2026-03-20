"""Tests for built-in environments."""

from agent_harness.core.action import Action, ActionType
from agent_harness.core.tool import ToolSpec
from agent_harness.envs.code_exec import CodeExecutionEnv
from agent_harness.envs.math import MathReasoningEnv
from agent_harness.envs.tool_call import ToolCallingEnv, ToolDef


# --- MathReasoningEnv ---

class TestMathReasoningEnv:
    def test_reset(self):
        env = MathReasoningEnv()
        obs = env.reset({"prompt": "What is 2+2?", "answer": "4"})
        assert "2+2" in obs.content
        assert len(obs.available_tools) > 0

    def test_calculator_tool(self):
        env = MathReasoningEnv()
        env.reset({"prompt": "What is 6*7?", "answer": "42"})
        action = Action.tool("calculator", {"expression": "6*7"})
        obs, done = env.step(action)
        assert "42" in obs.content
        assert done is False

    def test_calculator_error(self):
        env = MathReasoningEnv()
        env.reset({"prompt": "test"})
        action = Action.tool("calculator", {"expression": "invalid!@#"})
        obs, done = env.step(action)
        assert done is False

    def test_python_exec_tool(self):
        env = MathReasoningEnv()
        env.reset({"prompt": "Calculate factorial of 5", "answer": "120"})
        action = Action.tool("python_exec", {"code": "result = 1\nfor i in range(1,6): result *= i"})
        obs, done = env.step(action)
        assert done is False

    def test_finish_action(self):
        env = MathReasoningEnv()
        env.reset({"prompt": "What is 1+1?", "answer": "2"})
        obs, done = env.step(Action.finish("2"))
        assert done is True

    def test_max_turns(self):
        env = MathReasoningEnv(max_turns=2)
        env.reset({"prompt": "test"})
        env.step(Action.text("thinking..."))
        obs, done = env.step(Action.text("still thinking..."))
        assert done is True

    def test_get_ground_truth(self):
        env = MathReasoningEnv()
        env.reset({"prompt": "Q?", "answer": "42"})
        assert env.get_ground_truth() == "42"

    def test_get_available_tools(self):
        env = MathReasoningEnv(tools=["calculator"])
        tools = env.get_available_tools()
        assert len(tools) == 1
        assert tools[0].name == "calculator"

    def test_state_snapshot(self):
        env = MathReasoningEnv()
        env.reset({"prompt": "test"})
        env.step(Action.text("hi"))
        state = env.get_state_snapshot()
        assert state["turn_count"] == 1
        assert state["done"] is False

    def test_unavailable_tool(self):
        env = MathReasoningEnv(tools=["calculator"])
        env.reset({"prompt": "test"})
        action = Action.tool("python_exec", {"code": "print(1)"})
        obs, done = env.step(action)
        assert "not available" in obs.content.lower() or "error" in obs.content.lower()

    def test_disabled_tools(self):
        env = MathReasoningEnv(tools=[])
        assert len(env.get_available_tools()) == 0

    def test_already_done(self):
        env = MathReasoningEnv()
        env.reset({"prompt": "test"})
        env.step(Action.finish("done"))
        obs, done = env.step(Action.text("more"))
        assert done is True
        assert "already" in obs.content.lower()


# --- CodeExecutionEnv ---

class TestCodeExecutionEnv:
    def test_reset(self):
        env = CodeExecutionEnv()
        obs = env.reset({"prompt": "Write a function", "test_code": "assert True"})
        assert "function" in obs.content.lower()

    def test_run_code(self):
        env = CodeExecutionEnv()
        env.reset({"prompt": "test"})
        action = Action.tool("run_code", {"code": "print('hello')"})
        obs, done = env.step(action)
        assert "hello" in obs.content
        assert done is False

    def test_run_tests_pass(self):
        env = CodeExecutionEnv()
        env.reset({"prompt": "test", "test_code": "assert add(2, 3) == 5"})
        action = Action.tool("run_tests", {"code": "def add(a, b): return a + b"})
        obs, done = env.step(action)
        assert "passed" in obs.content.lower()

    def test_run_tests_fail(self):
        env = CodeExecutionEnv()
        env.reset({"prompt": "test", "test_code": "assert add(2, 3) == 5"})
        action = Action.tool("run_tests", {"code": "def add(a, b): return a - b"})
        obs, done = env.step(action)
        assert any(r.is_error for r in obs.tool_results)

    def test_finish(self):
        env = CodeExecutionEnv()
        env.reset({"prompt": "test"})
        obs, done = env.step(Action.finish("def foo(): pass"))
        assert done is True

    def test_get_tools(self):
        env = CodeExecutionEnv()
        tools = env.get_available_tools()
        names = {t.name for t in tools}
        assert "run_code" in names
        assert "run_tests" in names


# --- ToolCallingEnv ---

class TestToolCallingEnv:
    def _make_env(self):
        def search(query: str) -> str:
            return f"Results for: {query}"

        def calculate(expression: str) -> str:
            return str(eval(expression))  # noqa: S307

        tools = {
            "search": ToolDef(
                spec=ToolSpec(name="search", description="Search"),
                fn=search,
            ),
            "calculate": ToolDef(
                spec=ToolSpec(name="calculate", description="Calculate"),
                fn=calculate,
            ),
        }
        return ToolCallingEnv(tools=tools)

    def test_reset(self):
        env = self._make_env()
        obs = env.reset({"prompt": "Find info about Python"})
        assert "Python" in obs.content
        assert "search" in obs.available_tools

    def test_tool_call(self):
        env = self._make_env()
        env.reset({"prompt": "test"})
        obs, done = env.step(Action.tool("search", {"query": "hello"}))
        assert "Results for: hello" in obs.content
        assert done is False

    def test_unknown_tool(self):
        env = self._make_env()
        env.reset({"prompt": "test"})
        obs, done = env.step(Action.tool("nonexistent", {}))
        assert any(r.is_error for r in obs.tool_results)

    def test_tool_error(self):
        def bad_fn():
            raise ValueError("Something went wrong")

        env = ToolCallingEnv(tools={
            "bad": ToolDef(spec=ToolSpec(name="bad"), fn=bad_fn),
        })
        env.reset({"prompt": "test"})
        obs, done = env.step(Action.tool("bad", {}))
        assert any(r.is_error for r in obs.tool_results)

    def test_finish(self):
        env = self._make_env()
        env.reset({"prompt": "test"})
        obs, done = env.step(Action.finish("answer"))
        assert done is True

    def test_max_turns(self):
        env = ToolCallingEnv(max_turns=2)
        env.reset({"prompt": "test"})
        env.step(Action.text("a"))
        obs, done = env.step(Action.text("b"))
        assert done is True

    def test_register_tool(self):
        env = ToolCallingEnv()
        env.register_tool("greet", ToolDef(
            spec=ToolSpec(name="greet", description="Greet"),
            fn=lambda name: f"Hello, {name}!",
        ))
        env.reset({"prompt": "test"})
        obs, done = env.step(Action.tool("greet", {"name": "World"}))
        assert "Hello, World!" in obs.content

    def test_get_available_tools(self):
        env = self._make_env()
        tools = env.get_available_tools()
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert "search" in names
        assert "calculate" in names
