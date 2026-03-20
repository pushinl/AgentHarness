"""Tests for core protocol types."""

from agent_harness.core.action import Action, ActionType, Observation, ToolCall, ToolResult
from agent_harness.core.tool import ParameterType, ToolParameter, ToolSpec
from agent_harness.core.trajectory import Trajectory, Turn


class TestAction:
    def test_text_action(self):
        a = Action.text("Hello world")
        assert a.action_type == ActionType.TEXT
        assert a.content == "Hello world"
        assert a.tool_calls == []

    def test_tool_action(self):
        a = Action.tool("calculator", {"expr": "1+1"}, call_id="c1")
        assert a.action_type == ActionType.TOOL_CALL
        assert len(a.tool_calls) == 1
        assert a.tool_calls[0].tool_name == "calculator"
        assert a.tool_calls[0].arguments == {"expr": "1+1"}

    def test_finish_action(self):
        a = Action.finish("42")
        assert a.action_type == ActionType.FINISH
        assert a.content == "42"


class TestToolResult:
    def test_success(self):
        r = ToolResult(tool_name="calc", output="2")
        assert r.success is True

    def test_error(self):
        r = ToolResult(tool_name="calc", error="division by zero", is_error=True)
        assert r.success is False


class TestObservation:
    def test_simple(self):
        obs = Observation.simple("The answer is 42")
        assert obs.content == "The answer is 42"

    def test_from_tool_result(self):
        result = ToolResult(tool_name="calc", output="42")
        obs = Observation.from_tool_result(result)
        assert obs.content == "42"
        assert len(obs.tool_results) == 1

    def test_from_error_result(self):
        result = ToolResult(tool_name="calc", error="fail", is_error=True)
        obs = Observation.from_tool_result(result)
        assert "Error" in obs.content


class TestToolSpec:
    def test_to_openai_schema(self):
        spec = ToolSpec(
            name="search",
            description="Search the web",
            parameters=[
                ToolParameter(
                    name="query",
                    param_type=ParameterType.STRING,
                    description="Search query",
                    required=True,
                ),
                ToolParameter(
                    name="limit",
                    param_type=ParameterType.INTEGER,
                    description="Max results",
                    required=False,
                    default=10,
                ),
            ],
        )
        schema = spec.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "search"
        props = schema["function"]["parameters"]["properties"]
        assert "query" in props
        assert "limit" in props
        assert props["query"]["type"] == "string"
        assert schema["function"]["parameters"]["required"] == ["query"]


class TestTrajectory:
    def _make_trajectory(self) -> Trajectory:
        traj = Trajectory(task={"prompt": "What is 2+2?"}, env_name="test")
        traj.add_turn(
            action=Action.tool("calculator", {"expr": "2+2"}),
            observation=Observation.simple("4"),
            reward=0.5,
        )
        traj.add_turn(
            action=Action.finish("The answer is 4"),
            observation=Observation.simple(""),
            reward=1.0,
        )
        return traj

    def test_add_turn(self):
        traj = self._make_trajectory()
        assert traj.num_turns == 2
        assert traj.turns[0].turn_number == 0
        assert traj.turns[1].turn_number == 1

    def test_get_final_answer(self):
        traj = self._make_trajectory()
        assert traj.get_final_answer() == "The answer is 4"

    def test_to_messages(self):
        traj = self._make_trajectory()
        msgs = traj.to_messages()
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "What is 2+2?"

    def test_serialization_roundtrip(self):
        traj = self._make_trajectory()
        data = traj.to_dict()
        restored = Trajectory.from_dict(data)
        assert restored.num_turns == traj.num_turns
        assert restored.task == traj.task
        assert restored.env_name == traj.env_name
        assert restored.turns[0].action.tool_calls[0].tool_name == "calculator"

    def test_empty_trajectory(self):
        traj = Trajectory()
        assert traj.num_turns == 0
        assert traj.get_final_answer() is None
        assert traj.to_messages() == []

    def test_actions_and_observations(self):
        traj = self._make_trajectory()
        assert len(traj.actions) == 2
        assert len(traj.observations) == 2
        assert traj.actions[0].action_type == ActionType.TOOL_CALL
        assert traj.actions[1].action_type == ActionType.FINISH
