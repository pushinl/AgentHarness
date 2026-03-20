"""Tests for the reward engine."""

from agent_harness.core.action import Action, ActionType, Observation, ToolCall, ToolResult
from agent_harness.core.trajectory import Trajectory
from agent_harness.rewards import (
    CreditAssigner,
    RewardComposer,
    contains_match,
    exact_match,
    format_follows,
    fuzzy_match,
    length_penalty,
    structured_output,
    tool_call_valid,
    tool_usage_rate,
    trajectory_efficiency,
)


def _make_math_trajectory(answer: str = "42", n_turns: int = 2) -> Trajectory:
    traj = Trajectory(task={"prompt": "What is 6*7?"})
    for i in range(n_turns - 1):
        traj.add_turn(
            action=Action.tool("calculator", {"expr": "6*7"}),
            observation=Observation.simple("42"),
        )
    traj.add_turn(
        action=Action.finish(answer),
        observation=Observation.simple(""),
    )
    return traj


def _make_tool_trajectory(tool_names: list[str], valid: list[bool]) -> Trajectory:
    traj = Trajectory(task={"prompt": "test"})
    for name, is_valid in zip(tool_names, valid):
        tc = ToolCall(tool_name=name, arguments={}, call_id=f"c_{name}")
        action = Action(action_type=ActionType.TOOL_CALL, tool_calls=[tc])
        tr = ToolResult(
            call_id=f"c_{name}",
            tool_name=name,
            output="ok" if is_valid else None,
            error=None if is_valid else "failed",
            is_error=not is_valid,
        )
        obs = Observation(content="ok" if is_valid else "error", tool_results=[tr])
        traj.add_turn(action=action, observation=obs)
    traj.add_turn(action=Action.finish("done"), observation=Observation.simple(""))
    return traj


# --- ExactMatch ---

class TestExactMatch:
    def test_correct_answer(self):
        traj = _make_math_trajectory("42")
        r = exact_match(key="answer")
        assert r.compute(traj, ground_truth="42") == 1.0

    def test_wrong_answer(self):
        traj = _make_math_trajectory("43")
        r = exact_match()
        assert r.compute(traj, ground_truth="42") == 0.0

    def test_case_insensitive(self):
        traj = _make_math_trajectory("Hello World")
        r = exact_match()
        assert r.compute(traj, ground_truth="hello world") == 1.0

    def test_no_ground_truth(self):
        traj = _make_math_trajectory("42")
        r = exact_match()
        assert r.compute(traj) == 0.0

    def test_extract_number(self):
        traj = _make_math_trajectory("The answer is 42.")
        r = exact_match(extract_number=True)
        assert r.compute(traj, ground_truth="42") == 1.0


# --- FuzzyMatch ---

class TestFuzzyMatch:
    def test_exact(self):
        traj = _make_math_trajectory("hello")
        r = fuzzy_match()
        assert r.compute(traj, ground_truth="hello") == 1.0

    def test_similar(self):
        traj = _make_math_trajectory("hello world")
        r = fuzzy_match()
        score = r.compute(traj, ground_truth="hello worlds")
        assert 0.8 < score < 1.0

    def test_threshold(self):
        traj = _make_math_trajectory("abc")
        r = fuzzy_match(threshold=0.9)
        score = r.compute(traj, ground_truth="xyz")
        assert score == 0.0


# --- ContainsMatch ---

class TestContainsMatch:
    def test_contains(self):
        traj = _make_math_trajectory("The answer is 42")
        r = contains_match()
        assert r.compute(traj, ground_truth="42") == 1.0

    def test_not_contains(self):
        traj = _make_math_trajectory("The answer is 43")
        r = contains_match()
        assert r.compute(traj, ground_truth="42") == 0.0


# --- ToolCallValid ---

class TestToolCallValid:
    def test_all_valid(self):
        traj = _make_tool_trajectory(["calc", "search"], [True, True])
        r = tool_call_valid()
        assert r.compute(traj) == 1.0

    def test_some_invalid(self):
        traj = _make_tool_trajectory(["calc", "search"], [True, False])
        r = tool_call_valid()
        assert r.compute(traj) == 0.5

    def test_no_tools(self):
        traj = _make_math_trajectory("42", n_turns=1)
        r = tool_call_valid()
        assert r.compute(traj) == 1.0

    def test_with_available_tools(self):
        traj = _make_tool_trajectory(["calc", "unknown"], [True, True])
        r = tool_call_valid(available_tools=["calc", "search"])
        score = r.compute(traj)
        assert score == 0.5  # "unknown" is not in available_tools


# --- TrajectoryEfficiency ---

class TestTrajectoryEfficiency:
    def test_one_turn(self):
        traj = _make_math_trajectory("42", n_turns=1)
        r = trajectory_efficiency(max_turns=10)
        assert r.compute(traj) == 1.0

    def test_max_turns(self):
        traj = _make_math_trajectory("42", n_turns=10)
        r = trajectory_efficiency(max_turns=10)
        assert r.compute(traj) == 0.0

    def test_mid_turns(self):
        traj = _make_math_trajectory("42", n_turns=5)
        r = trajectory_efficiency(max_turns=10)
        score = r.compute(traj)
        assert 0.0 < score < 1.0


# --- FormatFollows ---

class TestFormatFollows:
    def test_matches(self):
        traj = _make_math_trajectory("```python\nprint(42)\n```")
        r = format_follows(pattern=r"```python.*```")
        assert r.compute(traj) == 1.0

    def test_no_match(self):
        traj = _make_math_trajectory("just text")
        r = format_follows(pattern=r"```python.*```")
        assert r.compute(traj) == 0.0


# --- LengthPenalty ---

class TestLengthPenalty:
    def test_within_limit(self):
        traj = _make_math_trajectory("short")
        r = length_penalty(max_tokens=100)
        assert r.compute(traj) == 1.0

    def test_over_limit(self):
        traj = _make_math_trajectory("a" * 150)
        r = length_penalty(max_tokens=100)
        score = r.compute(traj)
        assert 0.0 < score < 1.0

    def test_way_over_limit(self):
        traj = _make_math_trajectory("a" * 300)
        r = length_penalty(max_tokens=100)
        score = r.compute(traj)
        assert score == 0.0


# --- StructuredOutput ---

class TestStructuredOutput:
    def test_valid_json(self):
        traj = _make_math_trajectory('{"answer": 42}')
        r = structured_output(format_type="json")
        assert r.compute(traj) == 1.0

    def test_invalid_json(self):
        traj = _make_math_trajectory("not json")
        r = structured_output(format_type="json")
        assert r.compute(traj) == 0.0

    def test_json_in_code_block(self):
        traj = _make_math_trajectory('```json\n{"answer": 42}\n```')
        r = structured_output(format_type="json")
        assert r.compute(traj) == 1.0


# --- RewardComposer ---

class TestRewardComposer:
    def test_weighted_sum(self):
        traj = _make_math_trajectory("42", n_turns=2)
        composer = RewardComposer([
            exact_match(weight=0.7),
            trajectory_efficiency(max_turns=10, weight=0.3),
        ])
        score = composer.compute(traj, ground_truth="42")
        # exact_match = 1.0, efficiency for 2 turns out of 10 ≈ 0.889
        assert 0.9 < score <= 1.0

    def test_breakdown(self):
        traj = _make_math_trajectory("42")
        composer = RewardComposer([
            exact_match(weight=0.5),
            trajectory_efficiency(max_turns=10, weight=0.5),
        ])
        breakdown = composer.compute_breakdown(traj, ground_truth="42")
        assert "exact_match" in breakdown
        assert "trajectory_efficiency" in breakdown
        assert breakdown["exact_match"] == 1.0

    def test_weighted_breakdown(self):
        traj = _make_math_trajectory("42")
        composer = RewardComposer([
            exact_match(weight=0.6),
            trajectory_efficiency(max_turns=10, weight=0.4),
        ])
        wb = composer.compute_weighted_breakdown(traj, ground_truth="42")
        total = sum(wb.values())
        assert abs(total - composer.compute(traj, ground_truth="42")) < 1e-6

    def test_empty_composer(self):
        composer = RewardComposer([])
        traj = _make_math_trajectory("42")
        assert composer.compute(traj) == 0.0


# --- CreditAssigner ---

class TestCreditAssigner:
    def test_outcome_only(self):
        traj = _make_math_trajectory("42", n_turns=3)
        traj.total_reward = 0.8
        assigner = CreditAssigner(strategy="outcome_only")
        rewards = assigner.assign(traj)
        assert len(rewards) == 3
        assert all(r == 0.8 for r in rewards)

    def test_outcome_with_fn(self):
        traj = _make_math_trajectory("42", n_turns=2)
        assigner = CreditAssigner(
            strategy="outcome_only",
            trajectory_reward_fn=exact_match(),
        )
        rewards = assigner.assign(traj, ground_truth="42")
        assert all(r == 1.0 for r in rewards)

    def test_hybrid(self):
        traj = _make_math_trajectory("42", n_turns=2)
        traj.total_reward = 1.0
        assigner = CreditAssigner(
            strategy="hybrid",
            turn_reward_fn=trajectory_efficiency(max_turns=5),
            turn_weight=0.3,
            trajectory_weight=0.7,
        )
        rewards = assigner.assign(traj)
        assert len(rewards) == 2
        # trajectory_weight * 1.0 + turn_weight * efficiency
        assert all(0.0 < r <= 1.0 for r in rewards)

    def test_apply_writes_to_turns(self):
        traj = _make_math_trajectory("42", n_turns=2)
        traj.total_reward = 0.5
        assigner = CreditAssigner(strategy="outcome_only")
        assigner.apply(traj)
        assert all(t.reward == 0.5 for t in traj.turns)

    def test_empty_trajectory(self):
        traj = Trajectory()
        assigner = CreditAssigner()
        assert assigner.assign(traj) == []
