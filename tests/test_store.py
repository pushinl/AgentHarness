"""Tests for Trajectory Store and Curriculum Scheduler."""

import tempfile
from pathlib import Path

from agent_harness.core.action import Action, Observation
from agent_harness.core.trajectory import Trajectory
from agent_harness.store.curriculum import CurriculumScheduler, Stage
from agent_harness.store.trajectory import TrajectoryStore


def _make_traj(answer: str, reward: float, n_turns: int = 2, env: str = "test", success: bool = False) -> Trajectory:
    traj = Trajectory(task={"prompt": "Q?"}, total_reward=reward, env_name=env, success=success)
    for _ in range(n_turns - 1):
        traj.add_turn(Action.tool("calc", {}), Observation.simple("ok"))
    traj.add_turn(Action.finish(answer), Observation.simple(""))
    return traj


# --- TrajectoryStore ---

class TestTrajectoryStore:
    def test_add_and_len(self):
        store = TrajectoryStore()
        store.add(_make_traj("42", 1.0))
        store.add(_make_traj("43", 0.5))
        assert len(store) == 2

    def test_add_batch(self):
        store = TrajectoryStore()
        store.add_batch([_make_traj("a", 0.1), _make_traj("b", 0.2)])
        assert len(store) == 2

    def test_filter_by_reward(self):
        store = TrajectoryStore()
        store.add_batch([_make_traj("a", 0.3), _make_traj("b", 0.7), _make_traj("c", 0.9)])
        filtered = store.filter(min_reward=0.5)
        assert len(filtered) == 2

    def test_filter_by_turns(self):
        store = TrajectoryStore()
        store.add_batch([
            _make_traj("a", 0.5, n_turns=2),
            _make_traj("b", 0.5, n_turns=5),
            _make_traj("c", 0.5, n_turns=8),
        ])
        assert len(store.filter(max_turns=5)) == 2
        assert len(store.filter(min_turns=5)) == 2

    def test_filter_by_env(self):
        store = TrajectoryStore()
        store.add(_make_traj("a", 0.5, env="math"))
        store.add(_make_traj("b", 0.5, env="code"))
        assert len(store.filter(env_name="math")) == 1

    def test_filter_success_only(self):
        store = TrajectoryStore()
        store.add(_make_traj("a", 1.0, success=True))
        store.add(_make_traj("b", 0.0, success=False))
        assert len(store.filter(success_only=True)) == 1

    def test_filter_custom(self):
        store = TrajectoryStore()
        store.add_batch([_make_traj("short", 0.5), _make_traj("a very long answer", 0.5)])
        result = store.filter(custom_fn=lambda t: len(t.get_final_answer() or "") > 10)
        assert len(result) == 1

    def test_sample(self):
        store = TrajectoryStore()
        store.add_batch([_make_traj(str(i), 0.1 * i) for i in range(20)])
        sample = store.sample(5, seed=42)
        assert len(sample) == 5

    def test_sort_by_reward(self):
        store = TrajectoryStore()
        store.add_batch([_make_traj("a", 0.3), _make_traj("b", 0.9), _make_traj("c", 0.1)])
        sorted_trajs = store.sort_by_reward()
        assert sorted_trajs[0].total_reward == 0.9
        assert sorted_trajs[-1].total_reward == 0.1

    def test_statistics(self):
        store = TrajectoryStore()
        store.add_batch([
            _make_traj("a", 0.4, n_turns=2, success=True),
            _make_traj("b", 0.8, n_turns=4, success=False),
        ])
        stats = store.statistics()
        assert stats["count"] == 2
        assert abs(stats["reward_mean"] - 0.6) < 1e-9
        assert stats["success_rate"] == 0.5

    def test_empty_statistics(self):
        store = TrajectoryStore()
        stats = store.statistics()
        assert stats["count"] == 0

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = TrajectoryStore(tmpdir)
            store.add_batch([_make_traj("42", 1.0), _make_traj("43", 0.5)])
            store.save()

            loaded = TrajectoryStore.load(tmpdir)
            assert len(loaded) == 2
            assert loaded[0].get_final_answer() == "42"
            assert loaded[1].total_reward == 0.5

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = TrajectoryStore(tmpdir)
            original = _make_traj("hello", 0.7, n_turns=3, env="test_env", success=True)
            store.add(original)
            store.save()

            loaded = TrajectoryStore.load(tmpdir)
            restored = loaded[0]
            assert restored.get_final_answer() == "hello"
            assert restored.total_reward == 0.7
            assert restored.num_turns == 3
            assert restored.env_name == "test_env"
            assert restored.success is True

    def test_clear(self):
        store = TrajectoryStore()
        store.add_batch([_make_traj("a", 0.1) for _ in range(5)])
        assert len(store) == 5
        store.clear()
        assert len(store) == 0

    def test_iteration(self):
        store = TrajectoryStore()
        store.add_batch([_make_traj(str(i), 0.1) for i in range(3)])
        answers = [t.get_final_answer() for t in store]
        assert answers == ["0", "1", "2"]


# --- CurriculumScheduler ---

class TestCurriculumScheduler:
    def _default_stages(self) -> list[Stage]:
        return [
            Stage(name="easy", difficulty="easy", epochs=2, promotion_threshold=0.7),
            Stage(name="medium", difficulty="medium", epochs=3, promotion_threshold=0.8),
            Stage(name="hard", difficulty="hard", epochs=5, promotion_threshold=0.9),
        ]

    def test_initial_state(self):
        sched = CurriculumScheduler(stages=self._default_stages())
        assert sched.current_stage_index == 0
        assert sched.current_stage is not None
        assert sched.current_stage.name == "easy"
        assert not sched.is_complete

    def test_promotion_by_threshold(self):
        sched = CurriculumScheduler(stages=self._default_stages())
        # High reward → immediate promotion
        promoted = sched.update(mean_reward=0.8)
        assert promoted is True
        assert sched.current_stage.name == "medium"

    def test_promotion_by_epoch_exhaustion(self):
        sched = CurriculumScheduler(stages=self._default_stages())
        # Low rewards, but exhaust epochs
        sched.update(mean_reward=0.3)  # epoch 0
        promoted = sched.update(mean_reward=0.3)  # epoch 1 (exhausted)
        assert promoted is True
        assert sched.current_stage.name == "medium"

    def test_complete_all_stages(self):
        sched = CurriculumScheduler(stages=self._default_stages())
        # Fast-track through all stages
        sched.update(mean_reward=0.9)  # easy → medium
        sched.update(mean_reward=0.9)  # medium → hard
        sched.update(mean_reward=0.95)  # hard → complete
        assert sched.is_complete

    def test_progress(self):
        stages = [
            Stage(name="a", difficulty="easy", epochs=2, promotion_threshold=0.7),
            Stage(name="b", difficulty="hard", epochs=2, promotion_threshold=0.7),
        ]
        sched = CurriculumScheduler(stages=stages)
        assert sched.progress == 0.0
        sched.update(mean_reward=0.5)  # epoch 0 of stage a
        assert 0.0 < sched.progress < 1.0

    def test_get_current_tasks(self):
        sched = CurriculumScheduler(stages=self._default_stages())
        all_tasks = [
            {"prompt": "1+1", "difficulty": "easy"},
            {"prompt": "integral", "difficulty": "medium"},
            {"prompt": "proof", "difficulty": "hard"},
        ]
        tasks = sched.get_current_tasks(all_tasks)
        assert len(tasks) == 1
        assert tasks[0]["difficulty"] == "easy"

    def test_dict_stages(self):
        sched = CurriculumScheduler(stages=[
            {"difficulty": "easy", "epochs": 2, "promotion_threshold": 0.7},
            {"difficulty": "hard", "epochs": 3},
        ])
        assert sched.current_stage.difficulty == "easy"
        assert len(sched.stages) == 2

    def test_history(self):
        sched = CurriculumScheduler(stages=self._default_stages())
        sched.update(mean_reward=0.5)
        sched.update(mean_reward=0.8)
        history = sched.get_history()
        assert len(history) == 2
        assert history[0]["reward"] == 0.5

    def test_reset(self):
        sched = CurriculumScheduler(stages=self._default_stages())
        sched.update(mean_reward=0.9)
        sched.reset()
        assert sched.current_stage_index == 0
        assert sched.current_epoch == 0

    def test_empty_scheduler(self):
        sched = CurriculumScheduler()
        assert sched.is_complete
        assert sched.progress == 1.0
