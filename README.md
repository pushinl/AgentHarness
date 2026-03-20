<p align="center">
  <h1 align="center">AgentHarness</h1>
  <p align="center">
    <strong>Harness any task. Compose any reward. Train any agent.</strong>
  </p>
  <p align="center">
    The Universal Harness Engine for Agentic RL — bridging the gap between environments, rewards, and training frameworks.
  </p>
  <p align="center">
    English | <a href="./README_CN.md">中文</a>
  </p>
  <p align="center">
    <a href="#quick-start">Quick Start</a> &bull;
    <a href="#architecture">Architecture</a> &bull;
    <a href="#examples">Examples</a> &bull;
    <a href="#api-reference">API Reference</a> &bull;
    <a href="#contributing">Contributing</a>
  </p>
</p>

---

## Why AgentHarness?

The agentic RL ecosystem has three critical gaps:

```
  Training Frameworks          Environments / Benchmarks        Reward Engineering
  ─────────────────          ─────────────────────────        ──────────────────
  veRL, OpenRLHF, TRL        SWE-bench, WebArena, OSWorld     ad-hoc functions
                  \                    |                       /
                   \                   |                      /
                    ╳──── No unified Harness layer ────╳
```

**Every project re-invents** the wiring between environments, rewards, and training.
AgentHarness solves this once — a standard protocol layer that lets you:

- **Define any task** as a trainable environment in 5 methods
- **Compose rewards** declaratively, like building blocks
- **Debug reward hacking** before it ruins your training run
- **Plug into any trainer** (veRL, OpenRLHF, TRL) with one line

Think of it as **Gymnasium for Agentic RL**, plus a **Reward Engineering workbench**.

## Features

| Feature | Description |
|---------|-------------|
| **Environment Protocol** | Universal 5-method interface — any benchmark becomes trainable |
| **Reward Composer** | Declarative DSL to compose 12+ builtin reward signals |
| **Reward Debugger** | Visualize distributions, detect hacking, A/B compare rewards |
| **Credit Assignment** | Outcome-only, turn-level, or hybrid credit strategies |
| **Trajectory Store** | Collect, filter, persist (JSONL), and sample trajectories |
| **Curriculum Scheduler** | Automatic difficulty progression with promotion thresholds |
| **Training Backends** | Adapters for veRL, OpenRLHF, TRL (swap with one line) |
| **CLI** | `agent-harness info`, `stats`, `debug` commands |

## Installation

```bash
pip install agent-harness

# With dev tools
pip install agent-harness[dev]

# With all optional dependencies
pip install agent-harness[all]
```

**From source:**

```bash
git clone https://github.com/pushinl/AgentHarness.git
cd AgentHarness
pip install -e ".[dev]"
```

**Requirements:** Python 3.10+

## Quick Start

### 5-Minute Math Reasoning Pipeline

```python
import agent_harness as ah

# 1. Define tasks
tasks = [
    {"prompt": "What is 12 * 15?", "answer": "180"},
    {"prompt": "What is 99 + 101?", "answer": "200"},
    {"prompt": "What is 2**10?",    "answer": "1024"},
]

# 2. Create environment
env = ah.envs.MathReasoningEnv(tools=["calculator", "python_exec"], max_turns=6)

# 3. Compose reward (like building blocks)
reward = ah.RewardComposer([
    ah.rewards.exact_match(key="answer", weight=0.6, extract_number=True),
    ah.rewards.tool_call_valid(weight=0.15),
    ah.rewards.trajectory_efficiency(max_turns=6, weight=0.15),
    ah.rewards.length_penalty(max_tokens=200, weight=0.1),
])

# 4. Wire everything together
harness = ah.Harness(
    env=env,
    reward=reward,
    credit=ah.CreditAssigner("hybrid"),
    backend=ah.backends.VeRLBackend(model="Qwen/Qwen2.5-7B", algorithm="grpo"),
)

# 5. Define your agent (or use your LLM)
def my_agent(observation: str, available_tools: list[str]) -> ah.Action:
    # Your agent logic here...
    return ah.Action.tool("calculator", {"expression": "12*15"})

# 6. Train
harness.train(tasks=tasks, agent_fn=my_agent, iterations=50)

# 7. Debug rewards
from agent_harness.debug import RewardDebugger
debugger = RewardDebugger(reward)
report = debugger.analyze(harness.trajectories)
print(report.summary())
```

## Architecture

```
                            AgentHarness
┌────────────────────────────────────────────────────────────┐
│                                                            │
│  ┌──────────────┐  ┌───────────────┐  ┌────────────────┐  │
│  │  Environment  │  │    Reward     │  │   Trajectory   │  │
│  │   Protocol    │  │    Engine     │  │     Store      │  │
│  │              │  │               │  │                │  │
│  │  • AgentEnv  │  │  • Composer   │  │  • Collect     │  │
│  │  • MathEnv   │  │  • 12+ builtins│ │  • Filter      │  │
│  │  • CodeEnv   │  │  • Debugger   │  │  • Persist     │  │
│  │  • ToolEnv   │  │  • Credit     │  │  • Curriculum  │  │
│  └──────┬───────┘  └──────┬────────┘  └───────┬────────┘  │
│         │                 │                    │           │
│         └─────────────────┼────────────────────┘           │
│                           │                                │
│               ┌───────────▼───────────┐                    │
│               │    Harness Runtime    │                    │
│               │    (Orchestrator)     │                    │
│               └───────────┬───────────┘                    │
│                           │                                │
│  ┌────────────────────────▼─────────────────────────────┐  │
│  │              Training Backend Adapters                │  │
│  │   ┌──────┐   ┌──────────┐   ┌─────┐   ┌──────────┐  │  │
│  │   │ veRL │   │ OpenRLHF │   │ TRL │   │  Custom  │  │  │
│  │   └──────┘   └──────────┘   └─────┘   └──────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

## Core Concepts

### 1. Environment Protocol

Any task becomes trainable by implementing the `AgentEnv` protocol — just 5 methods:

```python
from agent_harness import AgentEnv, Action, Observation

class MyCustomEnv(AgentEnv):
    def reset(self, task: dict) -> Observation:
        """Initialize env with a task, return first observation."""
        ...

    def step(self, action: Action) -> tuple[Observation, bool]:
        """Execute action, return (observation, done)."""
        ...

    def get_ground_truth(self):
        """Return correct answer (for verifiable reward)."""
        ...

    def get_available_tools(self) -> list[ToolSpec]:
        """Return tools the agent can use."""
        ...

    def get_state_snapshot(self) -> dict:
        """Return env state (for debugging / reward computation)."""
        ...
```

**Built-in environments:**

| Environment | Description | Tools |
|-------------|-------------|-------|
| `MathReasoningEnv` | Math problem solving | `calculator`, `python_exec` |
| `CodeExecutionEnv` | Code generation + test verification | `run_code`, `run_tests` |
| `ToolCallingEnv` | Generic tool-use (register any callable) | User-defined |

### 2. Reward Composer

Compose multi-signal reward functions declaratively:

```python
reward = ah.RewardComposer([
    # Correctness
    ah.rewards.exact_match(key="answer", weight=0.5),
    ah.rewards.fuzzy_match(weight=0.1),

    # Tool use quality
    ah.rewards.tool_call_valid(weight=0.15),
    ah.rewards.trajectory_efficiency(max_turns=10, weight=0.1),

    # Output quality
    ah.rewards.format_follows(pattern=r"\d+", weight=0.05),
    ah.rewards.length_penalty(max_tokens=500, weight=0.05),
    ah.rewards.structured_output(format_type="json", weight=0.05),
])
```

**All 12 builtin reward functions:**

| Function | Signal | Description |
|----------|--------|-------------|
| `exact_match` | Correctness | Exact string/number match with ground truth |
| `fuzzy_match` | Correctness | Fuzzy similarity (SequenceMatcher) |
| `contains_match` | Correctness | Answer contains ground truth (or vice versa) |
| `code_passes_tests` | Correctness | Generated code passes unit tests |
| `code_executable` | Quality | Code runs without errors |
| `tool_call_valid` | Tool use | Tool calls reference valid tools and succeed |
| `tool_usage_rate` | Tool use | Penalizes not using available tools |
| `trajectory_efficiency` | Efficiency | Fewer turns = higher reward |
| `format_follows` | Format | Output matches a regex pattern |
| `length_penalty` | Format | Penalizes overly long outputs |
| `structured_output` | Format | Valid JSON/XML output |
| `llm_judge` | Quality | LLM-as-a-judge scoring (pluggable client) |

### 3. Reward Debugger

**AgentHarness's killer feature** — no other project provides reward debugging tools:

```python
from agent_harness.debug import RewardDebugger

debugger = RewardDebugger(reward)

# Full analysis with hacking detection
report = debugger.analyze(trajectories, ground_truth="42")
print(report.summary())
```

Output:

```
============================================================
              Reward Debug Report
============================================================
  Trajectories analyzed: 100
  Composite score: 0.682 (std=0.234)

  Component              Mean    Std    Min    Max Risk
  --------------------------------------------------------
  exact_match           0.620  0.488  0.000  1.000 Low
  tool_call_valid       0.950  0.080  0.500  1.000 High
  trajectory_efficiency 0.780  0.150  0.200  1.000 Low
  length_penalty        0.920  0.060  0.700  1.000 Medium

  Alerts:
  [ALERT] tool_call_valid: Near-saturated: 92% of trajectories score > 0.9
        -> Increase difficulty, tighten criteria, or reduce weight
  [WARN] length_penalty: Suspiciously low variance (std=0.0600)
        -> Check if reward is too easy to game
============================================================
```

**A/B compare two reward functions:**

```python
comparison = debugger.compare(reward_v1, reward_v2, trajectories)
print(comparison.summary())
# Shows: mean, std, correlation, ranking agreement, per-trajectory diff
```

### 4. Credit Assignment

Assign per-turn rewards with three strategies:

```python
# All turns get the same final reward
credit = ah.CreditAssigner("outcome_only")

# Each turn scored individually
credit = ah.CreditAssigner("turn_level", turn_reward_fn=my_reward)

# Weighted mix
credit = ah.CreditAssigner(
    "hybrid",
    turn_reward_fn=ah.rewards.trajectory_efficiency(max_turns=10),
    turn_weight=0.3,
    trajectory_weight=0.7,
)
```

### 5. Trajectory Store & Curriculum

```python
from agent_harness.store import TrajectoryStore, CurriculumScheduler

# Persist and query trajectories
store = TrajectoryStore("./my_trajectories")
store.add_batch(trajectories)
store.save()                                # JSONL format
good = store.filter(min_reward=0.5, max_turns=10)
stats = store.statistics()                  # mean, min, max, success_rate

# Curriculum learning
curriculum = CurriculumScheduler(stages=[
    {"difficulty": "easy",   "epochs": 2, "promotion_threshold": 0.7},
    {"difficulty": "medium", "epochs": 3, "promotion_threshold": 0.8},
    {"difficulty": "hard",   "epochs": 5},
])
```

### 6. Training Backends

Swap training frameworks with one line:

```python
from agent_harness.backends import VeRLBackend, OpenRLHFBackend, TRLBackend

harness = ah.Harness(
    env=env,
    reward=reward,
    backend=VeRLBackend(model="Qwen/Qwen2.5-7B", algorithm="grpo"),
    # backend=OpenRLHFBackend(model="meta-llama/Llama-3-8B"),
    # backend=TRLBackend(model="mistralai/Mistral-7B"),
)
```

## Examples

Three complete runnable examples in [`examples/`](examples/):

### Math Reasoning ([`examples/math_reasoning.py`](examples/math_reasoning.py))

Full pipeline: calculator tools → reward composition → training → debugging → persistence.

```bash
python examples/math_reasoning.py
```

```
[Phase 1] Collecting trajectories...
  ✓ What is 12 * 15?     agent=180   gt=180   reward=0.970
  ✓ What is 99 + 101?    agent=200   gt=200   reward=0.970
  ...
[Phase 2] Training (5 iterations, offline)...
[Phase 3] Reward Debug Report
[Phase 4] Evaluation — Success Rate: 62.5%
[Phase 5] Saved 8 trajectories
```

### Code Generation ([`examples/code_generation.py`](examples/code_generation.py))

Code env with test verification + A/B reward comparison.

```bash
python examples/code_generation.py
```

### Tool Calling + Curriculum ([`examples/tool_calling_curriculum.py`](examples/tool_calling_curriculum.py))

Custom tool registration + 3-stage curriculum (easy → medium → hard).

```bash
python examples/tool_calling_curriculum.py
```

## CLI

```bash
# Show available components
agent-harness info

# Trajectory store statistics
agent-harness stats ./trajectories

# Debug rewards on stored trajectories
agent-harness debug ./trajectories
```

## API Reference

### Top-level imports

```python
import agent_harness as ah

# Core types
ah.Action              # Agent action (text, tool_call, finish)
ah.Observation         # Environment observation
ah.Trajectory          # Full episode recording
ah.AgentEnv            # Environment protocol (abstract)
ah.ToolSpec            # Tool definition

# Reward
ah.Reward              # Base reward class
ah.RewardComposer      # Weighted reward composition
ah.CreditAssigner      # Per-turn credit assignment

# Orchestrator
ah.Harness             # Main training orchestrator

# Submodules
ah.envs                # Built-in environments
ah.rewards             # Reward functions
ah.backends            # Training backend adapters
ah.debug               # Reward debugger
ah.store               # Trajectory store + curriculum
```

### Harness

```python
harness = ah.Harness(env, reward, credit?, store?, curriculum?, backend?, config?)

# Collect trajectories
traj  = harness.collect_trajectory(task, agent_fn)
batch = harness.collect_batch(tasks, agent_fn)

# Train
history = harness.train(tasks?, agent_fn?, iterations=10, batch_size=8, trajectories?)

# Evaluate
metrics = harness.evaluate(tasks, agent_fn)
# → {"mean_reward", "success_rate", "mean_turns", "num_tasks"}

# Stats
stats = harness.get_stats()
```

### Agent Function Signature

```python
def my_agent(observation: str, available_tools: list[str]) -> ah.Action:
    """
    Args:
        observation: Text from the environment.
        available_tools: Names of tools the agent can call.
    Returns:
        An Action — one of:
          ah.Action.text("reasoning...")
          ah.Action.tool("tool_name", {"arg": "value"})
          ah.Action.finish("final answer")
    """
```

## Project Structure

```
agent_harness/
├── __init__.py              # Public API
├── harness.py               # Harness orchestrator
├── core/
│   ├── action.py            # Action, Observation, ToolCall, ToolResult
│   ├── env.py               # AgentEnv protocol
│   ├── tool.py              # ToolSpec, ToolParameter
│   └── trajectory.py        # Trajectory, Turn
├── rewards/
│   ├── base.py              # Reward, RewardComposer
│   ├── credit.py            # CreditAssigner
│   └── builtin/             # 12 builtin reward functions
│       ├── match.py         # exact_match, fuzzy_match, contains_match
│       ├── code.py          # code_passes_tests, code_executable
│       ├── tool.py          # tool_call_valid, trajectory_efficiency, tool_usage_rate
│       ├── format.py        # format_follows, length_penalty, structured_output
│       └── judge.py         # llm_judge
├── envs/
│   ├── math.py              # MathReasoningEnv
│   ├── code_exec.py         # CodeExecutionEnv
│   └── tool_call.py         # ToolCallingEnv, ToolDef
├── store/
│   ├── trajectory.py        # TrajectoryStore
│   └── curriculum.py        # CurriculumScheduler
├── backends/
│   └── adapter.py           # TrainingBackend, VeRL, OpenRLHF, TRL, Dummy
├── debug/
│   └── debugger.py          # RewardDebugger, hacking detection, A/B comparison
└── cli/
    └── main.py              # CLI entry point

examples/
├── math_reasoning.py        # Full math pipeline
├── code_generation.py       # Code gen + A/B comparison
└── tool_calling_curriculum.py # Custom tools + curriculum

tests/                       # 165 tests (unit + integration)
├── test_core.py
├── test_rewards.py
├── test_debugger.py
├── test_envs.py
├── test_store.py
├── test_harness.py
└── test_integration.py      # 36 end-to-end integration tests
```

## Comparison with Existing Projects

| Capability | AgentHarness | Agent-R1 | ToolRL | GEM | reward-composer |
|---|---|---|---|---|---|
| Standard env protocol | **Yes** | No (veRL-coupled) | No | Partial | No |
| Composable rewards | **12 builtins** | Ad-hoc | Ad-hoc | No | Basic |
| Reward debugging | **Yes** | No | No | No | No |
| Hacking detection | **Yes** | No | No | No | No |
| A/B reward comparison | **Yes** | No | No | No | No |
| Credit assignment | **3 strategies** | Outcome only | Outcome only | No | No |
| Training-agnostic | **Yes** | veRL only | veRL only | N/A | N/A |
| Trajectory persistence | **Yes** | No | No | No | No |
| Curriculum learning | **Yes** | No | No | No | No |
| CLI tools | **Yes** | No | No | No | No |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run only integration tests
pytest tests/test_integration.py -v

# Lint
ruff check agent_harness/
```

## Roadmap

- [ ] Async environment execution (Ray-based parallel collection)
- [ ] Web-based Reward Debugger UI (Gradio)
- [ ] Parquet trajectory storage for large-scale datasets
- [ ] Full veRL integration with live training
- [ ] WebBrowsingEnv (Playwright), SQLQueryEnv, FileEditingEnv
- [ ] Reward hacking detection with historical trend analysis
- [ ] arXiv technical report

## Citation

```bibtex
@software{agentharness2025,
  title  = {AgentHarness: The Universal Harness Engine for Agentic RL},
  author = {pushinl},
  url    = {https://github.com/pushinl/AgentHarness},
  year   = {2025},
}
```

## License

[Apache-2.0](LICENSE)
