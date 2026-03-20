# AgentHarness

> **Harness any task. Compose any reward. Train any agent.**

The Universal Harness Engine for Agentic RL — bridging the gap between environments, rewards, and training frameworks.

## Installation

```bash
pip install agent-harness
```

## Quick Start

```python
import agent_harness as ah

# 1. Define environment
env = ah.envs.MathReasoningEnv(
    dataset="gsm8k",
    tools=["calculator", "python_exec"],
)

# 2. Compose reward
reward = ah.RewardComposer([
    ah.rewards.exact_match(key="answer", weight=0.7),
    ah.rewards.tool_call_valid(weight=0.15),
    ah.rewards.trajectory_efficiency(max_turns=8, weight=0.15),
])

# 3. Configure and train
harness = ah.Harness(
    env=env,
    reward=reward,
    credit=ah.CreditAssigner("hybrid"),
    backend=ah.backends.veRL(model="Qwen/Qwen2.5-7B", algorithm="grpo"),
)
harness.train(iterations=50)

# 4. Debug reward
ah.debug.RewardDebugger(reward).visualize(harness.trajectories)
```

## License

Apache-2.0
