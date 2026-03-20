"""AgentHarness: The Universal Harness Engine for Agentic RL.

Harness any task. Compose any reward. Train any agent.
"""

__version__ = "0.1.0"

from agent_harness.core import AgentEnv, Action, Observation, Trajectory, ToolSpec
from agent_harness.rewards import Reward, RewardComposer, CreditAssigner
from agent_harness.harness import Harness
from agent_harness import envs, rewards, backends, debug, store

__all__ = [
    # Version
    "__version__",
    # Core
    "AgentEnv",
    "Action",
    "Observation",
    "Trajectory",
    "ToolSpec",
    # Rewards
    "Reward",
    "RewardComposer",
    "CreditAssigner",
    # Harness
    "Harness",
    # Submodules
    "backends",
    "debug",
    "envs",
    "rewards",
    "store",
]
