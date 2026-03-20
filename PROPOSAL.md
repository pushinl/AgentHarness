# AgentHarness: The Universal Harness Engine for Agentic RL

## 一句话定位

> **连接 Environment、Reward、Training 三大断层的统一 Harness 层 —— 让任何任务 5 分钟变成可训练的 Agent RL 环境。**

---

## 1. 为什么这个项目目前没人做？（Gap Analysis）

### 当前生态地图

```
┌─────────────────────────────────────────────────────────┐
│                    现有开源项目分布                        │
├──────────────┬──────────────┬───────────────────────────┤
│  训练框架层   │  环境/评测层   │  Reward 工程层             │
│              │              │                           │
│  veRL ★4.9k  │  SWE-bench   │  reward-composer (基础)    │
│  OpenRLHF    │  WebArena    │  ToolRL (特定场景)          │
│  TRL ★11k   │  OSWorld     │  Text2Reward (机器人)       │
│  Agent-R1    │  AgentBench  │                           │
│  ToolBrain   │  GEM         │  ← 极度碎片化，无统一方案    │
│              │  MATH/GSM8K  │                           │
└──────┬───────┴──────┬───────┴───────────┬───────────────┘
       │              │                   │
       │      ❌ 没有统一的 Harness 层 ❌    │
       │              │                   │
       └──────────────┴───────────────────┘
```

### 三大断层（核心痛点）

| 断层 | 痛点 | 现状 |
|------|------|------|
| **Environment → Reward** | 每个环境的 reward 都要从零写，无法复用 | 大家都在写 ad-hoc reward functions |
| **Reward → Training** | reward signal 格式不统一，换训练框架要重写 | veRL/OpenRLHF/TRL 各有私有接口 |
| **Environment → Training** | trajectory 收集、格式、replay 无标准 | 每个项目自己实现一套 |

### 竞品不足分析

| 项目 | Stars | 做了什么 | 没做什么 |
|------|-------|---------|---------|
| **Agent-R1** | ~2k | 端到端 agent RL | 强耦合 veRL，不可拆分复用 |
| **ToolRL** | ~800 | Tool-use 的 RL 训练 | 只关注 tool calling 场景 |
| **GEM** | ~200 | Gym-like 环境接口 | 没有 reward 工程，没有训练集成 |
| **reward-composer** | ~100 | GRPO 的可组合 reward | 只做了 reward 拼接，没有环境和训练 |
| **Agent Lightning** | ~12k | 给已有 agent 加 RL | 闭环在微软自己生态内 |

**关键洞察**：没有一个项目同时解决 Environment 标准化 + Reward 可组合 + Training 可插拔 + Trajectory 管理这四件事。

---

## 2. AgentHarness 架构设计

### 核心理念

```
"任何任务都可以被 harness 化" —— 就像 pytest 之于测试、Gymnasium 之于传统 RL
```

### 整体架构

```
                          AgentHarness
┌──────────────────────────────────────────────────────┐
│                                                      │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │  Environment │  │   Reward     │  │  Trajectory │ │
│  │  Protocol    │  │   Engine     │  │  Store      │ │
│  │             │  │              │  │             │ │
│  │ • EnvSpec   │  │ • Composable │  │ • Collect   │ │
│  │ • Sandbox   │  │ • Verifiable │  │ • Replay    │ │
│  │ • Multi-turn│  │ • Debuggable │  │ • Filter    │ │
│  │ • Async     │  │ • Per-turn   │  │ • Curriculum│ │
│  └──────┬──────┘  └──────┬───────┘  └──────┬──────┘ │
│         │                │                 │        │
│         └────────────────┼─────────────────┘        │
│                          │                          │
│              ┌───────────▼──────────┐               │
│              │   Harness Runtime    │               │
│              │  (Orchestrator)      │               │
│              └───────────┬──────────┘               │
│                          │                          │
│  ┌───────────────────────▼───────────────────────┐  │
│  │           Training Backend Adapters           │  │
│  │  ┌──────┐  ┌────────┐  ┌─────┐  ┌─────────┐  │  │
│  │  │ veRL │  │OpenRLHF│  │ TRL │  │ Custom  │  │  │
│  │  └──────┘  └────────┘  └─────┘  └─────────┘  │  │
│  └───────────────────────────────────────────────┘  │
│                                                      │
└──────────────────────────────────────────────────────┘
```

### 2.1 Environment Protocol（环境协议层）

**核心创新**：定义一个 `AgentEnv` 协议，任何任务 benchmark 只需实现 5 个方法就能接入。

```python
from agent_harness import AgentEnv, Observation, Action

class MyTaskEnv(AgentEnv):
    """任何任务只需实现这个协议"""

    def reset(self, task: dict) -> Observation:
        """初始化环境，返回初始观察"""

    def step(self, action: Action) -> tuple[Observation, bool]:
        """执行动作，返回 (观察, 是否结束)"""

    def get_ground_truth(self) -> Any:
        """返回 ground truth（可选，用于 verifiable reward）"""

    def get_available_tools(self) -> list[ToolSpec]:
        """返回当前可用工具列表"""

    def get_state_snapshot(self) -> dict:
        """返回环境状态快照（用于 reward 计算和调试）"""
```

**预置环境适配器**（开箱即用）：

```python
from agent_harness.envs import (
    CodeExecutionEnv,    # 代码执行 + 单测验证
    WebBrowsingEnv,      # 网页浏览 (接入 Playwright)
    ToolCallingEnv,      # 通用 tool-use
    MathReasoningEnv,    # 数学推理
    ShellCommandEnv,     # 终端操作
    SQLQueryEnv,         # 数据库查询
    APIInteractionEnv,   # API 调用
    FileEditingEnv,      # 文件编辑（类 SWE-bench）
)
```

### 2.2 Reward Engine（奖励引擎 —— 核心差异化）

**这是整个项目最大的创新点。**

#### 2.2.1 声明式 Reward DSL

```python
from agent_harness.rewards import (
    Reward, RewardComposer,
    exact_match, fuzzy_match, code_passes_tests,
    tool_call_valid, format_follows, length_penalty,
    llm_judge, trajectory_efficiency,
)

# 像搭积木一样组合 reward
reward = RewardComposer([
    # 结果正确性 (权重 0.6)
    exact_match(key="answer", weight=0.6),

    # 工具使用质量 (权重 0.2)
    tool_call_valid(weight=0.1),
    trajectory_efficiency(max_turns=10, weight=0.1),

    # 格式和质量 (权重 0.2)
    format_follows(pattern=r"```python.*```", weight=0.1),
    length_penalty(max_tokens=2000, weight=0.1),
])
```

#### 2.2.2 多级 Credit Assignment（多级信用分配）

```python
from agent_harness.rewards import CreditAssigner

assigner = CreditAssigner(
    strategy="hybrid",  # "outcome_only" | "turn_level" | "hybrid"

    # Turn-level 策略
    turn_rewards={
        "tool_success": 0.2,      # 工具调用成功
        "info_gain": "auto",       # 自动计算信息增益 (IGPO)
        "critic_score": "model",   # 用 critic model 打分 (CriticSearch)
    },

    # Trajectory-level 策略
    trajectory_reward="final_answer_correct",

    # 混合权重
    turn_weight=0.3,
    trajectory_weight=0.7,
)
```

#### 2.2.3 Reward Debugger（奖励调试器 —— 独家功能）

```python
from agent_harness.debug import RewardDebugger

debugger = RewardDebugger(reward_fn=reward)

# 在已有 trajectory 上可视化 reward 分布
debugger.visualize(trajectories, output="reward_report.html")

# 检测 reward hacking
debugger.detect_hacking(trajectories)

# A/B 对比两个 reward function
debugger.compare(reward_v1, reward_v2, trajectories)
```

输出示例：

```
╔══════════════════════════════════════════════════════╗
║              Reward Debug Report                     ║
╠══════════════════════════════════════════════════════╣
║ Component         │ Mean  │ Std   │ Hacking Risk    ║
║───────────────────┼───────┼───────┼─────────────────║
║ exact_match       │ 0.42  │ 0.49  │ Low             ║
║ tool_call_valid   │ 0.87  │ 0.15  │ Medium          ║
║ format_follows    │ 0.95  │ 0.08  │ High (!)        ║
║ length_penalty    │ 0.71  │ 0.22  │ Low             ║
╠══════════════════════════════════════════════════════╣
║ Alert: format_follows reward is near-saturated       ║
║ (95% of trajectories score > 0.9)                    ║
║ Suggestion: increase difficulty or reduce weight     ║
╚══════════════════════════════════════════════════════╝
```

### 2.3 Trajectory Store（轨迹管理）

```python
from agent_harness.trajectory import TrajectoryStore, CurriculumScheduler

# 统一的 trajectory 格式
store = TrajectoryStore("./trajectories")

# 收集
store.collect(env, agent, n_episodes=1000)

# 过滤
good_trajectories = store.filter(min_reward=0.5, max_turns=15)

# 课程学习调度
curriculum = CurriculumScheduler(
    stages=[
        {"difficulty": "easy", "epochs": 2},
        {"difficulty": "medium", "epochs": 3},
        {"difficulty": "hard", "epochs": 5},
    ],
    promotion_threshold=0.7,
)
```

### 2.4 Training Backend Adapters（训练后端适配）

```python
from agent_harness.backends import veRLBackend, OpenRLHFBackend, TRLBackend

# 一行代码切换训练框架
harness = AgentHarness(
    env=MyTaskEnv(),
    reward=reward,
    assigner=assigner,
    store=store,
    backend=veRLBackend(
        algorithm="grpo",
        model="Qwen/Qwen2.5-7B",
        num_gpus=4,
    ),
)

# 开始训练
harness.train(num_iterations=100)
```

---

## 3. Slogan

> **"Harness any task. Compose any reward. Train any agent."**

---

## 4. 为什么能成为高星项目？

### 4.1 受众分析

```
目标受众 (按优先级):
1. 学术研究者 — 需要快速搭建 agentic RL 实验
2. 企业 AI 团队 — 需要可复用的 agent 训练基础设施
3. 开源开发者 — 需要标准化的 agent 环境接口
4. Benchmark 作者 — 需要让自己的 benchmark 可训练
```

### 4.2 差异化竞争力

```
                    AgentHarness 独有价值
                    ═══════════════════

  1. Reward Debugger        → 没有任何项目做过 reward 可视化调试
  2. 声明式 Reward DSL      → 比 reward-composer 更完整、更强大
  3. 统一 Environment 协议   → 比 GEM 更标准化、适配更多场景
  4. 训练框架无关            → 不绑定任何特定 RL 框架
  5. Trajectory 课程学习     → 内置 curriculum scheduling
  6. Hacking 检测           → 自动检测 reward hacking 风险
```

### 4.3 星星增长策略

| 阶段 | 时间 | 目标 | 关键动作 |
|------|------|------|---------|
| **v0.1** | Month 1-2 | 核心协议 + Reward Engine | 发布 Environment Protocol 和 Reward DSL |
| **v0.2** | Month 3-4 | 预置环境 + 调试器 | 支持 5+ 预置环境，Reward Debugger |
| **v0.3** | Month 5-6 | 训练集成 | 对接 veRL + OpenRLHF |
| **v1.0** | Month 7-8 | 完整发布 | 论文 + Tutorial + Benchmark 复现 |

### 4.4 星星增长引擎

1. **论文驱动**：发 arXiv 技术报告，学术圈自发传播
2. **教程驱动**：复现 DeepSeek-R1 / Agent-R1 的训练用我们的框架，写教程
3. **生态驱动**：让 SWE-bench / WebArena 等 benchmark 官方推荐使用
4. **痛点驱动**：Reward Debugger 解决真实痛点，自然口碑传播

---

## 5. 技术栈

| 组件 | 技术 | 理由 |
|------|------|------|
| 核心语言 | Python 3.10+ | 生态兼容 |
| 异步运行 | asyncio + Ray | 大规模并行环境 |
| 环境沙箱 | Docker / E2B | 安全隔离 |
| 存储 | Arrow / Parquet | 高效 trajectory 存储 |
| 可视化 | Rich (CLI) + Gradio (Web) | 调试界面 |
| 测试 | pytest + hypothesis | 属性测试 |
| 文档 | MkDocs Material | 现代文档 |

---

## 6. MVP 目录结构（v0.1）

```
agent_harness/
├── __init__.py
├── core/
│   ├── env.py          # AgentEnv protocol
│   ├── action.py       # Action / Observation 类型
│   ├── tool.py         # ToolSpec 定义
│   └── trajectory.py   # Trajectory 数据结构
├── rewards/
│   ├── base.py         # Reward 基类
│   ├── composer.py     # RewardComposer
│   ├── credit.py       # CreditAssigner
│   └── builtin/
│       ├── match.py    # exact_match, fuzzy_match
│       ├── code.py     # code_passes_tests
│       ├── tool.py     # tool_call_valid
│       ├── format.py   # format_follows
│       └── judge.py    # llm_judge
├── envs/
│   ├── code_exec.py    # CodeExecutionEnv
│   ├── math.py         # MathReasoningEnv
│   └── tool_call.py    # ToolCallingEnv
├── store/
│   ├── trajectory.py   # TrajectoryStore
│   └── curriculum.py   # CurriculumScheduler
├── backends/
│   └── verl.py         # veRL adapter
├── debug/
│   └── debugger.py     # RewardDebugger
└── cli/
    └── main.py         # CLI 入口
```

---

## 7. 完整示例：5 分钟让 GSM8K 变成 Agent RL 训练任务

```python
import agent_harness as ah

# 1. 定义环境（或用预置的）
env = ah.envs.MathReasoningEnv(
    dataset="gsm8k",
    tools=["calculator", "python_exec"],
)

# 2. 组合 reward（3 行搞定）
reward = ah.RewardComposer([
    ah.rewards.exact_match(key="answer", weight=0.7),
    ah.rewards.tool_call_valid(weight=0.15),
    ah.rewards.trajectory_efficiency(max_turns=8, weight=0.15),
])

# 3. 配置训练
harness = ah.Harness(
    env=env,
    reward=reward,
    credit=ah.CreditAssigner("hybrid"),
    backend=ah.backends.veRL(
        model="Qwen/Qwen2.5-7B",
        algorithm="grpo",
    ),
)

# 4. 一键训练
harness.train(iterations=50)

# 5. 调试 reward
ah.debug.RewardDebugger(reward).visualize(harness.trajectories)
```

---

## 8. 对标定位

```
  传统 RL 世界                    Agentic RL 世界
  ════════════                    ════════════════

  Gymnasium (环境标准)      →     AgentHarness (环境标准)
  Stable-Baselines3 (训练)  →     veRL/OpenRLHF (训练)
  WandB (监控)             →     AgentHarness Debugger (监控)

  我们是 Agentic RL 的 "Gymnasium + 超级 Reward 工程台"
```

---

## 9. 预期影响力

| 指标 | 6 个月 | 12 个月 |
|------|--------|---------|
| GitHub Stars | 1k-3k | 5k-10k |
| 论文引用 | 发布技术报告 | 顶会投稿 (NeurIPS/ICML) |
| 社区 | Discord 500+ | 外部贡献者 20+ |
| 适配环境 | 5-8 个 | 20+ 个 |

---

## 10. 总结

**AgentHarness 填补的是 Agentic RL 生态中最关键的空白 —— Harness 层。**

现有项目要么是训练框架（不管环境和 reward），要么是环境（不管训练），要么是 reward 工具（只做拼接）。没有人把这三者用一个统一的、优雅的协议连接起来，并提供 reward 调试这个被严重忽视但极其重要的功能。

**这就是 AgentHarness 的机会。**
