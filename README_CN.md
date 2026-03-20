<p align="center">
  <h1 align="center">AgentHarness</h1>
  <p align="center">
    <strong>驾驭任何任务，组合任何奖励，训练任何智能体。</strong>
  </p>
  <p align="center">
    面向 Agentic RL 的通用 Harness 引擎 —— 连接环境、奖励与训练框架之间的断层。
  </p>
  <p align="center">
    <a href="./README.md">English</a> | 中文
  </p>
  <p align="center">
    <a href="#快速开始">快速开始</a> &bull;
    <a href="#整体架构">整体架构</a> &bull;
    <a href="#完整示例">完整示例</a> &bull;
    <a href="#api-参考">API 参考</a> &bull;
    <a href="#参与贡献">参与贡献</a>
  </p>
</p>

---

## 为什么需要 AgentHarness？

当前 Agentic RL 生态存在三个关键断层：

```
  训练框架                      环境 / 评测基准                   奖励工程
  ──────                      ──────────────                   ────────
  veRL, OpenRLHF, TRL         SWE-bench, WebArena, OSWorld     各写各的 ad-hoc 函数
                  \                    |                       /
                   \                   |                      /
                    ╳──── 缺少统一的 Harness 层 ────╳
```

**每个项目都在重复造轮子** —— 环境接口不统一、奖励函数不可复用、换训练框架要重写。
AgentHarness 提供一个标准化的协议层，让你可以：

- **5 个方法** 将任意任务定义为可训练环境
- **像搭积木一样** 声明式组合多维度奖励信号
- **训练前检测** Reward Hacking，避免白白浪费算力
- **一行代码** 切换训练后端（veRL / OpenRLHF / TRL）

> 定位：**Agentic RL 的 Gymnasium** + **Reward 工程工作台**

## 核心特性

| 特性 | 说明 |
|------|------|
| **环境协议 (AgentEnv)** | 通用 5 方法接口 —— 任何 Benchmark 可即插即用 |
| **奖励组合器 (RewardComposer)** | 声明式 DSL，内置 12+ 种奖励信号，加权求和 |
| **奖励调试器 (RewardDebugger)** | 分布可视化、Hacking 检测、A/B 对比 —— 独家功能 |
| **信用分配 (CreditAssigner)** | 支持 outcome_only / turn_level / hybrid 三种策略 |
| **轨迹管理 (TrajectoryStore)** | 收集、过滤、持久化 (JSONL)、随机采样 |
| **课程调度 (CurriculumScheduler)** | 按难度自动升级，支持晋升阈值 |
| **训练后端适配器** | veRL / OpenRLHF / TRL 适配器，一行切换 |
| **命令行工具 (CLI)** | `agent-harness info`、`stats`、`debug` |

## 安装

```bash
pip install agent-harness

# 安装开发工具
pip install agent-harness[dev]

# 安装所有可选依赖
pip install agent-harness[all]
```

**从源码安装：**

```bash
git clone https://github.com/pushinl/AgentHarness.git
cd AgentHarness
pip install -e ".[dev]"
```

**环境要求：** Python 3.10+

## 快速开始

### 5 分钟搭建数学推理训练流水线

```python
import agent_harness as ah

# 1. 定义任务
tasks = [
    {"prompt": "What is 12 * 15?", "answer": "180"},
    {"prompt": "What is 99 + 101?", "answer": "200"},
    {"prompt": "What is 2**10?",    "answer": "1024"},
]

# 2. 创建环境
env = ah.envs.MathReasoningEnv(tools=["calculator", "python_exec"], max_turns=6)

# 3. 组合奖励（像搭积木一样）
reward = ah.RewardComposer([
    ah.rewards.exact_match(key="answer", weight=0.6, extract_number=True),
    ah.rewards.tool_call_valid(weight=0.15),
    ah.rewards.trajectory_efficiency(max_turns=6, weight=0.15),
    ah.rewards.length_penalty(max_tokens=200, weight=0.1),
])

# 4. 组装 Harness
harness = ah.Harness(
    env=env,
    reward=reward,
    credit=ah.CreditAssigner("hybrid"),
    backend=ah.backends.VeRLBackend(model="Qwen/Qwen2.5-7B", algorithm="grpo"),
)

# 5. 定义你的 Agent（或接入你的 LLM）
def my_agent(observation: str, available_tools: list[str]) -> ah.Action:
    # 你的 agent 逻辑...
    return ah.Action.tool("calculator", {"expression": "12*15"})

# 6. 训练
harness.train(tasks=tasks, agent_fn=my_agent, iterations=50)

# 7. 调试奖励
from agent_harness.debug import RewardDebugger
debugger = RewardDebugger(reward)
report = debugger.analyze(harness.trajectories)
print(report.summary())
```

## 整体架构

```
                            AgentHarness
┌────────────────────────────────────────────────────────────┐
│                                                            │
│  ┌──────────────┐  ┌───────────────┐  ┌────────────────┐  │
│  │   环境协议    │  │   奖励引擎    │  │   轨迹管理     │  │
│  │  Environment │  │    Reward     │  │   Trajectory   │  │
│  │   Protocol   │  │    Engine     │  │     Store      │  │
│  │              │  │               │  │                │  │
│  │ • AgentEnv   │  │ • Composer    │  │ • 收集         │  │
│  │ • MathEnv    │  │ • 12+ 内置    │  │ • 过滤         │  │
│  │ • CodeEnv    │  │ • Debugger    │  │ • 持久化       │  │
│  │ • ToolEnv    │  │ • Credit      │  │ • 课程学习     │  │
│  └──────┬───────┘  └──────┬────────┘  └───────┬────────┘  │
│         │                 │                    │           │
│         └─────────────────┼────────────────────┘           │
│                           │                                │
│               ┌───────────▼───────────┐                    │
│               │   Harness 运行时       │                    │
│               │    (编排调度器)         │                    │
│               └───────────┬───────────┘                    │
│                           │                                │
│  ┌────────────────────────▼─────────────────────────────┐  │
│  │              训练后端适配器                            │  │
│  │   ┌──────┐   ┌──────────┐   ┌─────┐   ┌──────────┐  │  │
│  │   │ veRL │   │ OpenRLHF │   │ TRL │   │  自定义   │  │  │
│  │   └──────┘   └──────────┘   └─────┘   └──────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

## 核心概念

### 1. 环境协议 (AgentEnv)

任何任务只需实现 `AgentEnv` 的 5 个方法，即可接入 AgentHarness 进行训练：

```python
from agent_harness import AgentEnv, Action, Observation

class MyCustomEnv(AgentEnv):
    def reset(self, task: dict) -> Observation:
        """用任务初始化环境，返回初始观察。"""
        ...

    def step(self, action: Action) -> tuple[Observation, bool]:
        """执行动作，返回 (观察, 是否结束)。"""
        ...

    def get_ground_truth(self):
        """返回正确答案（用于可验证奖励）。"""
        ...

    def get_available_tools(self) -> list[ToolSpec]:
        """返回当前可用工具列表。"""
        ...

    def get_state_snapshot(self) -> dict:
        """返回环境状态快照（用于调试 / 奖励计算）。"""
        ...
```

**内置环境：**

| 环境 | 说明 | 工具 |
|------|------|------|
| `MathReasoningEnv` | 数学推理 | `calculator`、`python_exec` |
| `CodeExecutionEnv` | 代码生成 + 单测验证 | `run_code`、`run_tests` |
| `ToolCallingEnv` | 通用工具调用（注册任意可调用对象） | 用户自定义 |

### 2. 奖励组合器 (RewardComposer)

声明式组合多维度奖励信号：

```python
reward = ah.RewardComposer([
    # 正确性
    ah.rewards.exact_match(key="answer", weight=0.5),
    ah.rewards.fuzzy_match(weight=0.1),

    # 工具使用质量
    ah.rewards.tool_call_valid(weight=0.15),
    ah.rewards.trajectory_efficiency(max_turns=10, weight=0.1),

    # 输出质量
    ah.rewards.format_follows(pattern=r"\d+", weight=0.05),
    ah.rewards.length_penalty(max_tokens=500, weight=0.05),
    ah.rewards.structured_output(format_type="json", weight=0.05),
])
```

**全部 12 种内置奖励函数：**

| 函数 | 信号维度 | 说明 |
|------|---------|------|
| `exact_match` | 正确性 | 精确匹配（支持数字提取、大小写归一化） |
| `fuzzy_match` | 正确性 | 模糊匹配（基于 SequenceMatcher） |
| `contains_match` | 正确性 | 包含关系匹配 |
| `code_passes_tests` | 正确性 | 代码通过单元测试 |
| `code_executable` | 质量 | 代码可无错运行 |
| `tool_call_valid` | 工具使用 | 工具调用合法且成功 |
| `tool_usage_rate` | 工具使用 | 惩罚不使用可用工具 |
| `trajectory_efficiency` | 效率 | 越少轮次完成 → 越高奖励 |
| `format_follows` | 格式 | 输出匹配指定正则 |
| `length_penalty` | 格式 | 惩罚过长输出 |
| `structured_output` | 格式 | 有效 JSON/XML 输出 |
| `llm_judge` | 质量 | LLM 评委打分（可插拔客户端） |

### 3. 奖励调试器 (RewardDebugger)

**AgentHarness 的杀手级功能** —— 目前没有任何其他项目提供奖励调试工具：

```python
from agent_harness.debug import RewardDebugger

debugger = RewardDebugger(reward)

# 完整分析 + Hacking 检测
report = debugger.analyze(trajectories, ground_truth="42")
print(report.summary())
```

输出示例：

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
  [ALERT] tool_call_valid: 接近饱和：92% 的轨迹得分 > 0.9
        -> 建议：提高难度、收紧标准或降低权重
  [WARN] length_penalty: 方差过低 (std=0.0600)
        -> 建议：检查是否太容易被 game
============================================================
```

**A/B 对比两个奖励函数：**

```python
comparison = debugger.compare(reward_v1, reward_v2, trajectories)
print(comparison.summary())
# 输出：均值、标准差、相关系数、排名一致率、逐轨迹差异
```

### 4. 信用分配 (CreditAssigner)

支持三种策略为每个 turn 分配奖励：

```python
# 所有 turn 获得相同的最终奖励
credit = ah.CreditAssigner("outcome_only")

# 每个 turn 独立计分
credit = ah.CreditAssigner("turn_level", turn_reward_fn=my_reward)

# 加权混合
credit = ah.CreditAssigner(
    "hybrid",
    turn_reward_fn=ah.rewards.trajectory_efficiency(max_turns=10),
    turn_weight=0.3,
    trajectory_weight=0.7,
)
```

### 5. 轨迹管理与课程学习

```python
from agent_harness.store import TrajectoryStore, CurriculumScheduler

# 持久化和查询轨迹
store = TrajectoryStore("./my_trajectories")
store.add_batch(trajectories)
store.save()                                # JSONL 格式
good = store.filter(min_reward=0.5, max_turns=10)
stats = store.statistics()                  # 均值、最小值、最大值、成功率

# 课程学习
curriculum = CurriculumScheduler(stages=[
    {"difficulty": "easy",   "epochs": 2, "promotion_threshold": 0.7},
    {"difficulty": "medium", "epochs": 3, "promotion_threshold": 0.8},
    {"difficulty": "hard",   "epochs": 5},
])
```

### 6. 训练后端适配器

一行切换训练框架：

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

## 完整示例

[`examples/`](examples/) 目录下包含三个可直接运行的完整示例：

### 数学推理 ([`examples/math_reasoning.py`](examples/math_reasoning.py))

完整流水线：计算器工具 → 奖励组合 → 训练 → 奖励调试 → 轨迹持久化。

```bash
python examples/math_reasoning.py
```

```
[Phase 1] 收集轨迹...
  ✓ What is 12 * 15?     agent=180   gt=180   reward=0.970
  ✓ What is 99 + 101?    agent=200   gt=200   reward=0.970
  ...
[Phase 2] 训练（5 轮迭代，离线模式）...
[Phase 3] Reward 调试报告
[Phase 4] 评估 — 成功率: 62.5%
[Phase 5] 保存 8 条轨迹
```

### 代码生成 ([`examples/code_generation.py`](examples/code_generation.py))

代码执行环境 + 单测验证 + A/B 奖励对比。

```bash
python examples/code_generation.py
```

### 工具调用 + 课程学习 ([`examples/tool_calling_curriculum.py`](examples/tool_calling_curriculum.py))

自定义工具注册 + 三阶段课程（easy → medium → hard）。

```bash
python examples/tool_calling_curriculum.py
```

## 命令行工具

```bash
# 查看可用组件
agent-harness info

# 查看轨迹库统计信息
agent-harness stats ./trajectories

# 在已有轨迹上调试奖励函数
agent-harness debug ./trajectories
```

## API 参考

### 顶层导入

```python
import agent_harness as ah

# 核心类型
ah.Action              # Agent 动作（text / tool_call / finish）
ah.Observation         # 环境观察
ah.Trajectory          # 完整轨迹记录
ah.AgentEnv            # 环境协议（抽象类）
ah.ToolSpec            # 工具定义

# 奖励
ah.Reward              # 奖励基类
ah.RewardComposer      # 加权奖励组合器
ah.CreditAssigner      # 逐 turn 信用分配

# 编排器
ah.Harness             # 主训练编排器

# 子模块
ah.envs                # 内置环境
ah.rewards             # 奖励函数
ah.backends            # 训练后端适配器
ah.debug               # 奖励调试器
ah.store               # 轨迹管理 + 课程学习
```

### Harness 编排器

```python
harness = ah.Harness(env, reward, credit?, store?, curriculum?, backend?, config?)

# 收集轨迹
traj  = harness.collect_trajectory(task, agent_fn)
batch = harness.collect_batch(tasks, agent_fn)

# 训练（在线 / 离线两种模式）
history = harness.train(tasks?, agent_fn?, iterations=10, batch_size=8, trajectories?)

# 评估
metrics = harness.evaluate(tasks, agent_fn)
# → {"mean_reward", "success_rate", "mean_turns", "num_tasks"}

# 获取统计
stats = harness.get_stats()
```

### Agent 函数签名

```python
def my_agent(observation: str, available_tools: list[str]) -> ah.Action:
    """
    参数:
        observation: 来自环境的文本观察。
        available_tools: 当前可调用的工具名列表。
    返回:
        一个 Action —— 以下三种之一：
          ah.Action.text("推理中...")
          ah.Action.tool("tool_name", {"arg": "value"})
          ah.Action.finish("最终答案")
    """
```

## 项目结构

```
agent_harness/
├── __init__.py              # 公开 API
├── harness.py               # Harness 编排器
├── core/
│   ├── action.py            # Action, Observation, ToolCall, ToolResult
│   ├── env.py               # AgentEnv 协议
│   ├── tool.py              # ToolSpec, ToolParameter
│   └── trajectory.py        # Trajectory, Turn
├── rewards/
│   ├── base.py              # Reward, RewardComposer
│   ├── credit.py            # CreditAssigner
│   └── builtin/             # 12 种内置奖励函数
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
│   └── debugger.py          # RewardDebugger, Hacking 检测, A/B 对比
└── cli/
    └── main.py              # CLI 入口

examples/                    # 3 个完整可运行示例
├── math_reasoning.py        # 数学推理全流程
├── code_generation.py       # 代码生成 + A/B 对比
└── tool_calling_curriculum.py # 自定义工具 + 课程学习

tests/                       # 165 个测试（单元 + 集成）
├── test_core.py
├── test_rewards.py
├── test_debugger.py
├── test_envs.py
├── test_store.py
├── test_harness.py
└── test_integration.py      # 36 个端到端集成测试
```

## 与现有项目对比

| 能力 | AgentHarness | Agent-R1 | ToolRL | GEM | reward-composer |
|------|-------------|----------|--------|-----|-----------------|
| 标准环境协议 | **支持** | 否 (强耦合 veRL) | 否 | 部分 | 否 |
| 可组合奖励 | **12 种内置** | Ad-hoc | Ad-hoc | 否 | 基础 |
| 奖励调试 | **支持** | 否 | 否 | 否 | 否 |
| Hacking 检测 | **支持** | 否 | 否 | 否 | 否 |
| A/B 奖励对比 | **支持** | 否 | 否 | 否 | 否 |
| 信用分配 | **3 种策略** | 仅 outcome | 仅 outcome | 否 | 否 |
| 训练框架无关 | **支持** | 仅 veRL | 仅 veRL | N/A | N/A |
| 轨迹持久化 | **支持** | 否 | 否 | 否 | 否 |
| 课程学习 | **支持** | 否 | 否 | 否 | 否 |
| CLI 工具 | **支持** | 否 | 否 | 否 | 否 |

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行全部测试
pytest tests/ -v

# 仅运行集成测试
pytest tests/test_integration.py -v

# 代码检查
ruff check agent_harness/
```

## 路线图

- [ ] 异步环境执行（基于 Ray 的并行轨迹收集）
- [ ] Web 版 Reward Debugger UI（Gradio）
- [ ] Parquet 格式轨迹存储（面向大规模数据集）
- [ ] veRL 完整集成（实际训练循环）
- [ ] 更多内置环境：WebBrowsingEnv (Playwright)、SQLQueryEnv、FileEditingEnv
- [ ] 基于历史趋势的 Reward Hacking 检测
- [ ] arXiv 技术报告

## 引用

```bibtex
@software{agentharness2025,
  title  = {AgentHarness: The Universal Harness Engine for Agentic RL},
  author = {pushinl},
  url    = {https://github.com/pushinl/AgentHarness},
  year   = {2025},
}
```

## 许可证

[Apache-2.0](LICENSE)
