"""CLI entry point for AgentHarness."""

from __future__ import annotations

import click

from agent_harness import __version__


@click.group()
@click.version_option(version=__version__)
def cli() -> None:
    """AgentHarness — Harness any task. Compose any reward. Train any agent."""
    pass


@cli.command()
def info() -> None:
    """Show AgentHarness info and available components."""
    from agent_harness import envs, rewards

    click.echo(f"AgentHarness v{__version__}")
    click.echo()
    click.echo("Available Environments:")
    for env_cls in [envs.MathReasoningEnv, envs.CodeExecutionEnv, envs.ToolCallingEnv]:
        click.echo(f"  - {env_cls.__name__}")
    click.echo()
    click.echo("Available Reward Functions:")
    reward_fns = [
        "exact_match", "fuzzy_match", "contains_match",
        "code_passes_tests", "code_executable",
        "tool_call_valid", "trajectory_efficiency", "tool_usage_rate",
        "format_follows", "length_penalty", "structured_output",
        "llm_judge",
    ]
    for name in reward_fns:
        click.echo(f"  - {name}")
    click.echo()
    click.echo("Available Training Backends:")
    click.echo("  - VeRLBackend")
    click.echo("  - OpenRLHFBackend")
    click.echo("  - TRLBackend")
    click.echo("  - DummyBackend")


@cli.command()
@click.argument("store_path", type=click.Path())
def stats(store_path: str) -> None:
    """Show statistics for a trajectory store."""
    from agent_harness.store import TrajectoryStore

    store = TrajectoryStore.load(store_path)
    stat = store.statistics()

    if stat["count"] == 0:
        click.echo("No trajectories found.")
        return

    click.echo(f"Trajectory Store: {store_path}")
    click.echo(f"  Count: {stat['count']}")
    click.echo(f"  Reward: mean={stat['reward_mean']:.3f}, "
               f"min={stat['reward_min']:.3f}, max={stat['reward_max']:.3f}")
    click.echo(f"  Turns:  mean={stat['turns_mean']:.1f}, "
               f"min={stat['turns_min']}, max={stat['turns_max']}")
    click.echo(f"  Success Rate: {stat['success_rate']:.1%}")


@cli.command()
@click.argument("store_path", type=click.Path())
@click.option("--reward", "-r", default="exact_match", help="Reward function to debug")
def debug(store_path: str, reward: str) -> None:
    """Debug reward function on stored trajectories."""
    from agent_harness.debug import RewardDebugger
    from agent_harness.rewards import RewardComposer, exact_match, trajectory_efficiency

    store = _load_store(store_path)
    if not store:
        return

    # Simple reward selection
    reward_fn = RewardComposer([exact_match(weight=0.7), trajectory_efficiency(weight=0.3)])
    debugger = RewardDebugger(reward_fn)
    report = debugger.analyze(list(store))
    click.echo(report.summary())


def _load_store(path: str):
    from agent_harness.store import TrajectoryStore
    store = TrajectoryStore.load(path)
    if len(store) == 0:
        click.echo("No trajectories found.")
        return None
    return store


if __name__ == "__main__":
    cli()
