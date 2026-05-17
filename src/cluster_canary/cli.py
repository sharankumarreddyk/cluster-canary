"""Command-line entrypoint for the cluster_canary package."""

from __future__ import annotations

import click

from cluster_canary import __version__


@click.group(help="cluster-canary — MLOps starter scaffold CLI.")
@click.version_option(__version__, prog_name="cluster_canary")
def main() -> None:
    """Top-level CLI group."""


@main.command()
def info() -> None:
    """Print package and environment info."""
    import os
    import platform

    click.echo(f"cluster_canary            v{__version__}")
    click.echo(f"python          {platform.python_version()}")
    click.echo(f"platform        {platform.system()} {platform.release()}")
    click.echo(f"mlflow uri      {os.environ.get('MLFLOW_TRACKING_URI', 'unset')}")
    click.echo(f"mlflow exp      {os.environ.get('MLFLOW_EXPERIMENT_NAME', 'unset')}")


if __name__ == "__main__":
    main()
