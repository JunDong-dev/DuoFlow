#!/usr/bin/env python

"""Train DuoFlow from a YAML configuration."""

import os
from pathlib import Path

import typer
import yaml
from rich.console import Console

from data_splits import validate_training_data
from motionspeech_trainer.config import MotionSpeechTrainerConfig
from motionspeech_trainer.trainer import MotionSpeechTrainer

console = Console()
app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Train DuoFlow using a YAML configuration.",
)


def expand_environment(value):
    """Expand environment variables in nested YAML values."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_environment(item) for key, item in value.items()}
    return value


def find_unexpanded(value, path="config"):
    """Return YAML paths that still contain unresolved environment variables."""
    if isinstance(value, str):
        return [path] if "${" in value else []
    if isinstance(value, list):
        return [item for index, child in enumerate(value) for item in find_unexpanded(child, f"{path}[{index}]")]
    if isinstance(value, dict):
        return [item for key, child in value.items() for item in find_unexpanded(child, f"{path}.{key}")]
    return []


@app.command()
def main(
    config_path: str | None = typer.Argument(
        None,
        help="Path to YAML configuration file; optional when --resume-from points to a run directory",
    ),
    resume_from: str | None = typer.Option(
        None,
        "--resume-from",
        help="Run/checkpoints directory or checkpoint file to resume",
    ),
    disable_progress_bars: bool = typer.Option(
        False,
        "--disable-progress-bars",
        help="Disable progress bars (useful for multi-process runs)",
    ),
) -> None:
    """Train the model using the provided configuration file."""
    if config_path is None:
        if resume_from is None:
            typer.echo("Error: provide CONFIG_PATH or --resume-from RUN_DIR.")
            raise typer.Exit(code=1)
        resume_path = Path(resume_from).expanduser().resolve()
        search_start = resume_path if resume_path.is_dir() else resume_path.parent
        candidates = [
            candidate / "training_config.yaml"
            for candidate in (search_start, *search_start.parents)
        ]
        config_path = next((path for path in candidates if path.is_file()), None)
        if config_path is None:
            typer.echo(f"Error: no training_config.yaml found for {resume_path}.")
            raise typer.Exit(code=1)
    else:
        config_path = Path(config_path)
    if not config_path.exists():
        typer.echo(f"Error: Configuration file {config_path} does not exist.")
        raise typer.Exit(code=1)

    with open(config_path, "r") as file:
        try:
            config_data = yaml.safe_load(file)
        except yaml.constructor.ConstructorError:
            file.seek(0)
            config_data = yaml.full_load(file)
    config_data = expand_environment(config_data)
    unresolved = find_unexpanded(config_data)
    if unresolved:
        typer.echo("Error: unresolved environment variables at: " + ", ".join(unresolved))
        raise typer.Exit(code=1)

    if resume_from is not None:
        config_data.setdefault("checkpoints", {})["resume_from_checkpoint"] = resume_from
        config_data.setdefault("model", {})["load_checkpoint"] = None

    try:
        validate_training_data(config_data)
        trainer_config = MotionSpeechTrainerConfig(**config_data)
    except Exception as e:
        typer.echo(f"Error: Invalid configuration data: {e}")
        raise typer.Exit(code=1) from e

    trainer = MotionSpeechTrainer(trainer_config)
    trainer.train(disable_progress_bars=disable_progress_bars)


if __name__ == "__main__":
    app()
