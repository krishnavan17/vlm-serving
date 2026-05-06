"""Typer CLI for graphics-validator.

Usage:
    graphics-validator run --video PATH [--golden PATH] [--config PATH] [--out DIR]
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="graphics-validator",
    help="Automated graphics validation pipeline for PC games and Windows display.",
    add_completion=False,
    no_args_is_help=True,
)


@app.command()
def run(
    video: Path = typer.Option(..., "--video", help="Path to input video file."),
    golden: Optional[Path] = typer.Option(
        None, "--golden", help="Path to golden/reference video for Layer-3 comparison."
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", help="Path to YAML config file (defaults to built-in config)."
    ),
    out: Path = typer.Option(
        Path("reports"), "--out", help="Output directory for reports."
    ),
) -> None:
    """Run the graphics validation pipeline on a video file."""
    import yaml

    from graphics_validator.pipeline import Pipeline
    from graphics_validator.utils.logging import get_logger

    log = get_logger(__name__)

    # Load configuration
    if config is None:
        default_cfg = Path(__file__).parent.parent.parent / "config" / "default.yaml"
        config = default_cfg if default_cfg.exists() else None

    cfg: dict = {}
    if config and config.exists():
        with open(config) as fh:
            cfg = yaml.safe_load(fh) or {}
        log.info(f"Loaded config from {config}")
    else:
        log.warning("No config file found — using built-in defaults.")

    if not video.exists():
        typer.echo(f"[error] Video file not found: {video}", err=True)
        raise typer.Exit(code=1)

    if golden is not None and not golden.exists():
        typer.echo(f"[error] Golden video not found: {golden}", err=True)
        raise typer.Exit(code=1)

    pipeline = Pipeline(cfg=cfg, out_dir=out)
    report = pipeline.run(video_path=video, golden_path=golden)

    typer.echo(f"Report written to: {report.run_dir}")


@app.command()
def version() -> None:
    """Print the graphics-validator version."""
    from graphics_validator import __version__

    typer.echo(f"graphics-validator {__version__}")


if __name__ == "__main__":
    app()
