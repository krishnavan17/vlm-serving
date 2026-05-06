"""Rich-based logger for graphics_validator."""
from __future__ import annotations

import logging

try:
    from rich.logging import RichHandler as _RichHandler
    from rich.console import Console as _Console

    _rich_available = True
except ImportError:
    _rich_available = False

_root_configured = False


def _configure_root() -> None:
    global _root_configured
    if _root_configured:
        return
    _root_configured = True

    if _rich_available:
        handler = _RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_path=False,
            console=_Console(stderr=True),
        )
    else:
        handler = logging.StreamHandler()

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[handler],
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured with Rich (or plain) output."""
    _configure_root()
    return logging.getLogger(name)
