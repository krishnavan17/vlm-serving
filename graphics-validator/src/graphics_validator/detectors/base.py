"""Base types for detectors: Flag, Severity, and Detector protocol."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

Severity = Literal["low", "medium", "high", "critical"]

_SEVERITY_ORDER: dict[str, int] = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "critical": 3,
}


def severity_gte(a: Severity, b: Severity) -> bool:
    """Return True if severity *a* is greater than or equal to *b*."""
    return _SEVERITY_ORDER[a] >= _SEVERITY_ORDER[b]


@dataclass
class Flag:
    """A single anomaly flag raised by one detector layer.

    Attributes
    ----------
    layer:
        Detector layer name, e.g. ``"layer1"``, ``"layer2"``.
    kind:
        Anomaly kind, e.g. ``"BSOD"``, ``"FROZEN"``, ``"FLICKER"``.
    severity:
        One of ``"low"``, ``"medium"``, ``"high"``, ``"critical"``.
    score:
        Numeric confidence score in ``[0, 1]``.
    detail:
        Optional dictionary with detector-specific metadata.
    """

    layer: str
    kind: str
    severity: Severity
    score: float
    detail: dict[str, Any] | None = field(default=None)

    def __str__(self) -> str:
        detail_str = f" detail={self.detail}" if self.detail else ""
        return (
            f"[{self.layer}] {self.kind} severity={self.severity} "
            f"score={self.score:.3f}{detail_str}"
        )


@runtime_checkable
class Detector(Protocol):
    """Protocol that all detector layers must implement."""

    def detect(self, frame: np.ndarray, idx: int, **kw: Any) -> list[Flag]:
        """Analyse *frame* and return a (possibly empty) list of :class:`Flag` objects."""
        ...
