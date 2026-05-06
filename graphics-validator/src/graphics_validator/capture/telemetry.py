"""Telemetry stubs: PresentMon CSV + Windows Event Log (TDR id 4101).

These are placeholder implementations. Real parsing of PresentMon output
and Windows Event Log requires platform-specific tooling.

TODO:
    - Parse PresentMon CSV: columns PresentRuntime, msBetweenPresents,
      msInPresentAPI, Dropped, AllowsTearing, etc.
    - Parse Windows Event Log: filter Source="nvlddmkm" EventID=4101 (TDR).
    - Expose TDR timestamps and frame-time series as structured data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FrameTimeSample:
    """One row from a PresentMon capture."""

    frame_index: int
    ms_between_presents: float
    dropped: bool
    allows_tearing: bool


@dataclass
class TDREvent:
    """A single GPU TDR (Timeout Detection and Recovery) event."""

    timestamp_s: float
    description: str


@dataclass
class TelemetryData:
    """Aggregated telemetry for one validation run."""

    frame_times: list[FrameTimeSample] = field(default_factory=list)
    tdr_events: list[TDREvent] = field(default_factory=list)


def parse_presentmon_csv(path: Path) -> list[FrameTimeSample]:
    """Parse a PresentMon CSV file and return frame-time samples.

    TODO: Implement real CSV parsing using ``csv.DictReader``.
          Expected columns: ProcessName, msBetweenPresents, Dropped,
          AllowsTearing, msInPresentAPI, Runtime.
    """
    # Stub — returns empty list
    _ = path
    return []


def parse_windows_event_log(log_path: Path | None = None) -> list[TDREvent]:
    """Parse Windows Event Log for TDR events (EventID 4101, Source nvlddmkm).

    TODO: On Windows use ``pywin32`` / ``wevtapi`` to query:
        ``Get-WinEvent -FilterHashtable @{LogName='System'; Id=4101}``
          or use ``win32evtlog.OpenEventLog``.
    """
    # Stub — returns empty list
    _ = log_path
    return []
