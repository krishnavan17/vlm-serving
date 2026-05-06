"""Pipeline orchestrator: iterates VideoSource, runs enabled layers in order,
accumulates Flags, calls Layer 4 only when flags exist and severity ≥ trigger_min_severity.

Usage:
    pipeline = Pipeline(cfg=cfg, out_dir=Path("reports"))
    report = pipeline.run(video_path=Path("game.mp4"), golden_path=None)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from graphics_validator.capture.video_source import VideoSource
from graphics_validator.detectors.base import Flag, severity_gte
from graphics_validator.detectors.layer1_cheap import make_layer1
from graphics_validator.detectors.layer2_temporal import make_layer2
from graphics_validator.detectors.layer3_reference import make_layer3
from graphics_validator.detectors.layer4_vlm import Layer4VLM
from graphics_validator.reporting.report import Report, make_report
from graphics_validator.utils.logging import get_logger

log = get_logger(__name__)


class Pipeline:
    """Multi-layer graphics validation pipeline.

    Parameters
    ----------
    cfg:
        Full YAML config dict (already parsed).
    out_dir:
        Root output directory for reports.
    """

    def __init__(self, cfg: dict | None = None, out_dir: Path | str = "reports") -> None:
        self._cfg = cfg or {}
        self._out_dir = Path(out_dir)

        pipeline_cfg = self._cfg.get("pipeline", {})
        self._fps_sample: float = float(pipeline_cfg.get("fps_sample", 4.0))
        self._trigger_min: str = pipeline_cfg.get("trigger_min_severity", "medium")

        self._l1 = make_layer1(self._cfg)
        self._l2 = make_layer2(self._cfg, fps_sample=self._fps_sample)
        self._l3 = make_layer3(self._cfg)
        self._l4 = Layer4VLM(self._cfg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        video_path: Path,
        golden_path: Path | None = None,
    ) -> Report:
        """Run the full pipeline and return the completed :class:`Report`."""
        report = make_report(self._cfg, out_dir=self._out_dir)
        log.info(f"Starting run {report.run_dir.name} — video: {video_path}")

        with VideoSource(video_path, fps_sample=self._fps_sample) as src:
            total = src.frame_count
            golden_src = (
                VideoSource(golden_path, fps_sample=self._fps_sample)
                if golden_path is not None
                else None
            )
            try:
                self._process_frames(src, golden_src, report, total)
            finally:
                if golden_src is not None:
                    golden_src.release()

        report.finalise()
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_frames(
        self,
        src: VideoSource,
        golden_src: VideoSource | None,
        report: Report,
        total: int,
    ) -> None:
        try:
            from rich.progress import Progress, BarColumn, TimeRemainingColumn, TextColumn

            progress = Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeRemainingColumn(),
            )
            task = progress.add_task("Validating frames", total=total)
            use_rich = True
        except ImportError:
            progress = None
            task = None
            use_rich = False

        if use_rich:
            progress.start()  # type: ignore[union-attr]

        try:
            for idx, frame in src:
                ts = src.timestamp(idx)
                flags: list[Flag] = []

                # Layer 1 — cheap heuristics
                flags.extend(self._l1.detect(frame, idx))

                # Layer 2 — temporal
                flags.extend(self._l2.detect(frame, idx))

                # Layer 3 — reference (only if golden available)
                if golden_src is not None:
                    try:
                        golden_frame = golden_src.frame_at(idx)
                    except (IndexError, Exception):
                        golden_frame = None
                    flags.extend(self._l3.detect(frame, idx, golden_frame=golden_frame))

                # Layer 4 — VLM (only when triggered)
                vlm_response: dict | None = None
                if flags and self._should_trigger_vlm(flags):
                    vlm_response = self._l4.describe(frame, flags)

                # Record event if any flags were raised
                if flags:
                    report.record(
                        frame_idx=idx,
                        timestamp_s=ts,
                        flags=flags,
                        frame_bgr=frame,
                        vlm_response=vlm_response,
                    )

                if use_rich:
                    progress.advance(task)  # type: ignore[union-attr]
        finally:
            if use_rich:
                progress.stop()  # type: ignore[union-attr]

    def _should_trigger_vlm(self, flags: list[Flag]) -> bool:
        """Return True if any flag meets or exceeds the configured minimum severity."""
        return any(severity_gte(f.severity, self._trigger_min) for f in flags)  # type: ignore[arg-type]
