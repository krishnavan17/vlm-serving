"""Qwen2.5-VL prompt template for graphics bug reports.

Exports
-------
SYSTEM_PROMPT : str
    System-role prompt establishing the QA engineer persona.
RESPONSE_SCHEMA : dict
    JSON Schema (draft-07) describing the required response structure.
build_user_prompt(flags, context) -> str
    Builds the per-frame user prompt from detector flags and optional context.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graphics_validator.detectors.base import Flag

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = (
    "You are a senior graphics QA engineer reviewing automated capture from PC game and "
    "Windows desktop validation runs. You analyze a single frame plus a list of automated "
    "detector flags and produce a structured bug report. You are precise, terse, and never "
    "invent details that are not visible in the frame. If evidence is insufficient, say so "
    "and lower your confidence. You must respond with **valid JSON only**, matching the schema "
    "provided. Do not include markdown, comments, or prose outside the JSON."
)

# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

RESPONSE_SCHEMA: dict = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": [
        "category",
        "severity",
        "description",
        "evidence",
        "suggested_root_cause",
        "confidence",
        "false_positive_likelihood",
        "recommended_action",
    ],
    "properties": {
        "category": {
            "type": "string",
            "enum": [
                "bsod",
                "black_screen",
                "frozen_frame",
                "tearing",
                "flicker",
                "missing_texture",
                "shader_artifact",
                "geometry_artifact",
                "hud_ui_issue",
                "color_hdr_issue",
                "perceptual_diff",
                "other",
                "no_anomaly",
            ],
        },
        "severity": {
            "type": "string",
            "enum": ["low", "medium", "high", "critical"],
        },
        "description": {
            "type": "string",
            "description": "What is visible in the frame. Maximum ~80 words.",
        },
        "evidence": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Concrete visual cues observed in the frame.",
        },
        "suggested_root_cause": {
            "type": "string",
            "description": "Best guess at the underlying cause.",
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Confidence in the assessment, 0–1.",
        },
        "false_positive_likelihood": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Likelihood that the detector flags are a false alarm, 0–1.",
        },
        "recommended_action": {
            "type": "string",
            "description": "Next step for a QA engineer.",
        },
    },
    "additionalProperties": False,
}

# Compact inline schema used inside the prompt body
_SCHEMA_INLINE = """{
  "category": "<one of: bsod|black_screen|frozen_frame|tearing|flicker|missing_texture|shader_artifact|geometry_artifact|hud_ui_issue|color_hdr_issue|perceptual_diff|other|no_anomaly>",
  "severity": "<low|medium|high|critical>",
  "description": "<≤80 words, what is visible>",
  "evidence": ["<specific visual cue>", "..."],
  "suggested_root_cause": "<best guess>",
  "confidence": <0-1>,
  "false_positive_likelihood": <0-1>,
  "recommended_action": "<next step for QA>"
}"""


# ---------------------------------------------------------------------------
# User prompt builder
# ---------------------------------------------------------------------------


def build_user_prompt(
    flags: list[Flag],
    context: dict | None = None,
) -> str:
    """Build the per-frame user prompt.

    Parameters
    ----------
    flags:
        List of :class:`~graphics_validator.detectors.base.Flag` objects raised
        by earlier pipeline layers.
    context:
        Optional dict with fields such as ``game``, ``build``, ``scene``,
        ``gpu``, ``driver``.

    Returns
    -------
    str
        Formatted prompt string ready to be sent to the VLM.
    """
    lines: list[str] = [
        "You are reviewing one captured frame from an automated graphics validation run.",
        "",
        "## Detector flags",
    ]

    if flags:
        for flag in flags:
            detail_str = f" detail={json.dumps(flag.detail)}" if flag.detail else ""
            lines.append(
                f"- [{flag.layer}] {flag.kind} severity={flag.severity} "
                f"score={flag.score:.3f}{detail_str}"
            )
    else:
        lines.append("- (none — frame was forwarded without heuristic flags)")

    if context:
        lines.append("")
        lines.append("## Context")
        for k, v in context.items():
            lines.append(f"{k}: {v}")

    lines += [
        "",
        "## Your task",
        "Analyze the attached frame. Treat the flags above as hints, not ground truth.",
        "Return a JSON object that conforms to this schema:",
        "",
        _SCHEMA_INLINE,
        "",
        "Rules:",
        "- Respond with VALID JSON ONLY. No markdown, no commentary.",
        "- Treat the detector flags as hints, not ground truth — verify them visually.",
        "- If the frame looks normal despite flags, set category='no_anomaly' with high false_positive_likelihood.",
        "- Cite concrete visual evidence; do not speculate beyond what is visible.",
        "- Prefer specific visual evidence over generic descriptions.",
        "- Do not speculate beyond what is visible in the frame.",
    ]

    return "\n".join(lines)
