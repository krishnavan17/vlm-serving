"""Tests for the prompt builder and response schema.

Validates:
- build_user_prompt mentions every flag kind passed in.
- RESPONSE_SCHEMA contains all required keys.
- Prompt structure matches the expected contract.
"""
from __future__ import annotations

import pytest

from graphics_validator.detectors.base import Flag
from graphics_validator.prompts.graphics_bug_report import (
    RESPONSE_SCHEMA,
    SYSTEM_PROMPT,
    build_user_prompt,
)

# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

REQUIRED_SCHEMA_KEYS = {
    "category",
    "severity",
    "description",
    "evidence",
    "suggested_root_cause",
    "confidence",
    "false_positive_likelihood",
    "recommended_action",
}


class TestResponseSchema:
    def test_schema_has_required_keys(self):
        """RESPONSE_SCHEMA must declare all required fields."""
        declared = set(RESPONSE_SCHEMA.get("required", []))
        missing = REQUIRED_SCHEMA_KEYS - declared
        assert not missing, f"RESPONSE_SCHEMA missing required keys: {missing}"

    def test_schema_properties_match_required(self):
        """Every required key must have a matching properties entry."""
        props = set(RESPONSE_SCHEMA.get("properties", {}).keys())
        required = set(RESPONSE_SCHEMA.get("required", []))
        missing_props = required - props
        assert not missing_props, f"Schema properties missing entries for: {missing_props}"

    def test_category_enum_contains_expected_values(self):
        """category enum must include at minimum 'bsod' and 'no_anomaly'."""
        enum_vals = RESPONSE_SCHEMA["properties"]["category"]["enum"]
        assert "bsod" in enum_vals
        assert "no_anomaly" in enum_vals

    def test_severity_enum(self):
        """severity enum must include the four severity levels."""
        enum_vals = RESPONSE_SCHEMA["properties"]["severity"]["enum"]
        for level in ("low", "medium", "high", "critical"):
            assert level in enum_vals


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    def test_system_prompt_non_empty(self):
        assert SYSTEM_PROMPT and len(SYSTEM_PROMPT) > 50

    def test_system_prompt_mentions_json(self):
        assert "JSON" in SYSTEM_PROMPT or "json" in SYSTEM_PROMPT.lower()


# ---------------------------------------------------------------------------
# build_user_prompt
# ---------------------------------------------------------------------------


def _make_flags() -> list[Flag]:
    return [
        Flag(
            layer="layer1",
            kind="BSOD",
            severity="critical",
            score=0.98,
            detail={"blue_dominance": 0.91, "std": 12.4},
        ),
        Flag(
            layer="layer2",
            kind="FLICKER",
            severity="medium",
            score=0.62,
            detail={"peak_hz": 12.0},
        ),
    ]


class TestBuildUserPrompt:
    def test_prompt_contains_each_flag_kind(self):
        """build_user_prompt must include every flag kind from the input list."""
        flags = _make_flags()
        prompt = build_user_prompt(flags)
        for flag in flags:
            assert flag.kind in prompt, f"Prompt missing flag kind: {flag.kind}"

    def test_prompt_contains_layer_names(self):
        """Prompt must reference each flag's layer."""
        flags = _make_flags()
        prompt = build_user_prompt(flags)
        for flag in flags:
            assert flag.layer in prompt

    def test_prompt_contains_severity(self):
        """Prompt must mention each flag's severity."""
        flags = _make_flags()
        prompt = build_user_prompt(flags)
        for flag in flags:
            assert flag.severity in prompt

    def test_prompt_contains_json_instruction(self):
        """Prompt must instruct the model to respond with JSON only."""
        prompt = build_user_prompt(_make_flags())
        assert "JSON" in prompt or "json" in prompt.lower()

    def test_prompt_contains_no_anomaly_instruction(self):
        """Prompt must instruct model to use no_anomaly for false positives."""
        prompt = build_user_prompt(_make_flags())
        assert "no_anomaly" in prompt

    def test_prompt_with_context(self):
        """Context dict values must appear in the generated prompt."""
        flags = _make_flags()
        ctx = {
            "game": "Cyberpunk 2077",
            "build": "2.13.0-internal",
            "scene": "corpo_plaza_night",
            "gpu": "NVIDIA RTX 4080 / driver 555.85",
        }
        prompt = build_user_prompt(flags, context=ctx)
        for key, val in ctx.items():
            assert val in prompt, f"Context value missing from prompt: {val!r}"

    def test_prompt_without_flags(self):
        """Empty flag list should still produce a valid prompt."""
        prompt = build_user_prompt([])
        assert "Detector flags" in prompt
        assert "no_anomaly" in prompt

    def test_prompt_schema_inline_contains_required_keys(self):
        """The inline schema snippet in the prompt must mention all required keys."""
        prompt = build_user_prompt(_make_flags())
        for key in REQUIRED_SCHEMA_KEYS:
            assert key in prompt, f"Inline schema in prompt missing key: {key}"
