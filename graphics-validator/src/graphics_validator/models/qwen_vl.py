"""Qwen2.5-VL model wrappers with a common interface.

Provides:
- ``OllamaQwenVL``: thin Ollama HTTP client (re-exported from layer4_vlm)
- ``HFQwenVL``: HuggingFace transformers backend (lazy torch import)
"""
from __future__ import annotations

# Re-export from the canonical location to keep the public API stable.
from graphics_validator.detectors.layer4_vlm import HFQwenVL, OllamaQwenVL

__all__ = ["OllamaQwenVL", "HFQwenVL"]
