"""Layer-4 VLM detector: Qwen2.5-VL via Ollama (default) or HuggingFace transformers.

The VLM layer is only invoked when earlier layers have raised at least one flag
whose severity meets the pipeline's ``trigger_min_severity``.

Two backends are supported:
- ``OllamaQwenVL`` (default): calls the Ollama HTTP API (``/api/generate``)
  hosted by this repo's docker-compose.yml setup.
- ``HFQwenVL`` (optional): loads the model via HuggingFace ``transformers``
  (requires ``pip install graphics-validator[hf]``).

The active backend is selected via the ``vlm.backend`` config key or the
``VLM_HOST`` / ``VLM_MODEL`` environment variables.
"""
from __future__ import annotations

import base64
import json
import os
from typing import Any

import cv2
import numpy as np

from graphics_validator.detectors.base import Flag
from graphics_validator.prompts.graphics_bug_report import (
    SYSTEM_PROMPT,
    build_user_prompt,
)
from graphics_validator.utils.logging import get_logger

log = get_logger(__name__)


def _encode_frame_b64(frame_bgr: np.ndarray) -> str:
    """Encode a BGR frame as base64-encoded PNG string."""
    ok, buf = cv2.imencode(".png", frame_bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


class OllamaQwenVL:
    """Queries Qwen2.5-VL via the Ollama HTTP API.

    Parameters
    ----------
    host:
        Ollama server URL, e.g. ``http://localhost:11434``.
    model:
        Ollama model tag, e.g. ``qwen2.5vl``.
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "qwen2.5vl",
        timeout: int = 60,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def describe(self, frame_bgr: np.ndarray, flags: list[Flag], context: dict | None = None) -> dict:
        """Send *frame_bgr* + *flags* to Ollama and return parsed JSON response."""
        import requests  # noqa: PLC0415

        b64 = _encode_frame_b64(frame_bgr)
        user_prompt = build_user_prompt(flags, context)

        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": f"{SYSTEM_PROMPT}\n\n{user_prompt}",
            "images": [b64],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.1},
        }

        try:
            resp = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            raw_text = resp.json().get("response", "")
            return self._parse_json(raw_text)
        except requests.exceptions.ConnectionError:
            log.warning(
                f"Ollama not reachable at {self.host} — Layer 4 skipped for this frame."
            )
            return {"raw": None, "skipped": True, "reason": "ollama_unreachable"}
        except Exception as exc:
            log.warning(f"Layer-4 Ollama request failed: {exc}")
            return {"raw": str(exc), "skipped": True, "reason": "request_error"}

    @staticmethod
    def _parse_json(text: str) -> dict:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            log.warning("Layer 4: VLM response is not valid JSON — returning raw text.")
            return {"raw": text}


class HFQwenVL:
    """Queries Qwen2.5-VL via HuggingFace ``transformers`` (optional backend).

    Requires: ``pip install graphics-validator[hf]``

    Parameters
    ----------
    model_id:
        HuggingFace model repository, e.g. ``Qwen/Qwen2.5-VL-7B-Instruct``.
    device:
        PyTorch device string (``"cuda"``, ``"cpu"``, ``"mps"``).
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
    ) -> None:
        self.model_id = model_id
        self.device = device
        self._model: Any = None
        self._processor: Any = None

    def _load(self) -> None:
        """Lazy-load the model and processor (torch import deferred here)."""
        if self._model is not None:
            return
        try:
            import torch  # noqa: PLC0415
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # noqa: PLC0415

            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                device_map=self.device,
            )
        except ImportError as exc:
            raise ImportError(
                "HuggingFace backend requires: pip install graphics-validator[hf]"
            ) from exc

    def describe(self, frame_bgr: np.ndarray, flags: list[Flag], context: dict | None = None) -> dict:
        """Run inference with the local HF model."""
        try:
            self._load()
            from PIL import Image  # noqa: PLC0415
            import torch  # noqa: PLC0415

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            user_prompt = build_user_prompt(flags, context)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ]

            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._processor(
                text=[text], images=[pil_image], return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                output_ids = self._model.generate(**inputs, max_new_tokens=512)

            generated = output_ids[0][inputs["input_ids"].shape[1]:]
            raw_text = self._processor.decode(generated, skip_special_tokens=True)

            return OllamaQwenVL._parse_json(raw_text)
        except Exception as exc:
            log.warning(f"HF Layer-4 inference failed: {exc}")
            return {"raw": str(exc), "skipped": True, "reason": "hf_error"}


class Layer4VLM:
    """Unified VLM layer that delegates to Ollama or HF backend.

    Parameters
    ----------
    cfg:
        Full pipeline config dict.  Reads ``vlm`` sub-key.
    """

    def __init__(self, cfg: dict | None = None) -> None:
        c = (cfg or {}).get("vlm", {})
        backend = os.environ.get("VLM_BACKEND", c.get("backend", "ollama"))
        host = os.environ.get("VLM_HOST", c.get("host", "http://localhost:11434"))
        model = os.environ.get("VLM_MODEL", c.get("model", "qwen2.5vl"))
        timeout = int(c.get("timeout", 60))

        if backend == "hf":
            self._backend: OllamaQwenVL | HFQwenVL = HFQwenVL(model_id=model)
        else:
            self._backend = OllamaQwenVL(host=host, model=model, timeout=timeout)

    def describe(self, frame_bgr: np.ndarray, flags: list[Flag], context: dict | None = None) -> dict:
        """Describe *frame_bgr* given the detector *flags*.  Returns a parsed dict."""
        return self._backend.describe(frame_bgr, flags, context)
