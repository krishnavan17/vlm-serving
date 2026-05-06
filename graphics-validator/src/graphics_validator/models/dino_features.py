"""DINOv2 feature extractor — lazy torch import so the CLI works without GPU.

Usage:
    from graphics_validator.models.dino_features import DinoFeatureExtractor
    extractor = DinoFeatureExtractor()          # loads model on first call
    embedding = extractor.embed(frame_bgr)      # torch.Tensor on CPU/GPU
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch


class DinoFeatureExtractor:
    """Wraps the DINOv2 ViT-S/14 model for per-frame embedding.

    Requires: ``pip install graphics-validator[dino]``
    (installs torch + torchvision)

    The model is lazy-loaded on the first call to :meth:`embed`.
    """

    def __init__(self, model_name: str = "dinov2_vits14", device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model: Any = None
        self._transforms: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # noqa: PLC0415
            import torchvision.transforms as T  # noqa: PLC0415

            self._model = torch.hub.load("facebookresearch/dinov2", self.model_name)
            self._model.eval().to(self.device)

            self._transforms = T.Compose(
                [
                    T.Resize(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        except ImportError as exc:
            raise ImportError(
                "DINOv2 requires: pip install graphics-validator[dino]"
            ) from exc

    def embed(self, frame_bgr: np.ndarray) -> Any:
        """Return a 1-D CLS-token embedding tensor for *frame_bgr*.

        Parameters
        ----------
        frame_bgr:
            BGR uint8 frame from OpenCV.

        Returns
        -------
        torch.Tensor
            Embedding vector, shape ``(embed_dim,)`` on :attr:`device`.
        """
        self._load()
        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
        import cv2  # noqa: PLC0415

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tensor = self._transforms(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self._model(tensor)  # shape (1, embed_dim)

        return features.squeeze(0)
