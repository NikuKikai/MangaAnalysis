from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from saliency.config import load_config
from saliency.model import create_model_from_config
from saliency.metrics import sigmoid_map
from saliency.datasets.common import IMAGENET_MEAN, IMAGENET_STD, resize_with_padding


def resolve_device(device_name: str) -> torch.device:
    """Resolve the configured device name into a CUDA device for GPU-only reading simulation."""
    if device_name != "auto":
        device = torch.device(device_name)
        if device.type != "cuda":
            raise RuntimeError("Reading simulation requires a CUDA device.")
        return device
    if not torch.cuda.is_available():
        raise RuntimeError("Reading simulation requires CUDA, but no CUDA device is available.")
    return torch.device("cuda")


class SaliencyInference:
    """Load a trained saliency model and run page-level inference on PIL images."""

    def __init__(self, config_path: str, checkpoint_path: str) -> None:
        """Create a saliency inference wrapper from one config file and one checkpoint."""
        self.config_path = str(Path(config_path).resolve())
        self.checkpoint_path = str(Path(checkpoint_path).resolve())
        self.config = load_config(self.config_path)
        self.device = resolve_device(self.config.training.device)

        # Build the exact architecture described by the training config so checkpoints remain compatible.
        self.model = create_model_from_config(self.config.model)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device)
        self.model.eval()
        self.input_size = self.config.dataset.input_size

    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert a PIL image into the normalized model input tensor."""
        resized = resize_with_padding(image.convert("RGB"), self.input_size, Image.Resampling.BILINEAR, (0, 0, 0))
        array = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        return tensor.unsqueeze(0)

    def _normalize_saliency_tensor(self, saliency_tensor: torch.Tensor) -> torch.Tensor:
        """Normalize one model-resolution saliency tensor in place-compatible tensor form."""
        saliency = saliency_tensor.to(dtype=torch.float32)
        saliency = saliency - saliency.min()
        saliency = saliency / torch.clamp(saliency.max(), min=1e-8)
        return saliency

    def _format_prediction(self, saliency_tensor: torch.Tensor, original_size: tuple[int, int]) -> tuple[np.ndarray, Image.Image]:
        """Resize and normalize raw model output so callers receive the legacy result shape."""
        saliency = self._normalize_saliency_tensor(saliency_tensor).detach().cpu().numpy()
        heatmap_image = Image.fromarray(np.clip(saliency * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
        if heatmap_image.size != original_size:
            heatmap_image = heatmap_image.resize(original_size, Image.Resampling.BILINEAR)
        resized_saliency = np.asarray(heatmap_image, dtype=np.float32) / 255.0
        resized_saliency = resized_saliency - float(resized_saliency.min())
        resized_saliency = resized_saliency / max(float(resized_saliency.max()), 1e-8)
        return resized_saliency, heatmap_image

    def predict_tensor_raw(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run inference from a preprocessed tensor and keep the normalized saliency map on GPU."""
        model_input = input_tensor.to(self.device)
        with torch.no_grad():
            logits = self.model(model_input)
            saliency_tensor = sigmoid_map(logits).squeeze(0).squeeze(0)
        return self._normalize_saliency_tensor(saliency_tensor)

    def predict_tensor(self, input_tensor: torch.Tensor, original_size: tuple[int, int]) -> tuple[np.ndarray, Image.Image]:
        """Run inference from a preprocessed NCHW tensor instead of a PIL image."""
        return self._format_prediction(self.predict_tensor_raw(input_tensor), original_size)
