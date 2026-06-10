from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from saliency.config import load_config
from saliency.metrics import sigmoid_map
from saliency.model import SaliencyUNet
from saliency.datasets.common import IMAGENET_MEAN, IMAGENET_STD, resize_with_padding


def resolve_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class SaliencyInference:
    def __init__(self, config_path: str, checkpoint_path: str) -> None:
        self.config_path = str(Path(config_path).resolve())
        self.checkpoint_path = str(Path(checkpoint_path).resolve())
        self.config = load_config(self.config_path)
        self.device = resolve_device(self.config.training.device)
        self.model = SaliencyUNet(
            encoder_pretrained=False,
            base_channels=self.config.model.base_channels,
        )
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device)
        self.model.eval()
        self.input_size = self.config.dataset.input_size

    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        resized = resize_with_padding(image.convert("RGB"), self.input_size, Image.Resampling.BILINEAR, (0, 0, 0))
        array = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        return tensor.unsqueeze(0)

    def predict(self, image: Image.Image) -> tuple[np.ndarray, Image.Image]:
        original_size = image.size
        input_tensor = self._image_to_tensor(image).to(self.device)
        with torch.no_grad():
            logits = self.model(input_tensor)
            saliency_tensor = sigmoid_map(logits).squeeze(0).squeeze(0).detach().cpu().numpy()

        saliency = saliency_tensor.astype(np.float32)
        saliency = saliency - float(saliency.min())
        saliency = saliency / max(float(saliency.max()), 1e-8)
        heatmap_image = Image.fromarray(np.clip(saliency * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
        if heatmap_image.size != original_size:
            heatmap_image = heatmap_image.resize(original_size, Image.Resampling.BILINEAR)
        resized_saliency = np.asarray(heatmap_image, dtype=np.float32) / 255.0
        resized_saliency = resized_saliency - float(resized_saliency.min())
        resized_saliency = resized_saliency / max(float(resized_saliency.max()), 1e-8)
        return resized_saliency, heatmap_image
