from __future__ import annotations

from .v1_res18_unet import SaliencyUNetV1
from .v2_res18_light_decoder import SaliencyUNetV2


MODEL_BUILDERS = {
    "v1": SaliencyUNetV1,
    "v2": SaliencyUNetV2,
}


def create_model_from_config(model_config) -> object:
    architecture_name = getattr(model_config, "architecture_name", "v1")
    if architecture_name not in MODEL_BUILDERS:
        raise ValueError(f"Unsupported architecture: {architecture_name}")
    builder = MODEL_BUILDERS[architecture_name]
    return builder(
        encoder_pretrained=model_config.encoder_pretrained,
        base_channels=model_config.base_channels,
        width_mult=getattr(model_config, "width_mult", 1.0),
    )
