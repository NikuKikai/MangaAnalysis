from __future__ import annotations

from .models import MODEL_BUILDERS, SaliencyUNetV1, SaliencyUNetV2, create_model_from_config

# Backward-compatible alias for existing training and export code paths.
SaliencyUNet = SaliencyUNetV1

__all__ = [
    "MODEL_BUILDERS",
    "SaliencyUNet",
    "SaliencyUNetV1",
    "SaliencyUNetV2",
    "create_model_from_config",
]
