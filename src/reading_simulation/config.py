from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(slots=True)
class SimulationConfig:
    """Top-level configuration shared by every reading-simulation strategy."""

    strategy: Literal["saliency_only", "panel_guided"] = field(
        default="saliency_only",
        metadata={"doc": "Strategy identifier used by the simulator dispatcher."},
    )
    default_roi_half_size_ratio: float = field(
        default=0.25,
        metadata={"doc": "Half of the default square ROI size as a ratio of page height."},
    )
    clear_radius_ratio: float = field(default=0.06, metadata={"doc": "Radius of the sharp foveal area as a ratio of page height."})
    max_blur_strength: float = field(default=7.0, metadata={"doc": "Maximum peripheral blur strength at the ROI outer radius."})
    history_sigma_ratio: float = field(default=0.047, metadata={"doc": "Sigma of the revisit-suppression Gaussian as a ratio of page height."})
    history_decay: float = field(default=0.94, metadata={"doc": "Per-step decay applied to the history map before adding the new fixation bump."})
    history_alpha: float = field(default=3.0, metadata={"doc": "Strength of history-based inhibition."})
    distance_sigma_ratio: float = field(default=0.183, metadata={"doc": "Sigma of the distance regularizer as a ratio of page height."})
    threshold_score: float = field(default=0.15, metadata={"doc": "Absolute minimum final candidate score kept after rescoring."})
    nms_radius_ratio: float = field(default=0.013, metadata={"doc": "Non-maximum suppression radius as a ratio of page height."})
    top_k: int = field(default=8, metadata={"doc": "Maximum number of local saliency peaks to keep per step."})
    steps: int | None = field(default=12, metadata={"doc": "Optional maximum number of transition steps after the initial fixation."})
