from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class CandidatePoint:
    """Scored candidate fixation point expressed in both ROI and page coordinates."""

    roi_x: int
    roi_y: int
    page_x: float
    page_y: float
    saliency_score: float
    history_value: float
    inhibition_score: float
    distance_score: float
    final_score: float

def extract_local_maxima_torch(
    score_map: torch.Tensor,
    nms_radius: int,
    top_k: int,
    min_score: float = 0.0,
) -> list[tuple[int, int, float]]:
    """Return top local maxima from a 2D torch score map using tensor max-pooling."""
    if score_map.ndim != 2 or score_map.numel() == 0:
        return []

    if nms_radius <= 0:
        pooled = score_map
    else:
        kernel_size = int(nms_radius) * 2 + 1
        pooled = F.max_pool2d(
            score_map.unsqueeze(0).unsqueeze(0),
            kernel_size=kernel_size,
            stride=1,
            padding=int(nms_radius),
        ).squeeze(0).squeeze(0)

    # Keep only true local maxima above the requested minimum score.
    peak_mask = (score_map >= float(min_score)) & (score_map >= pooled - 1e-8)
    peak_indices = torch.nonzero(peak_mask, as_tuple=False)
    if peak_indices.numel() == 0:
        return []

    peak_scores = score_map[peak_mask]
    keep_count = min(int(top_k), int(peak_scores.numel()))
    top_scores, top_order = torch.topk(peak_scores, k=keep_count, largest=True, sorted=True)
    top_indices = peak_indices[top_order]
    return [
        (int(index[0].item()), int(index[1].item()), float(score.item()))
        for index, score in zip(top_indices, top_scores, strict=False)
    ]
