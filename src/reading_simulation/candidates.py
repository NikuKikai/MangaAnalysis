from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class CandidatePoint:
    roi_x: int
    roi_y: int
    page_x: float
    page_y: float
    saliency_score: float
    history_value: float
    inhibition_score: float
    distance_score: float
    final_score: float


def _max_pool_2d(array: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return array
    height, width = array.shape
    pooled = np.empty_like(array)
    for y in range(height):
        top = max(0, y - radius)
        bottom = min(height, y + radius + 1)
        for x in range(width):
            left = max(0, x - radius)
            right = min(width, x + radius + 1)
            pooled[y, x] = float(array[top:bottom, left:right].max())
    return pooled


def extract_local_maxima(
    saliency_map: np.ndarray,
    threshold_ratio: float,
    nms_radius: int,
    top_k: int,
) -> list[tuple[int, int, float]]:
    if saliency_map.size == 0:
        return []

    peak_threshold = float(saliency_map.max()) * threshold_ratio
    pooled = _max_pool_2d(saliency_map, nms_radius)
    mask = (saliency_map >= peak_threshold) & (saliency_map >= pooled - 1e-8)
    points = np.argwhere(mask)
    ranked = sorted(
        ((int(y), int(x), float(saliency_map[y, x])) for y, x in points),
        key=lambda item: item[2],
        reverse=True,
    )
    return ranked[:top_k]
