from __future__ import annotations

import numpy as np


def create_history_map(page_height: int, page_width: int) -> np.ndarray:
    return np.zeros((page_height, page_width), dtype=np.float32)


def add_fixation_to_history(
    history_map: np.ndarray,
    fixation: tuple[float, float],
    sigma: float,
    amplitude: float = 1.0,
) -> None:
    height, width = history_map.shape
    center_x = float(fixation[0])
    center_y = float(fixation[1])
    radius = max(1, int(np.ceil(3.0 * sigma)))

    left = max(0, int(np.floor(center_x)) - radius)
    right = min(width, int(np.floor(center_x)) + radius + 1)
    top = max(0, int(np.floor(center_y)) - radius)
    bottom = min(height, int(np.floor(center_y)) + radius + 1)

    if left >= right or top >= bottom:
        return

    grid_x, grid_y = np.meshgrid(np.arange(left, right, dtype=np.float32), np.arange(top, bottom, dtype=np.float32))
    dx = grid_x - center_x
    dy = grid_y - center_y
    gaussian = np.exp(-0.5 * (dx * dx + dy * dy) / max(sigma * sigma, 1e-6))
    history_map[top:bottom, left:right] += amplitude * gaussian


def inhibition_factor(history_value: float, alpha: float) -> float:
    return float(np.exp(-alpha * history_value))
