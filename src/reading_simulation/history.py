from __future__ import annotations

import numpy as np

from panel_order.types import Rect


def create_history_map(page_height: int, page_width: int) -> np.ndarray:
    """Create an empty page-sized history map."""
    return np.zeros((page_height, page_width), dtype=np.float32)


def add_fixation_to_history(
    history_map: np.ndarray,
    fixation: tuple[float, float],
    sigma: float,
    decay: float = 1.0,
    amplitude: float = 1.0,
    clip_rect: Rect | None = None,
) -> None:
    """Add a Gaussian fixation bump to the history map, optionally clipped to a rectangle."""
    height, width = history_map.shape
    history_map *= min(0.999, max(float(decay), 0.0))
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

    # Optionally restrict the history write to a panel rectangle for panel-guided progress tracking.
    if clip_rect is not None:
        valid_x = (grid_x >= float(clip_rect.x)) & (grid_x <= float(clip_rect.right))
        valid_y = (grid_y >= float(clip_rect.y)) & (grid_y <= float(clip_rect.bottom))
        gaussian = gaussian * (valid_x & valid_y)

    history_map[top:bottom, left:right] += amplitude * gaussian


def inhibition_factor(history_value: float, alpha: float) -> float:
    """Convert one history-map value into an inhibition-of-return multiplier."""
    return float(np.exp(-alpha * history_value))
