from __future__ import annotations

import numpy as np
import torch

from panel_order.types import Rect


def create_history_map(page_height: int, page_width: int, device: torch.device) -> torch.Tensor:
    """Create an empty page-sized history map on the requested torch device."""
    return torch.zeros((page_height, page_width), dtype=torch.float32, device=device)


def add_fixation_to_history(
    history_map: torch.Tensor,
    fixation: tuple[float, float],
    sigma: float,
    decay: float = 1.0,
    amplitude: float = 1.0,
    clip_rect: Rect | None = None,
) -> None:
    """Add a Gaussian fixation bump to a device-resident history map, optionally clipped to a rectangle."""
    height, width = history_map.shape
    history_map.mul_(min(0.999, max(float(decay), 0.0)))
    center_x = float(fixation[0])
    center_y = float(fixation[1])
    radius = max(1, int(np.ceil(3.0 * sigma)))

    left = max(0, int(np.floor(center_x)) - radius)
    right = min(width, int(np.floor(center_x)) + radius + 1)
    top = max(0, int(np.floor(center_y)) - radius)
    bottom = min(height, int(np.floor(center_y)) + radius + 1)

    if left >= right or top >= bottom:
        return

    # Build the local Gaussian patch directly on the target device so history never leaves GPU memory.
    y_coords = torch.arange(top, bottom, device=history_map.device, dtype=torch.float32)
    x_coords = torch.arange(left, right, device=history_map.device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    dx = grid_x - center_x
    dy = grid_y - center_y
    gaussian = torch.exp(-0.5 * (dx * dx + dy * dy) / max(float(sigma) * float(sigma), 1e-6))

    # Optionally restrict the history write to the active panel rectangle.
    if clip_rect is not None:
        valid_x = (grid_x >= float(clip_rect.x)) & (grid_x <= float(clip_rect.right))
        valid_y = (grid_y >= float(clip_rect.y)) & (grid_y <= float(clip_rect.bottom))
        gaussian = gaussian * (valid_x & valid_y).to(torch.float32)

    history_map[top:bottom, left:right].add_(float(amplitude) * gaussian)


def materialize_history_map(history_map: torch.Tensor) -> np.ndarray:
    """Materialize a device-resident history map for visualization, serialization, or other result exports."""
    return history_map.detach().cpu().numpy().astype(np.float32, copy=False)


def inhibition_factor(history_value: float, alpha: float) -> float:
    """Convert one history-map value into an inhibition-of-return multiplier."""
    return float(np.exp(-alpha * history_value))
