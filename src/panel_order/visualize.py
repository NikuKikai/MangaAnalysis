from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .types import AnalysisResult


def render_panel_overlay(image: np.ndarray, result: AnalysisResult) -> np.ndarray:
    """Draw detected panel boxes and reading-order labels on top of an image."""
    # Normalize the source image to BGR so OpenCV drawing always works.
    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = image.copy()

    # Map stable panel ids to human-facing 1-based reading-order labels.
    panel_to_order = {panel_id: index + 1 for index, panel_id in enumerate(result.reading_order)}
    palette = [
        (44, 127, 184),
        (55, 180, 95),
        (238, 119, 51),
        (170, 51, 119),
        (102, 194, 165),
        (141, 160, 203),
    ]

    # Render one rectangle and label per panel for quick visual inspection.
    for panel in result.panels:
        color = palette[panel_to_order[panel.panel_id] % len(palette)]
        x, y = panel.rect.x, panel.rect.y
        right, bottom = panel.rect.right, panel.rect.bottom
        cv2.rectangle(canvas, (x, y), (right, bottom), color, 3)
        label = f"{panel_to_order[panel.panel_id]}:P{panel.panel_id}"
        cv2.rectangle(canvas, (x, max(0, y - 28)), (x + 104, y), color, -1)
        cv2.putText(canvas, label, (x + 5, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


def save_panel_overlay(image: np.ndarray, result: AnalysisResult, output_path: str) -> None:
    """Render an overlay image and save it to disk."""
    rendered = render_panel_overlay(image, result)
    cv2.imwrite(str(Path(output_path)), rendered)
