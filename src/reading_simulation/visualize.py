from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .types import SimulationResult


def _normalize_history(history_map: np.ndarray) -> np.ndarray:
    """Normalize a history map for visualization."""
    max_value = max(float(history_map.max()), 1e-8)
    return np.clip(history_map / max_value, 0.0, 1.0)


def _colorize_history(history_map: np.ndarray) -> np.ndarray:
    """Convert a normalized history map into a simple RGB heatmap."""
    normalized = _normalize_history(history_map)
    red = np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)
    green = np.clip(np.power(normalized, 0.6) * 180.0, 0.0, 255.0).astype(np.uint8)
    blue = np.clip((1.0 - normalized) * 90.0, 0.0, 255.0).astype(np.uint8)
    return np.stack([red, green, blue], axis=-1)


def _draw_fixation_path(base_image: Image.Image, result: SimulationResult) -> Image.Image:
    """Draw the realized scanpath plus one off-route candidate arrow per fixation."""
    image = base_image.copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    fixations = result.fixations

    if len(fixations) >= 2:
        draw.line([(float(x), float(y)) for x, y in fixations], fill=(0, 212, 255), width=4)

    # Scale arrow thickness by the strongest off-route score observed in the whole trace.
    max_outsider_score = 0.0
    if result.analysis is not None:
        for step in result.analysis.step_scores:
            max_outsider_score = max(max_outsider_score, float(step.score))

    for index, (x, y) in enumerate(fixations):
        radius = 8
        left = int(round(x)) - radius
        top = int(round(y)) - radius
        right = int(round(x)) + radius
        bottom = int(round(y)) + radius
        draw.ellipse((left, top, right, bottom), fill=(255, 80, 80), outline=(255, 255, 255), width=2)
        draw.text((int(round(x)) + 10, int(round(y)) + 8), str(index), fill=(255, 245, 0), font=font)

    # Draw distraction arrows from each fixation toward the strongest off-route candidate.
    if result.analysis is not None:
        for step in result.analysis.step_scores:
            if step.candidate is None or step.step_index >= len(result.fixations):
                continue
            start = result.fixations[step.step_index]
            end = (float(step.candidate.page_x), float(step.candidate.page_y))
            normalized = 0.0 if max_outsider_score <= 1e-8 else float(step.score) / max_outsider_score
            width = max(1, int(round(1 + normalized * 7)))
            color = (255, 64, 64) if step.stronger_than_actual_next else (255, 192, 0)
            _draw_arrow(draw, start, end, color=color, width=width)

    return image


def _blend_history(base_image: Image.Image, history_map: np.ndarray) -> Image.Image:
    """Overlay the final history map on top of the original page."""
    base = np.asarray(base_image.convert("RGB"), dtype=np.float32)
    heat = _colorize_history(history_map).astype(np.float32)
    blended = base * 0.45 + heat * 0.55
    return Image.fromarray(np.clip(blended, 0.0, 255.0).astype(np.uint8), mode="RGB")


def _label_panel(panel: Image.Image, title: str) -> Image.Image:
    """Wrap an image panel with a title bar."""
    title_height = 34
    canvas = Image.new("RGB", (panel.width, panel.height + title_height), (20, 20, 20))
    canvas.paste(panel, (0, title_height))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((12, 10), title, fill=(240, 240, 240), font=font)
    return canvas


def _draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[float, float], end: tuple[float, float], color: tuple[int, int, int], width: int) -> None:
    """Draw a simple arrow from one point to another."""
    import math

    draw.line([start, end], fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    head_length = 14 + width
    left = (
        end[0] - head_length * math.cos(angle - math.pi / 7.0),
        end[1] - head_length * math.sin(angle - math.pi / 7.0),
    )
    right = (
        end[0] - head_length * math.cos(angle + math.pi / 7.0),
        end[1] - head_length * math.sin(angle + math.pi / 7.0),
    )
    draw.polygon([end, left, right], fill=color)


def build_visualization(result: SimulationResult) -> np.ndarray:
    """Build the default two-panel visualization for one simulation result."""
    page_image = Image.open(result.page_path).convert("RGB")
    path_panel = _label_panel(_draw_fixation_path(page_image, result), "Fixation Path")
    history_panel = _label_panel(_blend_history(page_image, result.history_map), "History Suppression Map")

    canvas = Image.new("RGB", (path_panel.width + history_panel.width, max(path_panel.height, history_panel.height)), (0, 0, 0))
    canvas.paste(path_panel, (0, 0))
    canvas.paste(history_panel, (path_panel.width, 0))
    return np.asarray(canvas, dtype=np.uint8)


def show_visualization(result: SimulationResult) -> None:
    """Display the visualization with OpenCV when available, or save it as a fallback."""
    image = build_visualization(result)
    output_path = Path(result.page_path).with_name(f"{Path(result.page_path).stem}_reading_path.png")

    try:
        import cv2

        cv2.imshow("Reading Simulation", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    except Exception:
        Image.fromarray(image).save(output_path)
        print("cv2 is unavailable or window display failed.")
        print(f"Saved visualization to: {output_path}")


def save_visualization(result: SimulationResult, output_path: str) -> None:
    """Save the visualization image to disk."""
    image = build_visualization(result)
    Image.fromarray(image).save(Path(output_path))
