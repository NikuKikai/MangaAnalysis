from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .simulator import SimulationResult


def _normalize_history(history_map: np.ndarray) -> np.ndarray:
    max_value = max(float(history_map.max()), 1e-8)
    return np.clip(history_map / max_value, 0.0, 1.0)


def _colorize_history(history_map: np.ndarray) -> np.ndarray:
    normalized = _normalize_history(history_map)
    red = np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)
    green = np.clip(np.power(normalized, 0.6) * 180.0, 0.0, 255.0).astype(np.uint8)
    blue = np.clip((1.0 - normalized) * 90.0, 0.0, 255.0).astype(np.uint8)
    return np.stack([red, green, blue], axis=-1)


def _draw_fixation_path(base_image: Image.Image, fixations: list[tuple[float, float]]) -> Image.Image:
    image = base_image.copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    if len(fixations) >= 2:
        draw.line([(float(x), float(y)) for x, y in fixations], fill=(0, 212, 255), width=4)

    for index, (x, y) in enumerate(fixations):
        radius = 8
        left = int(round(x)) - radius
        top = int(round(y)) - radius
        right = int(round(x)) + radius
        bottom = int(round(y)) + radius
        draw.ellipse((left, top, right, bottom), fill=(255, 80, 80), outline=(255, 255, 255), width=2)
        draw.text((int(round(x)) + 10, int(round(y)) + 8), str(index), fill=(255, 245, 0), font=font)

    return image


def _blend_history(base_image: Image.Image, history_map: np.ndarray) -> Image.Image:
    base = np.asarray(base_image.convert("RGB"), dtype=np.float32)
    heat = _colorize_history(history_map).astype(np.float32)
    blended = base * 0.45 + heat * 0.55
    return Image.fromarray(np.clip(blended, 0.0, 255.0).astype(np.uint8), mode="RGB")


def _label_panel(panel: Image.Image, title: str) -> Image.Image:
    title_height = 34
    canvas = Image.new("RGB", (panel.width, panel.height + title_height), (20, 20, 20))
    canvas.paste(panel, (0, title_height))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((12, 10), title, fill=(240, 240, 240), font=font)
    return canvas


def build_visualization(result: SimulationResult) -> np.ndarray:
    page_image = Image.open(result.page_path).convert("RGB")
    path_panel = _label_panel(_draw_fixation_path(page_image, result.fixations), "Fixation Path")
    history_panel = _label_panel(_blend_history(page_image, result.history_map), "History Suppression Map")

    canvas = Image.new("RGB", (path_panel.width + history_panel.width, max(path_panel.height, history_panel.height)), (0, 0, 0))
    canvas.paste(path_panel, (0, 0))
    canvas.paste(history_panel, (path_panel.width, 0))
    return np.asarray(canvas, dtype=np.uint8)


def show_visualization(result: SimulationResult) -> None:
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
    image = build_visualization(result)
    Image.fromarray(image).save(Path(output_path))
