from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter


@dataclass(slots=True)
class RoiWindow:
    image: Image.Image
    fixation_x: float
    fixation_y: float
    origin_x: int
    origin_y: int
    crop_left: int
    crop_top: int
    crop_right: int
    crop_bottom: int
    page_width: int
    page_height: int

    @property
    def fixation_in_roi(self) -> tuple[float, float]:
        return float(self.fixation_x - self.crop_left + self.origin_x), float(self.fixation_y - self.crop_top + self.origin_y)


def extract_roi(page_image: Image.Image, fixation: tuple[float, float], roi_size: int) -> RoiWindow:
    page_width, page_height = page_image.size
    center_x = int(round(fixation[0]))
    center_y = int(round(fixation[1]))
    half = roi_size // 2

    left = center_x - half
    top = center_y - half
    right = left + roi_size
    bottom = top + roi_size

    crop_left = max(0, left)
    crop_top = max(0, top)
    crop_right = min(page_width, right)
    crop_bottom = min(page_height, bottom)

    cropped = page_image.crop((crop_left, crop_top, crop_right, crop_bottom))
    roi = Image.new("RGB", (roi_size, roi_size), (0, 0, 0))
    origin_x = crop_left - left
    origin_y = crop_top - top
    roi.paste(cropped, (origin_x, origin_y))

    return RoiWindow(
        image=roi,
        fixation_x=float(fixation[0]),
        fixation_y=float(fixation[1]),
        origin_x=origin_x,
        origin_y=origin_y,
        crop_left=crop_left,
        crop_top=crop_top,
        crop_right=crop_right,
        crop_bottom=crop_bottom,
        page_width=page_width,
        page_height=page_height,
    )


def apply_inverse_radial_blur(
    roi_image: Image.Image,
    fixation_in_roi: tuple[float, float],
    clear_radius: float,
    blur_level_count: int,
    max_blur_strength: float,
) -> Image.Image:
    sharp = np.asarray(roi_image, dtype=np.float32)
    level_count = max(int(blur_level_count), 2)
    blur_levels = np.linspace(0.0, max(float(max_blur_strength), 0.0), num=level_count, dtype=np.float32)
    blurred_stack: list[np.ndarray] = []
    for blur_strength in blur_levels:
        if blur_strength <= 1e-6:
            blurred_stack.append(sharp)
        else:
            blurred_image = roi_image.filter(ImageFilter.GaussianBlur(radius=float(blur_strength)))
            blurred_stack.append(np.asarray(blurred_image, dtype=np.float32))

    height, width = sharp.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    dx = grid_x - float(fixation_in_roi[0])
    dy = grid_y - float(fixation_in_roi[1])
    distance = np.sqrt(dx * dx + dy * dy)
    max_radius = max(min(width, height) * 0.5, clear_radius + 1e-6)

    target_blur = np.zeros_like(distance, dtype=np.float32)
    if max_radius > clear_radius:
        normalized = np.clip((distance - clear_radius) / (max_radius - clear_radius), 0.0, 1.0)
        target_blur = normalized * float(max_blur_strength)

    output = np.empty_like(sharp)
    for level_index in range(level_count - 1):
        lower = float(blur_levels[level_index])
        upper = float(blur_levels[level_index + 1])
        if level_index == level_count - 2:
            mask = (target_blur >= lower) & (target_blur <= upper)
        else:
            mask = (target_blur >= lower) & (target_blur < upper)
        if not np.any(mask):
            continue

        denom = max(upper - lower, 1e-6)
        alpha = ((target_blur - lower) / denom).astype(np.float32)
        alpha = np.clip(alpha, 0.0, 1.0)
        lower_image = blurred_stack[level_index]
        upper_image = blurred_stack[level_index + 1]
        blended = lower_image * (1.0 - alpha[..., None]) + upper_image * alpha[..., None]
        output[mask] = blended[mask]

    output_image = Image.fromarray(np.clip(output, 0.0, 255.0).astype(np.uint8), mode="RGB")

    # try:
    #     import cv2

    #     preview = cv2.cvtColor(np.asarray(output), cv2.COLOR_RGB2BGR)
    #     cv2.imshow("Inverse Radial Blur ROI", preview)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    # except Exception:
    #     pass

    return output_image
