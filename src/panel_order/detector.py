from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .types import Panel, Rect


@dataclass(slots=True)
class PanelDetectionConfig:
    """Thresholds and search ranges used by the panel detector."""

    adaptive_block_size: int = 31
    adaptive_c: int = 10
    close_kernel_size: int = 5
    min_area_ratio: float = 0.01
    max_area_ratio: float = 0.92
    min_rectangularity: float = 0.82
    max_polygon_vertices: int = 10
    min_aspect_ratio: float = 0.12
    max_aspect_ratio: float = 8.0
    min_border_score: float = 0.35
    refine_margin_ratio: float = 0.18
    max_refine_margin_px: int = 80
    border_band_px: int = 5
    overlap_iou_threshold: float = 0.55
    contain_threshold: float = 0.9


class PanelDetector:
    """Detect rectangular manga panels from a single page image."""

    def __init__(self, config: PanelDetectionConfig | None = None) -> None:
        """Create a detector with either the provided config or default thresholds."""
        self.config = config or PanelDetectionConfig()

    def detect(self, image: np.ndarray) -> list[Panel]:
        """Run the full panel detection pipeline on an input image."""
        # Stage 1: Normalize the input into a grayscale image and extract panel lines.
        gray = self._to_grayscale(image)
        line_mask = self._build_line_mask(gray)

        # Stage 2: Convert contours into panel candidates with geometric filtering.
        raw_candidates = self._extract_candidates(gray, line_mask)

        # Stage 3: Remove duplicates and assign stable panel ids in page order.
        panels = self._deduplicate(raw_candidates)
        ordered = sorted(panels, key=lambda panel: (panel.rect.y, panel.rect.x))
        return [Panel(panel_id=index, rect=panel.rect, score=panel.score) for index, panel in enumerate(ordered)]

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert a color page image into grayscale when needed."""
        if image.ndim == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _build_line_mask(self, gray: np.ndarray) -> np.ndarray:
        """Build a binary mask that highlights panel borders and gutters."""
        # Blur first so adaptive thresholding reacts to panel lines instead of screentone noise.
        blurred = cv2.GaussianBlur(gray, (5, 5), 0.0)
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.config.adaptive_block_size,
            self.config.adaptive_c,
        )

        # Close small gaps so broken border strokes can still form stable contours.
        kernel = np.ones((self.config.close_kernel_size, self.config.close_kernel_size), dtype=np.uint8)
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    def _extract_candidates(self, gray: np.ndarray, line_mask: np.ndarray) -> list[Panel]:
        """Extract panel candidates from the line mask using contour geometry."""
        contours, _hierarchy = cv2.findContours(line_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        page_area = gray.shape[0] * gray.shape[1]
        candidates: list[Panel] = []

        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            area = width * height

            # Filter out contours that are too small to be panels or too large to be useful.
            if area < page_area * self.config.min_area_ratio or area > page_area * self.config.max_area_ratio:
                continue

            aspect_ratio = float(width) / float(max(height, 1))
            if aspect_ratio < self.config.min_aspect_ratio or aspect_ratio > self.config.max_aspect_ratio:
                continue

            # Keep only contours that are close to rectangular panel outlines.
            contour_area = float(cv2.contourArea(contour))
            rectangularity = contour_area / float(max(area, 1))
            if rectangularity < self.config.min_rectangularity:
                continue

            perimeter = cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(polygon) > self.config.max_polygon_vertices:
                continue

            # Refine the raw contour bounds by snapping each side toward stronger border lines.
            refined = self._refine_rect(line_mask, Rect(x, y, width, height))
            border_score = self._border_score(line_mask, refined)
            if border_score < self.config.min_border_score:
                continue

            score = rectangularity * 0.6 + border_score * 0.4
            candidates.append(Panel(panel_id=-1, rect=refined, score=float(score)))

        return candidates

    def _refine_rect(self, line_mask: np.ndarray, rect: Rect) -> Rect:
        """Refine a rectangle by searching for stronger borders near each side."""
        margin_x = min(self.config.max_refine_margin_px, max(6, int(rect.width * self.config.refine_margin_ratio)))
        margin_y = min(self.config.max_refine_margin_px, max(6, int(rect.height * self.config.refine_margin_ratio)))
        band = self.config.border_band_px

        # Search inward from each edge so loose contour boxes collapse onto visible panel borders.
        left = self._best_vertical_boundary(line_mask, rect.x, rect.y, rect.height, +1, margin_x, band)
        right = self._best_vertical_boundary(line_mask, rect.right - band, rect.y, rect.height, -1, margin_x, band) + band
        top = self._best_horizontal_boundary(line_mask, rect.y, rect.x, rect.width, +1, margin_y, band)
        bottom = self._best_horizontal_boundary(line_mask, rect.bottom - band, rect.x, rect.width, -1, margin_y, band) + band

        if right <= left + 10 or bottom <= top + 10:
            return rect
        return Rect(left, top, right - left, bottom - top)

    def _best_vertical_boundary(
        self,
        line_mask: np.ndarray,
        edge_x: int,
        top: int,
        height: int,
        direction: int,
        max_shift: int,
        band: int,
    ) -> int:
        """Find the strongest nearby vertical border band for one rectangle side."""
        height_limit, width_limit = line_mask.shape
        top = max(0, top)
        bottom = min(height_limit, top + height)
        best_x = int(np.clip(edge_x, 0, width_limit - band))
        best_value = -1.0

        # Prefer strong border evidence, but penalize long shifts so boundaries stay near the seed box.
        for shift in range(max_shift + 1):
            x = int(np.clip(edge_x + direction * shift, 0, width_limit - band))
            score = float(line_mask[top:bottom, x : x + band].mean()) / 255.0
            score -= 0.3 * (float(shift) / float(max(max_shift, 1)))
            if score > best_value:
                best_value = score
                best_x = x

        return best_x

    def _best_horizontal_boundary(
        self,
        line_mask: np.ndarray,
        edge_y: int,
        left: int,
        width: int,
        direction: int,
        max_shift: int,
        band: int,
    ) -> int:
        """Find the strongest nearby horizontal border band for one rectangle side."""
        height_limit, width_limit = line_mask.shape
        left = max(0, left)
        right = min(width_limit, left + width)
        best_y = int(np.clip(edge_y, 0, height_limit - band))
        best_value = -1.0

        # Use the same border search strategy vertically for the top and bottom edges.
        for shift in range(max_shift + 1):
            y = int(np.clip(edge_y + direction * shift, 0, height_limit - band))
            score = float(line_mask[y : y + band, left:right].mean()) / 255.0
            score -= 0.3 * (float(shift) / float(max(max_shift, 1)))
            if score > best_value:
                best_value = score
                best_y = y

        return best_y

    def _border_score(self, line_mask: np.ndarray, rect: Rect) -> float:
        """Score how strongly all four rectangle sides align with panel borders."""
        band = min(self.config.border_band_px, rect.width, rect.height)
        if band <= 0:
            return 0.0
        top = float(line_mask[rect.y : rect.y + band, rect.x : rect.right].mean()) / 255.0
        bottom = float(line_mask[rect.bottom - band : rect.bottom, rect.x : rect.right].mean()) / 255.0
        left = float(line_mask[rect.y : rect.bottom, rect.x : rect.x + band].mean()) / 255.0
        right = float(line_mask[rect.y : rect.bottom, rect.right - band : rect.right].mean()) / 255.0
        return (top + bottom + left + right) / 4.0

    def _deduplicate(self, candidates: list[Panel]) -> list[Panel]:
        """Remove overlapping candidates while keeping the highest-quality box."""
        kept: list[Panel] = []
        ranked = sorted(candidates, key=lambda panel: (panel.score, panel.rect.area), reverse=True)
        for candidate in ranked:
            should_keep = True
            for existing in kept:
                # Suppress candidates that are almost contained by, or heavily overlap with, a kept box.
                if existing.rect.contains_ratio(candidate.rect) >= self.config.contain_threshold:
                    should_keep = False
                    break
                if candidate.rect.iou(existing.rect) >= self.config.overlap_iou_threshold:
                    should_keep = False
                    break
            if should_keep:
                kept.append(candidate)

        return sorted(kept, key=lambda panel: (panel.rect.y, panel.rect.x))
