from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from panel_order.types import Rect

from .candidates import CandidatePoint, extract_local_maxima
from .config import SimulationConfig
from .history import inhibition_factor
from .retina import RoiWindow, apply_inverse_radial_blur, extract_roi
from .saliency_inference import SaliencyInference


@dataclass(slots=True)
class RoiEvaluation:
    """Candidate evaluation result for one ROI size at one fixation."""

    roi_window: RoiWindow
    candidates: list[CandidatePoint]


class SaliencyScanpathEngine:
    """Shared saliency engine used by all reading-simulation strategies."""

    def __init__(self, inference: SaliencyInference, config: SimulationConfig) -> None:
        """Create a reusable saliency engine with shared scoring logic."""
        self.inference = inference
        self.config = config

    def scale_float(self, page_height: int, ratio: float, minimum: float = 1.0) -> float:
        """Scale a ratio by page height and clamp it to a minimum value."""
        return max(float(page_height) * ratio, minimum)

    def scale_int(self, page_height: int, ratio: float, minimum: int = 1) -> int:
        """Scale a ratio by page height, round it, and clamp it to a minimum value."""
        return max(int(round(float(page_height) * ratio)), minimum)

    def distance_score(self, from_point: tuple[float, float], to_point: tuple[float, float], distance_sigma: float) -> float:
        """Compute the Gaussian distance regularizer used by both strategies."""
        dx = float(to_point[0] - from_point[0])
        dy = float(to_point[1] - from_point[1])
        distance_sq = dx * dx + dy * dy
        sigma_sq = max(distance_sigma * distance_sigma, 1e-6)
        return float(np.exp(-0.5 * distance_sq / sigma_sq))

    def create_default_roi_size(self, page_height: int) -> int:
        """Return the default fixation-centered square ROI size."""
        half_size = self.scale_float(page_height, self.config.default_roi_half_size_ratio)
        return max(int(round(half_size * 2.0)), 1)

    def create_full_page_roi_size(self, page_width: int, page_height: int) -> int:
        """Return the square ROI size that covers the full page."""
        return max(page_width, page_height)

    def evaluate_roi(
        self,
        page_image: Image.Image,
        fixation: tuple[float, float],
        roi_size: int,
        history_map: np.ndarray,
        page_height: int,
    ) -> RoiEvaluation:
        """Run preprocessing, saliency inference, and candidate scoring for one ROI."""
        # Build the retina-like ROI around the current fixation.
        roi_window = extract_roi(page_image, fixation, roi_size)
        clear_radius = self.scale_float(page_height, self.config.clear_radius_ratio)
        retinal_roi = apply_inverse_radial_blur(
            roi_window.image,
            roi_window.fixation_in_roi,
            clear_radius,
            self.config.blur_level_count,
            self.config.max_blur_strength,
        )

        # Infer saliency on the processed ROI and rescore it in page coordinates.
        saliency_map, _saliency_image = self.inference.predict(retinal_roi)
        distance_sigma = self.scale_float(page_height, self.config.distance_sigma_ratio)
        nms_radius = self.scale_int(page_height, self.config.nms_radius_ratio)
        candidates = self._score_candidates(saliency_map, roi_window, fixation, history_map, distance_sigma, nms_radius)
        return RoiEvaluation(roi_window=roi_window, candidates=candidates)

    def filter_candidates_to_rect(self, candidates: list[CandidatePoint], rect: Rect | None) -> list[CandidatePoint]:
        """Keep only candidates that fall inside a target rectangle."""
        if rect is None:
            return list(candidates)
        filtered = []
        for candidate in candidates:
            if rect.x <= candidate.page_x <= rect.right and rect.y <= candidate.page_y <= rect.bottom:
                filtered.append(candidate)
        return filtered

    def candidate_to_point(self, candidate: CandidatePoint) -> tuple[float, float]:
        """Convert a candidate record into a fixation tuple."""
        return (float(candidate.page_x), float(candidate.page_y))

    def _score_candidates(
        self,
        saliency_map: np.ndarray,
        roi_window: RoiWindow,
        current_fixation: tuple[float, float],
        history_map: np.ndarray,
        distance_sigma: float,
        nms_radius: int,
    ) -> list[CandidatePoint]:
        """Score ROI saliency peaks in page coordinates using history and distance terms."""
        final_score_map = np.zeros_like(saliency_map, dtype=np.float32)
        saliency_height, saliency_width = saliency_map.shape
        if saliency_height == 0 or saliency_width == 0:
            return []

        # First build a full rescored map in ROI space so NMS runs on final transition scores.
        for roi_y in range(saliency_height):
            for roi_x in range(saliency_width):
                page_x = roi_window.crop_left + (float(roi_x) + 0.5) - float(roi_window.origin_x)
                page_y = roi_window.crop_top + (float(roi_y) + 0.5) - float(roi_window.origin_y)
                if page_x < 0.0 or page_x >= roi_window.page_width or page_y < 0.0 or page_y >= roi_window.page_height:
                    continue
                sample_y = int(np.clip(round(page_y), 0, roi_window.page_height - 1))
                sample_x = int(np.clip(round(page_x), 0, roi_window.page_width - 1))
                history_value = float(history_map[sample_y, sample_x])
                inhib = inhibition_factor(history_value, self.config.history_alpha)
                dist = self.distance_score(current_fixation, (page_x, page_y), distance_sigma)
                final_score_map[roi_y, roi_x] = float(saliency_map[roi_y, roi_x]) * inhib * dist

        # Then turn the final-score peaks back into detailed candidate records.
        peaks = extract_local_maxima(final_score_map, nms_radius, self.config.top_k)
        scored: list[CandidatePoint] = []
        for roi_y, roi_x, _peak_score in peaks:
            page_x = roi_window.crop_left + (float(roi_x) + 0.5) - float(roi_window.origin_x)
            page_y = roi_window.crop_top + (float(roi_y) + 0.5) - float(roi_window.origin_y)
            if page_x < 0 or page_x >= roi_window.page_width or page_y < 0 or page_y >= roi_window.page_height:
                continue

            sample_y = int(np.clip(round(page_y), 0, roi_window.page_height - 1))
            sample_x = int(np.clip(round(page_x), 0, roi_window.page_width - 1))
            history_value = float(history_map[sample_y, sample_x])
            inhib = inhibition_factor(history_value, self.config.history_alpha)
            dist = self.distance_score(current_fixation, (page_x, page_y), distance_sigma)
            final_score = float(final_score_map[roi_y, roi_x])
            scored.append(
                CandidatePoint(
                    roi_x=int(roi_x),
                    roi_y=int(roi_y),
                    page_x=page_x,
                    page_y=page_y,
                    saliency_score=float(saliency_map[roi_y, roi_x]),
                    history_value=history_value,
                    inhibition_score=inhib,
                    distance_score=dist,
                    final_score=final_score,
                )
            )

        scored.sort(key=lambda item: item.final_score, reverse=True)
        return [candidate for candidate in scored if candidate.final_score >= self.config.threshold_score]
