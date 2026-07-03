from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch
from PIL import Image

from panel_order.types import Rect

from .candidates import CandidatePoint, extract_local_maxima_torch
from .config import SimulationConfig
from .retina import RoiWindow, WgpuRoiPreprocessor, extract_roi
from .saliency_inference import SaliencyInference


@dataclass(slots=True)
class RoiEvaluation:
    """Candidate evaluation result for one ROI size at one fixation."""

    roi_window: RoiWindow
    candidates: list[CandidatePoint]
    timings: dict[str, float]


class SaliencyScanpathEngine:
    """Shared saliency engine used by all reading-simulation strategies."""

    def __init__(self, inference: SaliencyInference, config: SimulationConfig) -> None:
        """Create a reusable saliency engine with shared scoring logic."""
        self.inference = inference
        self.config = config
        self._roi_grid_cache: dict[tuple[int, int, str], tuple[torch.Tensor, torch.Tensor]] = {}
        self._gpu_preprocessor = WgpuRoiPreprocessor(model_size=int(self.inference.input_size))
        self._scoring_device = self.inference.device
        self.history_device = self._scoring_device

    def scale_float(self, page_height: int, ratio: float, minimum: float = 1.0) -> float:
        """Scale a ratio by page height and clamp it to a minimum value."""
        return max(float(page_height) * ratio, minimum)

    def scale_int(self, page_height: int, ratio: float, minimum: int = 1) -> int:
        """Scale a ratio by page height, round it, and clamp it to a minimum value."""
        return max(int(round(float(page_height) * ratio)), minimum)

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
        history_map: torch.Tensor,
        page_height: int,
    ) -> RoiEvaluation:
        """Run preprocessing, saliency inference, and candidate scoring for one ROI."""
        timings: dict[str, float] = {}

        # Build the retina-like ROI around the current fixation.
        t0 = perf_counter()
        roi_window = extract_roi(page_image, fixation, roi_size)
        timings["extract_roi"] = perf_counter() - t0

        clear_radius = self.scale_float(page_height, self.config.clear_radius_ratio)
        t0 = perf_counter()
        preprocessed_tensor = self._gpu_preprocessor.preprocess(
            page_image=page_image,
            roi_window=roi_window,
            clear_radius=clear_radius,
            max_blur_strength=self.config.max_blur_strength,
        )
        timings["apply_inverse_radial_blur"] = perf_counter() - t0

        # Infer saliency on the processed ROI and rescore it in page coordinates.
        t0 = perf_counter()
        saliency_map = self.inference.predict_tensor_raw(preprocessed_tensor)
        timings["saliency_predict"] = perf_counter() - t0
        distance_sigma = self.scale_float(page_height, self.config.distance_sigma_ratio)
        nms_radius = max(1, int(round((page_height * self.config.nms_radius_ratio * saliency_map.shape[0]) / max(roi_size, 1))))
        t0 = perf_counter()
        candidates = self._score_candidates(saliency_map, roi_window, fixation, history_map, distance_sigma, nms_radius)
        timings["score_candidates_total"] = perf_counter() - t0
        return RoiEvaluation(roi_window=roi_window, candidates=candidates, timings=timings)

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
        saliency_map: torch.Tensor,
        roi_window: RoiWindow,
        current_fixation: tuple[float, float],
        history_map: torch.Tensor,
        distance_sigma: float,
        nms_radius: int,
    ) -> list[CandidatePoint]:
        """Score ROI saliency peaks on GPU using history and distance terms."""
        saliency_height, saliency_width = saliency_map.shape
        if saliency_height == 0 or saliency_width == 0:
            return []

        device = self._scoring_device
        saliency_tensor = saliency_map.to(device=device, dtype=torch.float32)
        grid_y, grid_x = self._get_roi_grids(saliency_height, saliency_width, device)

        # Convert ROI coordinates into page coordinates once, then reuse them for history and distance scoring.
        roi_scale_x = float(roi_window.image.width) / float(saliency_width)
        roi_scale_y = float(roi_window.image.height) / float(saliency_height)
        page_x = (grid_x * roi_scale_x) + float(roi_window.crop_left - roi_window.origin_x)
        page_y = (grid_y * roi_scale_y) + float(roi_window.crop_top - roi_window.origin_y)
        valid_mask = (
            (page_x >= 0.0)
            & (page_x < float(roi_window.page_width))
            & (page_y >= 0.0)
            & (page_y < float(roi_window.page_height))
        )

        # Gather the history map at the rounded page positions in one tensor pass.
        history_tensor = history_map.to(device=device, dtype=torch.float32)
        sample_x = torch.round(page_x).clamp(0, roi_window.page_width - 1).to(torch.int64)
        sample_y = torch.round(page_y).clamp(0, roi_window.page_height - 1).to(torch.int64)
        history_values = history_tensor[sample_y, sample_x]

        # Apply the same inhibition-of-return and distance terms used by the legacy implementation.
        sigma_sq = max(float(distance_sigma) * float(distance_sigma), 1e-6)
        dx = page_x - float(current_fixation[0])
        dy = page_y - float(current_fixation[1])
        distance_score = torch.exp(-0.5 * (dx * dx + dy * dy) / sigma_sq)
        inhibition_score = torch.exp(-float(self.config.history_alpha) * history_values)
        final_score_tensor = saliency_tensor * inhibition_score * distance_score * valid_mask.to(torch.float32)

        # Run NMS on-device so we avoid the large CPU local-window scan.
        peaks = extract_local_maxima_torch(final_score_tensor, nms_radius, self.config.top_k, self.config.threshold_score)
        scored: list[CandidatePoint] = []
        for roi_y, roi_x, _peak_score in peaks:
            page_x_value = float(page_x[roi_y, roi_x].item())
            page_y_value = float(page_y[roi_y, roi_x].item())
            if page_x_value < 0.0 or page_x_value >= roi_window.page_width or page_y_value < 0.0 or page_y_value >= roi_window.page_height:
                continue
            final_score = float(final_score_tensor[roi_y, roi_x].item())
            scored.append(
                CandidatePoint(
                    roi_x=int(roi_x),
                    roi_y=int(roi_y),
                    page_x=page_x_value,
                    page_y=page_y_value,
                    saliency_score=float(saliency_tensor[roi_y, roi_x].item()),
                    history_value=float(history_values[roi_y, roi_x].item()),
                    inhibition_score=float(inhibition_score[roi_y, roi_x].item()),
                    distance_score=float(distance_score[roi_y, roi_x].item()),
                    final_score=final_score,
                )
            )

        scored.sort(key=lambda item: item.final_score, reverse=True)
        return [candidate for candidate in scored if candidate.final_score >= self.config.threshold_score]

    def _get_roi_grids(self, saliency_height: int, saliency_width: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Cache ROI-space coordinate grids so repeated steps avoid rebuilding them."""
        cache_key = (saliency_height, saliency_width, str(device))
        cached = self._roi_grid_cache.get(cache_key)
        if cached is not None:
            return cached
        y_coords = torch.arange(saliency_height, device=device, dtype=torch.float32) + 0.5
        x_coords = torch.arange(saliency_width, device=device, dtype=torch.float32) + 0.5
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        self._roi_grid_cache[cache_key] = (grid_y, grid_x)
        return grid_y, grid_x
