from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from PIL import Image

from .candidates import CandidatePoint, extract_local_maxima
from .history import add_fixation_to_history, create_history_map, inhibition_factor
from .retina import RoiWindow, apply_inverse_radial_blur, extract_roi
from .saliency_inference import SaliencyInference


@dataclass(slots=True)
class SimulationConfig:
    """
    Configuration for the fixation-sequence simulation.

    Attributes:
        roi_size_ratio: ROI crop size as a ratio of page height.
        clear_radius_ratio: Radius of the sharp foveal area as a ratio of page height.
        blur_level_count: Number of blur levels used to approximate continuous peripheral blur.
        max_blur_strength: Maximum peripheral blur strength at the ROI outer radius.
        history_sigma_ratio: Sigma of the revisit-suppression Gaussian as a ratio of page height.
        history_alpha: Strength of history-based inhibition.
        distance_sigma_ratio: Sigma of the distance regularizer as a ratio of page height.
        start_region_radius_ratio: Size of the top-right start region as a ratio of page height.
        start_corner_weight: Blend weight for top-right proximity when selecting the initial fixation.
        threshold_ratio: Candidate threshold relative to the maximum saliency value in the current ROI.
        nms_radius_ratio: Non-maximum suppression radius as a ratio of page height.
        top_k: Maximum number of local saliency peaks to keep per step.
        steps: Maximum number of transition steps after the initial fixation.
    """

    roi_size_ratio: float = field(default=0.64, metadata={"doc": "ROI crop size as a ratio of page height."})
    clear_radius_ratio: float = field(default=0.06, metadata={"doc": "Radius of the sharp foveal area as a ratio of page height."})
    blur_level_count: int = field(default=8, metadata={"doc": "Number of blur levels used to approximate continuous peripheral blur."})
    max_blur_strength: float = field(default=16.0, metadata={"doc": "Maximum peripheral blur strength at the ROI outer radius."})
    history_sigma_ratio: float = field(default=0.047, metadata={"doc": "Sigma of the revisit-suppression Gaussian as a ratio of page height."})
    history_alpha: float = field(default=2.0, metadata={"doc": "Strength of history-based inhibition."})
    distance_sigma_ratio: float = field(default=0.183, metadata={"doc": "Sigma of the distance regularizer as a ratio of page height."})
    start_region_radius_ratio: float = field(default=0.25, metadata={"doc": "Size of the top-right start region as a ratio of page height."})
    start_corner_weight: float = field(default=0.35, metadata={"doc": "Blend weight for top-right proximity when selecting the initial fixation."})
    threshold_ratio: float = field(default=0.55, metadata={"doc": "Candidate threshold relative to the maximum ROI saliency value."})
    nms_radius_ratio: float = field(default=0.013, metadata={"doc": "Non-maximum suppression radius as a ratio of page height."})
    top_k: int = field(default=8, metadata={"doc": "Maximum number of local saliency peaks to keep per step."})
    steps: int = field(default=8, metadata={"doc": "Maximum number of transition steps after the initial fixation."})


@dataclass(slots=True)
class StepDebug:
    step_index: int
    phase: str
    fixation: tuple[float, float] | None
    history_map: np.ndarray
    roi_window: RoiWindow
    retinal_roi: Image.Image
    saliency_map: np.ndarray
    saliency_image: Image.Image
    candidates: list[CandidatePoint]
    selected_candidate: CandidatePoint | None


@dataclass(slots=True)
class SimulationResult:
    page_path: str
    fixations: list[tuple[float, float]]
    history_map: np.ndarray
    step_debug: list[StepDebug]

    def to_json(self) -> dict:
        return {
            "page_path": self.page_path,
            "fixations": [[float(x), float(y)] for x, y in self.fixations],
            "steps": [
                {
                    "step_index": step.step_index,
                    "phase": step.phase,
                    "fixation": None if step.fixation is None else [float(step.fixation[0]), float(step.fixation[1])],
                    "selected_candidate": asdict(step.selected_candidate) if step.selected_candidate is not None else None,
                    "candidates": [asdict(candidate) for candidate in step.candidates],
                }
                for step in self.step_debug
            ],
        }


class ReadingSimulator:
    def __init__(self, inference: SaliencyInference, config: SimulationConfig | None = None) -> None:
        self.inference = inference
        self.config = config or SimulationConfig()

    def _scale_float(self, page_height: int, ratio: float, minimum: float = 1.0) -> float:
        return max(float(page_height) * ratio, minimum)

    def _scale_int(self, page_height: int, ratio: float, minimum: int = 1) -> int:
        return max(int(round(float(page_height) * ratio)), minimum)

    def _distance_score(self, from_point: tuple[float, float], to_point: tuple[float, float], distance_sigma: float) -> float:
        dx = float(to_point[0] - from_point[0])
        dy = float(to_point[1] - from_point[1])
        distance_sq = dx * dx + dy * dy
        sigma_sq = max(distance_sigma * distance_sigma, 1e-6)
        return float(np.exp(-0.5 * distance_sq / sigma_sq))

    def _create_start_region_window(self, page_image: Image.Image, page_height: int) -> RoiWindow:
        page_width, _ = page_image.size
        region_size = self._scale_int(page_height, self.config.start_region_radius_ratio)
        region_size = min(region_size, page_width, page_height)

        left = max(0, page_width - region_size)
        top = 0
        right = page_width
        bottom = region_size
        start_region = page_image.crop((left, top, right, bottom))
        return RoiWindow(
            image=start_region,
            fixation_x=0.0,
            fixation_y=0.0,
            origin_x=0,
            origin_y=0,
            crop_left=left,
            crop_top=top,
            crop_right=right,
            crop_bottom=bottom,
            page_width=page_width,
            page_height=page_height,
        )

    def _select_initial_step(self, page_image: Image.Image, page_height: int, nms_radius: int) -> StepDebug:
        roi_window = self._create_start_region_window(page_image, page_height)
        start_region = roi_window.image
        saliency_map, _ = self.inference.predict(start_region)

        region_height, region_width = saliency_map.shape
        grid_x, grid_y = np.meshgrid(np.arange(region_width, dtype=np.float32), np.arange(region_height, dtype=np.float32))
        proximity_x = 1.0 - (float(region_width - 1) - grid_x) / max(float(region_width - 1), 1.0)
        proximity_y = 1.0 - grid_y / max(float(region_height - 1), 1.0)
        corner_proximity = np.clip(0.5 * (proximity_x + proximity_y), 0.0, 1.0)

        weight = float(np.clip(self.config.start_corner_weight, 0.0, 1.0))
        combined = (1.0 - weight) * saliency_map + weight * corner_proximity
        peaks = extract_local_maxima(
            combined,
            self.config.threshold_ratio,
            nms_radius,
            self.config.top_k,
        )
        candidates: list[CandidatePoint] = []
        for roi_y, roi_x, score in peaks:
            page_x = float(roi_window.crop_left + roi_x)
            page_y = float(roi_window.crop_top + roi_y)
            candidates.append(
                CandidatePoint(
                    roi_x=int(roi_x),
                    roi_y=int(roi_y),
                    page_x=page_x,
                    page_y=page_y,
                    saliency_score=float(score),
                    history_value=0.0,
                    inhibition_score=1.0,
                    distance_score=1.0,
                    final_score=float(score),
                )
            )

        candidates.sort(key=lambda item: item.final_score, reverse=True)
        saliency_image = Image.fromarray(np.clip(combined * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
        return StepDebug(
            step_index=0,
            phase="initial",
            fixation=None,
            history_map=np.zeros((page_height, page_image.size[0]), dtype=np.float32),
            roi_window=roi_window,
            retinal_roi=start_region,
            saliency_map=combined,
            saliency_image=saliency_image,
            candidates=candidates,
            selected_candidate=candidates[0] if candidates else None,
        )

    def _score_candidates(
        self,
        peaks: list[tuple[int, int, float]],
        roi_window: RoiWindow,
        current_fixation: tuple[float, float],
        history_map: np.ndarray,
        distance_sigma: float,
    ) -> list[CandidatePoint]:
        scored: list[CandidatePoint] = []
        for roi_y, roi_x, saliency_score in peaks:
            page_x = float(roi_window.crop_left + (roi_x - roi_window.origin_x))
            page_y = float(roi_window.crop_top + (roi_y - roi_window.origin_y))
            if page_x < 0 or page_x >= roi_window.page_width or page_y < 0 or page_y >= roi_window.page_height:
                continue

            history_value = float(history_map[int(round(page_y)), int(round(page_x))])
            inhib = inhibition_factor(history_value, self.config.history_alpha)
            dist = self._distance_score(current_fixation, (page_x, page_y), distance_sigma)
            final_score = float(saliency_score) * inhib * dist
            scored.append(
                CandidatePoint(
                    roi_x=int(roi_x),
                    roi_y=int(roi_y),
                    page_x=page_x,
                    page_y=page_y,
                    saliency_score=float(saliency_score),
                    history_value=history_value,
                    inhibition_score=inhib,
                    distance_score=dist,
                    final_score=final_score,
                )
            )

        scored.sort(key=lambda item: item.final_score, reverse=True)
        return scored

    def simulate(self, page_path: str) -> SimulationResult:
        page_image = Image.open(page_path).convert("RGB")
        page_width, page_height = page_image.size
        roi_size = self._scale_int(page_height, self.config.roi_size_ratio)
        clear_radius = self._scale_float(page_height, self.config.clear_radius_ratio)
        history_sigma = self._scale_float(page_height, self.config.history_sigma_ratio)
        distance_sigma = self._scale_float(page_height, self.config.distance_sigma_ratio)
        nms_radius = self._scale_int(page_height, self.config.nms_radius_ratio)
        history_map = create_history_map(page_height, page_width)
        initial_step = self._select_initial_step(page_image, page_height, nms_radius)
        step_debug: list[StepDebug] = [initial_step]
        fixations: list[tuple[float, float]] = []

        if initial_step.selected_candidate is None:
            return SimulationResult(
                page_path=page_path,
                fixations=fixations,
                history_map=history_map,
                step_debug=step_debug,
            )

        current_fixation = (initial_step.selected_candidate.page_x, initial_step.selected_candidate.page_y)
        fixations.append(current_fixation)
        add_fixation_to_history(history_map, current_fixation, history_sigma)

        for step_index in range(1, self.config.steps + 1):
            history_snapshot = history_map.copy()
            roi_window = extract_roi(page_image, current_fixation, roi_size)
            retinal_roi = apply_inverse_radial_blur(
                roi_window.image,
                roi_window.fixation_in_roi,
                clear_radius,
                self.config.blur_level_count,
                self.config.max_blur_strength,
            )
            saliency_map, saliency_image = self.inference.predict(retinal_roi)
            peaks = extract_local_maxima(
                saliency_map,
                self.config.threshold_ratio,
                nms_radius,
                self.config.top_k,
            )
            candidates = self._score_candidates(peaks, roi_window, current_fixation, history_map, distance_sigma)
            selected = candidates[0] if candidates else None
            step_debug.append(
                StepDebug(
                    step_index=step_index,
                    phase="transition",
                    fixation=current_fixation,
                    history_map=history_snapshot,
                    roi_window=roi_window,
                    retinal_roi=retinal_roi,
                    saliency_map=saliency_map,
                    saliency_image=saliency_image,
                    candidates=candidates,
                    selected_candidate=selected,
                )
            )

            if selected is None:
                break

            current_fixation = (selected.page_x, selected.page_y)
            fixations.append(current_fixation)
            add_fixation_to_history(history_map, current_fixation, history_sigma)

        return SimulationResult(
            page_path=page_path,
            fixations=fixations,
            history_map=history_map,
            step_debug=step_debug,
        )
