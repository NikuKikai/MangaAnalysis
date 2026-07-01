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
        default_roi_half_size_ratio: Half of the default square ROI size as a ratio of page height.
        clear_radius_ratio: Radius of the sharp foveal area as a ratio of page height.
        blur_level_count: Number of blur levels used to approximate continuous peripheral blur.
        max_blur_strength: Maximum peripheral blur strength at the ROI outer radius.
        history_sigma_ratio: Sigma of the revisit-suppression Gaussian as a ratio of page height.
        history_decay: Per-step decay applied to the history map before the new fixation bump is added.
        history_alpha: Strength of history-based inhibition.
        distance_sigma_ratio: Sigma of the distance regularizer as a ratio of page height.
        threshold_score: Absolute minimum final candidate score kept after rescoring.
        nms_radius_ratio: Non-maximum suppression radius as a ratio of page height.
        top_k: Maximum number of local saliency peaks to keep per step.
        steps: Maximum number of transition steps after the initial fixation.
    """

    default_roi_half_size_ratio: float = field(
        default=0.25,
        metadata={"doc": "Half of the default square ROI size as a ratio of page height."},
    )
    clear_radius_ratio: float = field(default=0.06, metadata={"doc": "Radius of the sharp foveal area as a ratio of page height."})
    blur_level_count: int = field(default=8, metadata={"doc": "Number of blur levels used to approximate continuous peripheral blur."})
    max_blur_strength: float = field(default=7.0, metadata={"doc": "Maximum peripheral blur strength at the ROI outer radius."})
    history_sigma_ratio: float = field(default=0.047, metadata={"doc": "Sigma of the revisit-suppression Gaussian as a ratio of page height."})
    history_decay: float = field(default=0.94, metadata={"doc": "Per-step decay applied to the history map before adding the new fixation bump."})
    history_alpha: float = field(default=3.0, metadata={"doc": "Strength of history-based inhibition."})
    distance_sigma_ratio: float = field(default=0.183, metadata={"doc": "Sigma of the distance regularizer as a ratio of page height."})
    threshold_score: float = field(default=0.15, metadata={"doc": "Absolute minimum final candidate score kept after rescoring."})
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
    used_full_page_roi: bool = False


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
                    "used_full_page_roi": step.used_full_page_roi,
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

    def _create_default_roi(self, page_height: int) -> int:
        half_size = self._scale_float(page_height, self.config.default_roi_half_size_ratio)
        return max(int(round(half_size * 2.0)), 1)

    def _create_full_page_roi(self, page_width: int, page_height: int) -> int:
        return max(page_width, page_height)

    def _score_candidates(
        self,
        saliency_map: np.ndarray,
        roi_window: RoiWindow,
        current_fixation: tuple[float, float],
        history_map: np.ndarray,
        distance_sigma: float,
        nms_radius: int,
    ) -> list[CandidatePoint]:
        final_score_map = np.zeros_like(saliency_map, dtype=np.float32)
        saliency_height, saliency_width = saliency_map.shape
        if saliency_height == 0 or saliency_width == 0:
            return []

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
                dist = self._distance_score(current_fixation, (page_x, page_y), distance_sigma)
                final_score_map[roi_y, roi_x] = float(saliency_map[roi_y, roi_x]) * inhib * dist

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
            dist = self._distance_score(current_fixation, (page_x, page_y), distance_sigma)
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

    def _run_step(
        self,
        page_image: Image.Image,
        page_width: int,
        fixation: tuple[float, float],
        roi_size: int,
        history_map: np.ndarray,
        clear_radius: float,
        distance_sigma: float,
        nms_radius: int,
        step_index: int,
        phase: str,
    ) -> StepDebug:
        history_snapshot = history_map.copy()
        roi_window = extract_roi(page_image, fixation, roi_size)
        retinal_roi = apply_inverse_radial_blur(
            roi_window.image,
            roi_window.fixation_in_roi,
            clear_radius,
            self.config.blur_level_count,
            self.config.max_blur_strength,
        )
        saliency_map, saliency_image = self.inference.predict(retinal_roi)
        candidates = self._score_candidates(saliency_map, roi_window, fixation, history_map, distance_sigma, nms_radius)
        used_full_page_roi = False

        if not candidates:
            roi_window = extract_roi(page_image, fixation, self._create_full_page_roi(page_width, page_image.size[1]))
            retinal_roi = apply_inverse_radial_blur(
                roi_window.image,
                roi_window.fixation_in_roi,
                clear_radius,
                self.config.blur_level_count,
                self.config.max_blur_strength,
            )
            saliency_map, saliency_image = self.inference.predict(retinal_roi)
            candidates = self._score_candidates(saliency_map, roi_window, fixation, history_map, distance_sigma, nms_radius)
            used_full_page_roi = True

        selected = candidates[0] if candidates else None
        return StepDebug(
            step_index=step_index,
            phase=phase,
            fixation=fixation,
            history_map=history_snapshot,
            roi_window=roi_window,
            retinal_roi=retinal_roi,
            saliency_map=saliency_map,
            saliency_image=saliency_image,
            candidates=candidates,
            selected_candidate=selected,
            used_full_page_roi=used_full_page_roi,
        )

    def simulate(
        self,
        page_path: str,
        initial_fixation: tuple[float, float] | None = None,
        initial_roi_size: int | None = None,
    ) -> SimulationResult:
        page_image = Image.open(page_path).convert("RGB")
        page_width, page_height = page_image.size
        clear_radius = self._scale_float(page_height, self.config.clear_radius_ratio)
        history_sigma = self._scale_float(page_height, self.config.history_sigma_ratio)
        distance_sigma = self._scale_float(page_height, self.config.distance_sigma_ratio)
        nms_radius = self._scale_int(page_height, self.config.nms_radius_ratio)
        history_map = create_history_map(page_height, page_width)
        if initial_fixation is None:
            raise ValueError("initial_fixation is required to match the current web simulation flow.")

        current_fixation = (float(initial_fixation[0]), float(initial_fixation[1]))
        roi_size = initial_roi_size if initial_roi_size is not None else self._create_default_roi(page_height)
        fixations: list[tuple[float, float]] = [current_fixation]
        step_debug: list[StepDebug] = []

        for step_index in range(self.config.steps + 1):
            add_fixation_to_history(
                history_map,
                current_fixation,
                history_sigma,
                decay=self.config.history_decay,
            )
            step = self._run_step(
                page_image=page_image,
                page_width=page_width,
                fixation=current_fixation,
                roi_size=roi_size,
                history_map=history_map,
                clear_radius=clear_radius,
                distance_sigma=distance_sigma,
                nms_radius=nms_radius,
                step_index=step_index,
                phase="start" if step_index == 0 else "transition",
            )
            step_debug.append(step)
            if step.selected_candidate is None or step_index >= self.config.steps:
                break

            current_fixation = (step.selected_candidate.page_x, step.selected_candidate.page_y)
            fixations.append(current_fixation)
            roi_size = self._create_default_roi(page_height)

        return SimulationResult(
            page_path=page_path,
            fixations=fixations,
            history_map=history_map,
            step_debug=step_debug,
        )
