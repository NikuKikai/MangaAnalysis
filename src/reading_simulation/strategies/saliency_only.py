from __future__ import annotations

from PIL import Image

from ..config import SimulationConfig
from ..engine import SaliencyScanpathEngine
from ..history import add_fixation_to_history, create_history_map
from ..types import SimulationResult, StepState


class SaliencyOnlyStrategy:
    """Original saliency-only scanpath strategy preserved as a standalone policy."""

    def __init__(self, engine: SaliencyScanpathEngine, config: SimulationConfig) -> None:
        """Create the strategy wrapper around the shared saliency engine."""
        self.engine = engine
        self.config = config

    def simulate(
        self,
        page_path: str,
        initial_fixation: tuple[float, float] | None = None,
        initial_roi_size: int | None = None,
    ) -> SimulationResult:
        """Simulate a scanpath using only saliency, history, and distance scoring."""
        page_image = Image.open(page_path).convert("RGB")
        page_width, page_height = page_image.size
        history_sigma = self.engine.scale_float(page_height, self.config.history_sigma_ratio)
        history_map = create_history_map(page_height, page_width)
        if initial_fixation is None:
            raise ValueError("initial_fixation is required to match the current web simulation flow.")

        current_fixation = (float(initial_fixation[0]), float(initial_fixation[1]))
        roi_size = initial_roi_size if initial_roi_size is not None else self.engine.create_default_roi_size(page_height)
        fixations: list[tuple[float, float]] = [current_fixation]
        step_states: list[StepState] = []

        for step_index in range(self.config.steps + 1):
            # Update the full-page history map before scoring the next transition.
            add_fixation_to_history(history_map, current_fixation, history_sigma, decay=self.config.history_decay)

            # Evaluate the default local ROI and fall back to a full-page square ROI when needed.
            local_eval = self.engine.evaluate_roi(page_image, current_fixation, roi_size, history_map, page_height)
            fallback_eval = None
            selected_candidate = local_eval.candidates[0] if local_eval.candidates else None
            used_full_page_roi = False
            if selected_candidate is None:
                fallback_eval = self.engine.evaluate_roi(
                    page_image,
                    current_fixation,
                    self.engine.create_full_page_roi_size(page_width, page_height),
                    history_map,
                    page_height,
                )
                used_full_page_roi = True
                selected_candidate = fallback_eval.candidates[0] if fallback_eval.candidates else None

            selected_fixation = None if selected_candidate is None else self.engine.candidate_to_point(selected_candidate)
            step_states.append(
                StepState(
                    step_index=step_index,
                    phase="start" if step_index == 0 else "transition",
                    fixation=current_fixation,
                    current_panel_index=None,
                    current_panel_id=None,
                    next_panel_id=None,
                    local_roi=_roi_window_to_rect(local_eval.roi_window),
                    used_full_page_roi=used_full_page_roi,
                    local_candidates=local_eval.candidates,
                    fallback_candidates=[] if fallback_eval is None else fallback_eval.candidates,
                    selected_candidate=selected_candidate,
                    selected_fixation=selected_fixation,
                    selected_panel_id=None,
                    selected_source="local" if not used_full_page_roi else "full_page",
                    actual_transition_score=0.0 if selected_candidate is None else float(selected_candidate.final_score),
                    note="Original saliency-only strategy.",
                )
            )

            if selected_fixation is None or step_index >= self.config.steps:
                break

            current_fixation = selected_fixation
            fixations.append(current_fixation)
            roi_size = self.engine.create_default_roi_size(page_height)

        return SimulationResult(
            page_path=page_path,
            strategy="saliency_only",
            fixations=fixations,
            history_map=history_map,
            step_states=step_states,
        )


def _roi_window_to_rect(roi_window) -> object:
    """Convert a ROI window into a lightweight rectangle object."""
    from panel_order.types import Rect

    return Rect(
        x=int(roi_window.crop_left - roi_window.origin_x),
        y=int(roi_window.crop_top - roi_window.origin_y),
        width=int(roi_window.image.width),
        height=int(roi_window.image.height),
    )
