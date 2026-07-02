from __future__ import annotations

from PIL import Image
import numpy as np

from panel_order.detector import PanelDetector
from panel_order.layout import MangaLayoutAnalyzer
from panel_order.types import Panel, Rect

from ..analysis import analyze_reading_fluidity
from ..config import SimulationConfig
from ..engine import SaliencyScanpathEngine
from ..history import add_fixation_to_history, create_history_map
from ..types import SimulationResult, StepState


class PanelGuidedStrategy:
    """Panel-order-guided strategy that constrains transitions by manga layout."""

    def __init__(self, engine: SaliencyScanpathEngine, config: SimulationConfig) -> None:
        """Create the strategy wrapper around the shared saliency engine."""
        self.engine = engine
        self.config = config
        self.panel_detector = PanelDetector()
        self.layout_analyzer = MangaLayoutAnalyzer()

    def simulate(
        self,
        page_path: str,
        initial_fixation: tuple[float, float] | None = None,
        initial_roi_size: int | None = None,
    ) -> SimulationResult:
        """Simulate a scanpath that follows detected panel order while using saliency inside each panel."""
        page_image = Image.open(page_path).convert("RGB")
        page_array = np.asarray(page_image)
        page_width, page_height = page_image.size
        history_sigma = self.engine.scale_float(page_height, self.config.history_sigma_ratio)
        history_map = create_history_map(page_height, page_width)

        # Detect panels once and derive the Japanese reading order used by this strategy.
        panels = self.panel_detector.detect(page_array)
        if not panels:
            raise RuntimeError("Panel-guided strategy requires at least one detected panel.")
        _layout_tree, panel_reading_order = self.layout_analyzer.build_layout(panels)
        ordered_panels = [_panel_by_id(panels, panel_id) for panel_id in panel_reading_order]

        current_panel_index = 0
        if initial_fixation is None:
            initial_fixation = _panel_center(ordered_panels[0].rect)
        current_fixation = (float(initial_fixation[0]), float(initial_fixation[1]))
        current_panel_index = _panel_index_for_fixation(ordered_panels, current_fixation)
        roi_size = initial_roi_size if initial_roi_size is not None else self.engine.create_default_roi_size(page_height)
        fixations: list[tuple[float, float]] = [current_fixation]
        step_states: list[StepState] = []

        for step_index in range(self.config.steps + 1):
            active_panel_index = current_panel_index
            current_panel = ordered_panels[current_panel_index]
            next_panel = ordered_panels[current_panel_index + 1] if current_panel_index + 1 < len(ordered_panels) else None

            # Update history only inside the current panel so panel-to-panel progress stays localized.
            add_fixation_to_history(
                history_map,
                current_fixation,
                history_sigma,
                decay=self.config.history_decay,
                clip_rect=current_panel.rect,
            )

            # Always score candidates with the original local ROI settings first.
            local_eval = self.engine.evaluate_roi(page_image, current_fixation, roi_size, history_map, page_height)
            current_panel_candidates = self.engine.filter_candidates_to_rect(local_eval.candidates, current_panel.rect)
            next_panel_candidates = self.engine.filter_candidates_to_rect(local_eval.candidates, None if next_panel is None else next_panel.rect)

            fallback_candidates = []
            used_full_page_roi = False
            selected_candidate = None
            selected_fixation = None
            selected_panel_id = current_panel.panel_id
            selected_source = "none"
            note = "No candidate selected."

            # Priority 1: stay inside the current panel whenever a valid candidate exists.
            if current_panel_candidates:
                selected_candidate = current_panel_candidates[0]
                selected_fixation = self.engine.candidate_to_point(selected_candidate)
                selected_source = "current_panel"
                note = "Selected the best candidate inside the current panel."

            # Priority 2: if the current panel is exhausted, advance into the next panel from the local ROI.
            elif next_panel is not None and next_panel_candidates:
                selected_candidate = next_panel_candidates[0]
                selected_fixation = self.engine.candidate_to_point(selected_candidate)
                selected_panel_id = next_panel.panel_id
                selected_source = "next_panel_local"
                current_panel_index += 1
                note = "Current panel had no candidate, so the best next-panel candidate was selected from the local ROI."

            # Priority 3: if the local ROI still cannot enter the next panel, retry with a full-page ROI.
            elif next_panel is not None:
                full_page_eval = self.engine.evaluate_roi(
                    page_image,
                    current_fixation,
                    self.engine.create_full_page_roi_size(page_width, page_height),
                    history_map,
                    page_height,
                )
                used_full_page_roi = True
                fallback_candidates = self.engine.filter_candidates_to_rect(full_page_eval.candidates, next_panel.rect)
                if fallback_candidates:
                    selected_candidate = fallback_candidates[0]
                    selected_fixation = self.engine.candidate_to_point(selected_candidate)
                    selected_panel_id = next_panel.panel_id
                    selected_source = "next_panel_full_page"
                    current_panel_index += 1
                    note = "Selected the best next-panel candidate after expanding the ROI to the full page."
                else:
                    selected_fixation = _panel_center(next_panel.rect)
                    selected_panel_id = next_panel.panel_id
                    selected_source = "next_panel_center"
                    current_panel_index += 1
                    note = "No next-panel candidate existed even after full-page fallback, so the next panel center was used."

            # Stop only when the last panel is active and it no longer offers a candidate.
            else:
                note = "Reached the last panel and found no remaining candidate."

            step_states.append(
                StepState(
                    step_index=step_index,
                    phase="start" if step_index == 0 else "transition",
                    fixation=current_fixation,
                    current_panel_index=active_panel_index,
                    current_panel_id=current_panel.panel_id,
                    next_panel_id=None if next_panel is None else next_panel.panel_id,
                    local_roi=_roi_rect(page_image, current_fixation, roi_size),
                    used_full_page_roi=used_full_page_roi,
                    local_candidates=local_eval.candidates,
                    fallback_candidates=fallback_candidates,
                    selected_candidate=selected_candidate,
                    selected_fixation=selected_fixation,
                    selected_panel_id=selected_panel_id,
                    selected_source=selected_source,
                    actual_transition_score=0.0 if selected_candidate is None else float(selected_candidate.final_score),
                    note=note,
                )
            )

            if selected_fixation is None or step_index >= self.config.steps:
                break

            current_fixation = selected_fixation
            fixations.append(current_fixation)
            roi_size = self.engine.create_default_roi_size(page_height)

            if current_panel_index >= len(ordered_panels):
                break

        result = SimulationResult(
            page_path=page_path,
            strategy="panel_guided",
            fixations=fixations,
            history_map=history_map,
            step_states=step_states,
            panels=ordered_panels,
            panel_reading_order=panel_reading_order,
        )
        result.analysis = analyze_reading_fluidity(result)
        return result


def _panel_by_id(panels: list[Panel], panel_id: int) -> Panel:
    """Resolve one panel id from a detected panel list."""
    for panel in panels:
        if panel.panel_id == panel_id:
            return panel
    raise KeyError(f"Panel id {panel_id} was not found.")


def _panel_center(rect: Rect) -> tuple[float, float]:
    """Return the geometric center of a panel rectangle."""
    return (rect.center_x, rect.center_y)


def _panel_index_for_fixation(panels: list[Panel], fixation: tuple[float, float]) -> int:
    """Find the panel containing the fixation, or fall back to the first panel."""
    for index, panel in enumerate(panels):
        rect = panel.rect
        if rect.x <= fixation[0] <= rect.right and rect.y <= fixation[1] <= rect.bottom:
            return index
    return 0


def _roi_rect(page_image: Image.Image, fixation: tuple[float, float], roi_size: int) -> Rect:
    """Reconstruct the padded ROI rectangle used by the saliency engine."""
    from ..retina import extract_roi

    roi_window = extract_roi(page_image, fixation, roi_size)
    return Rect(
        x=int(roi_window.crop_left - roi_window.origin_x),
        y=int(roi_window.crop_top - roi_window.origin_y),
        width=int(roi_window.image.width),
        height=int(roi_window.image.height),
    )
