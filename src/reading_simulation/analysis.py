from __future__ import annotations

from dataclasses import asdict

from panel_order.types import Panel

from .candidates import CandidatePoint
from .types import FluidityAnalysis, FluidityStep, SimulationResult, StepState


def analyze_reading_fluidity(result: SimulationResult) -> FluidityAnalysis:
    """Compute reading fluidity from off-route saliency peaks at each step."""
    panel_by_id = {panel.panel_id: panel for panel in result.panels}
    step_scores: list[FluidityStep] = []

    for step in result.step_states:
        outsider, outsider_panel_id = _find_best_outsider(step, panel_by_id)
        score = 0.0 if outsider is None else float(outsider.final_score)
        stronger = outsider is not None and score > float(step.actual_transition_score)
        step_scores.append(
            FluidityStep(
                step_index=step.step_index,
                score=score,
                candidate=outsider,
                candidate_panel_id=outsider_panel_id,
                stronger_than_actual_next=stronger,
            )
        )

    non_empty_scores = [step.score for step in step_scores]
    mean_score = 0.0 if not non_empty_scores else sum(non_empty_scores) / float(len(non_empty_scores))
    max_score = 0.0 if not non_empty_scores else max(non_empty_scores)
    return FluidityAnalysis(step_scores=step_scores, mean_score=mean_score, max_score=max_score)


def _find_best_outsider(step: StepState, panel_by_id: dict[int, Panel]) -> tuple[CandidatePoint | None, int | None]:
    """Return the best candidate that lies outside the current and next panels."""
    ignored_panel_ids = {panel_id for panel_id in [step.current_panel_id, step.next_panel_id] if panel_id is not None}
    best_candidate: CandidatePoint | None = None
    best_panel_id: int | None = None

    # Combine every candidate pool examined at this step so the metric sees fallback behavior too.
    for candidate in [*step.local_candidates, *step.fallback_candidates]:
        panel_id = _candidate_panel_id(candidate, panel_by_id)
        if panel_id in ignored_panel_ids:
            continue
        if best_candidate is None or candidate.final_score > best_candidate.final_score:
            best_candidate = candidate
            best_panel_id = panel_id

    return best_candidate, best_panel_id


def _candidate_panel_id(candidate: CandidatePoint, panel_by_id: dict[int, Panel]) -> int | None:
    """Map a candidate point to the panel that contains it, if any."""
    for panel_id, panel in panel_by_id.items():
        rect = panel.rect
        if rect.x <= candidate.page_x <= rect.right and rect.y <= candidate.page_y <= rect.bottom:
            return panel_id
    return None
