from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from panel_order.types import Panel, Rect

from .candidates import CandidatePoint


@dataclass(slots=True)
class StepState:
    """Lightweight per-step trace that excludes large intermediate arrays."""

    step_index: int
    phase: str
    fixation: tuple[float, float]
    current_panel_index: int | None
    current_panel_id: int | None
    next_panel_id: int | None
    local_roi: Rect
    used_full_page_roi: bool
    local_candidates: list[CandidatePoint] = field(default_factory=list)
    fallback_candidates: list[CandidatePoint] = field(default_factory=list)
    selected_candidate: CandidatePoint | None = None
    selected_fixation: tuple[float, float] | None = None
    selected_panel_id: int | None = None
    selected_source: str = "none"
    actual_transition_score: float = 0.0
    note: str = ""

    def to_json(self) -> dict:
        """Serialize the step state into a JSON-friendly dictionary."""
        return {
            "step_index": int(self.step_index),
            "phase": self.phase,
            "fixation": [float(self.fixation[0]), float(self.fixation[1])],
            "current_panel_index": self.current_panel_index,
            "current_panel_id": self.current_panel_id,
            "next_panel_id": self.next_panel_id,
            "local_roi": self.local_roi.to_json(),
            "used_full_page_roi": bool(self.used_full_page_roi),
            "local_candidates": [asdict(candidate) for candidate in self.local_candidates],
            "fallback_candidates": [asdict(candidate) for candidate in self.fallback_candidates],
            "selected_candidate": asdict(self.selected_candidate) if self.selected_candidate is not None else None,
            "selected_fixation": None if self.selected_fixation is None else [float(self.selected_fixation[0]), float(self.selected_fixation[1])],
            "selected_panel_id": self.selected_panel_id,
            "selected_source": self.selected_source,
            "actual_transition_score": float(self.actual_transition_score),
            "note": self.note,
        }


@dataclass(slots=True)
class FluidityStep:
    """Per-step reading-fluidity measurement based on off-route saliency attraction."""

    step_index: int
    score: float
    candidate: CandidatePoint | None
    candidate_panel_id: int | None
    stronger_than_actual_next: bool

    def to_json(self) -> dict:
        """Serialize the per-step fluidity measurement into a JSON-friendly dictionary."""
        return {
            "step_index": int(self.step_index),
            "score": float(self.score),
            "candidate": asdict(self.candidate) if self.candidate is not None else None,
            "candidate_panel_id": self.candidate_panel_id,
            "stronger_than_actual_next": bool(self.stronger_than_actual_next),
        }


@dataclass(slots=True)
class FluidityAnalysis:
    """Aggregate reading-fluidity analysis for the full scanpath."""

    step_scores: list[FluidityStep]
    mean_score: float
    max_score: float

    def to_json(self) -> dict:
        """Serialize the fluidity analysis into a JSON-friendly dictionary."""
        return {
            "step_scores": [step.to_json() for step in self.step_scores],
            "mean_score": float(self.mean_score),
            "max_score": float(self.max_score),
        }


@dataclass(slots=True)
class SimulationResult:
    """Unified simulation result returned by every reading strategy."""

    page_path: str
    strategy: str
    fixations: list[tuple[float, float]]
    history_map: np.ndarray
    step_states: list[StepState]
    panels: list[Panel] = field(default_factory=list)
    panel_reading_order: list[int] = field(default_factory=list)
    analysis: FluidityAnalysis | None = None

    def to_json(self) -> dict:
        """Serialize the simulation result into a JSON-friendly dictionary."""
        return {
            "page_path": self.page_path,
            "strategy": self.strategy,
            "fixations": [[float(x), float(y)] for x, y in self.fixations],
            "panels": [panel.to_json() for panel in self.panels],
            "panel_reading_order": [int(panel_id) for panel_id in self.panel_reading_order],
            "steps": [step.to_json() for step in self.step_states],
            "analysis": None if self.analysis is None else self.analysis.to_json(),
        }
