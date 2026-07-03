import type { Candidate, FluidityAnalysis, FluidityStep, PanelBox, PanelGuidedStepState, Point } from "../../types/simulation";
import { rectContainsPoint } from "./panelGuided";

/**
 * Compute the best off-route saliency candidate for one panel-guided step.
 */
export function analyzeFluidityStep(step: PanelGuidedStepState, panels: PanelBox[]): FluidityStep {
  const ignoredPanelIds = new Set<number>(
    [step.currentPanelId, step.nextPanelId].filter((panelId): panelId is number => panelId !== null),
  );

  let bestCandidate: Candidate | null = null;
  let bestPanelId: number | null = null;

  // Consider both local and fallback candidate pools so the metric sees every option the strategy inspected.
  for (const candidate of [...step.localCandidates, ...step.fallbackCandidates]) {
    const panelId = candidatePanelId(candidateToPoint(candidate), panels);
    if (panelId !== null && ignoredPanelIds.has(panelId)) {
      continue;
    }
    if (!bestCandidate || candidate.finalScore > bestCandidate.finalScore) {
      bestCandidate = candidate;
      bestPanelId = panelId;
    }
  }

  const score = bestCandidate?.finalScore ?? 0;
  return {
    stepIndex: step.stepIndex,
    score,
    candidate: bestCandidate,
    candidatePanelId: bestPanelId,
    strongerThanActualNext: bestCandidate !== null && score > step.actualTransitionScore,
  };
}

/**
 * Recompute the aggregate fluidity summary from the accumulated per-step measurements.
 */
export function summarizeFluidity(stepScores: FluidityStep[]): FluidityAnalysis {
  const values = stepScores.map((step) => step.score);
  const meanScore = values.length > 0 ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
  const maxScore = values.length > 0 ? Math.max(...values) : 0;
  return {
    stepScores,
    meanScore,
    maxScore,
  };
}

/**
 * Resolve the detected panel that contains a point, if any.
 */
export function candidatePanelId(point: Point, panels: PanelBox[]): number | null {
  for (const panel of panels) {
    if (rectContainsPoint(panel.rect, point)) {
      return panel.panelId;
    }
  }
  return null;
}

/**
 * Convert one candidate into a page-space point for spatial tests.
 */
export function candidateToPoint(candidate: Candidate): Point {
  return {
    x: candidate.pageX,
    y: candidate.pageY,
  };
}
