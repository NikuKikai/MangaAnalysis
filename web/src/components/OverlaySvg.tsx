import { useMemo } from "react";
import { useShallow } from "zustand/react/shallow";
import { useSimulationStore } from "../store/simulationStore";
import { pagePointToScreen, pageRectToScreenRect, roiToScreenRect, type ImageRect } from "../core/utils/roi";
import type { Point } from "../types/simulation";

type OverlaySvgProps = {
  viewportWidth: number;
  viewportHeight: number;
  imageRect: ImageRect | null;
  imageWidth: number;
  imageHeight: number;
  dragRoi: { x: number; y: number; size: number } | null;
};

export function OverlaySvg(props: OverlaySvgProps) {
  const { viewportWidth, viewportHeight, imageRect, imageWidth, imageHeight, dragRoi } = props;
  const {
    currentRoi,
    currentFixation,
    pendingNextFixation,
    trajectory,
    candidates,
    panelBoxes,
    showPanelBoxes,
    showPreprocess,
    showSelectorOverlay,
    strategy,
    panelGuidedAnalysis,
  } = useSimulationStore(
    useShallow((state) => ({
      currentRoi: state.currentRoi,
      currentFixation: state.currentFixation,
      pendingNextFixation: state.pendingNextFixation,
      trajectory: state.trajectory,
      candidates: state.candidates,
      panelBoxes: state.panelBoxes,
      showPanelBoxes: state.display.showPanelBoxes,
      showPreprocess: state.display.showPreprocess,
      showSelectorOverlay: state.display.showSelectorOverlay,
      strategy: state.strategy,
      panelGuidedAnalysis: state.panelGuidedAnalysis,
    })),
  );

  const trajectoryPoints = useMemo(() => {
    if (!imageRect || trajectory.length === 0 || imageWidth <= 0 || imageHeight <= 0) {
      return [];
    }
    return trajectory.map((point) => pagePointToScreen(point, imageRect, imageWidth, imageHeight));
  }, [imageHeight, imageRect, imageWidth, trajectory]);

  const trajectoryPath = useMemo(() => {
    if (trajectoryPoints.length < 2) {
      return "";
    }
    return trajectoryPoints.map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`).join(" ");
  }, [trajectoryPoints]);

  const candidateDots = useMemo(() => {
    if (!showSelectorOverlay || !imageRect || imageWidth <= 0 || imageHeight <= 0) {
      return [];
    }
    return candidates.map((candidate, index) => ({
      id: `${candidate.pageX}-${candidate.pageY}-${index}`,
      alpha: 0.35 + Math.min(0.65, candidate.finalScore),
      point: pagePointToScreen({ x: candidate.pageX, y: candidate.pageY }, imageRect, imageWidth, imageHeight),
    }));
  }, [candidates, imageHeight, imageRect, imageWidth, showSelectorOverlay]);

  const panelBoxRects = useMemo(() => {
    if (!showPanelBoxes || !imageRect || imageWidth <= 0 || imageHeight <= 0) {
      return [];
    }
    return panelBoxes.map((panel) => ({
      id: `panel-${panel.panelId}`,
      rect: pageRectToScreenRect(panel.rect, imageRect, imageWidth, imageHeight),
      readingIndex: panel.readingIndex,
    }));
  }, [imageHeight, imageRect, imageWidth, panelBoxes, showPanelBoxes]);

  const currentRoiRect = useMemo(() => {
    if (!showPreprocess || !imageRect || !currentRoi || imageWidth <= 0 || imageHeight <= 0) {
      return null;
    }
    return roiToScreenRect(currentRoi, imageRect, imageWidth, imageHeight);
  }, [currentRoi, imageHeight, imageRect, imageWidth, showPreprocess]);

  const dragRoiRect = useMemo(() => {
    if (!imageRect || !dragRoi || imageWidth <= 0 || imageHeight <= 0) {
      return null;
    }
    return roiToScreenRect(dragRoi, imageRect, imageWidth, imageHeight);
  }, [dragRoi, imageHeight, imageRect, imageWidth]);

  const currentFixationPoint = useMemo(() => {
    if (!imageRect || !currentFixation || imageWidth <= 0 || imageHeight <= 0) {
      return null;
    }
    return pagePointToScreen(currentFixation, imageRect, imageWidth, imageHeight);
  }, [currentFixation, imageHeight, imageRect, imageWidth]);

  const pendingFixationPoint = useMemo(() => {
    if (!imageRect || !pendingNextFixation || imageWidth <= 0 || imageHeight <= 0) {
      return null;
    }
    return pagePointToScreen(pendingNextFixation, imageRect, imageWidth, imageHeight);
  }, [imageHeight, imageRect, imageWidth, pendingNextFixation]);

  const bestCandidateArrow = useMemo(() => {
    if (!showSelectorOverlay) {
      return null;
    }
    if (!currentFixationPoint || !pendingFixationPoint) {
      return null;
    }
    const dx = pendingFixationPoint.x - currentFixationPoint.x;
    const dy = pendingFixationPoint.y - currentFixationPoint.y;
    const length = Math.hypot(dx, dy);
    if (length < 1) {
      return null;
    }

    const ux = dx / length;
    const uy = dy / length;
    const startOffset = 10;
    const endOffset = 13;
    const startX = currentFixationPoint.x + ux * startOffset;
    const startY = currentFixationPoint.y + uy * startOffset;
    const endX = pendingFixationPoint.x - ux * endOffset;
    const endY = pendingFixationPoint.y - uy * endOffset;

    return {
      x1: startX,
      y1: startY,
      x2: endX,
      y2: endY,
    };
  }, [currentFixationPoint, pendingFixationPoint, showSelectorOverlay]);

  const fluidityArrows = useMemo(() => {
    if (!showSelectorOverlay || strategy !== "panel_guided" || !imageRect || imageWidth <= 0 || imageHeight <= 0 || !panelGuidedAnalysis) {
      return [];
    }

    const maxScore = Math.max(panelGuidedAnalysis.maxScore, 1e-8);
    return panelGuidedAnalysis.stepScores.flatMap((step) => {
      if (!step.candidate || step.stepIndex >= trajectory.length) {
        return [];
      }

      const start = pagePointToScreen(trajectory[step.stepIndex], imageRect, imageWidth, imageHeight);
      const end = pagePointToScreen({ x: step.candidate.pageX, y: step.candidate.pageY }, imageRect, imageWidth, imageHeight);
      const arrow = buildArrowGeometry(start, end, 1 + (step.score / maxScore) * 7);
      if (!arrow) {
        return [];
      }

      return [
        {
          ...arrow,
          id: `fluidity-${step.stepIndex}`,
          color: step.strongerThanActualNext ? "rgba(255, 72, 72, 0.42)" : "rgba(255, 192, 0, 0.42)",
          width: arrow.width,
        },
      ];
    });
  }, [imageHeight, imageRect, imageWidth, panelGuidedAnalysis, showSelectorOverlay, strategy, trajectory]);

  return (
    <svg
      className="stage-svg overlay-layer"
      viewBox={`0 0 ${viewportWidth} ${viewportHeight}`}
      width={viewportWidth}
      height={viewportHeight}
      aria-hidden="true"
    >
      <defs>
        <marker
          id="best-candidate-arrowhead"
          markerWidth="20"
          markerHeight="20"
          refX="14"
          refY="8"
          orient="auto"
          markerUnits="userSpaceOnUse"
        >
          <path className="overlay-best-candidate-arrowhead" d="M0 0L16 8L0 16Z" />
        </marker>
      </defs>

      {panelBoxRects.map((panel) => (
        <g key={panel.id}>
          <rect className="overlay-panel-box-shadow" x={panel.rect.x} y={panel.rect.y} width={panel.rect.width} height={panel.rect.height} />
          <rect className="overlay-panel-box" x={panel.rect.x} y={panel.rect.y} width={panel.rect.width} height={panel.rect.height} />
          {panel.readingIndex ? (
            <text
              className="overlay-panel-label"
              x={panel.rect.x + panel.rect.width / 2}
              y={panel.rect.y + panel.rect.height / 2}
              textAnchor="middle"
              dominantBaseline="central"
            >
              {panel.readingIndex}
            </text>
          ) : null}
        </g>
      ))}

      {fluidityArrows.map((arrow) => (
        <g key={arrow.id}>
          <line
            className="overlay-fluidity-arrow-shadow"
            x1={arrow.x1}
            y1={arrow.y1}
            x2={arrow.x2}
            y2={arrow.y2}
            strokeWidth={arrow.width + 4}
          />
          <line
            x1={arrow.x1}
            y1={arrow.y1}
            x2={arrow.x2}
            y2={arrow.y2}
            stroke={arrow.color}
            strokeWidth={arrow.width}
            strokeLinecap="round"
          />
          <polygon points={arrow.headPoints} fill={arrow.color} />
        </g>
      ))}

      {trajectoryPath ? (
        <>
          <path className="overlay-history-shadow" d={trajectoryPath} />
          <path className="overlay-history-line" d={trajectoryPath} />
        </>
      ) : null}

      {bestCandidateArrow ? (
        <>
          <line
            className="overlay-best-candidate-arrow-shadow"
            x1={bestCandidateArrow.x1}
            y1={bestCandidateArrow.y1}
            x2={bestCandidateArrow.x2}
            y2={bestCandidateArrow.y2}
          />
          <line
            className="overlay-best-candidate-arrow"
            x1={bestCandidateArrow.x1}
            y1={bestCandidateArrow.y1}
            x2={bestCandidateArrow.x2}
            y2={bestCandidateArrow.y2}
            markerEnd="url(#best-candidate-arrowhead)"
          />
        </>
      ) : null}

      {trajectoryPoints.map((point, index) => (
        <g key={`fixation-history-${index}`}>
          <circle className="overlay-history-dot-shadow" cx={point.x} cy={point.y} r={7} />
          <circle className="overlay-history-dot" cx={point.x} cy={point.y} r={5} />
        </g>
      ))}

      {candidateDots.map((candidate) => (
        <g key={candidate.id}>
          <circle className="overlay-candidate-shadow" cx={candidate.point.x} cy={candidate.point.y} r={5} />
          <circle
            cx={candidate.point.x}
            cy={candidate.point.y}
            r={3.25}
            fill={`rgba(255, 191, 71, ${candidate.alpha})`}
            stroke="rgba(34, 21, 5, 0.95)"
            strokeWidth="1.5"
          />
        </g>
      ))}

      {currentRoiRect ? (
        <g>
          <rect className="overlay-roi-shadow" x={currentRoiRect.x} y={currentRoiRect.y} width={currentRoiRect.width} height={currentRoiRect.height} />
          <rect className="overlay-roi" x={currentRoiRect.x} y={currentRoiRect.y} width={currentRoiRect.width} height={currentRoiRect.height} />
        </g>
      ) : null}

      {dragRoiRect ? (
        <g>
          <rect className="overlay-drag-roi-shadow" x={dragRoiRect.x} y={dragRoiRect.y} width={dragRoiRect.width} height={dragRoiRect.height} />
          <rect className="overlay-drag-roi" x={dragRoiRect.x} y={dragRoiRect.y} width={dragRoiRect.width} height={dragRoiRect.height} />
        </g>
      ) : null}

      {currentFixationPoint ? (
        <g>
          <circle className="overlay-current-fixation-shadow" cx={currentFixationPoint.x} cy={currentFixationPoint.y} r={9} />
          <circle className="overlay-current-fixation" cx={currentFixationPoint.x} cy={currentFixationPoint.y} r={6} />
          <circle className="overlay-current-fixation-core" cx={currentFixationPoint.x} cy={currentFixationPoint.y} r={2.5} />
        </g>
      ) : null}

      {pendingFixationPoint ? (
        <g>
          <circle className="overlay-pending-fixation-shadow" cx={pendingFixationPoint.x} cy={pendingFixationPoint.y} r={12} />
          <circle className="overlay-pending-fixation-ring" cx={pendingFixationPoint.x} cy={pendingFixationPoint.y} r={9} />
          <circle className="overlay-pending-fixation" cx={pendingFixationPoint.x} cy={pendingFixationPoint.y} r={4.5} />
        </g>
      ) : null}
    </svg>
  );
}

/**
 * Build a line plus triangular head for one overlay arrow.
 */
function buildArrowGeometry(start: Point, end: Point, width: number) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const length = Math.hypot(dx, dy);
  if (length < 1) {
    return null;
  }

  const ux = dx / length;
  const uy = dy / length;
  const startOffset = 10;
  const endOffset = 15;
  const x1 = start.x + ux * startOffset;
  const y1 = start.y + uy * startOffset;
  const x2 = end.x - ux * endOffset;
  const y2 = end.y - uy * endOffset;
  const headLength = 14 + width;
  const headSpread = 0.45;
  const leftX = end.x - headLength * (ux * Math.cos(headSpread) - uy * Math.sin(headSpread));
  const leftY = end.y - headLength * (uy * Math.cos(headSpread) + ux * Math.sin(headSpread));
  const rightX = end.x - headLength * (ux * Math.cos(-headSpread) - uy * Math.sin(-headSpread));
  const rightY = end.y - headLength * (uy * Math.cos(-headSpread) + ux * Math.sin(-headSpread));

  return {
    x1,
    y1,
    x2,
    y2,
    width,
    headPoints: `${end.x},${end.y} ${leftX},${leftY} ${rightX},${rightY}`,
  };
}
