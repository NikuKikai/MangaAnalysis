import { useMemo } from "react";
import { useShallow } from "zustand/react/shallow";
import { useSimulationStore } from "../store/simulationStore";
import { pagePointToScreen, pageRectToScreenRect, roiToScreenRect, type ImageRect } from "../core/utils/roi";

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
  const { currentRoi, currentFixation, pendingNextFixation, trajectory, candidates, panelBoxes, showPanelBoxes } = useSimulationStore(
    useShallow((state) => ({
      currentRoi: state.currentRoi,
      currentFixation: state.currentFixation,
      pendingNextFixation: state.pendingNextFixation,
      trajectory: state.trajectory,
      candidates: state.candidates,
      panelBoxes: state.panelBoxes,
      showPanelBoxes: state.display.showPanelBoxes,
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
    if (!imageRect || imageWidth <= 0 || imageHeight <= 0) {
      return [];
    }
    return candidates.map((candidate, index) => ({
      id: `${candidate.pageX}-${candidate.pageY}-${index}`,
      alpha: 0.35 + Math.min(0.65, candidate.finalScore),
      point: pagePointToScreen({ x: candidate.pageX, y: candidate.pageY }, imageRect, imageWidth, imageHeight),
    }));
  }, [candidates, imageHeight, imageRect, imageWidth]);

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
    if (!imageRect || !currentRoi || imageWidth <= 0 || imageHeight <= 0) {
      return null;
    }
    return roiToScreenRect(currentRoi, imageRect, imageWidth, imageHeight);
  }, [currentRoi, imageHeight, imageRect, imageWidth]);

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
  }, [currentFixationPoint, pendingFixationPoint]);

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
