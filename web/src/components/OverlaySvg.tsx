import { useMemo } from "react";
import { useShallow } from "zustand/react/shallow";
import { useSimulationStore } from "../store/simulationStore";
import { pagePointToScreen, roiToScreenRect, type ImageRect } from "../core/utils/roi";

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
  const { currentRoi, currentFixation, pendingNextFixation, trajectory, candidates } = useSimulationStore(
    useShallow((state) => ({
      currentRoi: state.currentRoi,
      currentFixation: state.currentFixation,
      pendingNextFixation: state.pendingNextFixation,
      trajectory: state.trajectory,
      candidates: state.candidates,
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

  return (
    <svg
      className="stage-svg overlay-layer"
      viewBox={`0 0 ${viewportWidth} ${viewportHeight}`}
      width={viewportWidth}
      height={viewportHeight}
      aria-hidden="true"
    >
      {trajectoryPath ? (
        <>
          <path className="overlay-history-shadow" d={trajectoryPath} />
          <path className="overlay-history-line" d={trajectoryPath} />
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
