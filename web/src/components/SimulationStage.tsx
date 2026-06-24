import type { DragEvent, PointerEvent } from "react";
import { useShallow } from "zustand/react/shallow";
import { createSquareFromDrag, screenToPagePoint } from "../core/simulation/roi";
import { useSimulationEngineContext } from "../app/SimulationRuntime";
import { useSimulationStore } from "../store/simulationStore";
import { OverlaySvg } from "./OverlaySvg";

export function SimulationStage() {
  const {
    refs: { baseCanvasRef, historyCanvasRef, preprocessCanvasRef, heatmapCanvasRef },
    overlay,
    interactionState,
    startClickStep,
    startBoxStep,
  } = useSimulationEngineContext();
  const { mode, dragStart, isDragging, loadImageFile, setDragState } = useSimulationStore(
    useShallow((state) => ({
      mode: state.mode,
      dragStart: state.dragStart,
      isDragging: state.isDragging,
      loadImageFile: state.loadImageFile,
      setDragState: state.setDragState,
    })),
  );
  const { image, imageRect } = interactionState;

  const getPagePoint = (clientX: number, clientY: number) => {
    if (!image || !imageRect) {
      return null;
    }
    return screenToPagePoint(clientX, clientY, imageRect, image.width, image.height);
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    void loadImageFile(event.dataTransfer.files?.[0] ?? null);
  };

  const handlePointerDown = (event: PointerEvent<HTMLDivElement>) => {
    const point = getPagePoint(event.clientX, event.clientY);
    if (!point) {
      return;
    }
    if (mode === "box") {
      setDragState(point, point, true);
    }
  };

  const handlePointerMove = (event: PointerEvent<HTMLDivElement>) => {
    if (!isDragging || !dragStart || mode !== "box") {
      return;
    }
    const point = getPagePoint(event.clientX, event.clientY);
    if (!point) {
      return;
    }
    setDragState(dragStart, point, true);
  };

  const handlePointerUp = (event: PointerEvent<HTMLDivElement>) => {
    const point = getPagePoint(event.clientX, event.clientY);
    if (!point) {
      setDragState(null, null, false);
      return;
    }

    if (mode === "click") {
      void startClickStep(point);
      return;
    }

    if (mode === "box" && dragStart) {
      const roi = createSquareFromDrag(dragStart, point);
      setDragState(null, null, false);
      if (roi.size >= 2) {
        void startBoxStep(roi);
      }
    }
  };

  return (
    <div
      className="app-shell"
      onDragOver={(event) => {
        event.preventDefault();
        event.dataTransfer.dropEffect = "copy";
      }}
      onDrop={handleDrop}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
    >
      <div className="canvas-stack">
        <canvas ref={baseCanvasRef} className="stage-canvas" />
        <canvas ref={preprocessCanvasRef} className="stage-canvas preprocess-layer" />
        <canvas ref={heatmapCanvasRef} className="stage-canvas heatmap-layer" />
        <canvas ref={historyCanvasRef} className="stage-canvas history-layer" />
        <OverlaySvg {...overlay} />
      </div>
    </div>
  );
}
