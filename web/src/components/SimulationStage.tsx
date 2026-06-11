import { useSimulationRuntime } from "../app/SimulationRuntime";
import { OverlaySvg } from "./OverlaySvg";

export function SimulationStage() {
  const {
    baseCanvasRef,
    preprocessCanvasRef,
    heatmapCanvasRef,
    fileInputRef,
    handlePointerDown,
    handlePointerMove,
    handlePointerUp,
    handleDrop,
    handleFileChange,
    overlay,
  } = useSimulationRuntime();

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
      onPointerUp={(event) => void handlePointerUp(event)}
    >
      <input
        ref={fileInputRef}
        className="visually-hidden"
        type="file"
        accept="image/*"
        onChange={handleFileChange}
      />

      <div className="canvas-stack">
        <canvas ref={baseCanvasRef} className="stage-canvas" />
        <canvas ref={preprocessCanvasRef} className="stage-canvas preprocess-layer" />
        <canvas ref={heatmapCanvasRef} className="stage-canvas heatmap-layer" />
        <OverlaySvg {...overlay} />
      </div>
    </div>
  );
}
