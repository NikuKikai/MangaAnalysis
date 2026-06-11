import { useSimulationRuntime } from "../app/SimulationRuntime";

export function SimulationStage() {
  const {
    baseCanvasRef,
    preprocessCanvasRef,
    heatmapCanvasRef,
    overlayCanvasRef,
    fileInputRef,
    handlePointerDown,
    handlePointerMove,
    handlePointerUp,
    handleDrop,
    handleFileChange,
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
        <canvas ref={overlayCanvasRef} className="stage-canvas overlay-layer" />
      </div>
    </div>
  );
}
