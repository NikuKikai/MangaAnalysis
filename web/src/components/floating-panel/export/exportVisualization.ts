import html2canvas from "html2canvas";
import type { Candidate, DisplayState, FluidityAnalysis, PanelBox, Point, RoiRect, SimulationStrategy } from "../../../types/simulation";
import { drawExportOverlays } from "./drawExportOverlays";
import { drawHeatmapExport } from "./drawHeatmapExport";
import { drawPreprocessExport } from "./drawPreprocessExport";

type ExportVisualizationParams = {
  image: {
    width: number;
    height: number;
    url: string;
  } | null;
  refs: {
    historyCanvas: HTMLCanvasElement | null;
  };
  exportBuffers: {
    preprocessPreview: Float32Array | null;
    heatmap: Float32Array | null;
    modelSize: number;
  };
  overlay: {
    viewportWidth: number;
    viewportHeight: number;
    imageRect: {
      x: number;
      y: number;
      width: number;
      height: number;
    } | null;
  };
  simulation: {
    currentRoi: RoiRect | null;
    currentFixation: Point | null;
    pendingNextFixation: Point | null;
    trajectory: Point[];
    candidates: Candidate[];
    panelBoxes: PanelBox[];
    panelGuidedAnalysis: FluidityAnalysis | null;
    strategy: SimulationStrategy;
    display: DisplayState;
  };
};

function stripFileExtension(filename: string) {
  const index = filename.lastIndexOf(".");
  return index > 0 ? filename.slice(0, index) : filename;
}

export async function exportVisualization({
  image,
  refs,
  exportBuffers,
  overlay,
  simulation,
}: ExportVisualizationParams) {
  if (!image || !overlay.imageRect) {
    return;
  }
  const imageRect = overlay.imageRect;

  const stageElement = document.querySelector(".canvas-stack") as HTMLElement | null;
  if (!stageElement) {
    return;
  }

  const stageCapture = await html2canvas(stageElement, {
    backgroundColor: null,
    logging: false,
    useCORS: true,
    scale: window.devicePixelRatio || 1,
  });

  const scaleX = stageCapture.width / Math.max(overlay.viewportWidth, 1);
  const scaleY = stageCapture.height / Math.max(overlay.viewportHeight, 1);
  const cropX = Math.round(imageRect.x * scaleX);
  const cropY = Math.round(imageRect.y * scaleY);
  const cropWidth = Math.max(1, Math.round(imageRect.width * scaleX));
  const cropHeight = Math.max(1, Math.round(imageRect.height * scaleY));

  const exportCanvas = document.createElement("canvas");
  exportCanvas.width = cropWidth;
  exportCanvas.height = cropHeight;
  const context = exportCanvas.getContext("2d");
  if (!context) {
    return;
  }

  context.drawImage(stageCapture, cropX, cropY, cropWidth, cropHeight, 0, 0, cropWidth, cropHeight);

  const drawViewportCanvas = (sourceCanvas: HTMLCanvasElement | null, enabled: boolean) => {
    if (!sourceCanvas || !enabled) {
      return;
    }
    const sourceScaleX = sourceCanvas.width / Math.max(overlay.viewportWidth, 1);
    const sourceScaleY = sourceCanvas.height / Math.max(overlay.viewportHeight, 1);
    const sx = Math.round(imageRect.x * sourceScaleX);
    const sy = Math.round(imageRect.y * sourceScaleY);
    const sw = Math.max(1, Math.round(imageRect.width * sourceScaleX));
    const sh = Math.max(1, Math.round(imageRect.height * sourceScaleY));
    context.drawImage(sourceCanvas, sx, sy, sw, sh, 0, 0, cropWidth, cropHeight);
  };

  if (simulation.display.showPreprocess && exportBuffers.preprocessPreview && simulation.currentRoi) {
    drawPreprocessExport(context, exportBuffers.preprocessPreview, exportBuffers.modelSize, simulation.currentRoi, image.width, image.height, cropWidth, cropHeight);
  }
  if (simulation.display.showHeatmap && exportBuffers.heatmap && simulation.currentRoi) {
    drawHeatmapExport(context, exportBuffers.heatmap, exportBuffers.modelSize, simulation.currentRoi, image.width, image.height, cropWidth, cropHeight);
  }
  drawViewportCanvas(refs.historyCanvas, simulation.display.showHistoryHeatmap);

  drawExportOverlays(context, {
    currentRoi: simulation.currentRoi
      ? {
          x: (simulation.currentRoi.x / image.width) * cropWidth,
          y: (simulation.currentRoi.y / image.height) * cropHeight,
          size: (simulation.currentRoi.size / image.width) * cropWidth,
        }
      : null,
    currentFixation: simulation.currentFixation
      ? {
          x: (simulation.currentFixation.x / image.width) * cropWidth,
          y: (simulation.currentFixation.y / image.height) * cropHeight,
        }
      : null,
    pendingNextFixation: simulation.pendingNextFixation
      ? {
          x: (simulation.pendingNextFixation.x / image.width) * cropWidth,
          y: (simulation.pendingNextFixation.y / image.height) * cropHeight,
        }
      : null,
    trajectory: simulation.trajectory.map((point) => ({
      x: (point.x / image.width) * cropWidth,
      y: (point.y / image.height) * cropHeight,
    })),
    candidates: simulation.candidates.map((candidate) => ({
      ...candidate,
      pageX: (candidate.pageX / image.width) * cropWidth,
      pageY: (candidate.pageY / image.height) * cropHeight,
    })),
    panelBoxes: simulation.panelBoxes.map((panel) => ({
      ...panel,
      rect: {
        x: (panel.rect.x / image.width) * cropWidth,
        y: (panel.rect.y / image.height) * cropHeight,
        width: (panel.rect.width / image.width) * cropWidth,
        height: (panel.rect.height / image.height) * cropHeight,
      },
    })),
    panelGuidedAnalysis: simulation.panelGuidedAnalysis
      ? {
          ...simulation.panelGuidedAnalysis,
          stepScores: simulation.panelGuidedAnalysis.stepScores.map((step) => ({
            ...step,
            candidate: step.candidate
              ? {
                  ...step.candidate,
                  pageX: (step.candidate.pageX / image.width) * cropWidth,
                  pageY: (step.candidate.pageY / image.height) * cropHeight,
                }
              : null,
          })),
        }
      : null,
    strategy: simulation.strategy,
    display: simulation.display,
  });

  const blob = await new Promise<Blob | null>((resolve) => exportCanvas.toBlob(resolve, "image/png"));
  if (!blob) {
    return;
  }

  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${stripFileExtension(image.url.split("/").pop() ?? "visualization")}-visualized.png`;
  anchor.click();
  URL.revokeObjectURL(url);
}
