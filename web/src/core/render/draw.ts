import type { Candidate, Point, RoiRect } from "../../types/simulation";
import type { ImageRect } from "../simulation/roi";

function setupCanvas(canvas: HTMLCanvasElement, width: number, height: number): CanvasRenderingContext2D {
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(width * ratio));
  canvas.height = Math.max(1, Math.floor(height * ratio));
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const context = canvas.getContext("2d");
  if (!context) {
    throw new Error("2D canvas context is unavailable.");
  }
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  return context;
}

export function resizeAndClear2dCanvas(canvas: HTMLCanvasElement, width: number, height: number): CanvasRenderingContext2D {
  return setupCanvas(canvas, width, height);
}

export function drawBaseImage(
  context: CanvasRenderingContext2D,
  bitmap: ImageBitmap,
  imageRect: ImageRect,
): void {
  context.clearRect(0, 0, context.canvas.width, context.canvas.height);
  context.imageSmoothingEnabled = true;
  context.drawImage(bitmap, imageRect.x, imageRect.y, imageRect.width, imageRect.height);
}

export function drawPreprocessPreview(params: {
  context: CanvasRenderingContext2D;
  preview: Uint8ClampedArray | null;
  previewSize: number;
  imageRect: ImageRect | null;
  imageWidth: number;
  imageHeight: number;
  roi: RoiRect | null;
  enabled: boolean;
}): void {
  const { context, preview, previewSize, imageRect, imageWidth, imageHeight, roi, enabled } = params;
  context.clearRect(0, 0, context.canvas.width, context.canvas.height);
  if (!enabled || !preview || !imageRect || !roi || previewSize <= 0) {
    return;
  }

  const imageData = new ImageData(preview, previewSize, previewSize);
  const scratchCanvas = document.createElement("canvas");
  scratchCanvas.width = previewSize;
  scratchCanvas.height = previewSize;
  const scratchContext = scratchCanvas.getContext("2d");
  if (!scratchContext) {
    throw new Error("2D canvas context is unavailable.");
  }
  scratchContext.putImageData(imageData, 0, 0);

  const roiRect = roiToScreenRect(roi, imageRect, imageWidth, imageHeight);
  context.imageSmoothingEnabled = true;
  context.drawImage(scratchCanvas, roiRect.x, roiRect.y, roiRect.width, roiRect.height);
}

function pagePointToScreen(point: Point, imageRect: ImageRect, imageWidth: number, imageHeight: number): Point {
  return {
    x: imageRect.x + (point.x / imageWidth) * imageRect.width,
    y: imageRect.y + (point.y / imageHeight) * imageRect.height,
  };
}

function roiToScreenRect(roi: RoiRect, imageRect: ImageRect, imageWidth: number, imageHeight: number): ImageRect {
  return {
    x: imageRect.x + (roi.x / imageWidth) * imageRect.width,
    y: imageRect.y + (roi.y / imageHeight) * imageRect.height,
    width: (roi.size / imageWidth) * imageRect.width,
    height: (roi.size / imageHeight) * imageRect.height,
  };
}

export function drawOverlay(params: {
  context: CanvasRenderingContext2D;
  viewportWidth: number;
  viewportHeight: number;
  imageRect: ImageRect | null;
  imageWidth: number;
  imageHeight: number;
  currentRoi: RoiRect | null;
  dragRoi: RoiRect | null;
  currentFixation: Point | null;
  pendingNextFixation: Point | null;
  trajectory: Point[];
  candidates: Candidate[];
  showHistory: boolean;
}): void {
  const {
    context,
    viewportWidth,
    viewportHeight,
    imageRect,
    imageWidth,
    imageHeight,
    currentRoi,
    dragRoi,
    currentFixation,
    pendingNextFixation,
    trajectory,
    candidates,
    showHistory,
  } = params;

  context.clearRect(0, 0, viewportWidth, viewportHeight);
  if (!imageRect) {
    return;
  }

  if (showHistory && trajectory.length > 1) {
    context.strokeStyle = "rgba(200, 200, 200, 0.45)";
    context.lineWidth = 1;
    context.beginPath();
    trajectory.forEach((point, index) => {
      const screen = pagePointToScreen(point, imageRect, imageWidth, imageHeight);
      if (index === 0) {
        context.moveTo(screen.x, screen.y);
      } else {
        context.lineTo(screen.x, screen.y);
      }
    });
    context.stroke();
  }

  if (showHistory) {
    trajectory.forEach((point) => {
      const screen = pagePointToScreen(point, imageRect, imageWidth, imageHeight);
      context.strokeStyle = "rgba(220, 220, 220, 0.85)";
      context.lineWidth = 1;
      context.beginPath();
      context.arc(screen.x, screen.y, 4, 0, Math.PI * 2);
      context.stroke();
    });
  }

  candidates.forEach((candidate) => {
    const screen = pagePointToScreen({ x: candidate.pageX, y: candidate.pageY }, imageRect, imageWidth, imageHeight);
    const alpha = 0.2 + Math.min(0.8, candidate.finalScore);
    context.fillStyle = `rgba(255, 180, 60, ${alpha})`;
    context.beginPath();
    context.arc(screen.x, screen.y, 2.5, 0, Math.PI * 2);
    context.fill();
  });

  if (currentRoi) {
    const roiRect = roiToScreenRect(currentRoi, imageRect, imageWidth, imageHeight);
    context.strokeStyle = "rgba(240, 240, 240, 0.92)";
    context.lineWidth = 1;
    context.strokeRect(roiRect.x + 0.5, roiRect.y + 0.5, roiRect.width, roiRect.height);
  }

  if (dragRoi) {
    const roiRect = roiToScreenRect(dragRoi, imageRect, imageWidth, imageHeight);
    context.strokeStyle = "rgba(255, 255, 255, 0.48)";
    context.lineWidth = 1;
    context.strokeRect(roiRect.x + 0.5, roiRect.y + 0.5, roiRect.width, roiRect.height);
  }

  if (currentFixation) {
    const screen = pagePointToScreen(currentFixation, imageRect, imageWidth, imageHeight);
    context.fillStyle = "rgba(77, 217, 227, 0.96)";
    context.beginPath();
    context.arc(screen.x, screen.y, 4.5, 0, Math.PI * 2);
    context.fill();
  }

  if (pendingNextFixation) {
    const screen = pagePointToScreen(pendingNextFixation, imageRect, imageWidth, imageHeight);
    context.strokeStyle = "rgba(255, 185, 76, 0.96)";
    context.lineWidth = 1.5;
    context.beginPath();
    context.arc(screen.x, screen.y, 7, 0, Math.PI * 2);
    context.stroke();
    context.fillStyle = "rgba(255, 185, 76, 0.96)";
    context.beginPath();
    context.arc(screen.x, screen.y, 3.5, 0, Math.PI * 2);
    context.fill();
  }
}
