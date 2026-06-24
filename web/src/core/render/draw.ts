import type { RoiRect } from "../../types/simulation";
import { roiToScreenRect, type ImageRect } from "../simulation/roi";

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

  const imageData = new ImageData(new Uint8ClampedArray(preview), previewSize, previewSize);
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
