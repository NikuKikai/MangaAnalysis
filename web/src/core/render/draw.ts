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

function colorizeHistoryValue(value: number): [number, number, number, number] {
  const v = Math.max(0, Math.min(1, value));
  const low = Math.min(1, v * 1.6);
  const high = Math.max(0, Math.min(1, (v - 0.28) / 0.72));
  return [
    Math.round(255 * (0.25 + high * 0.75)),
    Math.round(255 * (0.2 + low * 0.55)),
    Math.round(255 * (0.1 + (1 - high) * 0.18)),
    Math.round(210 * v),
  ];
}

export function drawHistoryHeatmap(params: {
  context: CanvasRenderingContext2D;
  historyMap: Float32Array | null;
  historyMapWidth: number;
  historyMapHeight: number;
  imageRect: ImageRect | null;
  enabled: boolean;
}): void {
  const { context, historyMap, historyMapWidth, historyMapHeight, imageRect, enabled } = params;
  context.clearRect(0, 0, context.canvas.width, context.canvas.height);
  if (!enabled || !historyMap || !imageRect || historyMapWidth <= 0 || historyMapHeight <= 0) {
    return;
  }

  let maxValue = 0;
  for (let index = 0; index < historyMap.length; index += 1) {
    if (historyMap[index] > maxValue) {
      maxValue = historyMap[index];
    }
  }
  if (maxValue <= 0) {
    return;
  }

  const rgba = new Uint8ClampedArray(historyMapWidth * historyMapHeight * 4);
  for (let index = 0; index < historyMap.length; index += 1) {
    const normalized = Math.min(1, historyMap[index] / maxValue);
    const [r, g, b, a] = colorizeHistoryValue(normalized);
    const base = index * 4;
    rgba[base] = r;
    rgba[base + 1] = g;
    rgba[base + 2] = b;
    rgba[base + 3] = a;
  }

  const imageData = new ImageData(rgba, historyMapWidth, historyMapHeight);
  const scratchCanvas = document.createElement("canvas");
  scratchCanvas.width = historyMapWidth;
  scratchCanvas.height = historyMapHeight;
  const scratchContext = scratchCanvas.getContext("2d");
  if (!scratchContext) {
    throw new Error("2D canvas context is unavailable.");
  }
  scratchContext.putImageData(imageData, 0, 0);

  context.imageSmoothingEnabled = true;
  context.drawImage(scratchCanvas, imageRect.x, imageRect.y, imageRect.width, imageRect.height);
}
