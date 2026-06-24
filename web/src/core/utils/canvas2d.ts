import type { ImageRect } from "./roi";

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
