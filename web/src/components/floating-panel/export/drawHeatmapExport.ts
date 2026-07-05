import type { RoiRect } from "../../../types/simulation";

function smoothstep(edge0: number, edge1: number, value: number) {
  const t = Math.max(0, Math.min(1, (value - edge0) / Math.max(edge1 - edge0, 1e-6)));
  return t * t * (3 - 2 * t);
}

export function drawHeatmapExport(
  context: CanvasRenderingContext2D,
  heatmap: Float32Array,
  size: number,
  roi: RoiRect,
  imageWidth: number,
  imageHeight: number,
  outputWidth: number,
  outputHeight: number,
) {
  const roiX = Math.round((roi.x / imageWidth) * outputWidth);
  const roiY = Math.round((roi.y / imageHeight) * outputHeight);
  const roiWidth = Math.max(1, Math.round((roi.size / imageWidth) * outputWidth));
  const roiHeight = Math.max(1, Math.round((roi.size / imageHeight) * outputHeight));
  const imageData = context.createImageData(roiWidth, roiHeight);

  for (let y = 0; y < roiHeight; y += 1) {
    const v = roiHeight <= 1 ? 0 : y / (roiHeight - 1);
    const sampleY = Math.min(size - 1, Math.max(0, Math.round(v * (size - 1))));
    for (let x = 0; x < roiWidth; x += 1) {
      const u = roiWidth <= 1 ? 0 : x / (roiWidth - 1);
      const sampleX = Math.min(size - 1, Math.max(0, Math.round(u * (size - 1))));
      const sampleValue = Math.max(0, Math.min(1, heatmap[sampleY * size + sampleX] ?? 0));
      const low = smoothstep(0, 0.55, sampleValue);
      const high = smoothstep(0.4, 1, sampleValue);
      const r = Math.max(0, Math.min(255, high * 255));
      const g = Math.max(0, Math.min(255, low * 255));
      const b = Math.max(0, Math.min(255, (0.2 + low * 0.5) * 255));
      const alpha = Math.max(0, Math.min(255, sampleValue * 0.72 * 255));
      const pixelIndex = (y * roiWidth + x) * 4;
      imageData.data[pixelIndex] = r;
      imageData.data[pixelIndex + 1] = g;
      imageData.data[pixelIndex + 2] = b;
      imageData.data[pixelIndex + 3] = alpha;
    }
  }

  const bufferCanvas = document.createElement("canvas");
  bufferCanvas.width = roiWidth;
  bufferCanvas.height = roiHeight;
  const bufferContext = bufferCanvas.getContext("2d");
  if (!bufferContext) {
    return;
  }
  bufferContext.putImageData(imageData, 0, 0);
  context.drawImage(bufferCanvas, roiX, roiY);
}
