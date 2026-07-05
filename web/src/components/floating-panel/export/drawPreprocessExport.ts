import type { RoiRect } from "../../../types/simulation";

export function drawPreprocessExport(
  context: CanvasRenderingContext2D,
  preview: Float32Array,
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
  const planeSize = size * size;

  for (let y = 0; y < roiHeight; y += 1) {
    const v = roiHeight <= 1 ? 0 : y / (roiHeight - 1);
    const sampleY = Math.min(size - 1, Math.max(0, Math.round(v * (size - 1))));
    for (let x = 0; x < roiWidth; x += 1) {
      const u = roiWidth <= 1 ? 0 : x / (roiWidth - 1);
      const sampleX = Math.min(size - 1, Math.max(0, Math.round(u * (size - 1))));
      const sampleIndex = sampleY * size + sampleX;
      const r = Math.max(0, Math.min(255, (preview[sampleIndex] * 0.229 + 0.485) * 255));
      const g = Math.max(0, Math.min(255, (preview[planeSize + sampleIndex] * 0.224 + 0.456) * 255));
      const b = Math.max(0, Math.min(255, (preview[planeSize * 2 + sampleIndex] * 0.225 + 0.406) * 255));
      const pixelIndex = (y * roiWidth + x) * 4;
      imageData.data[pixelIndex] = r;
      imageData.data[pixelIndex + 1] = g;
      imageData.data[pixelIndex + 2] = b;
      imageData.data[pixelIndex + 3] = 255;
    }
  }

  context.putImageData(imageData, roiX, roiY);
}
