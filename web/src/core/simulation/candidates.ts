import type { Candidate, Point, RoiRect } from "../../types/simulation";
import { historyIndex, inhibitionFactor } from "./history";

function slidingWindowMax1d(
  source: Float32Array,
  target: Float32Array,
  length: number,
  radius: number,
  sourceOffset: number,
  sourceStride: number,
  targetOffset: number,
  targetStride: number,
): void {
  const windowSize = radius * 2 + 1;
  const dequeIndices = new Int32Array(length);
  let head = 0;
  let tail = 0;

  for (let index = 0; index < length; index += 1) {
    const sourceIndex = sourceOffset + index * sourceStride;
    const value = source[sourceIndex];
    while (tail > head) {
      const previousIndex = dequeIndices[tail - 1];
      const previousValue = source[sourceOffset + previousIndex * sourceStride];
      if (previousValue > value) {
        break;
      }
      tail -= 1;
    }
    dequeIndices[tail] = index;
    tail += 1;

    const minIndex = index - windowSize + 1;
    while (tail > head && dequeIndices[head] < minIndex) {
      head += 1;
    }

    const centeredIndex = index - radius;
    if (centeredIndex >= 0) {
      const clampedCenter = centeredIndex;
      const leftBound = Math.max(0, clampedCenter - radius);
      while (tail > head && dequeIndices[head] < leftBound) {
        head += 1;
      }
      target[targetOffset + clampedCenter * targetStride] = source[sourceOffset + dequeIndices[head] * sourceStride];
    }
  }

  for (let centeredIndex = Math.max(0, length - radius); centeredIndex < length; centeredIndex += 1) {
    const leftBound = Math.max(0, centeredIndex - radius);
    while (tail > head && dequeIndices[head] < leftBound) {
      head += 1;
    }
    target[targetOffset + centeredIndex * targetStride] = source[sourceOffset + dequeIndices[head] * sourceStride];
  }
}

function maxPool2d(data: Float32Array, width: number, height: number, radius: number): Float32Array {
  if (radius <= 0) {
    return data.slice();
  }
  const horizontal = new Float32Array(data.length);
  const pooled = new Float32Array(data.length);

  for (let y = 0; y < height; y += 1) {
    slidingWindowMax1d(data, horizontal, width, radius, y * width, 1, y * width, 1);
  }
  for (let x = 0; x < width; x += 1) {
    slidingWindowMax1d(horizontal, pooled, height, radius, x, width, x, width);
  }

  return pooled;
}

function distanceScore(from: Point, to: Point, sigma: number): number {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const sigmaSq = Math.max(sigma * sigma, 1e-6);
  return Math.exp(-0.5 * (dx * dx + dy * dy) / sigmaSq);
}

export function normalizeHeatmap(data: Float32Array): Float32Array {
  let minValue = Number.POSITIVE_INFINITY;
  let maxValue = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < data.length; index += 1) {
    minValue = Math.min(minValue, data[index]);
    maxValue = Math.max(maxValue, data[index]);
  }
  const range = Math.max(maxValue - minValue, 1e-8);
  const normalized = new Float32Array(data.length);
  for (let index = 0; index < data.length; index += 1) {
    normalized[index] = (data[index] - minValue) / range;
  }
  return normalized;
}

export function extractCandidates(params: {
  heatmap: Float32Array;
  mapWidth: number;
  mapHeight: number;
  thresholdRatio: number;
  nmsRadius: number;
  topK: number;
  roi: RoiRect;
  imageWidth: number;
  imageHeight: number;
  currentFixation: Point;
  historyMap: Float32Array;
  historyMapWidth: number;
  historyMapHeight: number;
  historyAlpha: number;
  distanceSigma: number;
}): Candidate[] {
  const {
    heatmap,
    mapWidth,
    mapHeight,
    thresholdRatio,
    nmsRadius,
    topK,
    roi,
    imageWidth,
    imageHeight,
    currentFixation,
    historyMap,
    historyMapWidth,
    historyMapHeight,
    historyAlpha,
    distanceSigma,
  } = params;

  let maxValue = 0;
  for (let index = 0; index < heatmap.length; index += 1) {
    maxValue = Math.max(maxValue, heatmap[index]);
  }
  const threshold = maxValue * thresholdRatio;

  const pooled = maxPool2d(heatmap, mapWidth, mapHeight, nmsRadius);
  const candidates: Candidate[] = [];

  for (let y = 0; y < mapHeight; y += 1) {
    for (let x = 0; x < mapWidth; x += 1) {
      const index = y * mapWidth + x;
      const score = heatmap[index];
      if (score < threshold || score < pooled[index] - 1e-8) {
        continue;
      }
      const pageX = roi.x + ((x + 0.5) / mapWidth) * roi.size;
      const pageY = roi.y + ((y + 0.5) / mapHeight) * roi.size;
      if (pageX < 0 || pageY < 0 || pageX >= imageWidth || pageY >= imageHeight) {
        continue;
      }
      const px = Math.min(imageWidth - 1, Math.max(0, Math.round(pageX)));
      const py = Math.min(imageHeight - 1, Math.max(0, Math.round(pageY)));
      const historyValue =
        px < historyMapWidth && py < historyMapHeight ? historyMap[historyIndex(historyMapWidth, px, py)] : 0;
      const inhib = inhibitionFactor(historyValue, historyAlpha);
      const dist = distanceScore(currentFixation, { x: pageX, y: pageY }, distanceSigma);
      candidates.push({
        modelX: x,
        modelY: y,
        pageX,
        pageY,
        saliencyScore: score,
        historyValue,
        inhibitionScore: inhib,
        distanceScore: dist,
        finalScore: score * inhib * dist,
      });
    }
  }

  candidates.sort((left, right) => right.finalScore - left.finalScore);
  return candidates.slice(0, topK);
}
