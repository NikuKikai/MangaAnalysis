import type { Point } from "../../types/simulation";

export function createHistoryMap(width: number, height: number): Float32Array {
  return new Float32Array(width * height);
}

export function historyIndex(width: number, x: number, y: number): number {
  return y * width + x;
}

export function addFixationToHistory(
  historyMap: Float32Array,
  width: number,
  height: number,
  fixation: Point,
  sigma: number,
  amplitude = 1,
): void {
  const radius = Math.max(1, Math.ceil(3 * sigma));
  const left = Math.max(0, Math.floor(fixation.x) - radius);
  const right = Math.min(width, Math.floor(fixation.x) + radius + 1);
  const top = Math.max(0, Math.floor(fixation.y) - radius);
  const bottom = Math.min(height, Math.floor(fixation.y) + radius + 1);
  const sigmaSq = Math.max(sigma * sigma, 1e-6);

  for (let y = top; y < bottom; y += 1) {
    for (let x = left; x < right; x += 1) {
      const dx = x - fixation.x;
      const dy = y - fixation.y;
      const gaussian = Math.exp(-0.5 * (dx * dx + dy * dy) / sigmaSq);
      historyMap[historyIndex(width, x, y)] += amplitude * gaussian;
    }
  }
}

export function inhibitionFactor(historyValue: number, alpha: number): number {
  return Math.exp(-alpha * historyValue);
}
