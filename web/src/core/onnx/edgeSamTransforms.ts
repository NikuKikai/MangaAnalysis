import type { EdgeSamMask, EdgeSamPoint } from "./edgeSamTypes";

export const EDGE_SAM_MODEL_SIZE = 1024;
export const EDGE_SAM_MASK_SIZE = 256;

export type EdgeSamImageTransform = {
  originalWidth: number;
  originalHeight: number;
  inputWidth: number;
  inputHeight: number;
  scale: number;
};

export type EdgeSamPreprocessedImage = {
  tensor: Float32Array;
  transform: EdgeSamImageTransform;
};

export function transformEdgeSamPoints(points: EdgeSamPoint[], transform: EdgeSamImageTransform): {
  coords: Float32Array;
  labels: Float32Array;
} {
  const coords = new Float32Array(points.length * 2);
  const labels = new Float32Array(points.length);

  points.forEach((point, index) => {
    coords[index * 2] = point.x * transform.scale;
    coords[index * 2 + 1] = point.y * transform.scale;
    labels[index] = point.label;
  });

  return { coords, labels };
}

export function postprocessEdgeSamMasks(
  lowResMasks: Uint8Array,
  scores: Float32Array,
  transform: EdgeSamImageTransform,
): EdgeSamMask[] {
  const maskCount = Math.trunc(lowResMasks.length / (EDGE_SAM_MASK_SIZE * EDGE_SAM_MASK_SIZE));
  const masks: EdgeSamMask[] = [];
  const validWidth = Math.max(1, Math.min(EDGE_SAM_MASK_SIZE, Math.round((transform.inputWidth / EDGE_SAM_MODEL_SIZE) * EDGE_SAM_MASK_SIZE)));
  const validHeight = Math.max(1, Math.min(EDGE_SAM_MASK_SIZE, Math.round((transform.inputHeight / EDGE_SAM_MODEL_SIZE) * EDGE_SAM_MASK_SIZE)));

  for (let maskIndex = 0; maskIndex < maskCount; maskIndex += 1) {
    const maskOffset = maskIndex * EDGE_SAM_MASK_SIZE * EDGE_SAM_MASK_SIZE;

    masks.push({
      data: lowResMasks.slice(maskOffset, maskOffset + EDGE_SAM_MASK_SIZE * EDGE_SAM_MASK_SIZE),
      width: EDGE_SAM_MASK_SIZE,
      height: EDGE_SAM_MASK_SIZE,
      validWidth,
      validHeight,
      score: scores[maskIndex] ?? 0,
    });
  }

  return masks;
}
