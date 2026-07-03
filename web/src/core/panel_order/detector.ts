import { buildPanelLayout } from "./layout";
import { loadOpenCv } from "./opencv";
import type { PanelAnalysis, PanelBox, PanelRect } from "./types";

type PanelDetectionConfig = {
  adaptiveBlockSize: number;
  adaptiveC: number;
  closeKernelSize: number;
  minAreaRatio: number;
  maxAreaRatio: number;
  minRectangularity: number;
  maxPolygonVertices: number;
  minAspectRatio: number;
  maxAspectRatio: number;
  minBorderScore: number;
  refineMarginRatio: number;
  maxRefineMarginPx: number;
  borderBandPx: number;
  overlapIouThreshold: number;
  containThreshold: number;
};

const DEFAULT_CONFIG: PanelDetectionConfig = {
  adaptiveBlockSize: 31,
  adaptiveC: 10,
  closeKernelSize: 5,
  minAreaRatio: 0.01,
  maxAreaRatio: 0.92,
  minRectangularity: 0.82,
  maxPolygonVertices: 10,
  minAspectRatio: 0.12,
  maxAspectRatio: 8.0,
  minBorderScore: 0.35,
  refineMarginRatio: 0.18,
  maxRefineMarginPx: 80,
  borderBandPx: 5,
  overlapIouThreshold: 0.55,
  containThreshold: 0.9,
};

function rectRight(rect: PanelRect): number {
  return rect.x + rect.width;
}

function rectBottom(rect: PanelRect): number {
  return rect.y + rect.height;
}

function rectArea(rect: PanelRect): number {
  return rect.width * rect.height;
}

function rectIntersection(left: PanelRect, right: PanelRect): PanelRect | null {
  const x = Math.max(left.x, right.x);
  const y = Math.max(left.y, right.y);
  const r = Math.min(rectRight(left), rectRight(right));
  const b = Math.min(rectBottom(left), rectBottom(right));
  if (r <= x || b <= y) {
    return null;
  }
  return { x, y, width: r - x, height: b - y };
}

function rectIoU(left: PanelRect, right: PanelRect): number {
  const overlap = rectIntersection(left, right);
  if (!overlap) {
    return 0;
  }
  const unionArea = rectArea(left) + rectArea(right) - rectArea(overlap);
  return rectArea(overlap) / Math.max(unionArea, 1);
}

function containsRatio(container: PanelRect, candidate: PanelRect): number {
  const overlap = rectIntersection(container, candidate);
  if (!overlap) {
    return 0;
  }
  return rectArea(overlap) / Math.max(rectArea(candidate), 1);
}

function imageBitmapToImageData(bitmap: ImageBitmap): ImageData {
  const canvas = document.createElement("canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) {
    throw new Error("Failed to create a 2D canvas for panel detection.");
  }
  context.drawImage(bitmap, 0, 0);
  return context.getImageData(0, 0, bitmap.width, bitmap.height);
}

function buildLineMask(cv: any, gray: any, config: PanelDetectionConfig): any {
  const blurred = new cv.Mat();
  const binary = new cv.Mat();
  const lineMask = new cv.Mat();
  const kernel = cv.Mat.ones(config.closeKernelSize, config.closeKernelSize, cv.CV_8U);

  try {
    // Blur first so adaptive thresholding reacts to borders instead of screentone noise.
    cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0, 0, cv.BORDER_DEFAULT);
    cv.adaptiveThreshold(
      blurred,
      binary,
      255,
      cv.ADAPTIVE_THRESH_GAUSSIAN_C,
      cv.THRESH_BINARY_INV,
      config.adaptiveBlockSize,
      config.adaptiveC,
    );
    // Close small gaps so broken borders still form usable contours.
    cv.morphologyEx(binary, lineMask, cv.MORPH_CLOSE, kernel, new cv.Point(-1, -1), 1);
    return lineMask.clone();
  } finally {
    blurred.delete();
    binary.delete();
    lineMask.delete();
    kernel.delete();
  }
}

function sliceMean(cv: any, mat: any, x: number, y: number, width: number, height: number): number {
  const roi = mat.roi(new cv.Rect(x, y, width, height));
  try {
    const mean = cv.mean(roi);
    return mean[0] / 255.0;
  } finally {
    roi.delete();
  }
}

function bestVerticalBoundary(
  cv: any,
  lineMask: any,
  edgeX: number,
  top: number,
  height: number,
  direction: number,
  maxShift: number,
  band: number,
): number {
  const heightLimit = lineMask.rows as number;
  const widthLimit = lineMask.cols as number;
  const clampedTop = Math.max(0, top);
  const bottom = Math.min(heightLimit, clampedTop + height);
  let bestX = Math.max(0, Math.min(edgeX, widthLimit - band));
  let bestScore = -Infinity;

  for (let shift = 0; shift <= maxShift; shift += 1) {
    const x = Math.max(0, Math.min(edgeX + direction * shift, widthLimit - band));
    const score = sliceMean(cv, lineMask, x, clampedTop, band, bottom - clampedTop) - 0.3 * (shift / Math.max(maxShift, 1));
    if (score > bestScore) {
      bestScore = score;
      bestX = x;
    }
  }
  return bestX;
}

function bestHorizontalBoundary(
  cv: any,
  lineMask: any,
  edgeY: number,
  left: number,
  width: number,
  direction: number,
  maxShift: number,
  band: number,
): number {
  const heightLimit = lineMask.rows as number;
  const widthLimit = lineMask.cols as number;
  const clampedLeft = Math.max(0, left);
  const right = Math.min(widthLimit, clampedLeft + width);
  let bestY = Math.max(0, Math.min(edgeY, heightLimit - band));
  let bestScore = -Infinity;

  for (let shift = 0; shift <= maxShift; shift += 1) {
    const y = Math.max(0, Math.min(edgeY + direction * shift, heightLimit - band));
    const score = sliceMean(cv, lineMask, clampedLeft, y, right - clampedLeft, band) - 0.3 * (shift / Math.max(maxShift, 1));
    if (score > bestScore) {
      bestScore = score;
      bestY = y;
    }
  }
  return bestY;
}

function refineRect(cv: any, lineMask: any, rect: PanelRect, config: PanelDetectionConfig): PanelRect {
  const marginX = Math.min(config.maxRefineMarginPx, Math.max(6, Math.round(rect.width * config.refineMarginRatio)));
  const marginY = Math.min(config.maxRefineMarginPx, Math.max(6, Math.round(rect.height * config.refineMarginRatio)));
  const band = config.borderBandPx;

  const left = bestVerticalBoundary(cv, lineMask, rect.x, rect.y, rect.height, +1, marginX, band);
  const right = bestVerticalBoundary(cv, lineMask, rectRight(rect) - band, rect.y, rect.height, -1, marginX, band) + band;
  const top = bestHorizontalBoundary(cv, lineMask, rect.y, rect.x, rect.width, +1, marginY, band);
  const bottom = bestHorizontalBoundary(cv, lineMask, rectBottom(rect) - band, rect.x, rect.width, -1, marginY, band) + band;

  if (right <= left + 10 || bottom <= top + 10) {
    return rect;
  }
  return {
    x: left,
    y: top,
    width: right - left,
    height: bottom - top,
  };
}

function borderScore(cv: any, lineMask: any, rect: PanelRect, config: PanelDetectionConfig): number {
  const band = Math.min(config.borderBandPx, rect.width, rect.height);
  if (band <= 0) {
    return 0;
  }
  const top = sliceMean(cv, lineMask, rect.x, rect.y, rect.width, band);
  const bottom = sliceMean(cv, lineMask, rect.x, rectBottom(rect) - band, rect.width, band);
  const left = sliceMean(cv, lineMask, rect.x, rect.y, band, rect.height);
  const right = sliceMean(cv, lineMask, rectRight(rect) - band, rect.y, band, rect.height);
  return (top + bottom + left + right) / 4.0;
}

function deduplicateCandidates(candidates: PanelBox[], config: PanelDetectionConfig): PanelBox[] {
  const kept: PanelBox[] = [];
  const ranked = [...candidates].sort((left, right) => {
    if (left.score !== right.score) {
      return right.score - left.score;
    }
    return rectArea(right.rect) - rectArea(left.rect);
  });

  for (const candidate of ranked) {
    let shouldKeep = true;
    for (const existing of kept) {
      if (containsRatio(existing.rect, candidate.rect) >= config.containThreshold) {
        shouldKeep = false;
        break;
      }
      if (rectIoU(candidate.rect, existing.rect) >= config.overlapIouThreshold) {
        shouldKeep = false;
        break;
      }
    }
    if (shouldKeep) {
      kept.push(candidate);
    }
  }

  return kept.sort((left, right) => {
    if (left.rect.y !== right.rect.y) {
      return left.rect.y - right.rect.y;
    }
    return left.rect.x - right.rect.x;
  });
}

export async function analyzePanels(bitmap: ImageBitmap): Promise<PanelAnalysis> {
  const cv = (await loadOpenCv()) as any;
  const imageData = imageBitmapToImageData(bitmap);
  const source = cv.matFromImageData(imageData);
  const gray = new cv.Mat();
  const hierarchy = new cv.Mat();
  const contours = new cv.MatVector();
  const config = DEFAULT_CONFIG;

  try {
    cv.cvtColor(source, gray, cv.COLOR_RGBA2GRAY);
    const lineMask = buildLineMask(cv, gray, config);
    try {
      // Find contour candidates before filtering them with panel-specific geometry tests.
      cv.findContours(lineMask, contours, hierarchy, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE);
      const pageArea = gray.rows * gray.cols;
      const rawCandidates: PanelBox[] = [];

      for (let index = 0; index < contours.size(); index += 1) {
        const contour = contours.get(index);
        const polygon = new cv.Mat();
        try {
          const bounds = cv.boundingRect(contour);
          const rect: PanelRect = {
            x: bounds.x,
            y: bounds.y,
            width: bounds.width,
            height: bounds.height,
          };
          const area = rectArea(rect);
          if (area < pageArea * config.minAreaRatio || area > pageArea * config.maxAreaRatio) {
            continue;
          }

          const aspectRatio = rect.width / Math.max(rect.height, 1);
          if (aspectRatio < config.minAspectRatio || aspectRatio > config.maxAspectRatio) {
            continue;
          }

          const contourArea = cv.contourArea(contour);
          const rectangularity = contourArea / Math.max(area, 1);
          if (rectangularity < config.minRectangularity) {
            continue;
          }

          const perimeter = cv.arcLength(contour, true);
          cv.approxPolyDP(contour, polygon, 0.02 * perimeter, true);
          if (polygon.rows > config.maxPolygonVertices) {
            continue;
          }

          const refinedRect = refineRect(cv, lineMask, rect, config);
          const refinedBorderScore = borderScore(cv, lineMask, refinedRect, config);
          if (refinedBorderScore < config.minBorderScore) {
            continue;
          }

          rawCandidates.push({
            panelId: -1,
            rect: refinedRect,
            score: rectangularity * 0.6 + refinedBorderScore * 0.4,
          });
        } finally {
          polygon.delete();
          contour.delete();
        }
      }

      const panels = deduplicateCandidates(rawCandidates, config).map((panel, index) => ({
        ...panel,
        panelId: index,
      }));
      if (panels.length === 0) {
        throw new Error("No manga panels were detected.");
      }

      const { layoutTree, readingOrder } = buildPanelLayout(panels);
      return {
        imageWidth: bitmap.width,
        imageHeight: bitmap.height,
        panels,
        layoutTree,
        readingOrder,
      };
    } finally {
      lineMask.delete();
    }
  } finally {
    source.delete();
    gray.delete();
    hierarchy.delete();
    contours.delete();
  }
}
