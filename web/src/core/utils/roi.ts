import type { Point, RoiRect } from "../../types/simulation";
import type { PanelRect } from "../../types/simulation";

export type ImageRect = {
  x: number;
  y: number;
  width: number;
  height: number;
};

export function fitImageToViewport(
  imageWidth: number,
  imageHeight: number,
  viewportWidth: number,
  viewportHeight: number,
): ImageRect {
  const scale = Math.min(viewportWidth / imageWidth, viewportHeight / imageHeight);
  const width = imageWidth * scale;
  const height = imageHeight * scale;
  return {
    x: (viewportWidth - width) * 0.5,
    y: (viewportHeight - height) * 0.5,
    width,
    height,
  };
}

export function screenToPagePoint(
  clientX: number,
  clientY: number,
  imageRect: ImageRect,
  imageWidth: number,
  imageHeight: number,
): Point | null {
  const localX = clientX - imageRect.x;
  const localY = clientY - imageRect.y;
  if (localX < 0 || localY < 0 || localX > imageRect.width || localY > imageRect.height) {
    return null;
  }
  return {
    x: (localX / imageRect.width) * imageWidth,
    y: (localY / imageRect.height) * imageHeight,
  };
}

export function pagePointToScreen(
  point: Point,
  imageRect: ImageRect,
  imageWidth: number,
  imageHeight: number,
): Point {
  return {
    x: imageRect.x + (point.x / imageWidth) * imageRect.width,
    y: imageRect.y + (point.y / imageHeight) * imageRect.height,
  };
}

export function roiToScreenRect(roi: RoiRect, imageRect: ImageRect, imageWidth: number, imageHeight: number): ImageRect {
  return {
    x: imageRect.x + (roi.x / imageWidth) * imageRect.width,
    y: imageRect.y + (roi.y / imageHeight) * imageRect.height,
    width: (roi.size / imageWidth) * imageRect.width,
    height: (roi.size / imageHeight) * imageRect.height,
  };
}

export function pageRectToScreenRect(rect: PanelRect, imageRect: ImageRect, imageWidth: number, imageHeight: number): ImageRect {
  return {
    x: imageRect.x + (rect.x / imageWidth) * imageRect.width,
    y: imageRect.y + (rect.y / imageHeight) * imageRect.height,
    width: (rect.width / imageWidth) * imageRect.width,
    height: (rect.height / imageHeight) * imageRect.height,
  };
}

export function imageRectToPhysical(imageRect: ImageRect | null, ratio: number): ImageRect | null {
  if (!imageRect) {
    return null;
  }
  return {
    x: imageRect.x * ratio,
    y: imageRect.y * ratio,
    width: imageRect.width * ratio,
    height: imageRect.height * ratio,
  };
}

export function roiToPhysicalScreenRect(
  roi: RoiRect | null,
  imageRect: ImageRect | null,
  imageWidth: number,
  imageHeight: number,
  ratio: number,
): ImageRect | null {
  if (!roi || !imageRect || imageWidth <= 0 || imageHeight <= 0) {
    return null;
  }
  const roiRect = roiToScreenRect(roi, imageRect, imageWidth, imageHeight);
  return imageRectToPhysical(roiRect, ratio);
}

export function createCenteredSquareRoi(fixation: Point, halfSize: number): RoiRect {
  return {
    x: fixation.x - halfSize,
    y: fixation.y - halfSize,
    size: halfSize * 2,
  };
}

export function createFullPageSquareRoi(imageWidth: number, imageHeight: number): RoiRect {
  const size = Math.max(imageWidth, imageHeight);
  return {
    x: 0,
    y: 0,
    size,
  };
}

export function createSquareFromDrag(anchor: Point, current: Point): RoiRect {
  const dx = current.x - anchor.x;
  const dy = current.y - anchor.y;
  const size = Math.max(Math.abs(dx), Math.abs(dy));
  const signedX = dx >= 0 ? 1 : -1;
  const signedY = dy >= 0 ? 1 : -1;
  const endX = anchor.x + size * signedX;
  const endY = anchor.y + size * signedY;
  return {
    x: Math.min(anchor.x, endX),
    y: Math.min(anchor.y, endY),
    size,
  };
}

export function roiCenter(roi: RoiRect): Point {
  return {
    x: roi.x + roi.size * 0.5,
    y: roi.y + roi.size * 0.5,
  };
}
