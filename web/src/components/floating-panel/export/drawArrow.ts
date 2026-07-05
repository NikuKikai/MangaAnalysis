import type { Point } from "../../../types/simulation";

function buildArrowGeometry(start: Point, end: Point, startOffset: number, endOffset: number) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const length = Math.hypot(dx, dy);
  if (length < 1) {
    return null;
  }

  const ux = dx / length;
  const uy = dy / length;
  const startPoint = { x: start.x + ux * startOffset, y: start.y + uy * startOffset };
  const tipPoint = { x: end.x, y: end.y };
  const endPoint = { x: end.x - ux * endOffset, y: end.y - uy * endOffset };
  const headLength = endOffset + 4;
  const headSpread = 0.45;
  const left = {
    x: tipPoint.x - headLength * (ux * Math.cos(headSpread) - uy * Math.sin(headSpread)),
    y: tipPoint.y - headLength * (uy * Math.cos(headSpread) + ux * Math.sin(headSpread)),
  };
  const right = {
    x: tipPoint.x - headLength * (ux * Math.cos(-headSpread) - uy * Math.sin(-headSpread)),
    y: tipPoint.y - headLength * (uy * Math.cos(-headSpread) + ux * Math.sin(-headSpread)),
  };

  return {
    start: startPoint,
    end: endPoint,
    tip: tipPoint,
    left,
    right,
  };
}

export function drawDashedArrow(
  context: CanvasRenderingContext2D,
  start: Point,
  end: Point,
  style: {
    stroke: string;
    shadowStroke: string;
    strokeWidth: number;
    shadowWidth: number;
    dash: number[];
  },
) {
  const geometry = buildArrowGeometry(start, end, 10, 18);
  if (!geometry) {
    return;
  }

  context.save();
  context.setLineDash(style.dash);
  context.lineCap = "round";
  context.strokeStyle = style.shadowStroke;
  context.lineWidth = style.shadowWidth;
  context.beginPath();
  context.moveTo(geometry.start.x, geometry.start.y);
  context.lineTo(geometry.end.x, geometry.end.y);
  context.stroke();

  context.strokeStyle = style.stroke;
  context.lineWidth = style.strokeWidth;
  context.beginPath();
  context.moveTo(geometry.start.x, geometry.start.y);
  context.lineTo(geometry.end.x, geometry.end.y);
  context.stroke();
  context.setLineDash([]);

  context.fillStyle = style.stroke;
  context.strokeStyle = style.shadowStroke;
  context.lineWidth = 1.5;
  context.beginPath();
  context.moveTo(geometry.tip.x, geometry.tip.y);
  context.lineTo(geometry.left.x, geometry.left.y);
  context.lineTo(geometry.right.x, geometry.right.y);
  context.closePath();
  context.fill();
  context.stroke();
  context.restore();
}

export function drawSolidArrow(
  context: CanvasRenderingContext2D,
  start: Point,
  end: Point,
  style: {
    stroke: string;
    shadowStroke: string;
    strokeWidth: number;
  },
) {
  const geometry = buildArrowGeometry(start, end, 10, 15);
  if (!geometry) {
    return;
  }

  context.save();
  context.lineCap = "round";
  context.strokeStyle = style.shadowStroke;
  context.lineWidth = style.strokeWidth + 4;
  context.beginPath();
  context.moveTo(geometry.start.x, geometry.start.y);
  context.lineTo(geometry.end.x, geometry.end.y);
  context.stroke();
  context.strokeStyle = style.stroke;
  context.lineWidth = style.strokeWidth;
  context.beginPath();
  context.moveTo(geometry.start.x, geometry.start.y);
  context.lineTo(geometry.end.x, geometry.end.y);
  context.stroke();
  context.fillStyle = style.stroke;
  context.beginPath();
  context.moveTo(geometry.tip.x, geometry.tip.y);
  context.lineTo(geometry.left.x, geometry.left.y);
  context.lineTo(geometry.right.x, geometry.right.y);
  context.closePath();
  context.fill();
  context.restore();
}
