import type { Candidate, DisplayState, FluidityAnalysis, PanelBox, Point, RoiRect, SimulationStrategy } from "../../../types/simulation";
import { drawDashedArrow, drawSolidArrow } from "./drawArrow";

export function drawExportOverlays(
  context: CanvasRenderingContext2D,
  params: {
    currentRoi: RoiRect | null;
    currentFixation: Point | null;
    pendingNextFixation: Point | null;
    trajectory: Point[];
    candidates: Candidate[];
    panelBoxes: PanelBox[];
    panelGuidedAnalysis: FluidityAnalysis | null;
    strategy: SimulationStrategy;
    display: DisplayState;
  },
) {
  const {
    currentRoi,
    currentFixation,
    pendingNextFixation,
    trajectory,
    candidates,
    panelBoxes,
    panelGuidedAnalysis,
    strategy,
    display,
  } = params;

  if (display.showPanelBoxes) {
    for (const panel of panelBoxes) {
      context.save();
      context.strokeStyle = "rgba(0, 0, 0, 0.58)";
      context.lineWidth = 6;
      context.strokeRect(panel.rect.x, panel.rect.y, panel.rect.width, panel.rect.height);
      context.strokeStyle = "rgba(246, 250, 252, 0.97)";
      context.lineWidth = 2;
      context.strokeRect(panel.rect.x, panel.rect.y, panel.rect.width, panel.rect.height);
      context.restore();
    }
  }

  if (display.showSelectorOverlay && strategy === "saliency_only" && currentFixation && pendingNextFixation) {
    drawDashedArrow(context, currentFixation, pendingNextFixation, {
      stroke: "rgba(255, 214, 138, 0.48)",
      shadowStroke: "rgba(0, 0, 0, 0.32)",
      strokeWidth: 1.75,
      shadowWidth: 3.5,
      dash: [6, 6],
    });
  }

  if (trajectory.length >= 2) {
    context.save();
    context.beginPath();
    context.moveTo(trajectory[0].x, trajectory[0].y);
    for (let index = 1; index < trajectory.length; index += 1) {
      context.lineTo(trajectory[index].x, trajectory[index].y);
    }
    context.strokeStyle = "rgba(0, 0, 0, 0.6)";
    context.lineCap = "round";
    context.lineJoin = "round";
    context.lineWidth = 6;
    context.stroke();
    context.strokeStyle = "rgba(255, 236, 184, 0.96)";
    context.lineWidth = 2.75;
    context.stroke();
    context.restore();
  }

  if (display.showSelectorOverlay && strategy === "panel_guided" && panelGuidedAnalysis) {
    const maxScore = Math.max(panelGuidedAnalysis.maxScore, 1e-8);
    for (const step of panelGuidedAnalysis.stepScores) {
      if (!step.candidate || step.stepIndex >= trajectory.length) {
        continue;
      }
      const start = trajectory[step.stepIndex];
      const end = { x: step.candidate.pageX, y: step.candidate.pageY };
      drawSolidArrow(context, start, end, {
        stroke: step.strongerThanActualNext ? "rgba(255, 72, 72, 0.96)" : "rgba(255, 192, 0, 0.96)",
        shadowStroke: "rgba(0, 0, 0, 0.36)",
        strokeWidth: 1 + (step.score / maxScore) * 7,
      });
    }
  }

  for (const point of trajectory) {
    context.save();
    context.fillStyle = "rgba(0, 0, 0, 0.45)";
    context.beginPath();
    context.arc(point.x, point.y, 7, 0, Math.PI * 2);
    context.fill();
    context.fillStyle = "rgba(255, 245, 212, 0.94)";
    context.strokeStyle = "rgba(20, 22, 24, 0.95)";
    context.lineWidth = 2;
    context.beginPath();
    context.arc(point.x, point.y, 5, 0, Math.PI * 2);
    context.fill();
    context.stroke();
    context.restore();
  }

  if (display.showSelectorOverlay) {
    for (const candidate of candidates) {
      const alpha = 0.35 + Math.min(0.65, candidate.finalScore);
      context.save();
      context.fillStyle = "rgba(0, 0, 0, 0.4)";
      context.beginPath();
      context.arc(candidate.pageX, candidate.pageY, 5, 0, Math.PI * 2);
      context.fill();
      context.fillStyle = `rgba(255, 191, 71, ${alpha})`;
      context.strokeStyle = "rgba(34, 21, 5, 0.95)";
      context.lineWidth = 1.5;
      context.beginPath();
      context.arc(candidate.pageX, candidate.pageY, 3.25, 0, Math.PI * 2);
      context.fill();
      context.stroke();
      context.restore();
    }
  }

  if (display.showPreprocess && currentRoi) {
    context.save();
    context.strokeStyle = "rgba(0, 0, 0, 0.58)";
    context.lineWidth = 6;
    context.strokeRect(currentRoi.x, currentRoi.y, currentRoi.size, currentRoi.size);
    context.strokeStyle = "rgba(246, 250, 252, 0.97)";
    context.lineWidth = 2;
    context.strokeRect(currentRoi.x, currentRoi.y, currentRoi.size, currentRoi.size);
    context.restore();
  }

  if (currentFixation) {
    context.save();
    context.fillStyle = "rgba(0, 0, 0, 0.42)";
    context.beginPath();
    context.arc(currentFixation.x, currentFixation.y, 9, 0, Math.PI * 2);
    context.fill();
    context.fillStyle = "rgba(93, 237, 255, 0.96)";
    context.strokeStyle = "rgba(14, 24, 31, 0.95)";
    context.lineWidth = 2.5;
    context.beginPath();
    context.arc(currentFixation.x, currentFixation.y, 6, 0, Math.PI * 2);
    context.fill();
    context.stroke();
    context.fillStyle = "rgba(239, 254, 255, 0.98)";
    context.beginPath();
    context.arc(currentFixation.x, currentFixation.y, 2.5, 0, Math.PI * 2);
    context.fill();
    context.restore();
  }

  if (pendingNextFixation) {
    context.save();
    context.fillStyle = "rgba(0, 0, 0, 0.42)";
    context.beginPath();
    context.arc(pendingNextFixation.x, pendingNextFixation.y, 12, 0, Math.PI * 2);
    context.fill();
    context.strokeStyle = "rgba(255, 198, 92, 0.98)";
    context.lineWidth = 3;
    context.beginPath();
    context.arc(pendingNextFixation.x, pendingNextFixation.y, 9, 0, Math.PI * 2);
    context.stroke();
    context.fillStyle = "rgba(255, 198, 92, 0.98)";
    context.strokeStyle = "rgba(32, 20, 3, 0.95)";
    context.lineWidth = 2;
    context.beginPath();
    context.arc(pendingNextFixation.x, pendingNextFixation.y, 4.5, 0, Math.PI * 2);
    context.fill();
    context.stroke();
    context.restore();
  }
}
