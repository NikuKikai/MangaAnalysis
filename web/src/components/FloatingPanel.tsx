import { useEffect, useRef, useState, type ChangeEvent, type ReactNode } from "react";
import html2canvas from "html2canvas";
import { HiBars3 } from "react-icons/hi2";
import { IoEyeOutline, IoFolderOpenOutline, IoLanguageOutline, IoPlay, IoRefresh, IoSaveOutline } from "react-icons/io5";
import { FaGithub, FaXTwitter } from "react-icons/fa6";
import { LuMousePointer2 } from "react-icons/lu";
import { PiRectangle } from "react-icons/pi";
import { TbInfoCircle } from "react-icons/tb";
import { useTranslation } from "react-i18next";
import { useShallow } from "zustand/react/shallow";
import { useSimulationEngineContext } from "../app/SimulationRuntime";
import { IconButton } from "./IconButton";
import type { AppLanguage } from "../i18n/resources";
import type { Candidate, DisplayState, FluidityAnalysis, PanelBox, Point, RoiRect, SimulationStrategy } from "../types/simulation";
import { useSimulationStore } from "../store/simulationStore";

type SliderControlProps = {
  label: string;
  title: string;
  min: string;
  max: string;
  step: string;
  value: number;
  displayValue: string;
  onChange: (value: number) => void;
};

function SliderControl({ label, title, min, max, step, value, displayValue, onChange }: SliderControlProps) {
  return (
    <label className="slider-row" title={title}>
      <span className="slider-label">{label}</span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(event) => onChange(Number(event.target.value))} />
      <span className="slider-value">{displayValue}</span>
    </label>
  );
}

type SettingsSectionProps = {
  title: string;
  toggle?: ReactNode;
  children?: ReactNode;
};

function SettingsSection({ title, toggle, children }: SettingsSectionProps) {
  return (
    <section className="settings-section">
      <div className="settings-section-header">
        <span className="settings-section-title">{title}</span>
        {toggle ? <div className="settings-section-toggle">{toggle}</div> : null}
      </div>
      <div className="settings-section-body">{children}</div>
    </section>
  );
}

export function InfoDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { t } = useTranslation();
  if (!open) {
    return null;
  }

  return (
    <div className="dialog-backdrop" onClick={onClose} role="presentation">
      <div
        className="dialog-panel"
        role="dialog"
        aria-modal="true"
        aria-label={t("panel.info.dialogLabel")}
        onClick={(event) => event.stopPropagation()}
      >
        <div className="dialog-content">
          <div className="info-card">
            <p className="info-line">{t("panel.info.author")}</p>
            <p className="info-line info-contact">{t("panel.info.contact")}</p>
            <div className="info-links" aria-label={t("panel.info.socialLinks")}>
              <a
                className="info-link"
                href="https://github.com/nikukikai"
                target="_blank"
                rel="noopener noreferrer"
                aria-label={t("panel.info.github")}
                title={t("panel.info.github")}
              >
                <FaGithub />
              </a>
              <a
                className="info-link"
                href="https://x.com/nikukikai"
                target="_blank"
                rel="noopener noreferrer"
                aria-label={t("panel.info.x")}
                title={t("panel.info.x")}
              >
                <FaXTwitter />
              </a>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function stripFileExtension(filename: string) {
  const index = filename.lastIndexOf(".");
  return index > 0 ? filename.slice(0, index) : filename;
}

function drawPreprocessExport(
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
      const r = Math.max(0, Math.min(255, ((preview[sampleIndex] * 0.229 + 0.485) * 255)));
      const g = Math.max(0, Math.min(255, ((preview[planeSize + sampleIndex] * 0.224 + 0.456) * 255)));
      const b = Math.max(0, Math.min(255, ((preview[planeSize * 2 + sampleIndex] * 0.225 + 0.406) * 255)));
      const pixelIndex = (y * roiWidth + x) * 4;
      imageData.data[pixelIndex] = r;
      imageData.data[pixelIndex + 1] = g;
      imageData.data[pixelIndex + 2] = b;
      imageData.data[pixelIndex + 3] = 255;
    }
  }

  context.putImageData(imageData, roiX, roiY);
}

function drawHeatmapExport(
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

function smoothstep(edge0: number, edge1: number, value: number) {
  const t = Math.max(0, Math.min(1, (value - edge0) / Math.max(edge1 - edge0, 1e-6)));
  return t * t * (3 - 2 * t);
}

function drawExportOverlays(
  context: CanvasRenderingContext2D,
  params: {
    imageWidth: number;
    imageHeight: number;
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

function drawDashedArrow(
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

function drawSolidArrow(
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

export function FloatingPanel() {
  const { t, i18n } = useTranslation();
  const { refs, exportBuffers, overlay, handleNextStep } = useSimulationEngineContext();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const languageMenuRef = useRef<HTMLDivElement>(null);
  const [panelExpanded, setPanelExpanded] = useState(true);
  const [infoOpen, setInfoOpen] = useState(false);
  const [languageMenuOpen, setLanguageMenuOpen] = useState(false);
  const {
    image,
    currentRoi,
    currentFixation,
    pendingNextFixation,
    candidates,
    panelBoxes,
    panelGuidedAnalysis,
    trajectory,
    mode,
    strategy,
    display,
    settings,
    loadImageFile,
    setMode,
    setStrategy,
    updateSetting,
    toggleDisplay,
    clearSimulation,
  } =
    useSimulationStore(
      useShallow((state) => ({
        image: state.image,
        currentRoi: state.currentRoi,
        currentFixation: state.currentFixation,
        pendingNextFixation: state.pendingNextFixation,
        candidates: state.candidates,
        panelBoxes: state.panelBoxes,
        panelGuidedAnalysis: state.panelGuidedAnalysis,
        trajectory: state.trajectory,
        mode: state.mode,
        strategy: state.strategy,
        display: state.display,
        settings: state.settings,
        loadImageFile: state.loadImageFile,
        setMode: state.setMode,
        setStrategy: state.setStrategy,
        updateSetting: state.updateSetting,
        toggleDisplay: state.toggleDisplay,
        clearSimulation: state.clearSimulation,
      })),
    );
  const nextDisabled = !pendingNextFixation;
  const openImageDialog = () => fileInputRef.current?.click();
  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    void loadImageFile(event.target.files?.[0] ?? null);
    event.target.value = "";
  };
  const imageHeight = image?.height ?? 0;
  const defaultRoiHalfSizePx = imageHeight > 0 ? imageHeight * settings.defaultRoiHalfSizeRatio : 0;
  const defaultRoiSizePx = defaultRoiHalfSizePx * 2;
  const clearRadiusPx = imageHeight > 0 ? imageHeight * settings.clearRadiusRatio : 0;
  const historySigmaPx = imageHeight > 0 ? imageHeight * settings.historySigmaRatio : 0;
  const distanceSigmaPx = imageHeight > 0 ? imageHeight * settings.distanceSigmaRatio : 0;
  const referenceRoiSizePx = currentRoi?.size ?? defaultRoiSizePx;
  const nmsRadiusModelPx =
    imageHeight > 0 && referenceRoiSizePx > 0
      ? Math.max(1, Math.round((imageHeight * settings.nmsRadiusRatio * 512) / referenceRoiSizePx))
      : 0;
  const formatPixels = (value: number) => (value > 0 ? `${value.toFixed(1)} px` : t("panel.unknown"));
  const imageHeightLabel = imageHeight > 0 ? `${imageHeight} px` : t("panel.unknown");
  const roiSizeLabel = referenceRoiSizePx > 0 ? `${referenceRoiSizePx.toFixed(1)} px` : t("panel.unknown");
  const roiSourceLabel = currentRoi ? t("panel.roiSource.current") : t("panel.roiSource.default");
  const handleLanguageChange = (language: AppLanguage) => {
    setLanguageMenuOpen(false);
    void i18n.changeLanguage(language);
  };
  const strategyOptions: Array<{ id: SimulationStrategy; label: string }> = [
    { id: "saliency_only", label: t("panel.buttons.saliencyOnlyStrategy") },
    { id: "panel_guided", label: t("panel.buttons.panelGuidedStrategy") },
  ];

  useEffect(() => {
    if (!languageMenuOpen) {
      return;
    }

    const handlePointerDown = (event: PointerEvent) => {
      if (languageMenuRef.current && !languageMenuRef.current.contains(event.target as Node)) {
        setLanguageMenuOpen(false);
      }
    };

    window.addEventListener("pointerdown", handlePointerDown);
    return () => window.removeEventListener("pointerdown", handlePointerDown);
  }, [languageMenuOpen]);

  const handleSaveVisualization = async () => {
    if (!image || !overlay.imageRect) {
      return;
    }

    const stageElement = document.querySelector(".canvas-stack") as HTMLElement | null;
    if (!stageElement) {
      return;
    }

    const stageCapture = await html2canvas(stageElement, {
      backgroundColor: null,
      logging: false,
      useCORS: true,
      scale: window.devicePixelRatio || 1,
    });

    const scaleX = stageCapture.width / Math.max(overlay.viewportWidth, 1);
    const scaleY = stageCapture.height / Math.max(overlay.viewportHeight, 1);
    const cropX = Math.round(overlay.imageRect.x * scaleX);
    const cropY = Math.round(overlay.imageRect.y * scaleY);
    const cropWidth = Math.max(1, Math.round(overlay.imageRect.width * scaleX));
    const cropHeight = Math.max(1, Math.round(overlay.imageRect.height * scaleY));

    const exportCanvas = document.createElement("canvas");
    exportCanvas.width = cropWidth;
    exportCanvas.height = cropHeight;
    const context = exportCanvas.getContext("2d");
    if (!context) {
      return;
    }

    context.drawImage(stageCapture, cropX, cropY, cropWidth, cropHeight, 0, 0, cropWidth, cropHeight);

    const drawViewportCanvas = (sourceCanvas: HTMLCanvasElement | null, enabled: boolean) => {
      if (!sourceCanvas || !enabled) {
        return;
      }
      const sourceScaleX = sourceCanvas.width / Math.max(overlay.viewportWidth, 1);
      const sourceScaleY = sourceCanvas.height / Math.max(overlay.viewportHeight, 1);
      const sx = Math.round(overlay.imageRect!.x * sourceScaleX);
      const sy = Math.round(overlay.imageRect!.y * sourceScaleY);
      const sw = Math.max(1, Math.round(overlay.imageRect!.width * sourceScaleX));
      const sh = Math.max(1, Math.round(overlay.imageRect!.height * sourceScaleY));
      context.drawImage(sourceCanvas, sx, sy, sw, sh, 0, 0, cropWidth, cropHeight);
    };

    if (display.showPreprocess && exportBuffers.preprocessPreview && currentRoi && image.width > 0 && image.height > 0) {
      drawPreprocessExport(context, exportBuffers.preprocessPreview, exportBuffers.modelSize, currentRoi, image.width, image.height, cropWidth, cropHeight);
    }
    if (display.showHeatmap && exportBuffers.heatmap && currentRoi && image.width > 0 && image.height > 0) {
      drawHeatmapExport(context, exportBuffers.heatmap, exportBuffers.modelSize, currentRoi, image.width, image.height, cropWidth, cropHeight);
    }
    drawViewportCanvas(refs.historyCanvasRef.current, display.showHistoryHeatmap);

    drawExportOverlays(context, {
      imageWidth: cropWidth,
      imageHeight: cropHeight,
      currentRoi: currentRoi
        ? {
            x: (currentRoi.x / image.width) * cropWidth,
            y: (currentRoi.y / image.height) * cropHeight,
            size: (currentRoi.size / image.width) * cropWidth,
          }
        : null,
      currentFixation: currentFixation
        ? {
            x: (currentFixation.x / image.width) * cropWidth,
            y: (currentFixation.y / image.height) * cropHeight,
          }
        : null,
      pendingNextFixation: pendingNextFixation
        ? {
            x: (pendingNextFixation.x / image.width) * cropWidth,
            y: (pendingNextFixation.y / image.height) * cropHeight,
          }
        : null,
      trajectory: trajectory.map((point) => ({
        x: (point.x / image.width) * cropWidth,
        y: (point.y / image.height) * cropHeight,
      })),
      candidates: candidates.map((candidate) => ({
        ...candidate,
        pageX: (candidate.pageX / image.width) * cropWidth,
        pageY: (candidate.pageY / image.height) * cropHeight,
      })),
      panelBoxes: panelBoxes.map((panel) => ({
        ...panel,
        rect: {
          x: (panel.rect.x / image.width) * cropWidth,
          y: (panel.rect.y / image.height) * cropHeight,
          width: (panel.rect.width / image.width) * cropWidth,
          height: (panel.rect.height / image.height) * cropHeight,
        },
      })),
      panelGuidedAnalysis: panelGuidedAnalysis
        ? {
            ...panelGuidedAnalysis,
            stepScores: panelGuidedAnalysis.stepScores.map((step) => ({
              ...step,
              candidate: step.candidate
                ? {
                    ...step.candidate,
                    pageX: (step.candidate.pageX / image.width) * cropWidth,
                    pageY: (step.candidate.pageY / image.height) * cropHeight,
                  }
                : null,
            })),
          }
        : null,
      strategy,
      display,
    });

    const blob = await new Promise<Blob | null>((resolve) => exportCanvas.toBlob(resolve, "image/png"));
    if (!blob) {
      return;
    }

    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${stripFileExtension(image.url.split("/").pop() ?? "visualization")}-visualized.png`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <>
      <input
        ref={fileInputRef}
        className="visually-hidden"
        type="file"
        accept="image/*"
        onChange={handleFileChange}
      />
      <div className="panel panel-top-right panel-control">
        <div className="panel-toolbar">
          <IconButton label={t("panel.buttons.openImage")} onClick={openImageDialog}>
            <IoFolderOpenOutline />
          </IconButton>
          <div className="mode-group" role="group" aria-label={`${t("panel.buttons.clickMode")} / ${t("panel.buttons.boxMode")}`}>
            <IconButton active={mode === "click"} label={t("panel.buttons.clickMode")} onClick={() => setMode("click")}>
              <LuMousePointer2 />
            </IconButton>
            <IconButton active={mode === "box"} label={t("panel.buttons.boxMode")} onClick={() => setMode("box")}>
              <PiRectangle />
            </IconButton>
          </div>
          <IconButton label={t("panel.buttons.nextStep")} disabled={nextDisabled} onClick={() => void handleNextStep()}>
            <IoPlay />
          </IconButton>
          <IconButton label={t("panel.buttons.reset")} onClick={clearSimulation}>
            <IoRefresh />
          </IconButton>
          <div className="toolbar-spacer" />
          <IconButton
            active={panelExpanded}
            label={panelExpanded ? t("panel.buttons.collapse") : t("panel.buttons.expand")}
            onClick={() => setPanelExpanded((value) => !value)}
          >
            <HiBars3 />
          </IconButton>
        </div>
        {panelExpanded ? (
          <div className="settings-sections">
            <div className="panel-actions-row">
              <IconButton label={t("panel.buttons.information")} onClick={() => setInfoOpen(true)}>
                <TbInfoCircle />
              </IconButton>
              <div className="menu-anchor" ref={languageMenuRef}>
                <IconButton
                  active={languageMenuOpen}
                  label={t("panel.buttons.languageMenu")}
                  onClick={() => setLanguageMenuOpen((value) => !value)}
                >
                  <IoLanguageOutline />
                </IconButton>
                {languageMenuOpen ? (
                  <div className="panel-menu">
                    <button type="button" className="panel-menu-item" onClick={() => handleLanguageChange("en")}>
                      {t("panel.language.en")}
                    </button>
                    <button type="button" className="panel-menu-item" onClick={() => handleLanguageChange("ja")}>
                      {t("panel.language.ja")}
                    </button>
                  </div>
                ) : null}
              </div>
              <IconButton label={t("panel.buttons.save")} onClick={() => void handleSaveVisualization()}>
                <IoSaveOutline />
              </IconButton>
            </div>
            <SettingsSection
              title={t("panel.sections.panels")}
              toggle={
                <IconButton active={display.showPanelBoxes} label={t("panel.buttons.togglePanels")} onClick={() => toggleDisplay("showPanelBoxes")}>
                  <IoEyeOutline />
                </IconButton>
              }
            />

            <SettingsSection
              title={t("panel.sections.preprocess")}
              toggle={
                <IconButton active={display.showPreprocess} label={t("panel.buttons.togglePreprocess")} onClick={() => toggleDisplay("showPreprocess")}>
                  <IoEyeOutline />
                </IconButton>
              }
            >
              <SliderControl
                label={t("panel.sliders.blur.label")}
                title={t("panel.sliders.blur.tooltip", { value: settings.maxBlurStrength.toFixed(2) })}
                min="0"
                max="32"
                step="0.25"
                value={settings.maxBlurStrength}
                displayValue={settings.maxBlurStrength.toFixed(2)}
                onChange={(value) => updateSetting("maxBlurStrength", value)}
              />
              <SliderControl
                label={t("panel.sliders.fovea.label")}
                title={t("panel.sliders.fovea.tooltip", {
                  imageHeight: imageHeightLabel,
                  ratio: settings.clearRadiusRatio.toFixed(3),
                  pixels: formatPixels(clearRadiusPx),
                })}
                min="0.01"
                max="0.2"
                step="0.005"
                value={settings.clearRadiusRatio}
                displayValue={settings.clearRadiusRatio.toFixed(3)}
                onChange={(value) => updateSetting("clearRadiusRatio", value)}
              />
            </SettingsSection>

            <SettingsSection
              title={t("panel.sections.saliency")}
              toggle={
                <IconButton active={display.showHeatmap} label={t("panel.buttons.toggleSaliency")} onClick={() => toggleDisplay("showHeatmap")}>
                  <IoEyeOutline />
                </IconButton>
              }
            >
              <SliderControl
                label={t("panel.sliders.box.label")}
                title={t("panel.sliders.box.tooltip", {
                  imageHeight: imageHeightLabel,
                  ratio: settings.defaultRoiHalfSizeRatio.toFixed(3),
                  halfSize: formatPixels(defaultRoiHalfSizePx),
                  fullSize: formatPixels(defaultRoiSizePx),
                })}
                min="0.05"
                max="0.45"
                step="0.005"
                value={settings.defaultRoiHalfSizeRatio}
                displayValue={settings.defaultRoiHalfSizeRatio.toFixed(3)}
                onChange={(value) => updateSetting("defaultRoiHalfSizeRatio", value)}
              />
            </SettingsSection>

            <SettingsSection
              title={t("panel.sections.history")}
              toggle={
                <IconButton active={display.showHistoryHeatmap} label={t("panel.buttons.toggleHistory")} onClick={() => toggleDisplay("showHistoryHeatmap")}>
                  <IoEyeOutline />
                </IconButton>
              }
            >
              <SliderControl
                label={t("panel.sliders.sigma.label")}
                title={t("panel.sliders.sigma.tooltip", {
                  imageHeight: imageHeightLabel,
                  ratio: settings.historySigmaRatio.toFixed(3),
                  pixels: formatPixels(historySigmaPx),
                })}
                min="0.01"
                max="0.12"
                step="0.001"
                value={settings.historySigmaRatio}
                displayValue={settings.historySigmaRatio.toFixed(3)}
                onChange={(value) => updateSetting("historySigmaRatio", value)}
              />
              <SliderControl
                label={t("panel.sliders.hist.label")}
                title={t("panel.sliders.hist.tooltip", { value: settings.historyAlpha.toFixed(1) })}
                min="0"
                max="8"
                step="0.1"
                value={settings.historyAlpha}
                displayValue={settings.historyAlpha.toFixed(1)}
                onChange={(value) => updateSetting("historyAlpha", value)}
              />
              <SliderControl
                label={t("panel.sliders.decay.label")}
                title={t("panel.sliders.decay.tooltip", { value: settings.historyDecay.toFixed(3) })}
                min="0.7"
                max="0.995"
                step="0.005"
                value={settings.historyDecay}
                displayValue={settings.historyDecay.toFixed(3)}
                onChange={(value) => updateSetting("historyDecay", value)}
              />
            </SettingsSection>

            <SettingsSection
              title={t("panel.sections.selector")}
              toggle={
                <IconButton active={display.showSelectorOverlay} label={t("panel.buttons.toggleSelector")} onClick={() => toggleDisplay("showSelectorOverlay")}>
                  <IoEyeOutline />
                </IconButton>
              }
            >
              <div className="text-mode-group" role="group" aria-label={t("panel.sections.selector")}>
                {strategyOptions.map((option) => (
                  <button
                    key={option.id}
                    type="button"
                    className={`text-mode-button${strategy === option.id ? " is-active" : ""}`}
                    onClick={() => setStrategy(option.id)}
                  >
                    {option.label}
                  </button>
                ))}
              </div>
              <SliderControl
                label={t("panel.sliders.dist.label")}
                title={t("panel.sliders.dist.tooltip", {
                  imageHeight: imageHeightLabel,
                  ratio: settings.distanceSigmaRatio.toFixed(3),
                  pixels: formatPixels(distanceSigmaPx),
                })}
                min="0.05"
                max="0.4"
                step="0.005"
                value={settings.distanceSigmaRatio}
                displayValue={settings.distanceSigmaRatio.toFixed(3)}
                onChange={(value) => updateSetting("distanceSigmaRatio", value)}
              />
              <SliderControl
                label={t("panel.sliders.thresh.label")}
                title={t("panel.sliders.thresh.tooltip", { value: settings.thresholdRatio.toFixed(2) })}
                min="0.05"
                max="0.95"
                step="0.01"
                value={settings.thresholdRatio}
                displayValue={settings.thresholdRatio.toFixed(2)}
                onChange={(value) => updateSetting("thresholdRatio", value)}
              />
              <SliderControl
                label={t("panel.sliders.nms.label")}
                title={t("panel.sliders.nms.tooltip", {
                  imageHeight: imageHeightLabel,
                  ratio: settings.nmsRadiusRatio.toFixed(3),
                  roiSource: roiSourceLabel,
                  roiSize: roiSizeLabel,
                  cells: nmsRadiusModelPx,
                })}
                min="0.001"
                max="0.05"
                step="0.001"
                value={settings.nmsRadiusRatio}
                displayValue={settings.nmsRadiusRatio.toFixed(3)}
                onChange={(value) => updateSetting("nmsRadiusRatio", value)}
              />
              <SliderControl
                label={t("panel.sliders.topK.label")}
                title={t("panel.sliders.topK.tooltip", { value: settings.topK.toFixed(0) })}
                min="1"
                max="32"
                step="1"
                value={settings.topK}
                displayValue={settings.topK.toFixed(0)}
                onChange={(value) => updateSetting("topK", value)}
              />
            </SettingsSection>
          </div>
        ) : null}
      </div>
      <InfoDialog open={infoOpen} onClose={() => setInfoOpen(false)} />
    </>
  );
}
