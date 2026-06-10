import { useEffect, useMemo, useRef, useState } from "react";
import { FloatingPanel } from "../components/FloatingPanel";
import { LoadingOverlay } from "../components/LoadingOverlay";
import { HeatmapRenderer } from "../core/gpu/heatmapRenderer";
import { modelSize, RoiPreprocessor } from "../core/gpu/preprocess";
import { SaliencySession } from "../core/onnx/saliencySession";
import { extractCandidates, normalizeHeatmap } from "../core/simulation/candidates";
import { addFixationToHistory, createHistoryMap } from "../core/simulation/history";
import {
  createCenteredSquareRoi,
  createSquareFromDrag,
  fitImageToViewport,
  roiCenter,
  roiToScreenRect,
  screenToPagePoint,
  type ImageRect,
} from "../core/simulation/roi";
import { drawBaseImage, drawOverlay, drawPreprocessPreview, resizeAndClear2dCanvas } from "../core/render/draw";
import { useSimulationStore } from "../store/simulationStore";
import type { Point, PreprocessPreview, RoiRect, StepResult } from "../types/simulation";

type Engine = {
  device: GPUDevice;
  preprocessor: RoiPreprocessor;
  heatmapRenderer: HeatmapRenderer;
  session: SaliencySession;
};

function useViewportSize() {
  const [size, setSize] = useState({ width: window.innerWidth, height: window.innerHeight });

  useEffect(() => {
    const onResize = () => setSize({ width: window.innerWidth, height: window.innerHeight });
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  return size;
}

async function createEngine(heatmapCanvas: HTMLCanvasElement): Promise<Engine> {
  if (!navigator.gpu) {
    throw new Error("WebGPU is unavailable in this browser.");
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("WebGPU adapter request failed.");
  }
  const device = await adapter.requestDevice();
  const preprocessor = new RoiPreprocessor(device);
  const heatmapRenderer = new HeatmapRenderer(device, heatmapCanvas);
  const session = await SaliencySession.create();
  return { device, preprocessor, heatmapRenderer, session };
}

export function App() {
  const baseCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const preprocessCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const heatmapCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const engineRef = useRef<Engine | null>(null);
  const initStartedRef = useRef(false);

  const {
    loadingPhase,
    loadingLabel,
    errorMessage,
    image,
    mode,
    display,
    settings,
    historyMap,
    historyMapWidth,
    historyMapHeight,
    currentFixation,
    currentRoi,
    activeRoiHalfSizePx,
    pendingNextFixation,
    candidates,
    trajectory,
    currentPreprocess,
    dragStart,
    dragCurrent,
    isDragging,
    setLoadingState,
    setError,
    setWebgpuAvailable,
    setImage,
    setMode,
    updateSetting,
    toggleDisplay,
    setDragState,
    applyStepResult,
    updateHistoryMap,
    clearSimulation,
  } = useSimulationStore();

  const viewport = useViewportSize();

  const imageRect = useMemo<ImageRect | null>(() => {
    if (!image) {
      return null;
    }
    return fitImageToViewport(image.width, image.height, viewport.width, viewport.height);
  }, [image, viewport.height, viewport.width]);

  const dragRoi = useMemo<RoiRect | null>(() => {
    if (mode !== "box" || !dragStart || !dragCurrent || !isDragging) {
      return null;
    }
    return createSquareFromDrag(dragStart, dragCurrent);
  }, [dragCurrent, dragStart, isDragging, mode]);

  useEffect(() => {
    const initialize = async () => {
      const heatmapCanvas = heatmapCanvasRef.current;
      if (!heatmapCanvas || initStartedRef.current) {
        return;
      }
      // React StrictMode replays effects in development, but the WebGPU canvas context must stay single-owned.
      initStartedRef.current = true;
      try {
        setLoadingState("webgpu", "Initializing WebGPU");
        const engine = await createEngine(heatmapCanvas);
        engineRef.current = engine;
        setWebgpuAvailable(true);
        setLoadingState("model", "Loading Model");
        setLoadingState("ready", "Ready");
      } catch (error) {
        const message = error instanceof Error ? error.message : "Failed to initialize the demo.";
        setError(message);
      }
    };
    void initialize();
  }, [setError, setLoadingState, setWebgpuAvailable]);

  useEffect(() => {
    if (image && engineRef.current) {
      engineRef.current.preprocessor.setSourceImage(image.bitmap);
    }
  }, [image]);

  const loadImageFile = async (file: File | null) => {
    if (!file) {
      return;
    }
    const url = URL.createObjectURL(file);
    const bitmap = await createImageBitmap(file);
    engineRef.current?.preprocessor.setSourceImage(bitmap);
    setImage({
      bitmap,
      width: bitmap.width,
      height: bitmap.height,
      url,
    });
    clearSimulation();
  };

  useEffect(() => {
    const baseCanvas = baseCanvasRef.current;
    const preprocessCanvas = preprocessCanvasRef.current;
    const overlayCanvas = overlayCanvasRef.current;
    const heatmapCanvas = heatmapCanvasRef.current;
    const engine = engineRef.current;
    if (!baseCanvas || !preprocessCanvas || !overlayCanvas || !heatmapCanvas) {
      return;
    }

    const baseContext = resizeAndClear2dCanvas(baseCanvas, viewport.width, viewport.height);
    const preprocessContext = resizeAndClear2dCanvas(preprocessCanvas, viewport.width, viewport.height);
    const overlayContext = resizeAndClear2dCanvas(overlayCanvas, viewport.width, viewport.height);
    const ratio = window.devicePixelRatio || 1;
    heatmapCanvas.width = Math.max(1, Math.floor(viewport.width * ratio));
    heatmapCanvas.height = Math.max(1, Math.floor(viewport.height * ratio));
    heatmapCanvas.style.width = `${viewport.width}px`;
    heatmapCanvas.style.height = `${viewport.height}px`;
    engine?.heatmapRenderer.resize(heatmapCanvas.width, heatmapCanvas.height);

    if (image && imageRect) {
      drawBaseImage(baseContext, image.bitmap, imageRect);
    }

    drawPreprocessPreview({
      context: preprocessContext,
      preview: currentPreprocess?.rgba ?? null,
      previewSize: currentPreprocess?.size ?? 0,
      imageRect,
      imageWidth: image?.width ?? 0,
      imageHeight: image?.height ?? 0,
      roi: currentRoi,
      enabled: display.showPreprocess,
    });

    drawOverlay({
      context: overlayContext,
      viewportWidth: viewport.width,
      viewportHeight: viewport.height,
      imageRect,
      imageWidth: image?.width ?? 0,
      imageHeight: image?.height ?? 0,
      currentRoi,
      dragRoi,
      currentFixation,
      pendingNextFixation,
      trajectory,
      candidates,
      showHistory: display.showHistory,
    });

    if (engine) {
      const physicalRatio = window.devicePixelRatio || 1;
      engine.heatmapRenderer.render({
        imageRect:
          imageRect
            ? {
                x: imageRect.x * physicalRatio,
                y: imageRect.y * physicalRatio,
                width: imageRect.width * physicalRatio,
                height: imageRect.height * physicalRatio,
              }
            : ({
                x: 0,
                y: 0,
                width: 0,
                height: 0,
              } satisfies ImageRect),
        roiRect:
          imageRect && currentRoi
            ? (() => {
                const roiRect = roiToScreenRect(currentRoi, imageRect, image.width, image.height);
                return {
                  x: roiRect.x * physicalRatio,
                  y: roiRect.y * physicalRatio,
                  width: roiRect.width * physicalRatio,
                  height: roiRect.height * physicalRatio,
                };
              })()
            : null,
        enabled: Boolean(display.showHeatmap && imageRect && currentRoi),
      });
    }
  }, [
    viewport.width,
    viewport.height,
    image,
    imageRect,
    currentRoi,
    dragRoi,
    currentFixation,
    pendingNextFixation,
    trajectory,
    candidates,
    currentPreprocess,
    display.showHistory,
    display.showPreprocess,
    display.showHeatmap,
  ]);

  const buildPreprocessPreview = (input: Float32Array): PreprocessPreview => {
    const size = modelSize();
    const planeSize = size * size;
    const rgba = new Uint8ClampedArray(planeSize * 4);

    for (let index = 0; index < planeSize; index += 1) {
      const r = input[index] * 0.229 + 0.485;
      const g = input[planeSize + index] * 0.224 + 0.456;
      const b = input[planeSize * 2 + index] * 0.225 + 0.406;
      const base = index * 4;
      rgba[base] = Math.max(0, Math.min(255, Math.round(r * 255)));
      rgba[base + 1] = Math.max(0, Math.min(255, Math.round(g * 255)));
      rgba[base + 2] = Math.max(0, Math.min(255, Math.round(b * 255)));
      rgba[base + 3] = 255;
    }

    return { rgba, size };
  };

  const runStep = async (
    roi: RoiRect,
    fixation: Point,
    committedTrajectory: Point[],
    activeHistoryMap: Float32Array,
  ) => {
    const engine = engineRef.current;
    if (!engine || !image) {
      return;
    }

    const input = await engine.preprocessor.run(roi, fixation, image.height, settings);
    const preprocess = buildPreprocessPreview(input);
    const rawHeatmap = await engine.session.predict(input);
    const heatmap = normalizeHeatmap(rawHeatmap);
    const nmsRadius = Math.max(1, Math.round((image.height * settings.nmsRadiusRatio * modelSize()) / roi.size));
    const distanceSigma = Math.max(1, image.height * settings.distanceSigmaRatio);
    const scoredCandidates = extractCandidates({
      heatmap,
      mapWidth: modelSize(),
      mapHeight: modelSize(),
      thresholdRatio: settings.thresholdRatio,
      nmsRadius,
      topK: settings.topK,
      roi,
      imageWidth: image.width,
      imageHeight: image.height,
      currentFixation: fixation,
      historyMap: activeHistoryMap,
      historyMapWidth,
      historyMapHeight,
      historyAlpha: settings.historyAlpha,
      distanceSigma,
    });

    engine.heatmapRenderer.updateHeatmap(heatmap, modelSize());
    const stepResult: StepResult = {
      roi,
      fixation,
      heatmap,
      preprocess,
      candidates: scoredCandidates,
      pendingNextFixation:
        scoredCandidates.length > 0
          ? {
              x: scoredCandidates[0].pageX,
              y: scoredCandidates[0].pageY,
            }
          : null,
    };
    applyStepResult(stepResult, committedTrajectory);
  };

  const startClickStep = async (point: Point) => {
    if (!image) {
      return;
    }
    const halfSize = image.height * settings.clickRoiHalfSizeRatio;
    const roi = createCenteredSquareRoi(point, halfSize);
    const trajectorySeed = [point];
    const nextHistory = historyMap ? historyMap.slice() : createHistoryMap(image.width, image.height);
    addFixationToHistory(nextHistory, image.width, image.height, point, image.height * settings.historySigmaRatio);
    updateHistoryMap(nextHistory, image.width, image.height);
    await runStep(roi, point, trajectorySeed, nextHistory);
  };

  const startBoxStep = async (roi: RoiRect) => {
    const fixation = roiCenter(roi);
    if (!image) {
      return;
    }
    const trajectorySeed = [fixation];
    const nextHistory = historyMap ? historyMap.slice() : createHistoryMap(image.width, image.height);
    addFixationToHistory(nextHistory, image.width, image.height, fixation, image.height * settings.historySigmaRatio);
    updateHistoryMap(nextHistory, image.width, image.height);
    await runStep(roi, fixation, trajectorySeed, nextHistory);
  };

  const handlePointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!imageRect || !image) {
      return;
    }
    const point = screenToPagePoint(event.clientX, event.clientY, imageRect, image.width, image.height);
    if (!point) {
      return;
    }
    if (mode === "box") {
      setDragState(point, point, true);
    }
  };

  const handlePointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!isDragging || !dragStart || !imageRect || !image || mode !== "box") {
      return;
    }
    const point = screenToPagePoint(event.clientX, event.clientY, imageRect, image.width, image.height);
    if (!point) {
      return;
    }
    setDragState(dragStart, point, true);
  };

  const handlePointerUp = async (event: React.PointerEvent<HTMLDivElement>) => {
    if (!imageRect || !image) {
      return;
    }
    const point = screenToPagePoint(event.clientX, event.clientY, imageRect, image.width, image.height);
    if (!point) {
      setDragState(null, null, false);
      return;
    }

    if (mode === "click") {
      await startClickStep(point);
      return;
    }

    if (mode === "box" && dragStart) {
      const roi = createSquareFromDrag(dragStart, point);
      setDragState(null, null, false);
      if (roi.size >= 2) {
        await startBoxStep(roi);
      }
    }
  };

  const handleNextStep = async () => {
    if (!image || !pendingNextFixation) {
      return;
    }
    const halfSize = activeRoiHalfSizePx ?? image.height * settings.clickRoiHalfSizeRatio;
    const roi = createCenteredSquareRoi(pendingNextFixation, halfSize);
    const nextTrajectory = [...trajectory, pendingNextFixation];
    const nextHistory = historyMap ? historyMap.slice() : createHistoryMap(image.width, image.height);
    addFixationToHistory(
      nextHistory,
      image.width,
      image.height,
      pendingNextFixation,
      image.height * settings.historySigmaRatio,
    );
    updateHistoryMap(nextHistory, image.width, image.height);
    await runStep(roi, pendingNextFixation, nextTrajectory, nextHistory);
  };

  const openImageDialog = () => fileInputRef.current?.click();

  return (
    <div
      className="app-shell"
      onDragOver={(event) => {
        event.preventDefault();
        event.dataTransfer.dropEffect = "copy";
      }}
      onDrop={(event) => {
        event.preventDefault();
        void loadImageFile(event.dataTransfer.files?.[0] ?? null);
      }}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={(event) => void handlePointerUp(event)}
    >
      <input
        ref={fileInputRef}
        className="visually-hidden"
        type="file"
        accept="image/*"
        onChange={(event) => void loadImageFile(event.target.files?.[0] ?? null)}
      />

      <div className="canvas-stack">
        <canvas ref={baseCanvasRef} className="stage-canvas" />
        <canvas ref={preprocessCanvasRef} className="stage-canvas preprocess-layer" />
        <canvas ref={heatmapCanvasRef} className="stage-canvas heatmap-layer" />
        <canvas ref={overlayCanvasRef} className="stage-canvas overlay-layer" />
      </div>

      <FloatingPanel
        mode={mode}
        display={display}
        settings={settings}
        nextDisabled={!pendingNextFixation}
        onModeChange={setMode}
        onSettingChange={updateSetting}
        onToggleDisplay={toggleDisplay}
        onNextStep={() => void handleNextStep()}
        onReset={clearSimulation}
        onOpenImage={openImageDialog}
      />

      <LoadingOverlay visible={loadingPhase !== "ready"} label={loadingLabel} errorMessage={errorMessage} />
    </div>
  );
}
