import { createContext, useContext, useEffect, useMemo, useRef, useState, type PropsWithChildren, type RefObject } from "react";
import { useShallow } from "zustand/react/shallow";
import { HeatmapRenderer } from "../core/gpu/heatmapRenderer";
import { modelSize, RoiPreprocessor } from "../core/gpu/preprocess";
import { SaliencySession } from "../core/onnx/saliencySession";
import { drawBaseImage, drawHistoryHeatmap, drawPreprocessPreview, resizeAndClear2dCanvas } from "../core/render/draw";
import { extractCandidates, normalizeHeatmap } from "../core/simulation/candidates";
import { addFixationToHistory, createHistoryMap } from "../core/simulation/history";
import {
  createCenteredSquareRoi,
  createSquareFromDrag,
  fitImageToViewport,
  roiCenter,
  roiToScreenRect,
  type ImageRect,
} from "../core/simulation/roi";
import { useSimulationStore } from "../store/simulationStore";
import type { ImageResource, Point, PreprocessPreview, RoiRect, StepResult } from "../types/simulation";
import { useViewportSize } from "./useViewportSize";

type Engine = {
  device: GPUDevice;
  preprocessor: RoiPreprocessor;
  heatmapRenderer: HeatmapRenderer;
  session: SaliencySession;
};

type SimulationEngineContextValue = {
  refs: {
    baseCanvasRef: RefObject<HTMLCanvasElement>;
    historyCanvasRef: RefObject<HTMLCanvasElement>;
    preprocessCanvasRef: RefObject<HTMLCanvasElement>;
    heatmapCanvasRef: RefObject<HTMLCanvasElement>;
  };
  overlay: {
    viewportWidth: number;
    viewportHeight: number;
    imageRect: ImageRect | null;
    imageWidth: number;
    imageHeight: number;
    dragRoi: RoiRect | null;
  };
  interactionState: {
    image: ImageResource | null;
    imageRect: ImageRect | null;
  };
  loadImageFile: (file: File | null) => Promise<void>;
  startClickStep: (point: Point) => Promise<void>;
  startBoxStep: (roi: RoiRect) => Promise<void>;
  handleNextStep: () => Promise<void>;
};

function buildPreprocessPreview(input: Float32Array): PreprocessPreview {
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

const SimulationEngineContext = createContext<SimulationEngineContextValue | null>(null);

export function SimulationProvider({ children }: PropsWithChildren) {
  const baseCanvasRef = useRef<HTMLCanvasElement>(null);
  const historyCanvasRef = useRef<HTMLCanvasElement>(null);
  const preprocessCanvasRef = useRef<HTMLCanvasElement>(null);
  const heatmapCanvasRef = useRef<HTMLCanvasElement>(null);
  const engineRef = useRef<Engine | null>(null);
  const initStartedRef = useRef(false);
  const [imageRect, setImageRect] = useState<ImageRect | null>(null);

  const viewport = useViewportSize();
  const {
    image,
    display,
    settings,
    historyMap,
    historyMapWidth,
    historyMapHeight,
    currentRoi,
    activeRoiHalfSizePx,
    pendingNextFixation,
    trajectory,
    currentPreprocess,
    dragStart,
    dragCurrent,
    isDragging,
    mode,
    setLoadingState,
    setError,
    setWebgpuAvailable,
    setImage,
    applyStepOutcome,
    clearSimulation,
  } = useSimulationStore(
    useShallow((state) => ({
      image: state.image,
      display: state.display,
      settings: state.settings,
      historyMap: state.historyMap,
      historyMapWidth: state.historyMapWidth,
      historyMapHeight: state.historyMapHeight,
      currentRoi: state.currentRoi,
      activeRoiHalfSizePx: state.activeRoiHalfSizePx,
      pendingNextFixation: state.pendingNextFixation,
      trajectory: state.trajectory,
      currentPreprocess: state.currentPreprocess,
      dragStart: state.dragStart,
      dragCurrent: state.dragCurrent,
      isDragging: state.isDragging,
      mode: state.mode,
      setLoadingState: state.setLoadingState,
      setError: state.setError,
      setWebgpuAvailable: state.setWebgpuAvailable,
      setImage: state.setImage,
      applyStepOutcome: state.applyStepOutcome,
      clearSimulation: state.clearSimulation,
    })),
  );

  const dragRoi = useMemo<RoiRect | null>(() => {
    if (mode !== "box" || !dragStart || !dragCurrent || !isDragging) {
      return null;
    }
    return createSquareFromDrag(dragStart, dragCurrent);
  }, [dragCurrent, dragStart, isDragging, mode]);

  useEffect(() => {
    if (!image) {
      setImageRect(null);
      return;
    }
    setImageRect(fitImageToViewport(image.width, image.height, viewport.width, viewport.height));
  }, [image, viewport.height, viewport.width]);

  useEffect(() => {
    const initialize = async () => {
      const heatmapCanvas = heatmapCanvasRef.current;
      if (!heatmapCanvas || initStartedRef.current) {
        return;
      }
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

  useEffect(() => {
    const baseCanvas = baseCanvasRef.current;
    const historyCanvas = historyCanvasRef.current;
    const preprocessCanvas = preprocessCanvasRef.current;
    const heatmapCanvas = heatmapCanvasRef.current;
    const engine = engineRef.current;
    if (!baseCanvas || !historyCanvas || !preprocessCanvas || !heatmapCanvas) {
      return;
    }

    const baseContext = resizeAndClear2dCanvas(baseCanvas, viewport.width, viewport.height);
    const historyContext = resizeAndClear2dCanvas(historyCanvas, viewport.width, viewport.height);
    const preprocessContext = resizeAndClear2dCanvas(preprocessCanvas, viewport.width, viewport.height);
    const ratio = window.devicePixelRatio || 1;
    heatmapCanvas.width = Math.max(1, Math.floor(viewport.width * ratio));
    heatmapCanvas.height = Math.max(1, Math.floor(viewport.height * ratio));
    heatmapCanvas.style.width = `${viewport.width}px`;
    heatmapCanvas.style.height = `${viewport.height}px`;
    engine?.heatmapRenderer.resize(heatmapCanvas.width, heatmapCanvas.height);

    if (image && imageRect) {
      drawBaseImage(baseContext, image.bitmap, imageRect);
    }

    drawHistoryHeatmap({
      context: historyContext,
      historyMap,
      historyMapWidth,
      historyMapHeight,
      imageRect,
      enabled: display.showHistoryHeatmap,
    });

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
          imageRect && currentRoi && image
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
    currentPreprocess,
    historyMap,
    historyMapWidth,
    historyMapHeight,
    display.showPreprocess,
    display.showHeatmap,
    display.showHistoryHeatmap,
  ]);

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

  const runStep = async (roi: RoiRect, fixation: Point, committedTrajectory: Point[]) => {
    const engine = engineRef.current;
    if (!engine || !image) {
      return;
    }

    const nextHistoryMap = historyMap ? historyMap.slice() : createHistoryMap(image.width, image.height);
    addFixationToHistory(nextHistoryMap, image.width, image.height, fixation, image.height * settings.historySigmaRatio);

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
      historyMap: nextHistoryMap,
      historyMapWidth: image.width,
      historyMapHeight: image.height,
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
    applyStepOutcome({
      result: stepResult,
      committedTrajectory,
      historyMap: nextHistoryMap,
      historyMapWidth: image.width,
      historyMapHeight: image.height,
    });
  };

  const startClickStep = async (point: Point) => {
    if (!image) {
      return;
    }
    const halfSize = image.height * settings.clickRoiHalfSizeRatio;
    const roi = createCenteredSquareRoi(point, halfSize);
    await runStep(roi, point, [point]);
  };

  const startBoxStep = async (roi: RoiRect) => {
    const fixation = roiCenter(roi);
    await runStep(roi, fixation, [fixation]);
  };

  const handleNextStep = async () => {
    if (!image || !pendingNextFixation) {
      return;
    }
    const halfSize = activeRoiHalfSizePx ?? image.height * settings.clickRoiHalfSizeRatio;
    const roi = createCenteredSquareRoi(pendingNextFixation, halfSize);
    await runStep(roi, pendingNextFixation, [...trajectory, pendingNextFixation]);
  };

  const value: SimulationEngineContextValue = {
    refs: {
      baseCanvasRef,
      historyCanvasRef,
      preprocessCanvasRef,
      heatmapCanvasRef,
    },
    overlay: {
      viewportWidth: viewport.width,
      viewportHeight: viewport.height,
      imageRect,
      imageWidth: image?.width ?? 0,
      imageHeight: image?.height ?? 0,
      dragRoi,
    },
    interactionState: {
      image,
      imageRect,
    },
    loadImageFile,
    startClickStep,
    startBoxStep,
    handleNextStep,
  };

  return <SimulationEngineContext.Provider value={value}>{children}</SimulationEngineContext.Provider>;
}

export function useSimulationEngineContext() {
  const value = useContext(SimulationEngineContext);
  if (!value) {
    throw new Error("useSimulationEngineContext must be used inside SimulationProvider.");
  }
  return value;
}
