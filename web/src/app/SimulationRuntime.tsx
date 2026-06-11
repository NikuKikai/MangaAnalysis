import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
  type PointerEvent,
  type PropsWithChildren,
  type RefObject,
} from "react";
import { useShallow } from "zustand/react/shallow";
import { HeatmapRenderer } from "../core/gpu/heatmapRenderer";
import { modelSize, RoiPreprocessor } from "../core/gpu/preprocess";
import { SaliencySession } from "../core/onnx/saliencySession";
import { drawBaseImage, drawPreprocessPreview, resizeAndClear2dCanvas } from "../core/render/draw";
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
import { useSimulationStore } from "../store/simulationStore";
import type { Point, PreprocessPreview, RoiRect, StepResult } from "../types/simulation";
import { useViewportSize } from "./useViewportSize";

type Engine = {
  device: GPUDevice;
  preprocessor: RoiPreprocessor;
  heatmapRenderer: HeatmapRenderer;
  session: SaliencySession;
};

type SimulationRuntimeValue = {
  baseCanvasRef: RefObject<HTMLCanvasElement>;
  preprocessCanvasRef: RefObject<HTMLCanvasElement>;
  heatmapCanvasRef: RefObject<HTMLCanvasElement>;
  fileInputRef: RefObject<HTMLInputElement>;
  overlay: {
    viewportWidth: number;
    viewportHeight: number;
    imageRect: ImageRect | null;
    imageWidth: number;
    imageHeight: number;
    dragRoi: RoiRect | null;
  };
  handlePointerDown: (event: PointerEvent<HTMLDivElement>) => void;
  handlePointerMove: (event: PointerEvent<HTMLDivElement>) => void;
  handlePointerUp: (event: PointerEvent<HTMLDivElement>) => void;
  handleDrop: (event: DragEvent<HTMLDivElement>) => void;
  openImageDialog: () => void;
  handleFileChange: (event: ChangeEvent<HTMLInputElement>) => void;
  handleNextStep: () => void;
};

const SimulationRuntimeContext = createContext<SimulationRuntimeValue | null>(null);

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

export function SimulationRuntimeProvider({ children }: PropsWithChildren) {
  const baseCanvasRef = useRef<HTMLCanvasElement>(null);
  const preprocessCanvasRef = useRef<HTMLCanvasElement>(null);
  const heatmapCanvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const engineRef = useRef<Engine | null>(null);
  const initStartedRef = useRef(false);
  const [imageRect, setImageRect] = useState<ImageRect | null>(null);

  const viewport = useViewportSize();
  const {
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
    setDragState,
    applyStepResult,
    updateHistoryMap,
    clearSimulation,
  } = useSimulationStore(
    useShallow((state) => ({
      image: state.image,
      mode: state.mode,
      display: state.display,
      settings: state.settings,
      historyMap: state.historyMap,
      historyMapWidth: state.historyMapWidth,
      historyMapHeight: state.historyMapHeight,
      currentFixation: state.currentFixation,
      currentRoi: state.currentRoi,
      activeRoiHalfSizePx: state.activeRoiHalfSizePx,
      pendingNextFixation: state.pendingNextFixation,
      candidates: state.candidates,
      trajectory: state.trajectory,
      currentPreprocess: state.currentPreprocess,
      dragStart: state.dragStart,
      dragCurrent: state.dragCurrent,
      isDragging: state.isDragging,
      setLoadingState: state.setLoadingState,
      setError: state.setError,
      setWebgpuAvailable: state.setWebgpuAvailable,
      setImage: state.setImage,
      setDragState: state.setDragState,
      applyStepResult: state.applyStepResult,
      updateHistoryMap: state.updateHistoryMap,
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
    const preprocessCanvas = preprocessCanvasRef.current;
    const heatmapCanvas = heatmapCanvasRef.current;
    const engine = engineRef.current;
    if (!baseCanvas || !preprocessCanvas || !heatmapCanvas) {
      return;
    }

    const baseContext = resizeAndClear2dCanvas(baseCanvas, viewport.width, viewport.height);
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
    display.showPreprocess,
    display.showHeatmap,
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

  const handlePointerDown = (event: PointerEvent<HTMLDivElement>) => {
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

  const handlePointerMove = (event: PointerEvent<HTMLDivElement>) => {
    if (!isDragging || !dragStart || !imageRect || !image || mode !== "box") {
      return;
    }
    const point = screenToPagePoint(event.clientX, event.clientY, imageRect, image.width, image.height);
    if (!point) {
      return;
    }
    setDragState(dragStart, point, true);
  };

  const handlePointerUp = async (event: PointerEvent<HTMLDivElement>) => {
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

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    void loadImageFile(event.target.files?.[0] ?? null);
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    void loadImageFile(event.dataTransfer.files?.[0] ?? null);
  };

  const value = useMemo<SimulationRuntimeValue>(
    () => ({
      baseCanvasRef,
      preprocessCanvasRef,
      heatmapCanvasRef,
      fileInputRef,
      overlay: {
        viewportWidth: viewport.width,
        viewportHeight: viewport.height,
        imageRect,
        imageWidth: image?.width ?? 0,
        imageHeight: image?.height ?? 0,
        dragRoi,
      },
      handlePointerDown,
      handlePointerMove,
      handlePointerUp,
      handleDrop,
      openImageDialog,
      handleFileChange,
      handleNextStep,
    }),
    [handleDrop, handleFileChange, handleNextStep, handlePointerDown, handlePointerMove, handlePointerUp],
  );

  return <SimulationRuntimeContext.Provider value={value}>{children}</SimulationRuntimeContext.Provider>;
}

export function useSimulationRuntime() {
  const value = useContext(SimulationRuntimeContext);
  if (!value) {
    throw new Error("useSimulationRuntime must be used inside SimulationRuntimeProvider.");
  }
  return value;
}
