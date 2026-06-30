import { createContext, useContext, useEffect, useMemo, useRef, useState, type PropsWithChildren, type RefObject } from "react";
import { GpuCandidateSelector } from "../core/gpu/candidateSelector";
import { HeatmapRenderer } from "../core/gpu/heatmapRenderer";
import { HistoryRenderer } from "../core/gpu/historyRenderer";
import { modelSize, RoiPreprocessor } from "../core/gpu/preprocess";
import { PreprocessRenderer } from "../core/gpu/preprocessRenderer";
import { SaliencySession } from "../core/onnx/saliencySession";
import { drawBaseImage, resizeAndClear2dCanvas } from "../core/utils/canvas2d";
import {
  createCenteredSquareRoi,
  createFullPageSquareRoi,
  createSquareFromDrag,
  fitImageToViewport,
  imageRectToPhysical,
  roiCenter,
  roiToPhysicalScreenRect,
  type ImageRect,
} from "../core/utils/roi";
import { useSimulationStore } from "../store/simulationStore";
import type { Candidate, ImageResource, Point, RoiRect } from "../types/simulation";
import { useViewportSize } from "./useViewportSize";

type Engine = {
  device: GPUDevice;
  preprocessor: RoiPreprocessor;
  heatmapRenderer: HeatmapRenderer;
  historyRenderer: HistoryRenderer;
  preprocessRenderer: PreprocessRenderer;
  candidateSelector: GpuCandidateSelector;
  session: SaliencySession;
  historyMapWidth: number;
  historyMapHeight: number;
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
  startClickStep: (point: Point) => Promise<void>;
  startBoxStep: (roi: RoiRect) => Promise<void>;
  handleNextStep: () => Promise<void>;
};

async function createEngine(
  heatmapCanvas: HTMLCanvasElement,
  historyCanvas: HTMLCanvasElement,
  preprocessCanvas: HTMLCanvasElement,
): Promise<Engine> {
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
  const historyRenderer = new HistoryRenderer(device, historyCanvas);
  const preprocessRenderer = new PreprocessRenderer(device, preprocessCanvas);
  const candidateSelector = new GpuCandidateSelector(device);
  const session = await SaliencySession.create();
  return {
    device,
    preprocessor,
    heatmapRenderer,
    historyRenderer,
    preprocessRenderer,
    candidateSelector,
    session,
    historyMapWidth: 0,
    historyMapHeight: 0,
  };
}

const SimulationEngineContext = createContext<SimulationEngineContextValue | null>(null);

export function SimulationProvider({ children }: PropsWithChildren) {
  // Canvas layers are owned here so rendering effects can address them directly.
  const baseCanvasRef = useRef<HTMLCanvasElement>(null);
  const historyCanvasRef = useRef<HTMLCanvasElement>(null);
  const preprocessCanvasRef = useRef<HTMLCanvasElement>(null);
  const heatmapCanvasRef = useRef<HTMLCanvasElement>(null);
  const engineRef = useRef<Engine | null>(null);
  const initStartedRef = useRef(false);
  const [imageRect, setImageRect] = useState<ImageRect | null>(null);

  // Read store fields individually so each dependency stays explicit at call sites.
  const viewport = useViewportSize();
  const image = useSimulationStore((state) => state.image);
  const display = useSimulationStore((state) => state.display);
  const settings = useSimulationStore((state) => state.settings);
  const currentRoi = useSimulationStore((state) => state.currentRoi);
  const currentFixation = useSimulationStore((state) => state.currentFixation);
  const pendingNextFixation = useSimulationStore((state) => state.pendingNextFixation);
  const trajectory = useSimulationStore((state) => state.trajectory);
  const dragStart = useSimulationStore((state) => state.dragStart);
  const dragCurrent = useSimulationStore((state) => state.dragCurrent);
  const isDragging = useSimulationStore((state) => state.isDragging);
  const mode = useSimulationStore((state) => state.mode);
  const setLoadingState = useSimulationStore((state) => state.setLoadingState);
  const setError = useSimulationStore((state) => state.setError);
  const setWebgpuAvailable = useSimulationStore((state) => state.setWebgpuAvailable);
  const applyStepOutcome = useSimulationStore((state) => state.applyStepOutcome);

  const renderHistoryOverlay = (engine: Engine, nextImageRect: ImageRect | null) => {
    const physicalRatio = window.devicePixelRatio || 1;
    const scaledImageRect = imageRectToPhysical(nextImageRect, physicalRatio);
    engine.historyRenderer.render({
      imageRect: scaledImageRect,
      enabled: Boolean(display.showHistoryHeatmap && nextImageRect),
    });
  };

  // Keep the drag ROI derived from pointer state instead of storing redundant data.
  const dragRoi = useMemo<RoiRect | null>(() => {
    if (mode !== "box" || !dragStart || !dragCurrent || !isDragging) {
      return null;
    }
    return createSquareFromDrag(dragStart, dragCurrent);
  }, [dragCurrent, dragStart, isDragging, mode]);

  // Recompute the fitted image rectangle whenever the viewport or image changes.
  useEffect(() => {
    if (!image) {
      setImageRect(null);
      return;
    }
    setImageRect(fitImageToViewport(image.width, image.height, viewport.width, viewport.height));
  }, [image, viewport.height, viewport.width]);

  // Create the GPU engine once after the render targets become available.
  useEffect(() => {
    const initialize = async () => {
      const heatmapCanvas = heatmapCanvasRef.current;
      const historyCanvas = historyCanvasRef.current;
      const preprocessCanvas = preprocessCanvasRef.current;
      if (!heatmapCanvas || !historyCanvas || !preprocessCanvas || initStartedRef.current) {
        return;
      }
      initStartedRef.current = true;
      try {
        setLoadingState("webgpu");
        const engine = await createEngine(heatmapCanvas, historyCanvas, preprocessCanvas);
        engineRef.current = engine;
        setWebgpuAvailable(true);
        setLoadingState("model");
        setLoadingState("ready");
      } catch (error) {
        const message = error instanceof Error ? error.message : "Failed to initialize the demo.";
        setError(message);
      }
    };
    void initialize();
  }, [setError, setLoadingState, setWebgpuAvailable]);

  // Push the latest decoded bitmap into the GPU preprocessor source texture.
  useEffect(() => {
    if (image && engineRef.current) {
      engineRef.current.preprocessor.setSourceImage(image.bitmap);
      engineRef.current.historyMapWidth = image.width;
      engineRef.current.historyMapHeight = image.height;
      engineRef.current.historyRenderer.initialize(image.width, image.height);
      renderHistoryOverlay(engineRef.current, imageRect);
    }
  }, [image, imageRect, display.showHistoryHeatmap]);

  // Reset GPU-side simulation buffers when the logical simulation state is cleared.
  useEffect(() => {
    const engine = engineRef.current;
    if (
      !engine ||
      !image ||
      trajectory.length !== 0 ||
      currentRoi ||
      currentFixation ||
      pendingNextFixation
    ) {
      return;
    }
    engine.historyMapWidth = image.width;
    engine.historyMapHeight = image.height;
    engine.historyRenderer.initialize(image.width, image.height);
    renderHistoryOverlay(engine, imageRect);
  }, [image, imageRect, trajectory.length, currentRoi, currentFixation, pendingNextFixation, display.showHistoryHeatmap]);

  // Keep the GPU overlay canvases sized to the current viewport in physical pixels.
  useEffect(() => {
    const preprocessCanvas = preprocessCanvasRef.current;
    const historyCanvas = historyCanvasRef.current;
    const heatmapCanvas = heatmapCanvasRef.current;
    const engine = engineRef.current;
    if (!preprocessCanvas || !historyCanvas || !heatmapCanvas) {
      return;
    }

    const ratio = window.devicePixelRatio || 1;
    preprocessCanvas.width = Math.max(1, Math.floor(viewport.width * ratio));
    preprocessCanvas.height = Math.max(1, Math.floor(viewport.height * ratio));
    preprocessCanvas.style.width = `${viewport.width}px`;
    preprocessCanvas.style.height = `${viewport.height}px`;
    historyCanvas.width = Math.max(1, Math.floor(viewport.width * ratio));
    historyCanvas.height = Math.max(1, Math.floor(viewport.height * ratio));
    historyCanvas.style.width = `${viewport.width}px`;
    historyCanvas.style.height = `${viewport.height}px`;
    heatmapCanvas.width = Math.max(1, Math.floor(viewport.width * ratio));
    heatmapCanvas.height = Math.max(1, Math.floor(viewport.height * ratio));
    heatmapCanvas.style.width = `${viewport.width}px`;
    heatmapCanvas.style.height = `${viewport.height}px`;
    engine?.preprocessRenderer.resize(preprocessCanvas.width, preprocessCanvas.height);
    engine?.heatmapRenderer.resize(heatmapCanvas.width, heatmapCanvas.height);
    engine?.historyRenderer.resize(historyCanvas.width, historyCanvas.height);
  }, [viewport.width, viewport.height]);

  // Redraw the base image only when its source or fitted rectangle changes.
  useEffect(() => {
    const baseCanvas = baseCanvasRef.current;
    if (!baseCanvas) {
      return;
    }

    const baseContext = resizeAndClear2dCanvas(baseCanvas, viewport.width, viewport.height);
    if (image && imageRect) {
      drawBaseImage(baseContext, image.bitmap, imageRect);
    }
  }, [viewport.width, viewport.height, image, imageRect]);

  // Redraw the preprocess preview layer independently from the other overlays.
  useEffect(() => {
    const engine = engineRef.current;
    if (!engine) {
      return;
    }

    const physicalRatio = window.devicePixelRatio || 1;
    const scaledImageRect = imageRectToPhysical(imageRect, physicalRatio);
    const scaledRoiRect = roiToPhysicalScreenRect(
      currentRoi,
      imageRect,
      image?.width ?? 0,
      image?.height ?? 0,
      physicalRatio,
    );
    engine.preprocessRenderer.render({
      imageRect:
        scaledImageRect ??
        ({
          x: 0,
          y: 0,
          width: 0,
          height: 0,
        } satisfies ImageRect),
      roiRect: scaledRoiRect,
      enabled: display.showPreprocess,
    });
  }, [imageRect, image, currentRoi, display.showPreprocess]);

  // Upload and render the history overlay whenever its data or visibility changes.
  useEffect(() => {
    const engine = engineRef.current;
    if (!engine) {
      return;
    }

    renderHistoryOverlay(engine, imageRect);
  }, [imageRect, display.showHistoryHeatmap]);

  // Render the heatmap overlay from the latest GPU buffer and current ROI placement.
  useEffect(() => {
    const engine = engineRef.current;
    if (!engine) {
      return;
    }

    const physicalRatio = window.devicePixelRatio || 1;
    const scaledImageRect = imageRectToPhysical(imageRect, physicalRatio);
    const scaledRoiRect = roiToPhysicalScreenRect(
      currentRoi,
      imageRect,
      image?.width ?? 0,
      image?.height ?? 0,
      physicalRatio,
    );
    engine.heatmapRenderer.render({
      imageRect:
        scaledImageRect ??
        ({
          x: 0,
          y: 0,
          width: 0,
          height: 0,
        } satisfies ImageRect),
      roiRect: scaledRoiRect,
      enabled: Boolean(display.showHeatmap && imageRect && currentRoi),
    });
  }, [image, imageRect, currentRoi, display.showHeatmap]);

  const filterCandidatesByThreshold = (candidates: Candidate[]) =>
    candidates.filter((candidate) => candidate.finalScore >= settings.thresholdRatio);

  const runStepOnce = async (engine: Engine, roi: RoiRect, fixation: Point) => {
    const input = await engine.preprocessor.run(roi, fixation, image.height, settings);
    engine.preprocessRenderer.updatePreview(input, modelSize());
    const heatmap = await engine.session.predict(input);
    const nmsRadius = Math.max(1, Math.round((image.height * settings.nmsRadiusRatio * modelSize()) / roi.size));
    const distanceSigma = Math.max(1, image.height * settings.distanceSigmaRatio);
    engine.heatmapRenderer.updateHeatmap(heatmap, modelSize());
    const scoredCandidates = await engine.candidateSelector.select({
      heatmapBuffer: engine.heatmapRenderer.getBuffer(),
      historyBuffer: engine.historyRenderer.getBuffer(),
      mapSize: modelSize(),
      nmsRadius,
      topK: settings.topK,
      roi,
      imageWidth: image.width,
      imageHeight: image.height,
      currentFixation: fixation,
      historyMapWidth: engine.historyMapWidth,
      historyMapHeight: engine.historyMapHeight,
      historyAlpha: settings.historyAlpha,
      distanceSigma,
    });
    return {
      roi,
      candidates: filterCandidatesByThreshold(scoredCandidates),
    };
  };

  // Run one simulation step: update history, preprocess ROI input, infer heatmap, then select candidates.
  const runStep = async (initialRoi: RoiRect, fixation: Point, committedTrajectory: Point[]) => {
    const engine = engineRef.current;
    if (!engine || !image) {
      return;
    }
    if (engine.historyMapWidth !== image.width || engine.historyMapHeight !== image.height) {
      engine.historyMapWidth = image.width;
      engine.historyMapHeight = image.height;
      engine.historyRenderer.initialize(image.width, image.height);
    }
    engine.historyRenderer.accumulateFixation(
      fixation.x,
      fixation.y,
      image.height * settings.historySigmaRatio,
      settings.historyDecay,
    );
    {
      renderHistoryOverlay(engine, imageRect);
    }

    let stepResult = await runStepOnce(engine, initialRoi, fixation);
    if (stepResult.candidates.length === 0) {
      stepResult = await runStepOnce(engine, createFullPageSquareRoi(image.width, image.height), fixation);
    }

    const { roi, candidates: scoredCandidates } = stepResult;
    const pendingNextFixation =
      scoredCandidates.length > 0
        ? {
            x: scoredCandidates[0].pageX,
            y: scoredCandidates[0].pageY,
          }
        : null;
    applyStepOutcome({
      roi,
      fixation,
      candidates: scoredCandidates,
      pendingNextFixation,
      committedTrajectory,
    });
  };

  // Start a click-driven step using the configured square ROI around the click point.
  const startClickStep = async (point: Point) => {
    if (!image) {
      return;
    }
    const halfSize = image.height * settings.defaultRoiHalfSizeRatio;
    const roi = createCenteredSquareRoi(point, halfSize);
    await runStep(roi, point, [point]);
  };

  // Start a box-driven step using the user-drawn ROI and its center as fixation.
  const startBoxStep = async (roi: RoiRect) => {
    const fixation = roiCenter(roi);
    await runStep(roi, fixation, [fixation]);
  };

  // Continue the trajectory from the pending fixation chosen in the previous step.
  const handleNextStep = async () => {
    if (!image || !pendingNextFixation) {
      return;
    }
    const halfSize = image.height * settings.defaultRoiHalfSizeRatio;
    const roi = createCenteredSquareRoi(pendingNextFixation, halfSize);
    await runStep(roi, pendingNextFixation, [...trajectory, pendingNextFixation]);
  };

  // Expose rendering refs plus interaction entry points to the stage and controls.
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
