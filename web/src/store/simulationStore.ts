import { create } from "zustand";
import type {
  Candidate,
  DisplayState,
  FluidityAnalysis,
  ImageResource,
  LoadingPhase,
  MouseMode,
  PanelBox,
  PanelGuidedStepState,
  Point,
  RoiRect,
  SimulationStrategy,
  SimulationSettings,
} from "../types/simulation";

const defaultSettings: SimulationSettings = {
  maxBlurStrength: 7,
  clearRadiusRatio: 0.06,
  defaultRoiHalfSizeRatio: 0.25,
  historySigmaRatio: 0.047,
  historyDecay: 0.94,
  historyAlpha: 3,
  distanceSigmaRatio: 0.183,
  thresholdRatio: 0.15,
  nmsRadiusRatio: 0.013,
  topK: 8,
};

const defaultDisplay: DisplayState = {
  showPreprocess: true,
  showHeatmap: true,
  showHistoryHeatmap: false,
  showPanelBoxes: true,
  showSelectorOverlay: true,
};

type SimulationStore = {
  // Lifecycle and runtime availability.
  loadingPhase: LoadingPhase;
  errorMessage: string | null;
  webgpuAvailable: boolean;

  // Loaded page resource and initial ROI selection mode.
  image: ImageResource | null;
  mode: MouseMode;
  strategy: SimulationStrategy;

  // UI visibility flags and tunable simulation parameters.
  display: DisplayState;
  settings: SimulationSettings;

  // Latest committed step result used for rendering and stepping forward.
  currentFixation: Point | null;
  currentRoi: RoiRect | null;
  pendingNextFixation: Point | null;
  candidates: Candidate[];
  panelBoxes: PanelBox[];
  panelReadingOrder: number[];
  trajectory: Point[];
  panelGuidedCurrentPanelIndex: number | null;
  panelGuidedStepStates: PanelGuidedStepState[];
  panelGuidedAnalysis: FluidityAnalysis | null;

  // Transient box-drag interaction state.
  dragStart: Point | null;
  dragCurrent: Point | null;
  isDragging: boolean;

  // Store actions.
  setLoadingState: (phase: LoadingPhase) => void;
  setError: (message: string) => void;
  setWebgpuAvailable: (available: boolean) => void;
  setImage: (image: ImageResource | null) => void;
  loadImageFile: (file: File | null) => Promise<void>;
  setMode: (mode: MouseMode) => void;
  setStrategy: (strategy: SimulationStrategy) => void;
  updateSetting: <K extends keyof SimulationSettings>(key: K, value: SimulationSettings[K]) => void;
  toggleDisplay: (key: keyof DisplayState) => void;
  setDragState: (start: Point | null, current: Point | null, dragging: boolean) => void;
  setPanelDetection: (panelBoxes: PanelBox[], panelReadingOrder: number[]) => void;
  applyStepOutcome: (params: {
    roi: RoiRect;
    fixation: Point;
    candidates: Candidate[];
    pendingNextFixation: Point | null;
    committedTrajectory: Point[];
    panelGuidedCurrentPanelIndex?: number | null;
    panelGuidedStepState?: PanelGuidedStepState | null;
    panelGuidedAnalysis?: FluidityAnalysis | null;
  }) => void;
  commitPendingFixation: () => Point | null;
  clearSimulation: () => void;
};

export const useSimulationStore = create<SimulationStore>((set, get) => ({
  // Lifecycle and runtime availability.
  loadingPhase: "boot",
  errorMessage: null,
  webgpuAvailable: false,

  // Loaded page resource and initial ROI selection mode.
  image: null,
  mode: "click",
  strategy: "saliency_only",

  // UI visibility flags and tunable simulation parameters.
  display: defaultDisplay,
  settings: defaultSettings,

  // Latest committed step result used for rendering and stepping forward.
  currentFixation: null,
  currentRoi: null,
  pendingNextFixation: null,
  candidates: [],
  panelBoxes: [],
  panelReadingOrder: [],
  trajectory: [],
  panelGuidedCurrentPanelIndex: null,
  panelGuidedStepStates: [],
  panelGuidedAnalysis: null,

  // Transient box-drag interaction state.
  dragStart: null,
  dragCurrent: null,
  isDragging: false,

  // Store actions.
  setLoadingState: (phase) =>
    set({
      loadingPhase: phase,
      errorMessage: phase === "error" ? get().errorMessage : null,
    }),
  setError: (message) =>
    set({
      loadingPhase: "error",
      errorMessage: message,
    }),
  setWebgpuAvailable: (available) => set({ webgpuAvailable: available }),
  setImage: (image) =>
    set({
      image,
      currentFixation: null,
      currentRoi: null,
      pendingNextFixation: null,
      candidates: [],
      panelBoxes: [],
      panelReadingOrder: [],
      trajectory: [],
      panelGuidedCurrentPanelIndex: null,
      panelGuidedStepStates: [],
      panelGuidedAnalysis: null,
    }),
  loadImageFile: async (file) => {
    if (!file) {
      return;
    }
    const url = URL.createObjectURL(file);
    const bitmap = await createImageBitmap(file);
    get().setImage({
      bitmap,
      width: bitmap.width,
      height: bitmap.height,
      url,
    });
    get().clearSimulation();
  },
  setMode: (mode) => set({ mode }),
  setStrategy: (strategy) =>
    set({
      strategy,
      currentFixation: null,
      currentRoi: null,
      pendingNextFixation: null,
      candidates: [],
      trajectory: [],
      panelGuidedCurrentPanelIndex: null,
      panelGuidedStepStates: [],
      panelGuidedAnalysis: null,
      dragStart: null,
      dragCurrent: null,
      isDragging: false,
    }),
  updateSetting: (key, value) =>
    set((state) => ({
      settings: {
        ...state.settings,
        [key]: value,
      },
    })),
  toggleDisplay: (key) =>
    set((state) => ({
      display: {
        ...state.display,
        [key]: !state.display[key],
      },
    })),
  setDragState: (start, current, dragging) =>
    set({
      dragStart: start,
      dragCurrent: current,
      isDragging: dragging,
    }),
  setPanelDetection: (panelBoxes, panelReadingOrder) => set({ panelBoxes, panelReadingOrder }),
  applyStepOutcome: ({
    roi,
    fixation,
    candidates,
    pendingNextFixation,
    committedTrajectory,
    panelGuidedCurrentPanelIndex,
    panelGuidedStepState,
    panelGuidedAnalysis,
  }) =>
    set((state) => ({
      currentFixation: fixation,
      currentRoi: roi,
      pendingNextFixation,
      candidates,
      trajectory: committedTrajectory,
      panelGuidedCurrentPanelIndex:
        panelGuidedCurrentPanelIndex === undefined ? state.panelGuidedCurrentPanelIndex : panelGuidedCurrentPanelIndex,
      panelGuidedStepStates:
        panelGuidedStepState === undefined || panelGuidedStepState === null
          ? state.panelGuidedStepStates
          : panelGuidedStepState.stepIndex === 0
            ? [panelGuidedStepState]
            : [...state.panelGuidedStepStates, panelGuidedStepState],
      panelGuidedAnalysis:
        panelGuidedAnalysis === undefined
          ? state.panelGuidedAnalysis
          : panelGuidedAnalysis,
    })),
  commitPendingFixation: () => {
    const pending = get().pendingNextFixation;
    if (!pending) {
      return null;
    }
    set((state) => ({
      currentFixation: pending,
      trajectory: [...state.trajectory, pending],
    }));
    return pending;
  },
  clearSimulation: () =>
    set({
      currentFixation: null,
      currentRoi: null,
      pendingNextFixation: null,
      candidates: [],
      trajectory: [],
      panelGuidedCurrentPanelIndex: null,
      panelGuidedStepStates: [],
      panelGuidedAnalysis: null,
      dragStart: null,
      dragCurrent: null,
      isDragging: false,
    }),
}));
