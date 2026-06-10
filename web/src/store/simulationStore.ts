import { create } from "zustand";
import type {
  Candidate,
  DisplayState,
  ImageResource,
  LoadingPhase,
  MouseMode,
  Point,
  PreprocessPreview,
  RoiRect,
  SimulationSettings,
  StepResult,
} from "../types/simulation";

const defaultSettings: SimulationSettings = {
  maxBlurStrength: 16,
  clearRadiusRatio: 0.06,
  clickRoiHalfSizeRatio: 0.2,
  historySigmaRatio: 0.047,
  historyAlpha: 2,
  distanceSigmaRatio: 0.183,
  thresholdRatio: 0.55,
  nmsRadiusRatio: 0.013,
  topK: 8,
};

const defaultDisplay: DisplayState = {
  showPreprocess: true,
  showHeatmap: true,
  showHistory: true,
};

type SimulationStore = {
  loadingPhase: LoadingPhase;
  loadingLabel: string;
  errorMessage: string | null;
  webgpuAvailable: boolean;
  image: ImageResource | null;
  mode: MouseMode;
  display: DisplayState;
  settings: SimulationSettings;
  historyMap: Float32Array | null;
  historyMapWidth: number;
  historyMapHeight: number;
  currentFixation: Point | null;
  currentRoi: RoiRect | null;
  activeRoiHalfSizePx: number | null;
  pendingNextFixation: Point | null;
  candidates: Candidate[];
  trajectory: Point[];
  currentHeatmap: Float32Array | null;
  currentPreprocess: PreprocessPreview | null;
  dragStart: Point | null;
  dragCurrent: Point | null;
  isDragging: boolean;
  setLoadingState: (phase: LoadingPhase, label: string) => void;
  setError: (message: string) => void;
  setWebgpuAvailable: (available: boolean) => void;
  setImage: (image: ImageResource | null) => void;
  setMode: (mode: MouseMode) => void;
  updateSetting: <K extends keyof SimulationSettings>(key: K, value: SimulationSettings[K]) => void;
  toggleDisplay: (key: keyof DisplayState) => void;
  setDragState: (start: Point | null, current: Point | null, dragging: boolean) => void;
  applyStepResult: (result: StepResult, committedTrajectory: Point[]) => void;
  commitPendingFixation: () => Point | null;
  initializeHistoryMap: (width: number, height: number) => void;
  updateHistoryMap: (map: Float32Array, width: number, height: number) => void;
  clearSimulation: () => void;
};

export const useSimulationStore = create<SimulationStore>((set, get) => ({
  loadingPhase: "boot",
  loadingLabel: "Initializing",
  errorMessage: null,
  webgpuAvailable: false,
  image: null,
  mode: "click",
  display: defaultDisplay,
  settings: defaultSettings,
  historyMap: null,
  historyMapWidth: 0,
  historyMapHeight: 0,
  currentFixation: null,
  currentRoi: null,
  activeRoiHalfSizePx: null,
  pendingNextFixation: null,
  candidates: [],
  trajectory: [],
  currentHeatmap: null,
  currentPreprocess: null,
  dragStart: null,
  dragCurrent: null,
  isDragging: false,
  setLoadingState: (phase, label) =>
    set({
      loadingPhase: phase,
      loadingLabel: label,
      errorMessage: phase === "error" ? get().errorMessage : null,
    }),
  setError: (message) =>
    set({
      loadingPhase: "error",
      loadingLabel: "Error",
      errorMessage: message,
    }),
  setWebgpuAvailable: (available) => set({ webgpuAvailable: available }),
  setImage: (image) =>
    set({
      image,
      historyMap: image ? new Float32Array(image.width * image.height) : null,
      historyMapWidth: image?.width ?? 0,
      historyMapHeight: image?.height ?? 0,
      currentFixation: null,
      currentRoi: null,
      activeRoiHalfSizePx: null,
      pendingNextFixation: null,
      candidates: [],
      trajectory: [],
      currentHeatmap: null,
      currentPreprocess: null,
    }),
  setMode: (mode) => set({ mode }),
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
  applyStepResult: (result, committedTrajectory) =>
    set({
      currentFixation: result.fixation,
      currentRoi: result.roi,
      activeRoiHalfSizePx: result.roi.size * 0.5,
      pendingNextFixation: result.pendingNextFixation,
      candidates: result.candidates,
      currentHeatmap: result.heatmap,
      currentPreprocess: result.preprocess,
      trajectory: committedTrajectory,
    }),
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
  initializeHistoryMap: (width, height) =>
    set({
      historyMap: new Float32Array(width * height),
      historyMapWidth: width,
      historyMapHeight: height,
      trajectory: [],
      pendingNextFixation: null,
      candidates: [],
      currentHeatmap: null,
      currentPreprocess: null,
      currentFixation: null,
      currentRoi: null,
      activeRoiHalfSizePx: null,
    }),
  updateHistoryMap: (map, width, height) =>
    set({
      historyMap: map,
      historyMapWidth: width,
      historyMapHeight: height,
    }),
  clearSimulation: () =>
    set((state) => ({
      historyMap:
        state.image && state.image.width > 0 && state.image.height > 0
          ? new Float32Array(state.image.width * state.image.height)
          : null,
      historyMapWidth: state.image?.width ?? 0,
      historyMapHeight: state.image?.height ?? 0,
      currentFixation: null,
      currentRoi: null,
      activeRoiHalfSizePx: null,
      pendingNextFixation: null,
      candidates: [],
      trajectory: [],
      currentHeatmap: null,
      currentPreprocess: null,
      dragStart: null,
      dragCurrent: null,
      isDragging: false,
    })),
}));
