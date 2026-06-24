import { create } from "zustand";
import type {
  Candidate,
  DisplayState,
  ImageResource,
  LoadingPhase,
  MouseMode,
  Point,
  RoiRect,
  SimulationSettings,
} from "../types/simulation";

const defaultSettings: SimulationSettings = {
  maxBlurStrength: 7,
  clearRadiusRatio: 0.06,
  clickRoiHalfSizeRatio: 0.25,
  historySigmaRatio: 0.047,
  historyAlpha: 3,
  distanceSigmaRatio: 0.183,
  thresholdRatio: 0.55,
  nmsRadiusRatio: 0.013,
  topK: 8,
};

const defaultDisplay: DisplayState = {
  showPreprocess: true,
  showHeatmap: true,
  showHistoryHeatmap: false,
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
  currentPreprocess: Float32Array | null;
  dragStart: Point | null;
  dragCurrent: Point | null;
  isDragging: boolean;
  setLoadingState: (phase: LoadingPhase, label: string) => void;
  setError: (message: string) => void;
  setWebgpuAvailable: (available: boolean) => void;
  setImage: (image: ImageResource | null) => void;
  loadImageFile: (file: File | null) => Promise<void>;
  setMode: (mode: MouseMode) => void;
  updateSetting: <K extends keyof SimulationSettings>(key: K, value: SimulationSettings[K]) => void;
  toggleDisplay: (key: keyof DisplayState) => void;
  setDragState: (start: Point | null, current: Point | null, dragging: boolean) => void;
  applyStepOutcome: (params: {
    roi: RoiRect;
    fixation: Point;
    heatmap: Float32Array;
    preprocess: Float32Array;
    candidates: Candidate[];
    pendingNextFixation: Point | null;
    committedTrajectory: Point[];
    historyMap: Float32Array;
    historyMapWidth: number;
    historyMapHeight: number;
  }) => void;
  commitPendingFixation: () => Point | null;
  initializeHistoryMap: (width: number, height: number) => void;
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
  applyStepOutcome: ({
    roi,
    fixation,
    heatmap,
    preprocess,
    candidates,
    pendingNextFixation,
    committedTrajectory,
    historyMap,
    historyMapWidth,
    historyMapHeight,
  }) =>
    set({
      historyMap,
      historyMapWidth,
      historyMapHeight,
      currentFixation: fixation,
      currentRoi: roi,
      activeRoiHalfSizePx: roi.size * 0.5,
      pendingNextFixation,
      candidates,
      currentHeatmap: heatmap,
      currentPreprocess: preprocess,
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
