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
  currentFixation: Point | null;
  currentRoi: RoiRect | null;
  activeRoiHalfSizePx: number | null;
  pendingNextFixation: Point | null;
  candidates: Candidate[];
  trajectory: Point[];
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
    candidates: Candidate[];
    pendingNextFixation: Point | null;
    committedTrajectory: Point[];
  }) => void;
  commitPendingFixation: () => Point | null;
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
  currentFixation: null,
  currentRoi: null,
  activeRoiHalfSizePx: null,
  pendingNextFixation: null,
  candidates: [],
  trajectory: [],
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
      currentFixation: null,
      currentRoi: null,
      activeRoiHalfSizePx: null,
      pendingNextFixation: null,
      candidates: [],
      trajectory: [],
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
    candidates,
    pendingNextFixation,
    committedTrajectory,
  }) =>
    set({
      currentFixation: fixation,
      currentRoi: roi,
      activeRoiHalfSizePx: roi.size * 0.5,
      pendingNextFixation,
      candidates,
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
  clearSimulation: () =>
    set({
      currentFixation: null,
      currentRoi: null,
      activeRoiHalfSizePx: null,
      pendingNextFixation: null,
      candidates: [],
      trajectory: [],
      dragStart: null,
      dragCurrent: null,
      isDragging: false,
    }),
}));
