import { create } from "zustand";
import type {
  Candidate,
  DisplayState,
  ImageResource,
  LoadingPhase,
  MouseMode,
  PanelBox,
  Point,
  RoiRect,
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
};

type SimulationStore = {
  // Lifecycle and runtime availability.
  loadingPhase: LoadingPhase;
  errorMessage: string | null;
  webgpuAvailable: boolean;

  // Loaded page resource and initial ROI selection mode.
  image: ImageResource | null;
  mode: MouseMode;

  // UI visibility flags and tunable simulation parameters.
  display: DisplayState;
  settings: SimulationSettings;

  // Latest committed step result used for rendering and stepping forward.
  currentFixation: Point | null;
  currentRoi: RoiRect | null;
  pendingNextFixation: Point | null;
  candidates: Candidate[];
  panelBoxes: PanelBox[];
  trajectory: Point[];

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
  updateSetting: <K extends keyof SimulationSettings>(key: K, value: SimulationSettings[K]) => void;
  toggleDisplay: (key: keyof DisplayState) => void;
  setDragState: (start: Point | null, current: Point | null, dragging: boolean) => void;
  setPanelBoxes: (panelBoxes: PanelBox[]) => void;
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
  // Lifecycle and runtime availability.
  loadingPhase: "boot",
  errorMessage: null,
  webgpuAvailable: false,

  // Loaded page resource and initial ROI selection mode.
  image: null,
  mode: "click",

  // UI visibility flags and tunable simulation parameters.
  display: defaultDisplay,
  settings: defaultSettings,

  // Latest committed step result used for rendering and stepping forward.
  currentFixation: null,
  currentRoi: null,
  pendingNextFixation: null,
  candidates: [],
  panelBoxes: [],
  trajectory: [],

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
  setPanelBoxes: (panelBoxes) => set({ panelBoxes }),
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
      pendingNextFixation: null,
      candidates: [],
      trajectory: [],
      dragStart: null,
      dragCurrent: null,
      isDragging: false,
    }),
}));
