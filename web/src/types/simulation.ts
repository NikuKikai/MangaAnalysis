export type Point = {
  x: number;
  y: number;
};

export type RoiRect = {
  x: number;
  y: number;
  size: number;
};

export type MouseMode = "click" | "box";
export type SimulationStrategy = "saliency_only" | "panel_guided";

export type DisplayState = {
  showPreprocess: boolean;
  showHeatmap: boolean;
  showHistoryHeatmap: boolean;
  showPanelBoxes: boolean;
  showSelectorOverlay: boolean;
};

export type SimulationSettings = {
  maxBlurStrength: number;
  clearRadiusRatio: number;
  defaultRoiHalfSizeRatio: number;
  historySigmaRatio: number;
  historyDecay: number;
  historyAlpha: number;
  distanceSigmaRatio: number;
  thresholdRatio: number;
  nmsRadiusRatio: number;
  topK: number;
};

export type Candidate = {
  modelX: number;
  modelY: number;
  pageX: number;
  pageY: number;
  saliencyScore: number;
  historyValue: number;
  inhibitionScore: number;
  distanceScore: number;
  finalScore: number;
};

export type ImageResource = {
  bitmap: ImageBitmap;
  width: number;
  height: number;
  url: string;
};

export type PanelRect = {
  x: number;
  y: number;
  width: number;
  height: number;
};

export type PanelBox = {
  panelId: number;
  rect: PanelRect;
  score: number;
  readingIndex: number | null;
};

export type PanelGuidedSelectionSource =
  | "none"
  | "current_panel"
  | "next_panel_local"
  | "next_panel_full_page"
  | "next_panel_center";

export type PanelGuidedStepState = {
  stepIndex: number;
  phase: "start" | "transition";
  fixation: Point;
  currentPanelIndex: number | null;
  currentPanelId: number | null;
  nextPanelId: number | null;
  localRoi: RoiRect;
  usedFullPageRoi: boolean;
  localCandidates: Candidate[];
  fallbackCandidates: Candidate[];
  selectedCandidate: Candidate | null;
  selectedFixation: Point | null;
  selectedPanelId: number | null;
  selectedSource: PanelGuidedSelectionSource;
  actualTransitionScore: number;
  note: string;
};

export type FluidityStep = {
  stepIndex: number;
  score: number;
  candidate: Candidate | null;
  candidatePanelId: number | null;
  strongerThanActualNext: boolean;
};

export type FluidityAnalysis = {
  stepScores: FluidityStep[];
  meanScore: number;
  maxScore: number;
};

export type LoadingPhase = "boot" | "webgpu" | "model" | "ready" | "error";
