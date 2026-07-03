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

export type DisplayState = {
  showPreprocess: boolean;
  showHeatmap: boolean;
  showHistoryHeatmap: boolean;
  showPanelBoxes: boolean;
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

export type LoadingPhase = "boot" | "webgpu" | "model" | "ready" | "error";
