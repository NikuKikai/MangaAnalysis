export type EdgeSamPointLabel = 0 | 1;

export type EdgeSamPoint = {
  x: number;
  y: number;
  label: EdgeSamPointLabel;
};

export type EdgeSamMask = {
  data: Uint8Array;
  width: number;
  height: number;
  validWidth: number;
  validHeight: number;
  score: number;
};

export type EdgeSamPrediction = {
  masks: EdgeSamMask[];
  lowResLogits: Float32Array;
  lowResMaskUint8: Uint8Array;
};

export type EdgeSamPrompt = {
  points: EdgeSamPoint[];
};
