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
};

export type PanelLayoutNode = {
  kind: "leaf" | "horizontal" | "vertical";
  bounds: PanelRect;
  panelId?: number;
  children: PanelLayoutNode[];
};

export type PanelAnalysis = {
  imageWidth: number;
  imageHeight: number;
  panels: PanelBox[];
  layoutTree: PanelLayoutNode;
  readingOrder: number[];
};
