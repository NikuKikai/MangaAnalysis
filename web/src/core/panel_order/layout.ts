import type { PanelBox, PanelLayoutNode, PanelRect } from "./types";

type LayoutConfig = {
  minSplitGapPx: number;
  rowOverlapTolerance: number;
  columnOverlapTolerance: number;
};

const DEFAULT_LAYOUT_CONFIG: LayoutConfig = {
  minSplitGapPx: 12,
  rowOverlapTolerance: 0.2,
  columnOverlapTolerance: 0.2,
};

function rectRight(rect: PanelRect): number {
  return rect.x + rect.width;
}

function rectBottom(rect: PanelRect): number {
  return rect.y + rect.height;
}

function unionBounds(rects: PanelRect[]): PanelRect {
  const left = Math.min(...rects.map((rect) => rect.x));
  const top = Math.min(...rects.map((rect) => rect.y));
  const right = Math.max(...rects.map((rect) => rectRight(rect)));
  const bottom = Math.max(...rects.map((rect) => rectBottom(rect)));
  return { x: left, y: top, width: right - left, height: bottom - top };
}

function flattenReadingOrder(node: PanelLayoutNode): number[] {
  if (node.kind === "leaf") {
    if (node.panelId === undefined) {
      throw new Error("Leaf layout node is missing panelId.");
    }
    return [node.panelId];
  }
  return node.children.flatMap((child) => flattenReadingOrder(child));
}

function bestVerticalSplit(
  panels: PanelBox[],
  config: LayoutConfig,
): { score: number; firstGroup: PanelBox[]; secondGroup: PanelBox[] } | null {
  const ordered = [...panels].sort((left, right) => left.rect.x - right.rect.x);
  const prefixMaxRight: number[] = [];
  const suffixMinLeft: number[] = new Array(ordered.length);

  let currentMax = rectRight(ordered[0].rect);
  for (const panel of ordered) {
    currentMax = Math.max(currentMax, rectRight(panel.rect));
    prefixMaxRight.push(currentMax);
  }

  let currentMin = ordered[ordered.length - 1].rect.x;
  for (let index = ordered.length - 1; index >= 0; index -= 1) {
    currentMin = Math.min(currentMin, ordered[index].rect.x);
    suffixMinLeft[index] = currentMin;
  }

  let best: { score: number; firstGroup: PanelBox[]; secondGroup: PanelBox[] } | null = null;
  for (let index = 0; index < ordered.length - 1; index += 1) {
    const gap = suffixMinLeft[index + 1] - prefixMaxRight[index];
    if (gap < config.minSplitGapPx) {
      continue;
    }
    if (!best || gap > best.score) {
      best = {
        score: gap,
        firstGroup: ordered.slice(0, index + 1),
        secondGroup: ordered.slice(index + 1),
      };
    }
  }
  return best;
}

function bestHorizontalSplit(
  panels: PanelBox[],
  config: LayoutConfig,
): { score: number; firstGroup: PanelBox[]; secondGroup: PanelBox[] } | null {
  const ordered = [...panels].sort((left, right) => left.rect.y - right.rect.y);
  const prefixMaxBottom: number[] = [];
  const suffixMinTop: number[] = new Array(ordered.length);

  let currentMax = rectBottom(ordered[0].rect);
  for (const panel of ordered) {
    currentMax = Math.max(currentMax, rectBottom(panel.rect));
    prefixMaxBottom.push(currentMax);
  }

  let currentMin = ordered[ordered.length - 1].rect.y;
  for (let index = ordered.length - 1; index >= 0; index -= 1) {
    currentMin = Math.min(currentMin, ordered[index].rect.y);
    suffixMinTop[index] = currentMin;
  }

  let best: { score: number; firstGroup: PanelBox[]; secondGroup: PanelBox[] } | null = null;
  for (let index = 0; index < ordered.length - 1; index += 1) {
    const gap = suffixMinTop[index + 1] - prefixMaxBottom[index];
    if (gap < config.minSplitGapPx) {
      continue;
    }
    if (!best || gap > best.score) {
      best = {
        score: gap,
        firstGroup: ordered.slice(0, index + 1),
        secondGroup: ordered.slice(index + 1),
      };
    }
  }
  return best;
}

function groupIntoRows(panels: PanelBox[], config: LayoutConfig): PanelBox[][] {
  const ordered = [...panels].sort((left, right) => left.rect.y - right.rect.y);
  const rows: PanelBox[][] = [];
  let current: PanelBox[] = [];
  let currentBottom = -1;

  for (const panel of ordered) {
    if (current.length === 0) {
      current = [panel];
      currentBottom = rectBottom(panel.rect);
      continue;
    }

    const overlap = currentBottom - panel.rect.y;
    const minHeight = Math.min(...current.map((item) => item.rect.height), panel.rect.height);
    if (overlap >= minHeight * config.rowOverlapTolerance) {
      current.push(panel);
      currentBottom = Math.max(currentBottom, rectBottom(panel.rect));
      continue;
    }

    rows.push([...current].sort((left, right) => right.rect.x - left.rect.x));
    current = [panel];
    currentBottom = rectBottom(panel.rect);
  }

  if (current.length > 0) {
    rows.push([...current].sort((left, right) => right.rect.x - left.rect.x));
  }
  return rows;
}

function groupIntoColumns(panels: PanelBox[], config: LayoutConfig): PanelBox[][] {
  const ordered = [...panels].sort((left, right) => right.rect.x - left.rect.x);
  const columns: PanelBox[][] = [];
  let current: PanelBox[] = [];
  let currentLeft = Number.POSITIVE_INFINITY;

  for (const panel of ordered) {
    if (current.length === 0) {
      current = [panel];
      currentLeft = panel.rect.x;
      continue;
    }

    const overlap = rectRight(panel.rect) - currentLeft;
    const minWidth = Math.min(...current.map((item) => item.rect.width), panel.rect.width);
    if (overlap >= minWidth * config.columnOverlapTolerance) {
      current.push(panel);
      currentLeft = Math.min(currentLeft, panel.rect.x);
      continue;
    }

    columns.push([...current].sort((left, right) => left.rect.y - right.rect.y));
    current = [panel];
    currentLeft = panel.rect.x;
  }

  if (current.length > 0) {
    columns.push([...current].sort((left, right) => left.rect.y - right.rect.y));
  }
  return columns;
}

function buildRecursive(panels: PanelBox[], config: LayoutConfig): PanelLayoutNode {
  const bounds = unionBounds(panels.map((panel) => panel.rect));
  if (panels.length === 1) {
    return {
      kind: "leaf",
      bounds,
      panelId: panels[0].panelId,
      children: [],
    };
  }

  const vertical = bestVerticalSplit(panels, config);
  const horizontal = bestHorizontalSplit(panels, config);
  if (vertical || horizontal) {
    if (!horizontal || (vertical && vertical.score >= horizontal.score)) {
      return {
        kind: "vertical",
        bounds,
        // Japanese manga reads right group before left group.
        children: [buildRecursive(vertical!.secondGroup, config), buildRecursive(vertical!.firstGroup, config)],
      };
    }
    return {
      kind: "horizontal",
      bounds,
      children: [buildRecursive(horizontal.firstGroup, config), buildRecursive(horizontal.secondGroup, config)],
    };
  }

  const rows = groupIntoRows(panels, config);
  if (rows.length > 1) {
    return {
      kind: "horizontal",
      bounds,
      children: rows.map((row) => buildRecursive(row, config)),
    };
  }

  const columns = groupIntoColumns(panels, config);
  if (columns.length > 1) {
    return {
      kind: "vertical",
      bounds,
      children: columns.map((column) => buildRecursive(column, config)),
    };
  }

  const fallback = [...panels].sort((left, right) => {
    if (left.rect.y !== right.rect.y) {
      return left.rect.y - right.rect.y;
    }
    return right.rect.x - left.rect.x;
  });
  return {
    kind: "horizontal",
    bounds,
    children: fallback.map((panel) => buildRecursive([panel], config)),
  };
}

export function buildPanelLayout(
  panels: PanelBox[],
  config: Partial<LayoutConfig> = {},
): { layoutTree: PanelLayoutNode; readingOrder: number[] } {
  if (panels.length === 0) {
    throw new Error("At least one panel is required to build a layout tree.");
  }
  const resolvedConfig: LayoutConfig = { ...DEFAULT_LAYOUT_CONFIG, ...config };
  const layoutTree = buildRecursive(panels, resolvedConfig);
  return {
    layoutTree,
    readingOrder: flattenReadingOrder(layoutTree),
  };
}
