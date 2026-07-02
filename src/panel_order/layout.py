from __future__ import annotations

from dataclasses import dataclass

from .types import LayoutNode, Panel, Rect


@dataclass(slots=True)
class LayoutConfig:
    """Grouping and splitting thresholds for layout analysis."""

    min_split_gap_px: int = 12
    row_overlap_tolerance: float = 0.2
    column_overlap_tolerance: float = 0.2


class MangaLayoutAnalyzer:
    """Build a recursive layout tree and derive Japanese reading order."""

    def __init__(self, config: LayoutConfig | None = None) -> None:
        """Create a layout analyzer with the provided grouping configuration."""
        self.config = config or LayoutConfig()

    def build_layout(self, panels: list[Panel]) -> tuple[LayoutNode, list[int]]:
        """Build the layout tree and flatten it into reading order."""
        if not panels:
            raise ValueError("At least one panel is required to build a layout tree.")
        root = self._build_recursive(panels)
        return root, self._flatten_reading_order(root)

    def _build_recursive(self, panels: list[Panel]) -> LayoutNode:
        """Recursively partition panels into a tree of rows, columns, and leaves."""
        bounds = self._union_bounds([panel.rect for panel in panels])
        if len(panels) == 1:
            return LayoutNode(kind="leaf", bounds=bounds, panel_id=panels[0].panel_id)

        # Stage 1: Prefer explicit XY-cut splits when a clear empty gutter exists.
        split = self._best_xy_cut(panels)
        if split is not None:
            kind, first_group, second_group = split
            children = [self._build_recursive(first_group), self._build_recursive(second_group)]
            return LayoutNode(kind=kind, bounds=bounds, children=children)

        # Stage 2: Fall back to row grouping for top-to-bottom reading progression.
        rows = self._group_into_rows(panels)
        if len(rows) > 1:
            children = [self._build_recursive(row) for row in rows]
            return LayoutNode(kind="horizontal", bounds=bounds, children=children)

        # Stage 3: If rows fail, fall back to column grouping for right-to-left progression.
        columns = self._group_into_columns(panels)
        if len(columns) > 1:
            children = [self._build_recursive(column) for column in columns]
            return LayoutNode(kind="vertical", bounds=bounds, children=children)

        # Stage 4: Use a stable fallback order so every panel can still be serialized.
        fallback = sorted(panels, key=lambda panel: (panel.rect.center_y, -panel.rect.center_x))
        children = [self._build_recursive([panel]) for panel in fallback]
        return LayoutNode(kind="horizontal", bounds=bounds, children=children)

    def _best_xy_cut(self, panels: list[Panel]) -> tuple[str, list[Panel], list[Panel]] | None:
        """Choose the strongest empty-gutter split between vertical and horizontal cuts."""
        vertical = self._best_vertical_split(panels)
        horizontal = self._best_horizontal_split(panels)
        if vertical is None and horizontal is None:
            return None
        if vertical is None:
            _score, top_group, bottom_group = horizontal
            return "horizontal", top_group, bottom_group
        if horizontal is None:
            _score, left_group, right_group = vertical
            return "vertical", right_group, left_group
        if vertical[0] >= horizontal[0]:
            _score, left_group, right_group = vertical
            return "vertical", right_group, left_group
        _score, top_group, bottom_group = horizontal
        return "horizontal", top_group, bottom_group

    def _best_vertical_split(self, panels: list[Panel]) -> tuple[float, list[Panel], list[Panel]] | None:
        """Find the widest valid vertical gutter between two panel groups."""
        ordered = sorted(panels, key=lambda panel: panel.rect.x)
        prefix_max_right: list[int] = []
        suffix_min_left: list[int] = [0] * len(ordered)

        # Precompute occupied extents so every possible split can be scored in linear time.
        current_max = ordered[0].rect.right
        for panel in ordered:
            current_max = max(current_max, panel.rect.right)
            prefix_max_right.append(current_max)

        current_min = ordered[-1].rect.x
        for index in range(len(ordered) - 1, -1, -1):
            current_min = min(current_min, ordered[index].rect.x)
            suffix_min_left[index] = current_min

        best: tuple[float, list[Panel], list[Panel]] | None = None
        for index in range(len(ordered) - 1):
            # A valid vertical cut needs a visible empty gap between the left and right groups.
            gap = suffix_min_left[index + 1] - prefix_max_right[index]
            if gap < self.config.min_split_gap_px:
                continue
            score = float(gap)
            group_a = ordered[: index + 1]
            group_b = ordered[index + 1 :]
            if best is None or score > best[0]:
                best = (score, group_a, group_b)
        return best

    def _best_horizontal_split(self, panels: list[Panel]) -> tuple[float, list[Panel], list[Panel]] | None:
        """Find the widest valid horizontal gutter between two panel groups."""
        ordered = sorted(panels, key=lambda panel: panel.rect.y)
        prefix_max_bottom: list[int] = []
        suffix_min_top: list[int] = [0] * len(ordered)

        # Precompute occupied extents so every possible split can be scored in linear time.
        current_max = ordered[0].rect.bottom
        for panel in ordered:
            current_max = max(current_max, panel.rect.bottom)
            prefix_max_bottom.append(current_max)

        current_min = ordered[-1].rect.y
        for index in range(len(ordered) - 1, -1, -1):
            current_min = min(current_min, ordered[index].rect.y)
            suffix_min_top[index] = current_min

        best: tuple[float, list[Panel], list[Panel]] | None = None
        for index in range(len(ordered) - 1):
            # A valid horizontal cut needs a visible empty gap between the top and bottom groups.
            gap = suffix_min_top[index + 1] - prefix_max_bottom[index]
            if gap < self.config.min_split_gap_px:
                continue
            score = float(gap)
            group_a = ordered[: index + 1]
            group_b = ordered[index + 1 :]
            if best is None or score > best[0]:
                best = (score, group_a, group_b)
        return best

    def _group_into_rows(self, panels: list[Panel]) -> list[list[Panel]]:
        """Group panels into reading rows using vertical overlap."""
        ordered = sorted(panels, key=lambda panel: panel.rect.y)
        rows: list[list[Panel]] = []
        current: list[Panel] = []
        current_bottom = -1

        for panel in ordered:
            if not current:
                current = [panel]
                current_bottom = panel.rect.bottom
                continue

            # Panels that share enough vertical overlap belong to the same reading row.
            overlap = current_bottom - panel.rect.y
            min_height = min(min(item.rect.height for item in current), panel.rect.height)
            if overlap >= min_height * self.config.row_overlap_tolerance:
                current.append(panel)
                current_bottom = max(current_bottom, panel.rect.bottom)
                continue

            # Inside a row, Japanese manga panels are read from right to left.
            rows.append(sorted(current, key=lambda item: -item.rect.x))
            current = [panel]
            current_bottom = panel.rect.bottom

        if current:
            rows.append(sorted(current, key=lambda item: -item.rect.x))
        return rows

    def _group_into_columns(self, panels: list[Panel]) -> list[list[Panel]]:
        """Group panels into columns using horizontal overlap."""
        ordered = sorted(panels, key=lambda panel: panel.rect.x, reverse=True)
        columns: list[list[Panel]] = []
        current: list[Panel] = []
        current_left = 10**9

        for panel in ordered:
            if not current:
                current = [panel]
                current_left = panel.rect.x
                current_right = panel.rect.right
                continue

            # Panels that share enough horizontal overlap belong to the same column.
            overlap = panel.rect.right - current_left
            min_width = min(min(item.rect.width for item in current), panel.rect.width)
            if overlap >= min_width * self.config.column_overlap_tolerance:
                current.append(panel)
                current_left = min(current_left, panel.rect.x)
                current_right = max(current_right, panel.rect.right)
                continue

            columns.append(sorted(current, key=lambda item: item.rect.y))
            current = [panel]
            current_left = panel.rect.x
            current_right = panel.rect.right

        if current:
            columns.append(sorted(current, key=lambda item: item.rect.y))
        return columns

    def _flatten_reading_order(self, node: LayoutNode) -> list[int]:
        """Flatten the layout tree into the final reading order sequence."""
        if node.kind == "leaf":
            if node.panel_id is None:
                raise ValueError("Leaf node is missing panel_id.")
            return [node.panel_id]
        ordered: list[int] = []
        for child in node.children:
            ordered.extend(self._flatten_reading_order(child))
        return ordered

    def _union_bounds(self, rects: list[Rect]) -> Rect:
        """Return a single rectangle covering every rectangle in the input list."""
        bounds = rects[0]
        for rect in rects[1:]:
            bounds = bounds.union(rect)
        return bounds
