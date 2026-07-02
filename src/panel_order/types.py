from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Rect:
    """Axis-aligned rectangle in page coordinates."""

    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        """Return the exclusive right edge."""
        return self.x + self.width

    @property
    def bottom(self) -> int:
        """Return the exclusive bottom edge."""
        return self.y + self.height

    @property
    def area(self) -> int:
        """Return the rectangle area."""
        return self.width * self.height

    @property
    def center_x(self) -> float:
        """Return the horizontal center coordinate."""
        return self.x + self.width * 0.5

    @property
    def center_y(self) -> float:
        """Return the vertical center coordinate."""
        return self.y + self.height * 0.5

    def intersection(self, other: Rect) -> Rect | None:
        """Return the overlap rectangle, or None when the rectangles do not intersect."""
        left = max(self.x, other.x)
        top = max(self.y, other.y)
        right = min(self.right, other.right)
        bottom = min(self.bottom, other.bottom)
        if right <= left or bottom <= top:
            return None
        return Rect(left, top, right - left, bottom - top)

    def union(self, other: Rect) -> Rect:
        """Return the smallest rectangle covering both rectangles."""
        left = min(self.x, other.x)
        top = min(self.y, other.y)
        right = max(self.right, other.right)
        bottom = max(self.bottom, other.bottom)
        return Rect(left, top, right - left, bottom - top)

    def iou(self, other: Rect) -> float:
        """Return the intersection-over-union ratio against another rectangle."""
        intersection = self.intersection(other)
        if intersection is None:
            return 0.0
        union_area = self.area + other.area - intersection.area
        return float(intersection.area) / float(max(union_area, 1))

    def contains_ratio(self, other: Rect) -> float:
        """Return how much of another rectangle is covered by this rectangle."""
        intersection = self.intersection(other)
        if intersection is None:
            return 0.0
        return float(intersection.area) / float(max(other.area, 1))

    def to_json(self) -> dict:
        """Serialize the rectangle into a JSON-friendly dictionary."""
        return {
            "x": int(self.x),
            "y": int(self.y),
            "width": int(self.width),
            "height": int(self.height),
        }


@dataclass(slots=True)
class Panel:
    """Detected manga panel with a stable identifier and confidence score."""

    panel_id: int
    rect: Rect
    score: float

    def to_json(self) -> dict:
        """Serialize the panel into a JSON-friendly dictionary."""
        return {
            "panel_id": int(self.panel_id),
            "rect": self.rect.to_json(),
            "score": float(self.score),
        }


@dataclass(slots=True)
class LayoutNode:
    """Node in the recursive panel layout tree."""

    kind: str
    bounds: Rect
    panel_id: int | None = None
    children: list["LayoutNode"] = field(default_factory=list)

    def to_json(self) -> dict:
        """Serialize the layout node into a JSON-friendly dictionary."""
        payload = {
            "kind": self.kind,
            "bounds": self.bounds.to_json(),
        }
        if self.panel_id is not None:
            payload["panel_id"] = int(self.panel_id)
        if self.children:
            payload["children"] = [child.to_json() for child in self.children]
        return payload


@dataclass(slots=True)
class AnalysisResult:
    """Full page analysis output for panel detection and reading order."""

    page_path: str
    image_width: int
    image_height: int
    panels: list[Panel]
    layout_tree: LayoutNode
    reading_order: list[int]

    def to_json(self) -> dict:
        """Serialize the analysis result into a JSON-friendly dictionary."""
        return {
            "page_path": self.page_path,
            "image_width": int(self.image_width),
            "image_height": int(self.image_height),
            "panels": [panel.to_json() for panel in self.panels],
            "layout_tree": self.layout_tree.to_json(),
            "reading_order": [int(panel_id) for panel_id in self.reading_order],
        }
