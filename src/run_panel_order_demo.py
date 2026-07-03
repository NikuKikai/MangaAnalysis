from __future__ import annotations

import argparse
import json
import site
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
site.addsitedir(str(REPO_ROOT / ".venv" / "Lib" / "site-packages"))

import cv2

from panel_order import AnalysisResult, MangaLayoutAnalyzer, PanelDetector
from panel_order.visualize import save_panel_overlay

# Default sample page used by the demo entry point.
DEFAULT_IMAGE_PATH = REPO_ROOT / "samples" / "TARU_014.png"
# Default output directory for structured analysis results and overlay previews.
DEFAULT_OUTPUT_DIR = REPO_ROOT / "samples" / "panel_order_output"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the panel-order demo."""
    parser = argparse.ArgumentParser(description="Detect manga panels, infer a layout tree, and derive Japanese reading order.")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE_PATH, help="Input manga page path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for generated JSON and overlay files.")
    return parser.parse_args()


def build_output_paths(output_dir: Path) -> tuple[Path, Path]:
    """Build canonical output paths for panel-order demo artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "panel_order_result.json", output_dir / "panel_order_result.png"


def analyze_page(image_path: Path) -> tuple[AnalysisResult, cv2.typing.MatLike]:
    """Run the full panel detection and reading-order analysis for one page."""
    # Load the page image first so downstream code only receives a valid OpenCV image.
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")

    # Run panel detection before layout analysis because the tree is built on panel rectangles.
    detector = PanelDetector()
    panels = detector.detect(image)
    if not panels:
        raise RuntimeError("No manga panels were detected.")

    # Build the recursive layout tree and flatten it into Japanese reading order.
    layout_analyzer = MangaLayoutAnalyzer()
    layout_tree, reading_order = layout_analyzer.build_layout(panels)
    result = AnalysisResult(
        page_path=str(image_path),
        image_width=int(image.shape[1]),
        image_height=int(image.shape[0]),
        panels=panels,
        layout_tree=layout_tree,
        reading_order=reading_order,
    )
    return result, image


def main() -> None:
    """Execute the demo and write both JSON and overlay outputs."""
    args = parse_args()
    output_json_path, output_image_path = build_output_paths(args.output_dir)
    result, image = analyze_page(args.image)
    output_json_path.write_text(json.dumps(result.to_json(), indent=2, ensure_ascii=False), encoding="utf-8")
    save_panel_overlay(image, result, str(output_image_path))
    print(f"Detected {len(result.panels)} panels.")
    print("Reading order:", result.reading_order)
    print(f"Saved JSON to: {output_json_path}")
    print(f"Saved overlay to: {output_image_path}")


if __name__ == "__main__":
    main()
