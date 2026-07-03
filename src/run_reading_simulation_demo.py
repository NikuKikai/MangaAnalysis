from __future__ import annotations

import argparse
import json
import site
from pathlib import Path
from time import perf_counter

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
site.addsitedir(str(REPO_ROOT / ".venv" / "Lib" / "site-packages"))

from reading_simulation import ReadingSimulator, SimulationConfig
from reading_simulation.saliency_inference import SaliencyInference
from reading_simulation.visualize import save_visualization, show_visualization

# Default sample page used by the reading-simulation demo.
PAGE_IMAGE_PATH = REPO_ROOT / "samples" / "TARU_014.png"

CONFIG_PATH = REPO_ROOT / "configs" / "saliency" / "stage1_salicon_pretrained_512_v2w075.toml"
CHECKPOINT_PATH = REPO_ROOT / "runs" / "saliency" / "stage1_salicon_pretrained_512_v2w075" / "checkpoints" / "epoch_004.pt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "samples" / "reading_simulation_output"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the reading-simulation demo."""
    parser = argparse.ArgumentParser(description="Run manga reading simulation with selectable strategies.")
    parser.add_argument("--image", type=Path, default=PAGE_IMAGE_PATH, help="Input manga page path.")
    parser.add_argument(
        "--strategy",
        choices=["saliency_only", "panel_guided"],
        default="panel_guided",
        help="Reading-simulation strategy to run.",
    )
    parser.add_argument("--steps", type=int, default=None, help="Optional maximum number of transition steps.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for all generated files.")
    parser.add_argument("--show", action="store_true", help="Show the visualization in an OpenCV window when available.")
    return parser.parse_args()


def build_output_paths(output_dir: Path, strategy: str) -> tuple[Path, Path]:
    """Build the canonical image and JSON output paths inside one output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{strategy}_result.png", output_dir / f"{strategy}_result.json"


def main() -> None:
    """Execute the selected reading-simulation strategy on one page."""
    args = parse_args()
    output_image_path, output_json_path = build_output_paths(args.output_dir, args.strategy)
    inference_start = perf_counter()
    inference = SaliencyInference(str(CONFIG_PATH), str(CHECKPOINT_PATH))
    inference_elapsed = perf_counter() - inference_start
    simulator = ReadingSimulator(
        inference,
        SimulationConfig(
            strategy=args.strategy,
            default_roi_half_size_ratio=0.25,
            clear_radius_ratio=0.06,
            max_blur_strength=7.0,
            history_sigma_ratio=0.047,
            history_decay=0.94,
            history_alpha=3.0,
            distance_sigma_ratio=0.183,
            threshold_score=0.15,
            nms_radius_ratio=0.013,
            top_k=8,
            steps=args.steps,
        ),
    )
    with Image.open(args.image) as page_image:
        page_width, page_height = page_image.size
    initial_fixation = (page_width * 0.9, page_height * 0.1)
    simulate_start = perf_counter()
    result = simulator.simulate(str(args.image), initial_fixation=initial_fixation)
    simulate_elapsed = perf_counter() - simulate_start

    # Persist both the structured trace and the default visualization for inspection.
    save_start = perf_counter()
    save_visualization(result, str(output_image_path))
    save_elapsed = perf_counter() - save_start
    output_json_path.write_text(json.dumps(result.to_json(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Strategy: {result.strategy}")
    print(f"Fixations: {len(result.fixations)}")
    if result.analysis is not None:
        print(f"Reading fluidity mean score: {result.analysis.mean_score:.4f}")
        print(f"Reading fluidity max score: {result.analysis.max_score:.4f}")
    print(f"Model init: {inference_elapsed:.3f}s")
    print(f"Simulation: {simulate_elapsed:.3f}s")
    print(f"Visualization save: {save_elapsed:.3f}s")
    if result.timing_summary:
        print("Timing summary:")
        for key, elapsed in sorted(result.timing_summary.items(), key=lambda item: item[1], reverse=True):
            count = result.timing_counts.get(key, 1)
            average = elapsed / max(count, 1)
            print(f"  {key}: total={elapsed:.3f}s count={count} avg={average:.3f}s")
    print(f"Saved image to: {output_image_path}")
    print(f"Saved json to: {output_json_path}")
    if args.show:
        show_visualization(result)


if __name__ == "__main__":
    main()
