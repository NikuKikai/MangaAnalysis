from __future__ import annotations

from pathlib import Path

from reading_simulation.saliency_inference import SaliencyInference
from reading_simulation.simulator import ReadingSimulator, SimulationConfig
from reading_simulation.visualize import save_visualization, show_visualization


REPO_ROOT = Path(__file__).resolve().parent.parent
PAGE_IMAGE_PATH = REPO_ROOT / "samples" / "Q_005_006.png"
PAGE_IMAGE_PATH = REPO_ROOT / "samples" / "TARU_014.png"

CONFIG_PATH = REPO_ROOT / "configs" / "saliency" / "stage1_salicon_gpu_pretrained.toml"
CHECKPOINT_PATH = REPO_ROOT / "runs" / "saliency" / "stage1_salicon_gpu_pretrained" / "checkpoints" / "epoch_004.pt"
OUTPUT_IMAGE_PATH = REPO_ROOT / "samples" / "reading_simulation_result.png"
OUTPUT_JSON_PATH = REPO_ROOT / "samples" / "reading_simulation_result.json"


def main() -> None:
    inference = SaliencyInference(str(CONFIG_PATH), str(CHECKPOINT_PATH))
    simulator = ReadingSimulator(
        inference,
        SimulationConfig(
            roi_size_ratio=0.65,
            clear_radius_ratio=0.1,
            blur_level_count=8,
            max_blur_strength=16.0,
            history_sigma_ratio=0.047,
            history_alpha=2.0,
            distance_sigma_ratio=0.183,
            start_region_radius_ratio=0.3,
            start_corner_weight=0.35,
            threshold_ratio=0.55,
            nms_radius_ratio=0.013,
            top_k=8,
            steps=12,
        ),
    )
    result = simulator.simulate(str(PAGE_IMAGE_PATH))
    save_visualization(result, str(OUTPUT_IMAGE_PATH))
    OUTPUT_JSON_PATH.write_text(__import__("json").dumps(result.to_json(), indent=2), encoding="utf-8")
    print(f"Saved image to: {OUTPUT_IMAGE_PATH}")
    print(f"Saved json to: {OUTPUT_JSON_PATH}")
    show_visualization(result)


if __name__ == "__main__":
    main()
