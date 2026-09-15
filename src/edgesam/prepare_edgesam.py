from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from add_uint8_mask_output import add_uint8_mask_output


REPO_URL = "https://github.com/chongzhou96/EdgeSAM.git"
HF_REPO_ID = "chongzhou/EdgeSAM"
HF_REPO_TYPE = "space"
ENCODER_FILENAME = "edge_sam_3x_encoder.onnx"
DECODER_FILENAME = "edge_sam_3x_decoder.onnx"
DECODER_UINT8_FILENAME = "edge_sam_3x_decoder_uint8.onnx"


def run_command(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def ensure_edgesam_source(project_root: Path, source_dir: Path) -> None:
    if (source_dir / "setup.py").exists():
        return
    source_dir.parent.mkdir(parents=True, exist_ok=True)
    run_command(["git", "clone", "--depth", "1", REPO_URL, str(source_dir)], project_root)


def download_weights(project_root: Path, weights_dir: Path) -> None:
    from huggingface_hub import hf_hub_download

    weights_dir.mkdir(parents=True, exist_ok=True)
    for filename in (ENCODER_FILENAME, DECODER_FILENAME):
        hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            filename=f"weights/{filename}",
            local_dir=project_root / "src" / "edgesam",
        )


def install_edgesam(project_root: Path, source_dir: Path) -> None:
    run_command(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(project_root / ".venv" / "Scripts" / "python.exe"),
            "-e",
            str(source_dir),
            "--cache-dir",
            str(project_root / ".uv-cache"),
        ],
        project_root,
    )
    run_command(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(project_root / ".venv" / "Scripts" / "python.exe"),
            "huggingface-hub",
            "loralib",
            "yacs",
            "timm",
            "kornia",
            "--cache-dir",
            str(project_root / ".uv-cache"),
        ],
        project_root,
    )


def copy_frontend_models(project_root: Path, weights_dir: Path, frontend_models_dir: Path) -> None:
    frontend_models_dir.mkdir(parents=True, exist_ok=True)
    (frontend_models_dir / ENCODER_FILENAME).write_bytes((weights_dir / ENCODER_FILENAME).read_bytes())
    (frontend_models_dir / DECODER_FILENAME).write_bytes((weights_dir / DECODER_UINT8_FILENAME).read_bytes())


def prepare_edgesam(project_root: Path) -> None:
    source_dir = project_root / "third_party" / "edgesam"
    weights_dir = project_root / "src" / "edgesam" / "weights"
    frontend_models_dir = project_root / "web" / "public" / "models"

    ensure_edgesam_source(project_root, source_dir)
    install_edgesam(project_root, source_dir)
    download_weights(project_root, weights_dir)
    add_uint8_mask_output(weights_dir / DECODER_FILENAME, weights_dir / DECODER_UINT8_FILENAME)
    copy_frontend_models(project_root, weights_dir, frontend_models_dir)

    print(f"EdgeSAM source: {source_dir}")
    print(f"EdgeSAM weights: {weights_dir}")
    print(f"Frontend models: {frontend_models_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local EdgeSAM source and ONNX models.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Path to the MangaAnalysis project root.",
    )
    args = parser.parse_args()
    prepare_edgesam(args.project_root.resolve())


if __name__ == "__main__":
    main()
