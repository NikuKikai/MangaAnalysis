from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_SRC = Path(__file__).resolve().parents[2]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

import torch
from torch import nn

from saliency.config import load_config
from saliency.model import SaliencyUNet


# Keep these names stable for the WebGPU client and shader pipeline.
INPUT_NAME = "input_image"
OUTPUT_NAME = "saliency_map"


class SigmoidSaliencyWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the saliency checkpoint to ONNX for web inference.",
    )
    parser.add_argument(
        "--config",
        default="configs/saliency/stage1_salicon_gpu_pretrained.toml",
        help="Path to the model config file.",
    )
    parser.add_argument(
        "--checkpoint",
        default="runs/saliency/stage1_salicon_gpu_pretrained/checkpoints/epoch_004.pt",
        help="Path to the checkpoint file.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/saliency/stage1_salicon_gpu_pretrained/onnx",
        help="Directory for exported ONNX files.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--export-int8",
        action="store_true",
        help="Also export an experimental INT8 model when onnxruntime quantization is available.",
    )
    return parser.parse_args()


def build_model(config_path: str, checkpoint_path: str) -> tuple[nn.Module, int]:
    config = load_config(config_path)

    # Do not request pretrained encoder weights during export.
    # The checkpoint already contains the trained encoder parameters we need.
    model = SaliencyUNet(
        encoder_pretrained=False,
        base_channels=config.model.base_channels,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    wrapped = SigmoidSaliencyWrapper(model)
    wrapped.eval()
    return wrapped, config.dataset.input_size


def export_fp32_model(
    model: nn.Module,
    input_size: int,
    output_path: Path,
    opset: int,
) -> None:
    dummy = torch.randn(1, 3, input_size, input_size, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes=None,
        dynamo=False,
    )


def export_fp16_model_from_onnx(fp32_path: Path, fp16_path: Path) -> None:
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("FP16 export requires the 'onnx' package to be installed.") from exc

    try:
        from onnxconverter_common.float16 import convert_float_to_float16
    except ImportError as exc:
        raise RuntimeError(
            "FP16 export requires 'onnxconverter-common'. "
            "Install it in the existing uv environment if you want FP16 conversion."
        ) from exc

    model = onnx.load(fp32_path)
    model_fp16 = convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, fp16_path)


def export_int8_model(fp32_path: Path, int8_path: Path) -> None:
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:
        raise RuntimeError(
            "INT8 export requires 'onnxruntime' with quantization support."
        ) from exc

    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model, input_size = build_model(args.config, args.checkpoint)

    name_prefix = f"saliency_stage1_{input_size}_sigmoid"
    fp32_path = output_dir / f"{name_prefix}_fp32.onnx"
    fp16_path = output_dir / f"{name_prefix}_fp16.onnx"
    int8_path = output_dir / f"{name_prefix}_int8.onnx"

    export_fp32_model(model, input_size, fp32_path, args.opset)
    print(f"Exported FP32 ONNX to: {fp32_path}")
    print(f"Input name: {INPUT_NAME}")
    print(f"Output name: {OUTPUT_NAME}")
    print(f"Fixed input shape: 1x3x{input_size}x{input_size}")
    print(f"Fixed output shape: 1x1x{input_size}x{input_size}")

    try:
        export_fp16_model_from_onnx(fp32_path, fp16_path)
        print(f"Exported FP16 ONNX to: {fp16_path}")
    except Exception as exc:
        print(f"Skipped FP16 export: {exc}")

    if args.export_int8:
        try:
            export_int8_model(fp32_path, int8_path)
            print(f"Exported INT8 ONNX to: {int8_path}")
        except Exception as exc:
            print(f"Skipped INT8 export: {exc}")


if __name__ == "__main__":
    main()
