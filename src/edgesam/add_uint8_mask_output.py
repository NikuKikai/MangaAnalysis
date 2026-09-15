from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def add_uint8_mask_output(input_path: Path, output_path: Path) -> None:
    model = onnx.load(input_path)
    graph = model.graph

    if not any(output.name == "masks" for output in graph.output):
        raise ValueError("Expected decoder graph to have an output named 'masks'.")
    if any(output.name == "masks_uint8" for output in graph.output):
        raise ValueError("Decoder graph already has an output named 'masks_uint8'.")

    graph.initializer.append(
        helper.make_tensor(
            name="masks_uint8_scale_255",
            data_type=TensorProto.FLOAT,
            dims=[],
            vals=[255.0],
        )
    )

    graph.node.extend(
        [
            helper.make_node(
                "Sigmoid",
                inputs=["masks"],
                outputs=["masks_sigmoid"],
                name="MasksSigmoid",
            ),
            helper.make_node(
                "Mul",
                inputs=["masks_sigmoid", "masks_uint8_scale_255"],
                outputs=["masks_scaled_255"],
                name="MasksScale255",
            ),
            helper.make_node(
                "Cast",
                inputs=["masks_scaled_255"],
                outputs=["masks_uint8"],
                name="MasksCastUint8",
                to=TensorProto.UINT8,
            ),
        ]
    )

    masks_output = next(output for output in graph.output if output.name == "masks")
    graph.output.append(
        helper.make_tensor_value_info(
            "masks_uint8",
            TensorProto.UINT8,
            [
                dim.dim_value if dim.dim_value > 0 else dim.dim_param or None
                for dim in masks_output.type.tensor_type.shape.dim
            ],
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model)
    onnx.save(model, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add a uint8 soft-mask output to the EdgeSAM decoder ONNX model.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("src/edgesam/weights/edge_sam_3x_decoder.onnx"),
        help="Path to the original EdgeSAM decoder ONNX model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/edgesam/weights/edge_sam_3x_decoder_uint8.onnx"),
        help="Path to write the decoder ONNX model with the extra masks_uint8 output.",
    )
    args = parser.parse_args()
    add_uint8_mask_output(args.input, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
