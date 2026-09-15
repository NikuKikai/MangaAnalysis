# Local EdgeSAM Deployment

This folder contains a local checkout of the official EdgeSAM repository, installed into this project's existing uv-managed virtual environment.

## Installed Package

EdgeSAM is installed as an editable package from this folder:

```powershell
uv pip install --python .venv\Scripts\python.exe -e src\edgesam --cache-dir .uv-cache
```

Small runtime dependencies required by the current import path were installed into the same local environment:

```powershell
uv pip install --python .venv\Scripts\python.exe loralib yacs timm kornia --cache-dir .uv-cache
```

## ONNX Weights

The browser-friendly ONNX weights are stored here:

```text
src/edgesam/weights/edge_sam_3x_encoder.onnx
src/edgesam/weights/edge_sam_3x_decoder.onnx
```

They were downloaded from the official Hugging Face Space:

```text
chongzhou/EdgeSAM
```

## Smoke Test

Run this from the project root:

```powershell
.venv\Scripts\python.exe -c "from edge_sam.onnx import SamPredictorONNX; p=SamPredictorONNX('src/edgesam/weights/edge_sam_3x_encoder.onnx','src/edgesam/weights/edge_sam_3x_decoder.onnx'); print('onnx predictor ok')"
```

Expected output:

```text
onnx predictor ok
```

## Notes

The upstream `edge_sam.modeling.sam` module was patched so optional `mmdet/mmengine` detection-head imports are only required when those heads are explicitly used. Basic EdgeSAM and ONNX predictor imports do not need those heavy dependencies.
