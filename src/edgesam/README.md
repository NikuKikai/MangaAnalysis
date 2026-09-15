# Optional EdgeSAM Setup

EdgeSAM is an optional integration for local, non-commercial segmentation experiments.

This MIT-licensed repository does not include the EdgeSAM source code or EdgeSAM ONNX weights. The setup script downloads them locally from the upstream project and Hugging Face Space. Those downloaded files are governed by the EdgeSAM S-Lab License 1.0, which permits non-commercial use and redistribution under its terms. Commercial use requires permission from the EdgeSAM contributors.

Run from the project root:

```powershell
.venv\Scripts\python.exe src\edgesam\prepare_edgesam.py
```

The script performs these steps:

```text
clone EdgeSAM source to third_party/edgesam/
install it into the existing local uv .venv
download encoder/decoder ONNX weights to src/edgesam/weights/
generate a decoder with an extra masks_uint8 output
copy the browser models to web/public/models/
```

Generated/downloaded paths are intentionally ignored by git:

```text
third_party/edgesam/
src/edgesam/weights/
web/public/models/edge_sam_3x_encoder.onnx
web/public/models/edge_sam_3x_decoder.onnx
```
