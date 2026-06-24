

## Saliency Model Training

`/src/saliency/`


- Train:
``` ```

- Export onnx
```bash
uv run python src/saliency/scripts/export_saliency_onnx.py <FolderPathOfRun> --export-int8
# Currently used in demo
uv run python src/saliency/scripts/export_saliency_onnx.py runs/saliency/stage1_salicon_pretrained_512_v2w075
```

## Reading Simulation (python)


## Reading Simulation Demo (web)

`/web/`

- Deploy: need to remove ORT WASM file under `/dist/assets` manually.


