

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

`/src/reading_simulation/`

- The Python simulator now follows the current `web` algorithm:
  - caller supplies the initial fixation
  - default ROI size is `2 * page_height * default_roi_half_size_ratio`
  - history is updated with `history = history * decay + gaussian(fixation)`
  - candidate NMS runs on the final rescored map, not raw saliency
  - candidates are filtered by absolute `threshold_score`
  - if no candidate survives inside the local ROI, the step falls back to a full-page square ROI

- Demo:
```bash
uv run python src/run_reading_simulation_demo.py
```


## Reading Simulation Demo (web)

`/web/`

- Deploy: need to remove ORT WASM file under `/dist/assets` manually.


