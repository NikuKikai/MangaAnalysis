

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

- The package is now split into:
  - shared saliency infrastructure in `config.py`, `engine.py`, `types.py`, `analysis.py`
  - ROI extraction and GPU preprocessing in `retina.py`
  - strategy implementations in `/src/reading_simulation/strategies/`

- GPU execution:
  - candidate rescoring and NMS run on GPU with PyTorch tensors
  - history map residency
    - the simulation now keeps `history_map` as a device-side `torch.Tensor` during the algorithm flow
    - it is materialized only when building the final `SimulationResult`
  - ROI preprocessing
    - the preprocessing path is fixed to the `wgpu` compute-shader implementation aligned with the web version

- Available strategies:
  - `saliency_only`
    - preserves the previous behavior
    - caller supplies the initial fixation
    - default ROI size is `2 * page_height * default_roi_half_size_ratio`
    - history is updated with `history = history * decay + gaussian(fixation)`
    - candidate NMS runs on the final rescored map, not raw saliency
    - candidates are filtered by absolute `threshold_score`
    - if no candidate survives inside the local ROI, the step falls back to a full-page square ROI
  - `panel_guided`
    - uses detected manga panel order from `/src/panel_order/`
    - still uses the original saliency ROI settings
    - only accepts transitions into the current panel, or the next panel when the current panel is exhausted
    - writes history only inside the current panel
    - retries the next panel with a full-page ROI when the local ROI cannot enter it
    - falls back to the next panel center when even the full-page retry finds no next-panel candidate
    - records lightweight per-step state and computes a reading-fluidity score from the best off-route candidate

- Visualization:
  - draws the realized scanpath
  - draws one arrow per fixation toward the best candidate outside the current and next panels
  - uses arrow thickness to encode that candidate score
  - colors the arrow red when that off-route score is larger than the actual next-fixation score

- Demo:
```bash
python src/run_reading_simulation_demo.py --strategy panel_guided --steps 12
```

- Temporary timing instrumentation:
  - the demo currently prints per-stage timings
  - `SimulationResult.to_json()` also includes `timing_summary` and `timing_counts`
  - recent 2-step `panel_guided` run after optimization:
    - total simulation time: about `0.76s`
    - candidate rescoring: about `0.04s / ROI`
    - ROI preprocessing: about `0.07s / ROI`


## Reading Simulation Demo (web)

`/web/`

- Deploy: need to remove ORT WASM file under `/dist/assets` manually.


## Manga Panel Order (python)

`/src/panel_order/`

- Traditional CV2-based pipeline, no external model required.
- Detects rectangular manga panels from page borders.
- Builds a recursive XY-cut layout tree.
- Derives Japanese manga reading order:
  - rows are read top to bottom
  - panels inside a row are read right to left
- Best suited to pages with explicit panel borders and visible gutters.

- Demo:
```bash
python src/run_panel_order_demo.py
```

- Output:
  - `samples/panel_order_result.json`
  - `samples/panel_order_result.png`


