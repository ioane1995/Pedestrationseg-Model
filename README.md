# Pedestrian Instance Segmentation - End-to-End Computer Vision Project

Detects **and segments every pedestrian** in an image (instance segmentation) with a fine-tuned **Mask R-CNN**,
and ships the full ML workflow: data preparation -> training -> ablation experiments -> evaluation & error analysis
-> optimisation (speed/size) -> REST API + web demo.

* **Dataset:** [Penn-Fudan Pedestrian](https://www.cis.upenn.edu/~jshi/ped_html/) (170 images, 345 pedestrians). Fixed split: **120 train / 20 val / 30 test**.
* **Models:** Mask R-CNN ResNet-50-FPN (COCO-pretrained, fine-tuned) and a lightweight MobileNetV3-FPN Mask R-CNN.
* **Metrics:** COCO mAP@[.5:.95] and mAP@.5 for masks (primary) and boxes; precision/recall/F1, mask IoU, latency, size.
* **Runs on free resources:** Google Colab / Kaggle T4 (whole pipeline ~2 h). No paid services.

> Final numbers live in [`results/RESULTS.md`](results/RESULTS.md) (auto-generated) and in the report
> [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md).

## Project structure
```
src/
  data.py            dataset download, fixed splits, Dataset class, augmentations
  models.py          model factory (maskrcnn_r50, maskrcnn_mobilenet) + head replacement
  engine.py          train loop (AMP, warm-up, grad clipping), COCO evaluation, latency benchmark
  train.py           one experiment -> results/<name>/{best.pt,history.csv,metrics.json,curves.png}
  inference.py       Predictor class + CLI for new images
  matching.py        numpy instance matching (TP/FP/FN, FP taxonomy) - unit tested
  error_analysis.py  P/R/F1, threshold sweep, FP types, recall by size/crowdedness, worst-image pictures
  optimize.py        resolution sweep, dynamic INT8, FP16 autocast, ONNX Runtime
  visualize.py       mask/box overlay
app/api.py           FastAPI service (/predict, /predict/image, /health)
app/demo.py          Gradio web UI
scripts/             run_experiments.sh, run_all.sh, make_report_tables.py, export_slim.py
tests/               pytest: matching logic, splits, forward/backward of both models
notebooks/run_on_colab.ipynb   one-click pipeline on a free GPU
docs/                TECHNICAL_REPORT.md, AI_USAGE.md, INTERVIEW_PREP.md
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.10+
pip install -r requirements.txt                         # torch, torchvision, torchmetrics, pycocotools, fastapi, gradio, onnxruntime ...
pytest -q tests                                         # optional sanity check
```

## Reproduce everything (recommended: Colab/Kaggle GPU, see `notebooks/run_on_colab.ipynb`)
```bash
bash scripts/run_all.sh        # experiments -> best by VAL mAP -> error analysis -> optimisation -> results/RESULTS.md
```
Single experiment:
```bash
python -m src.train --name my_run --model maskrcnn_r50 --trainable-layers 3 --aug basic --epochs 15
```
The dataset is downloaded automatically into `data/`. Interrupted runs resume at experiment granularity (finished ones are skipped).
Experiments (see `scripts/run_experiments.sh`): E0 scratch, E1 frozen backbone, E2 fine-tune no-aug, E3 +basic aug, E4 +strong aug,
E5 full unfreeze, E6 lower LR, E7 AdamW, E8 MobileNetV3 Mask R-CNN.

## Getting the trained model
`run_all.sh` writes `models/best.pt` (fp32, ~170 MB) and `models/best_fp16.pt` (~85 MB, fits GitHub's 100 MB limit).
The model is published at: **`<PUT YOUR RELEASE / HUGGING FACE / DRIVE LINK HERE>`** - download it to `models/best.pt`.
If the link is unavailable, retrain with `bash scripts/run_all.sh` (deterministic splits, fixed seed).

## Use it
```bash
python -m src.inference --input some_image.jpg --output outputs        # CLI -> outputs/*_pred.png
python -m app.demo                                                       # web UI on http://127.0.0.1:7860  (--share for public link)
uvicorn app.api:app --port 8000                                          # REST API, Swagger docs at /docs
curl -F "file=@some_image.jpg" "http://localhost:8000/predict?score_thresh=0.5"
curl -F "file=@some_image.jpg" http://localhost:8000/predict/image -o result.png
```
`/predict` returns `{count, inference_ms, instances:[{box, score}]}` (add `include_masks=true` for COCO-RLE masks).
Docker (CPU): `docker build -t pedseg . && docker run -p 8000:8000 pedseg` (uses `models/best_fp16.pt`).

## Key design decisions (details in the report)
* Same fixed split for all runs; model selection on **val**, test touched once -> no test leakage.
* Transfer learning from COCO (person is a COCO class) - ablated against training from scratch.
* Ablations change **one factor at a time**; metrics are COCO mAP (the standard), plus P/R/F1 at a chosen operating point.
* Optimisation reports accuracy **and** latency/size together, so speed-ups are never "free" claims.
