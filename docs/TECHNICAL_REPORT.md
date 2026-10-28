# Technical Report - Pedestrian Instance Segmentation

> **Before submitting:** run the pipeline, then replace every `[FILL: ...]` with your real numbers/observations
> (all numbers are in `results/RESULTS.md`). Delete this note. Do not submit placeholders.

## 1. What problem does the project solve?
Given an image of a street scene, find **each pedestrian individually** and outline it with a pixel-accurate mask
(instance segmentation). Practical uses: people counting, crowd/safety analytics, privacy blurring of individuals,
driver-assistance perception. Instance (not semantic) segmentation is needed because touching/overlapping people must be
separated and counted.

## 2. Why this data, model and approach?
**Data - Penn-Fudan Pedestrian (170 images, 345 pedestrians).** Small enough to run *many* controlled experiments on a free
GPU (one run ~5-10 min), yet it has real instance masks, occlusion and varied pedestrian sizes. The task is clean (1 class),
so differences between experiments reflect the method, not label noise. Fixed split 120/20/30 (seed 42), identical for all runs;
validation for model selection, test reported once.

**Model - Mask R-CNN (ResNet-50-FPN), COCO-pretrained.** Two-stage detector + mask head gives detection *and* segmentation in one
model, a strong standard baseline, and *person* is already a COCO class, so transfer learning is highly effective with only 120 images.
Heads (box predictor, mask predictor) are replaced for 2 classes. A **MobileNetV3-FPN Mask R-CNN** is the efficiency-oriented alternative.

**Approach.** Fine-tune with SGD+momentum, 1-epoch linear warm-up, cosine LR, AMP, gradient clipping. The best epoch is chosen
on validation segm mAP. Metrics: COCO mAP@[.5:.95] and mAP@.5 (mask = primary, box = secondary), because they are the
standard, threshold-free ranking metrics for detection/segmentation; P/R/F1 at a fixed score threshold for the deployed operating point.

## 3. Experiments and results
All numbers: `results/RESULTS.md`. One factor changes at a time; same epochs, split and seed.

| ID | Change vs. previous / baseline | Question answered | Test segm mAP | Observation |
|---|---|---|---|---|
| E0 | random init, basic aug | Does transfer learning matter? | [FILL] | [FILL] |
| E1 | frozen backbone (heads only) | How much is gained by adapting features? | [FILL] | [FILL] |
| E2 | fine-tune top 3 stages, no aug | Standard baseline | [FILL] | [FILL] |
| E3 | + flip & colour jitter | Do light augmentations help with 120 images? | [FILL] | [FILL] |
| E4 | + photometric/zoom-out/IoU-crop | Does stronger augmentation help or hurt? | [FILL] | [FILL] |
| E5 | unfreeze all 5 stages | Is deeper adaptation better or does it overfit? | [FILL] | [FILL] |
| E6 | lr 0.0025 | LR sensitivity | [FILL] | [FILL] |
| E7 | AdamW 1e-4 | Optimiser sensitivity | [FILL] | [FILL] |
| E8 | MobileNetV3 Mask R-CNN | Accuracy/size cost of a light model | [FILL] | [FILL] |

Training curves: `results/<exp>/curves.png`. **Chosen model:** `[FILL: name]` (best validation segm mAP), test segm mAP `[FILL]`, bbox mAP `[FILL]`.
Caveat to state honestly: the val (20) and test (30) sets are small, so differences below ~[FILL, e.g. 1-2] mAP points are within noise;
a single seed was used per configuration (multiple seeds / k-fold CV would quantify variance).

### Error analysis (`results/<best>/error_analysis.json`, pictures in `error_analysis/`)
Operating point score >= 0.5: precision `[FILL]`, recall `[FILL]`, F1 `[FILL]`, mean matched mask IoU `[FILL]`.
False positives by type (duplicate / localisation / background): `[FILL]`. Recall by size (small/medium/large): `[FILL]`;
by crowdedness: `[FILL]`. Typical failures seen in the worst-image panels: `[FILL: e.g. heavily occluded people merged/missed,
small distant pedestrians, objects resembling people]`.

### Optimisation (`results/<best>/optimization.json`)
Measured on the test set, batch 1, accuracy **and** latency together.

| Variant | Size MB | Latency ms | Segm mAP | Verdict |
|---|---|---|---|---|
| fp32, short side 800 (baseline) | [FILL] | [FILL] | [FILL] | |
| fp32, 640 / 512 / 384 | [FILL] | [FILL] | [FILL] | [FILL: which resolution is the knee of the curve] |
| dynamic INT8 (Linear layers) | [FILL] | [FILL] | [FILL] | [FILL] |
| FP16 autocast (GPU) | [FILL] | [FILL] | [FILL] | [FILL] |
| ONNX Runtime | [FILL] | [FILL] | [FILL] | [FILL: or "export failed because ..."] |
| MobileNetV3 (E8) vs ResNet-50 | [FILL] | [FILL] | [FILL] | [FILL] |

Interpretation notes: Mask R-CNN resizes the short side to 800 px by default, while Penn-Fudan images are smaller, so the default *up-scales*;
reducing it is the cheapest speed-up. Dynamic INT8 only quantises `nn.Linear` (the box head), the conv backbone stays fp32, so gains are
modest - this is expected, not a bug. Fully quantising convs would need static PTQ/QAT, listed under future work.

## 4. Problems encountered and how they were solved
| Problem | Solution |
|---|---|
| Only 120 training images -> overfitting risk | COCO transfer learning, augmentation ablations, validation-based checkpoint selection |
| Comparing experiments fairly | Fixed splits independent of the seed; one factor changed per run; test set used once |
| Augmentation (crop/zoom) can create degenerate boxes/empty masks that crash or poison training | `SanitizeBoundingBoxes` removes boxes and their masks together |
| Non-finite loss / early divergence with SGD | 1-epoch linear warm-up, gradient clipping, AMP GradScaler, explicit finite-loss check with informative error |
| Model file (>100 MB) exceeds GitHub limit | fp16 checkpoint (`export_slim.py`) + release/HF link for fp32; `Predictor` casts back to fp32 |
| Mask R-CNN ONNX export is fragile (dynamic shapes, post-processing ops) | Isolated in a try/except: failure is recorded in results rather than breaking the pipeline |
| Free-tier sessions disconnect | Experiments are independent and skipped if finished (`metrics.json` exists) |
| [FILL: your own real problems, e.g. wrong mask dtype in metric, Colab OOM -> batch size 2 / AMP] | [FILL] |

## 5. Strengths and weaknesses
**Strengths:** complete reproducible pipeline (one command), controlled ablations, leakage-free evaluation, standard COCO metrics plus
operating-point analysis, error taxonomy, speed/size/accuracy trade-off table, working API + UI, unit tests for critical logic.

**Weaknesses / limits:** small dataset (single scene type: pedestrians on campus/street from Penn) -> weak evidence of generalisation to other
domains (night, fisheye, dense crowds); small val/test -> noisy comparisons; single seed per config; one class only; Mask R-CNN is relatively
heavy (not real-time on CPU); dynamic INT8 gives limited gains; no calibration of confidence scores.

## 6. What I would do with more time/resources
1. Evaluate cross-dataset (PASCAL VOC person / BDD100K) to measure domain shift; train on the union for robustness.
2. k-fold cross-validation + multiple seeds with confidence intervals for experiments.
3. Stronger/lighter architectures: YOLOv8-seg / RT-DETR-seg / Mask2Former; distillation from ResNet-50 into the MobileNet model.
4. True INT8 (static PTQ with calibration or QAT) and TensorRT/OpenVINO deployment; pruning.
5. Tune NMS / score thresholds on validation per use case; confidence calibration.
6. CI (lint + tests), experiment tracking (MLflow/W&B), data-version control, model monitoring hooks in the API, batch inference endpoint.
