# Interview preparation (read this, then READ THE CODE - you must be able to explain and modify everything)

Spend your time on: `src/data.py`, `src/models.py`, `src/engine.py`, `src/train.py`, `src/matching.py`. Run the code once, step through it with a debugger, change things.

## 1. 60-second pitch
"Instance segmentation of pedestrians on Penn-Fudan with Mask R-CNN fine-tuned from COCO. I ran nine controlled experiments (transfer learning,
frozen vs unfrozen backbone, augmentation strength, LR, optimiser, light architecture), picked the model on validation mAP, evaluated once on
a held-out test set, analysed errors by type/size/crowding, and optimised speed and size, always reporting accuracy next to latency. It is served by a FastAPI endpoint and a Gradio UI."

## 2. Questions you must answer
**Data.** Why 120/20/30 and why fixed? (comparability + no leakage; seed independent of training seed.) How are instance masks built? (`mask == id` for each non-zero id -> [N,H,W]; boxes from masks via `masks_to_boxes`.) Why label 1 and 2 classes? (0 = background in torchvision.) What does `SanitizeBoundingBoxes` do and why only for `strong` aug?
**Model.** What are the 4 Mask R-CNN stages (backbone+FPN -> RPN -> RoIAlign -> box/class head + mask head)? What are the losses (rpn objectness, rpn box, roi classifier, roi box, mask BCE)? Why RoIAlign vs RoIPool? Why replace predictors? What does `trainable_backbone_layers` do? Why is BatchNorm frozen (batch size 2 -> unreliable statistics)? How is the MobileNet variant assembled and what is pretrained in it?
**Training.** Why warm-up, cosine, grad clipping, AMP/GradScaler? Why SGD lr 0.005 with batch 2? What if loss is NaN? Why select the epoch on val, not test? What does overfitting look like in `curves.png`?
**Metrics.** Define IoU, precision, recall, AP, mAP@[.5:.95], mAP@.5. Difference between box mAP and mask mAP? Why can F1 at 0.5 and mAP rank models differently? How does greedy matching work and what is a "duplicate" FP?
**Results.** Which factor mattered most and why? Which differences are inside the noise (20-30 images!)? What would you do to prove significance (seeds, CV, bootstrap)?
**Optimisation.** Why does lowering the short side speed up inference? Why is dynamic INT8 only a small gain? What is the difference between dynamic, static PTQ and QAT? Why might ONNX export of Mask R-CNN fail? Why use autocast FP16 rather than `model.half()`?
**Deployment.** What happens on a non-image upload? How would you scale the API (batching, workers, GPU, queue)? Where is the model loaded and why once (`lru_cache`)?
**Weaknesses.** Small single-domain data, small eval sets, one seed, Mask R-CNN is heavy, scores uncalibrated.

## 3. "New requirement" drills (practise doing these in <15 min each)
| Likely request | Where to change |
|---|---|
| Add a second class (e.g. car) / new dataset | `data.py` labels + `num_classes` in `train.py`/`models.py`; metric `labels`; palette/text in `visualize.py` |
| Return only people larger than N px / ROI filter | filter in `Predictor.predict` or API |
| Count people per image / video input | API already returns `count`; loop frames with OpenCV and call `Predictor.predict` |
| Change IoU matching threshold / NMS threshold | `match_instances(iou_thr)`; `model.roi_heads.nms_thresh` |
| Add class weights / focal loss / different LR per layer | param groups in `train.py` |
| Add early stopping | break in the epoch loop when val mAP did not improve for k epochs |
| Add a metric (e.g. mean mask IoU, PQ) | `engine.evaluate_fn` / `error_analysis.py` |
| Export boxes in COCO JSON format | in `/predict` response builder |
| Batch endpoint | accept list of files, call model once with a list of tensors |
| Replace backbone (e.g. ResNet-101 / ConvNeXt) | `models.py` |
| Find the planted bug | check mask dtype, box format (xyxy vs xywh), channel order, normalisation, train/eval mode, split leakage, LR scheduler step placement |

## 4. Sanity checks to run yourself (and mention in the interview)
1. Feed ground truth as predictions into `evaluate_fn` -> mAP must be ~1.0.
2. Overfit 5 images for 30 epochs -> loss near 0 (proves the pipeline can learn).
3. Print a training sample after augmentation (`overlay`) to see that boxes/masks still align.
4. Check split overlap with `set(train) & set(test) == set()` (it's in the tests).
