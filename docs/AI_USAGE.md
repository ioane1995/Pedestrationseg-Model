# Use of AI tools

> **Edit this so it is 100% true for you** - the interviewers will ask. The draft below assumes you used an AI assistant (Claude)
> to scaffold the project and then verified it. Remove/adjust anything that does not match what you really did.

**Tools:** Claude (Anthropic) as a coding/design assistant.

**What I used it for**
- Proposing the project scope (Penn-Fudan instance segmentation, Mask R-CNN) and the experiment plan (ablations).
- Drafting the initial code structure (data pipeline, training loop, evaluation, API/demo) and documentation templates.
- Explaining unfamiliar parts (torchvision v2 transforms, COCO metrics, Mask R-CNN losses, quantisation, ONNX export).

**What I did myself / how I verified AI output**
- Read and ran every module; I can explain the data flow, loss terms, metrics and every hyper-parameter (see `docs/INTERVIEW_PREP.md`).
- Unit tests (`pytest`) for matching logic, splits, and forward/backward passes of both models.
- Checked against official sources: torchvision detection tutorial / docs, COCO evaluation protocol, torchmetrics docs.
- Sanity checks: no overlap between train/val/test; metrics on a trivial case (GT as prediction => mAP ~ 1.0) `[FILL if you did it]`;
  visual inspection of predictions and of the worst-case panels.
- Experimental claims in the report come only from my own runs (`results/`), not from the assistant.
- `[FILL: bugs you found in AI-generated code and how you fixed them; things you changed on purpose]`
