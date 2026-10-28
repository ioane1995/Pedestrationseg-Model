"""Training loop and evaluation (COCO-style bbox + segm mAP via torchmetrics/pycocotools)."""
import time
from collections import defaultdict

import torch


def _to_device(targets, device):
    return [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]


def train_one_epoch(model, optimizer, loader, device, scaler, epoch, clip=10.0):
    model.train()
    warmup = None
    if epoch == 0:  # linear LR warm-up in the first epoch avoids early divergence
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, total_iters=max(1, min(500, len(loader) - 1)))
    sums, n = defaultdict(float), 0
    for images, targets in loader:
        images = [i.to(device) for i in images]
        targets = _to_device(targets, device)
        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss: { {k: v.item() for k, v in loss_dict.items()} }")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()
        if warmup is not None:
            warmup.step()
        for k, v in loss_dict.items():
            sums[k] += v.item()
        sums["loss"] += loss.item()
        n += 1
    return {k: v / n for k, v in sums.items()}


def model_predict_fn(model, device, amp=False):
    """Returns f(image_tensor[3,H,W]) -> dict(boxes, labels, scores, masks)."""
    @torch.inference_mode()
    def fn(img):
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=amp and device.type == "cuda"):
            return model([img.to(device)])[0]
    return fn


def evaluate_fn(predict_fn, dataset, mask_thresh=0.5):
    """Runs predict_fn on every item of `dataset`; returns scalar COCO metrics for bbox and segm."""
    from torchmetrics.detection import MeanAveragePrecision
    metric = MeanAveragePrecision(iou_type=("bbox", "segm"), box_format="xyxy")
    for i in range(len(dataset)):
        img, tgt = dataset[i]
        out = predict_fn(img)
        masks = out["masks"].cpu()
        if masks.ndim == 4:
            masks = masks[:, 0]
        pred = {"boxes": out["boxes"].cpu().float(), "scores": out["scores"].cpu().float(),
                "labels": out["labels"].cpu().long(), "masks": masks > mask_thresh}
        gt = {"boxes": tgt["boxes"].float(), "labels": tgt["labels"].long(), "masks": tgt["masks"].bool()}
        metric.update([pred], [gt])
    res = metric.compute()
    return {k: float(v) for k, v in res.items() if torch.is_tensor(v) and v.ndim == 0}


def evaluate(model, dataset, device, amp=False):
    model.eval()
    return evaluate_fn(model_predict_fn(model, device, amp), dataset)


def measure_latency(predict_fn, images, device, warmup=3):
    """Per-image wall-clock latency in ms (batch size 1)."""
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()
    for _ in range(warmup):
        predict_fn(images[0])
    times = []
    for im in images:
        sync()
        t0 = time.perf_counter()
        predict_fn(im)
        sync()
        times.append((time.perf_counter() - t0) * 1000)
    t = torch.tensor(times)
    return {"latency_ms_mean": t.mean().item(), "latency_ms_median": t.median().item(),
            "latency_ms_p95": torch.quantile(t, 0.95).item(), "fps": 1000.0 / t.mean().item()}
