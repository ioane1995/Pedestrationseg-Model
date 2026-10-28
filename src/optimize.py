"""Optimisation study: speed / size / accuracy trade-offs of the chosen model.
    python -m src.optimize --exp E3_tl3_basic
Variants (each is measured on the TEST split, batch size 1; accuracy AND latency are reported):
  * fp32 at several input resolutions (short side 800 = torchvision default -> 640 / 512 / 384)
  * dynamic INT8 quantisation of the Linear layers (CPU)
  * FP16 autocast (GPU, only if CUDA is present)
  * ONNX Runtime export (best effort - Mask R-CNN export is fragile; failure is recorded, not fatal)
Writes results/<exp>/optimization.json
"""
import argparse
from pathlib import Path

import torch

from src.data import download_dataset, get_datasets
from src.engine import evaluate_fn, measure_latency, model_predict_fn
from src.inference import Predictor
from src.utils import file_size_mb, save_json, state_dict_size_mb

SIZES = [800, 640, 512, 384]


def row(name, device, min_size, size_mb, lat, metrics, note=""):
    return {"variant": name, "device": device, "min_size": min_size, "size_mb": size_mb, **lat,
            "segm_map": metrics["segm_map"], "segm_map50": metrics["segm_map_50"],
            "bbox_map": metrics["bbox_map"], "note": note}


def bench(name, fn, device, min_size, size_mb, ds, images, note=""):
    metrics = evaluate_fn(fn, ds)
    lat = measure_latency(fn, images, device)
    r = row(name, device.type, min_size, size_mb, lat, metrics, note)
    print(f"  {name:28s} {device.type:4s} size={min_size!s:>4} lat={lat['latency_ms_mean']:7.1f} ms "
          f"segm mAP={metrics['segm_map']:.3f}")
    return r


def try_onnx(exp_dir, ds, images, base_size_mb):
    try:
        import numpy as np
        import onnxruntime as ort
        pred = Predictor(exp_dir / "best.pt", device=torch.device("cpu"))
        pred.set_min_size(512)
        path = exp_dir / "model.onnx"
        kw = dict(opset_version=16, input_names=["image"], output_names=["boxes", "labels", "scores", "masks"],
                  dynamic_axes={"image": {0: "c", 1: "h", 2: "w"}, "boxes": {0: "n"}, "labels": {0: "n"},
                                "scores": {0: "n"}, "masks": {0: "n", 2: "mh", 3: "mw"}})
        try:
            torch.onnx.export(pred.model, ([images[0]],), str(path), dynamo=False, **kw)
        except TypeError:  # older torch has no `dynamo` kwarg
            torch.onnx.export(pred.model, ([images[0]],), str(path), **kw)
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

        def fn(img):
            b, l, s, m = sess.run(None, {"image": img.numpy().astype(np.float32)})
            return {"boxes": torch.from_numpy(b), "labels": torch.from_numpy(l),
                    "scores": torch.from_numpy(s), "masks": torch.from_numpy(m)}
        return bench("onnxruntime_fp32", fn, torch.device("cpu"), 512, file_size_mb(path), ds, images)
    except Exception as e:  # noqa: BLE001 - record and continue
        print(f"  ONNX export/run failed (recorded): {type(e).__name__}: {str(e)[:200]}")
        return {"variant": "onnxruntime_fp32", "device": "cpu", "note": f"FAILED: {type(e).__name__}: {str(e)[:300]}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--results", default="results")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--max-images", type=int, default=30)
    a = ap.parse_args()

    exp_dir = Path(a.results) / a.exp
    _, _, test_ds = get_datasets(download_dataset(a.data_root))
    images = [test_ds[i][0] for i in range(min(a.max_images, len(test_ds)))]
    ckpt_mb = file_size_mb(exp_dir / "best.pt")
    rows = []

    devices = [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])
    for dev in devices:
        for s in SIZES:
            pred = Predictor(exp_dir / "best.pt", device=dev)
            pred.set_min_size(s)
            fn = model_predict_fn(pred.model, dev)
            rows.append(bench("fp32", fn, dev, s, ckpt_mb, test_ds, images))
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        for s in (800, 512):
            pred = Predictor(exp_dir / "best.pt", device=dev)
            pred.set_min_size(s)
            rows.append(bench("fp16_autocast", model_predict_fn(pred.model, dev, amp=True), dev, s, ckpt_mb,
                              test_ds, images, "weights stay fp32; speed-up from fp16 compute"))

    cpu = torch.device("cpu")
    for s in (800, 512):
        try:
            pred = Predictor(exp_dir / "best.pt", device=cpu)
            pred.set_min_size(s)
            q = torch.ao.quantization.quantize_dynamic(pred.model, {torch.nn.Linear}, dtype=torch.qint8)
            rows.append(bench("dynamic_int8_linear", model_predict_fn(q, cpu), cpu, s, state_dict_size_mb(q),
                              test_ds, images, "only Linear (box head) layers are quantised"))
        except Exception as e:  # noqa: BLE001
            rows.append({"variant": "dynamic_int8_linear", "note": f"FAILED: {type(e).__name__}: {e}"})

    rows.append(try_onnx(exp_dir, test_ds, images, ckpt_mb))
    save_json({"exp": a.exp, "n_latency_images": len(images), "rows": rows}, str(exp_dir / "optimization.json"))


if __name__ == "__main__":
    main()
