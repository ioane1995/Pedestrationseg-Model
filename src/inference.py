"""Load a trained checkpoint and run it on new images (used by CLI, API and Gradio demo)."""
import argparse
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from src.models import build_model
from src.utils import get_device


class Predictor:
    def __init__(self, ckpt, device=None, score_thresh=0.5, min_size=None):
        self.device = device or get_device()
        self.score_thresh = score_thresh
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = ck["config"]
        self.model = build_model(cfg["model"], cfg["num_classes"], pretrained=False,
                                 min_size=min_size or cfg.get("min_size"), max_size=cfg.get("max_size"))
        sd = {k: (v.float() if v.is_floating_point() else v) for k, v in ck["state_dict"].items()}  # fp16 ckpt ok
        self.model.load_state_dict(sd)
        self.model.to(self.device).eval()

    def set_min_size(self, s):
        self.model.transform.min_size = (int(s),)

    @torch.inference_mode()
    def predict(self, image: Image.Image, score_thresh=None, raw=False):
        """Returns dict: boxes [N,4] (xyxy), scores [N], masks bool [N,H,W] (numpy) and inference_ms."""
        thr = self.score_thresh if score_thresh is None else score_thresh
        t = TF.to_tensor(image.convert("RGB")).to(self.device)
        t0 = time.perf_counter()
        out = self.model([t])[0]
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        keep = out["scores"] >= (0.0 if raw else thr)
        return {"boxes": out["boxes"][keep].cpu().numpy(), "scores": out["scores"][keep].cpu().numpy(),
                "masks": (out["masks"][keep][:, 0] > 0.5).cpu().numpy(), "inference_ms": ms}


def main():
    from src.visualize import overlay
    p = argparse.ArgumentParser(description="Run the model on an image or a folder of images")
    p.add_argument("--ckpt", default="models/best.pt")
    p.add_argument("--input", required=True, help="image file or directory")
    p.add_argument("--output", default="outputs")
    p.add_argument("--score-thresh", type=float, default=0.5)
    a = p.parse_args()
    pred = Predictor(a.ckpt, score_thresh=a.score_thresh)
    src = Path(a.input)
    files = sorted(src.glob("*")) if src.is_dir() else [src]
    Path(a.output).mkdir(parents=True, exist_ok=True)
    for f in files:
        if f.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            continue
        img = Image.open(f).convert("RGB")
        r = pred.predict(img)
        overlay(img, r["masks"], r["boxes"], r["scores"]).save(Path(a.output) / f"{f.stem}_pred.png")
        print(f"{f.name}: {len(r['scores'])} pedestrian(s), {r['inference_ms']:.0f} ms")


if __name__ == "__main__":
    main()
