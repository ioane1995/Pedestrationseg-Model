"""Gradio web demo.   python -m app.demo [--share]"""
import argparse
import os
from pathlib import Path

import gradio as gr

from src.inference import Predictor
from src.visualize import overlay

path = os.environ.get("MODEL_PATH", "models/best.pt")
if not Path(path).exists():
    path = "models/best_fp16.pt"
predictor = Predictor(path)


def run(image, thr, size):
    if image is None:
        return None, {}
    predictor.set_min_size(size)
    r = predictor.predict(image, thr)
    vis = overlay(image, r["masks"], r["boxes"], r["scores"])
    return vis, {"pedestrians": int(len(r["scores"])), "scores": [round(float(s), 3) for s in r["scores"]],
                 "inference_ms": round(r["inference_ms"], 1)}


demo = gr.Interface(
    fn=run,
    inputs=[gr.Image(type="pil", label="Input image"),
            gr.Slider(0.05, 0.95, value=0.5, step=0.05, label="Score threshold"),
            gr.Slider(320, 1024, value=800, step=32, label="Input short side (px) - speed/accuracy trade-off")],
    outputs=[gr.Image(label="Segmentation"), gr.JSON(label="Details")],
    title="Pedestrian instance segmentation (Mask R-CNN, Penn-Fudan)",
    examples=[[str(p), 0.5, 800] for p in sorted(Path("examples").glob("*"))[:4]] if Path("examples").exists() else None,
)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="public temporary link (useful on Colab)")
    demo.launch(share=ap.parse_args().share)
