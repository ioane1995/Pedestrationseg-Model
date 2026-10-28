"""REST API.   uvicorn app.api:app --host 0.0.0.0 --port 8000     (docs: http://localhost:8000/docs)
Env: MODEL_PATH (default models/best.pt, falls back to models/best_fp16.pt), SCORE_THRESH (default 0.5)
"""
import io
import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from PIL import Image

from src.inference import Predictor
from src.visualize import overlay

app = FastAPI(title="Pedestrian instance segmentation API", version="1.0")


@lru_cache(maxsize=1)
def get_predictor():
    path = os.environ.get("MODEL_PATH", "models/best.pt")
    if not Path(path).exists() and Path("models/best_fp16.pt").exists():
        path = "models/best_fp16.pt"
    if not Path(path).exists():
        raise HTTPException(503, f"model checkpoint not found at {path} - see README (Getting the model)")
    return Predictor(path, score_thresh=float(os.environ.get("SCORE_THRESH", 0.5)))


def _read(upload: UploadFile) -> Image.Image:
    try:
        return Image.open(io.BytesIO(upload.file.read())).convert("RGB")
    except Exception:
        raise HTTPException(400, "could not decode the uploaded file as an image")


@app.get("/health")
def health():
    p = get_predictor()
    return {"status": "ok", "device": str(p.device)}


@app.post("/predict")
def predict(file: UploadFile = File(...), score_thresh: float = Query(0.5, ge=0.0, le=1.0),
            include_masks: bool = Query(False, description="return masks as COCO RLE")):
    """JSON result: count, boxes (xyxy, pixels), scores, optional RLE masks."""
    img = _read(file)
    r = get_predictor().predict(img, score_thresh)
    out = {"width": img.width, "height": img.height, "count": int(len(r["scores"])),
           "inference_ms": round(r["inference_ms"], 1),
           "instances": [{"box": [round(float(v), 1) for v in b], "score": round(float(s), 4)}
                         for b, s in zip(r["boxes"], r["scores"])]}
    if include_masks:
        import numpy as np
        from pycocotools import mask as mu
        for inst, m in zip(out["instances"], r["masks"]):
            rle = mu.encode(np.asfortranarray(m.astype("uint8")))
            inst["mask_rle"] = {"size": rle["size"], "counts": rle["counts"].decode()}
    return out


@app.post("/predict/image")
def predict_image(file: UploadFile = File(...), score_thresh: float = Query(0.5, ge=0.0, le=1.0)):
    """PNG with masks + boxes drawn on the input image."""
    img = _read(file)
    r = get_predictor().predict(img, score_thresh)
    buf = io.BytesIO()
    overlay(img, r["masks"], r["boxes"], r["scores"]).save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png",
                    headers={"X-Pedestrian-Count": str(len(r["scores"]))})
