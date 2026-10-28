"""Draw instance masks + boxes on an image."""
import numpy as np
from PIL import Image, ImageDraw

PALETTE = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
           (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212)]


def overlay(image, masks, boxes, scores=None, title=None, alpha=0.45):
    """image: PIL; masks: bool [N,H,W]; boxes: [N,4] xyxy; scores: optional [N]."""
    base = np.asarray(image.convert("RGB")).astype(np.float32)
    for i, m in enumerate(masks):
        c = np.array(PALETTE[i % len(PALETTE)], dtype=np.float32)
        base[m] = (1 - alpha) * base[m] + alpha * c
    out = Image.fromarray(base.astype(np.uint8))
    d = ImageDraw.Draw(out)
    for i, b in enumerate(boxes):
        col = PALETTE[i % len(PALETTE)]
        d.rectangle([float(x) for x in b], outline=col, width=2)
        txt = "person" if scores is None else f"person {float(scores[i]):.2f}"
        d.text((float(b[0]) + 3, float(b[1]) + 2), txt, fill=(255, 255, 255))
    if title:
        d.rectangle([0, 0, 8 * len(title) + 8, 16], fill=(0, 0, 0))
        d.text((4, 2), title, fill=(255, 255, 255))
    return out
