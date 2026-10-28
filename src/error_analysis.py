"""Error analysis of the chosen model on the TEST split.
    python -m src.error_analysis --exp E3_tl3_basic
Writes results/<exp>/error_analysis.json and results/<exp>/error_analysis/*.png

Reports: precision / recall / F1 / mean mask IoU at the operating threshold, a threshold sweep,
false-positive taxonomy (duplicate / localisation / background), recall by pedestrian size and by
scene crowdedness, and side-by-side GT-vs-prediction pictures of the worst images.
"""
import argparse
from pathlib import Path

import numpy as np
from torchvision.transforms import functional as TF

from src.data import download_dataset, get_datasets
from src.inference import Predictor
from src.matching import match_instances
from src.utils import save_json
from src.visualize import overlay

THRESHOLDS = [0.3, 0.5, 0.7, 0.9]


def _prf(tp, fp, fn):
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r,
            "f1": 2 * p * r / max(p + r, 1e-9)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--results", default="results")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--op-thresh", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=6, help="number of worst images to visualise")
    a = ap.parse_args()

    out_dir = Path(a.results) / a.exp
    vis_dir = out_dir / "error_analysis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    pred = Predictor(out_dir / "best.pt", score_thresh=0.05)
    _, _, test_ds = get_datasets(download_dataset(a.data_root))

    # run the model once, keep everything with score >= 0.05 and filter per threshold afterwards
    cache = []
    for i in range(len(test_ds)):
        img_t, tgt = test_ds[i]
        pil = TF.to_pil_image(img_t)
        r = pred.predict(pil, raw=True)
        cache.append({"pil": pil, "r": r, "gt_masks": tgt["masks"].bool().numpy(),
                      "gt_boxes": tgt["boxes"].numpy()})

    def run(thr):
        per_img, ious, tot = [], [], dict(tp=0, fp=0, fn=0)
        fp_types = {"duplicate": 0, "localisation": 0, "background": 0}
        for c in cache:
            keep = c["r"]["scores"] >= thr
            pm, ps = c["r"]["masks"][keep], c["r"]["scores"][keep]
            tp, fp, fn, ft = match_instances(pm, ps, c["gt_masks"])
            for k, v in (("tp", len(tp)), ("fp", len(fp)), ("fn", len(fn))):
                tot[k] += v
            ious += [t[2] for t in tp]
            for t in ft.values():
                fp_types[t] += 1
            per_img.append({"tp": tp, "fp": fp, "fn": fn, "keep": keep})
        return per_img, tot, ious, fp_types

    sweep = {}
    for thr in THRESHOLDS:
        _, tot, ious, _ = run(thr)
        sweep[str(thr)] = {**_prf(**tot), "mean_tp_mask_iou": float(np.mean(ious)) if ious else 0.0}

    per_img, tot, ious, fp_types = run(a.op_thresh)
    summary = {"operating_threshold": a.op_thresh, **_prf(**tot),
               "mean_tp_mask_iou": float(np.mean(ious)) if ious else 0.0,
               "false_positive_types": fp_types, "threshold_sweep": sweep}

    # recall by GT size (terciles of mask area) and by crowdedness
    areas = np.array([m.sum() for c in cache for m in c["gt_masks"]])
    q1, q2 = np.quantile(areas, [1 / 3, 2 / 3])
    buckets = {"small": [0, 0], "medium": [0, 0], "large": [0, 0]}  # [found, total]
    crowd = {"1-2 people": [0, 0], "3+ people": [0, 0]}
    for c, pi in zip(cache, per_img):
        found = {t[1] for t in pi["tp"]}
        key_c = "3+ people" if len(c["gt_masks"]) >= 3 else "1-2 people"
        for g, m in enumerate(c["gt_masks"]):
            a_ = m.sum()
            key = "small" if a_ <= q1 else ("medium" if a_ <= q2 else "large")
            buckets[key][1] += 1
            crowd[key_c][1] += 1
            if g in found:
                buckets[key][0] += 1
                crowd[key_c][0] += 1
    summary["recall_by_size"] = {k: v[0] / max(v[1], 1) for k, v in buckets.items()}
    summary["recall_by_crowdedness"] = {k: v[0] / max(v[1], 1) for k, v in crowd.items()}
    summary["size_tercile_area_px"] = [float(q1), float(q2)]

    # worst images
    err = [len(pi["fp"]) + len(pi["fn"]) for pi in per_img]
    worst = [int(i) for i in np.argsort(err)[::-1][:a.k] if err[i] > 0]
    summary["worst_images"] = []
    for rank, i in enumerate(worst):
        c, pi = cache[i], per_img[i]
        keep = pi["keep"]
        left = overlay(c["pil"], c["gt_masks"], c["gt_boxes"], None, title="ground truth")
        right = overlay(c["pil"], c["r"]["masks"][keep], c["r"]["boxes"][keep], c["r"]["scores"][keep],
                        title=f"prediction (FP={len(pi['fp'])}, FN={len(pi['fn'])})")
        canvas = np.concatenate([np.asarray(left), np.asarray(right)], axis=1)
        from PIL import Image
        Image.fromarray(canvas).save(vis_dir / f"worst_{rank}_idx{i}.png")
        summary["worst_images"].append({"test_idx": i, "fp": len(pi["fp"]), "fn": len(pi["fn"])})

    save_json(summary, str(out_dir / "error_analysis.json"))
    print(f"precision {summary['precision']:.3f} recall {summary['recall']:.3f} "
          f"F1 {summary['f1']:.3f} | FP types {fp_types} | recall by size {summary['recall_by_size']}")


if __name__ == "__main__":
    main()
