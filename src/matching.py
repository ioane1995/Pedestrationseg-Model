"""Pure-numpy instance matching (no torch) so it can be unit-tested anywhere."""
import numpy as np


def mask_iou(pm, gm):
    """pm: bool [P,H,W], gm: bool [G,H,W] -> IoU matrix [P,G]."""
    if len(pm) == 0 or len(gm) == 0:
        return np.zeros((len(pm), len(gm)), dtype=np.float32)
    P = pm.reshape(len(pm), -1).astype(np.float32)
    G = gm.reshape(len(gm), -1).astype(np.float32)
    inter = P @ G.T
    union = P.sum(1)[:, None] + G.sum(1)[None, :] - inter
    return inter / np.maximum(union, 1.0)


def match_instances(pred_masks, pred_scores, gt_masks, iou_thr=0.5):
    """Greedy matching by descending score (COCO style).
    Returns (tp, fp, fn, fp_types): tp = [(pred_idx, gt_idx, iou)], fp/fn = index lists,
    fp_types = {pred_idx: 'duplicate' | 'localisation' | 'background'}."""
    iou = mask_iou(pred_masks, gt_masks)
    n_gt = iou.shape[1]
    used, tp, fp = set(), [], []
    for p in np.argsort(-np.asarray(pred_scores), kind="stable"):
        best_g, best_iou = -1, 0.0
        for g in range(n_gt):
            if g not in used and iou[p, g] > best_iou:
                best_g, best_iou = g, float(iou[p, g])
        if best_g >= 0 and best_iou >= iou_thr:
            used.add(best_g)
            tp.append((int(p), best_g, best_iou))
        else:
            fp.append(int(p))
    fn = [g for g in range(n_gt) if g not in used]
    fp_types = {}
    for p in fp:
        m = float(iou[p].max()) if n_gt else 0.0
        fp_types[p] = "duplicate" if m >= iou_thr else ("localisation" if m >= 0.1 else "background")
    return tp, fp, fn, fp_types
