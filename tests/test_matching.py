import numpy as np

from src.matching import match_instances, mask_iou


def box_mask(x0, y0, x1, y1, h=20, w=20):
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def test_iou_identity_and_disjoint():
    a = np.stack([box_mask(0, 0, 10, 10)])
    b = np.stack([box_mask(0, 0, 10, 10), box_mask(12, 12, 18, 18)])
    iou = mask_iou(a, b)
    assert iou.shape == (1, 2)
    assert abs(iou[0, 0] - 1.0) < 1e-6 and iou[0, 1] == 0.0


def test_tp_fp_fn_and_duplicate():
    gt = np.stack([box_mask(0, 0, 10, 10), box_mask(12, 12, 18, 18)])
    pred = np.stack([box_mask(0, 0, 10, 10),      # TP for gt0
                     box_mask(0, 0, 10, 10),      # duplicate of gt0 -> FP (duplicate)
                     box_mask(0, 15, 4, 19)])     # background FP; gt1 missed -> FN
    scores = np.array([0.9, 0.8, 0.7])
    tp, fp, fn, types = match_instances(pred, scores, gt)
    assert [t[:2] for t in tp] == [(0, 0)]
    assert sorted(fp) == [1, 2] and fn == [1]
    assert types[1] == "duplicate" and types[2] == "background"


def test_empty_predictions():
    gt = np.stack([box_mask(0, 0, 5, 5)])
    tp, fp, fn, _ = match_instances(np.zeros((0, 20, 20), bool), np.zeros(0), gt)
    assert tp == [] and fp == [] and fn == [0]


def test_higher_score_wins_gt():
    gt = np.stack([box_mask(0, 0, 10, 10)])
    pred = np.stack([box_mask(0, 0, 9, 10), box_mask(0, 0, 10, 10)])
    tp, fp, fn, _ = match_instances(pred, np.array([0.6, 0.95]), gt)
    assert tp[0][0] == 1 and fp == [0]
