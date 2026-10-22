# SPDX-License-Identifier: Apache-2.0
"""Pseudo-label audit: how correct were the labels a self-training method trained on, and how did that evolve?

Input per frame: positive pseudo-labels as xyxy boxes (+ score). A pseudo-label is correct if it matches a
non-ignored ground-truth box with IoU >= iou (greedy by score, one-to-one); labels whose centre falls inside an
ignore region are counted separately. Ground truth is used only for this offline audit.
"""
import numpy as np


def _iou(a, b):
    """a [N,4], b [M,4] xyxy -> [N,M]."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)


def gt_index(gt):
    by = {im["id"]: {"boxes": [], "ignore": [], "small": []} for im in gt["images"]}
    seen = set()
    for a in gt["annotations"]:
        x, y, w, h = a["bbox"]
        if a.get("iscrowd"):
            key = (a["image_id"], tuple(a["bbox"]))   # ignore regions are replicated per class
            if key not in seen:
                seen.add(key)
                by[a["image_id"]]["ignore"].append([x, y, x + w, y + h])
        else:
            by[a["image_id"]]["boxes"].append([x, y, x + w, y + h])
    return {k: {n: np.asarray(v, float).reshape(-1, 4) for n, v in d.items() if n != "small"} for k, d in by.items()}


def audit_frame(pl_boxes, pl_scores, g, iou=0.5):
    pl = np.asarray(pl_boxes, float).reshape(-1, 4)
    res = {"n": len(pl), "tp": 0, "in_ignore": 0, "n_gt": len(g["boxes"]), "tp_small": 0, "n_small": 0}
    if not len(pl):
        return res
    small = (pl[:, 2] - pl[:, 0]) * (pl[:, 3] - pl[:, 1]) < 32 * 32
    res["n_small"] = int(small.sum())
    cx, cy = (pl[:, 0] + pl[:, 2]) / 2, (pl[:, 1] + pl[:, 3]) / 2
    ign = np.zeros(len(pl), bool)
    for x1, y1, x2, y2 in g["ignore"]:
        ign |= (cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2)
    M = _iou(pl, g["boxes"])
    used = set()
    for i in np.argsort(-np.asarray(pl_scores, float)):
        if ign[i]:
            continue
        if M.shape[1]:
            cand = [(M[i, j], j) for j in range(M.shape[1]) if j not in used and M[i, j] >= iou]
            if cand:
                used.add(max(cand)[1])
                res["tp"] += 1
                res["tp_small"] += int(small[i])
    res["in_ignore"] = int(ign.sum())
    return res


def audit_stream(frames, gt, window=500, step=250, iou=0.5):
    """frames: list of (image_id, boxes_xyxy, scores) in stream order. Returns totals and a windowed curve."""
    G = gt_index(gt)
    per = [audit_frame(b, s, G[i], iou) for i, b, s in frames]

    def agg(rows):
        n = sum(r["n"] for r in rows)
        judged = n - sum(r["in_ignore"] for r in rows)
        tp = sum(r["tp"] for r in rows)
        ns = sum(r["n_small"] for r in rows)
        return {"pseudo_labels": n, "per_frame": n / max(1, len(rows)), "precision": tp / judged if judged else None,
                "gt_coverage": tp / max(1, sum(r["n_gt"] for r in rows)), "in_ignore_frac": (n - judged) / n if n else None,
                "precision_small": sum(r["tp_small"] for r in rows) / ns if ns else None}
    curve = []
    for s in range(0, max(1, len(per) - window + 1), step):
        curve.append({"start": s, "end": s + window, **agg(per[s:s + window])})
    return {"total": agg(per), "curve": curve}
