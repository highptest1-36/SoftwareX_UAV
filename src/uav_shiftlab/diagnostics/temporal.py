# SPDX-License-Identifier: Apache-2.0
"""Temporal diagnostics: AP over sliding windows of the stream (drift curves) and summary statistics.

Works on recorded online predictions of any method; ground truth is used only here, after the stream ended.
"""
import numpy as np

from ..evaluation.coco_eval import evaluate


def subset(gt, ids):
    ids = set(ids)
    return dict(gt, images=[im for im in gt["images"] if im["id"] in ids],
                annotations=[a for a in gt["annotations"] if a["image_id"] in ids])


def windowed_ap(gt, dets, stream_ids, window=500, step=250, metrics=("AP50", "AP50_tiny", "AP50_small", "P@0.5")):
    """Class-agnostic metrics on consecutive windows of the stream (in processing order)."""
    by_img = {}
    for d in dets:
        by_img.setdefault(d["image_id"], []).append(d)
    out, n = [], len(stream_ids)
    starts = list(range(0, max(1, n - window + 1), step))
    if starts[-1] + window < n:
        starts.append(n - window)
    for s in starts:
        ids = stream_ids[s:s + window]
        sub = subset(gt, ids)
        dd = [x for i in ids for x in by_img.get(i, [])]
        r = evaluate(sub, dd, per_sequence=False, modes=("agnostic",))["agnostic"] or {}
        out.append({"start": s, "end": s + len(ids), "n_gt": sum(1 for a in sub["annotations"] if not a.get("iscrowd")),
                    **{m: r.get(m) for m in metrics}})
    return out


def drift_summary(curve, key="AP50"):
    """late - early (mean of the last / first quarter of windows), slope per 1000 frames, min window."""
    v = np.array([w[key] for w in curve if w[key] is not None], float)
    if len(v) < 2:
        return {}
    q = max(1, len(v) // 4)
    x = np.array([(w["start"] + w["end"]) / 2 for w in curve if w[key] is not None], float)
    slope = float(np.polyfit(x / 1000.0, v, 1)[0])
    return {"early": float(v[:q].mean()), "late": float(v[-q:].mean()), "late_minus_early": float(v[-q:].mean() - v[:q].mean()),
            "slope_per_1000_frames": slope, "min": float(v.min()), "max": float(v.max())}
