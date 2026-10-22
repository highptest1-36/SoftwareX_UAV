# SPDX-License-Identifier: Apache-2.0
"""COCO-style evaluation for UAVDT.

Differences from pycocotools defaults:
  * maxDets = [1, 100, 300]  (UAVDT frames have up to 107 GT boxes); all metrics use 300.
  * extra area range 'tiny' = sqrt(area) < 16 px.
  * ignore regions are iscrowd=1 GT -> detections inside them are ignored.
  * class-agnostic mode ("vehicle") in addition to 3-class mAP.
  * per-sequence metrics and sequence-mean (each sequence weighted equally).

Detections: list of dicts {image_id, category_id, bbox[x,y,w,h], score}.
"""
import contextlib
import copy
import io
import json

import numpy as np
from pycocotools.coco import COCO

try:  # C++ implementation, verified identical to pycocotools (max |diff| 2e-16) and ~30x faster
    from faster_coco_eval import COCO as _FastCOCO
    from faster_coco_eval import COCOeval_faster as _FastEval
    BACKEND = "faster_coco_eval"
except ImportError:  # pragma: no cover
    from pycocotools.cocoeval import COCOeval as _PyEval
    BACKEND = "pycocotools"

AREA_RNG = {"all": [0, 1e10], "tiny": [0, 16 ** 2], "small": [0, 32 ** 2],
            "medium": [32 ** 2, 96 ** 2], "large": [96 ** 2, 1e10]}
MAX_DETS = [1, 100, 300]
_IOUS = np.linspace(0.5, 0.95, 10)


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def load_gt(gt, class_agnostic=False):
    """gt: path or dict (COCO json). Returns pycocotools COCO object."""
    d = json.load(open(gt)) if isinstance(gt, str) else copy.deepcopy(gt)
    if class_agnostic:
        seen = set()
        anns = []
        for a in d["annotations"]:
            if a.get("iscrowd", 0):
                key = (a["image_id"], tuple(a["bbox"]))
                if key in seen:
                    continue
                seen.add(key)
            anns.append(dict(a, category_id=1))
        d["annotations"] = anns
        d["categories"] = [{"id": 1, "name": "vehicle"}]
    coco = COCO()
    coco.dataset = d
    _quiet(coco.createIndex)
    return coco


def _ap(E, iou=None, area="all", cat_idx=None):
    """Mean AP from E.eval['precision'] [T,R,K,A,M]; -1 entries excluded."""
    p = E.eval["precision"]
    a = list(E.params.areaRngLbl).index(area)
    m = len(E.params.maxDets) - 1
    if iou is not None:
        p = p[[int(np.argmin(np.abs(_IOUS - iou)))]]
    p = p[:, :, :, a, m]
    if cat_idx is not None:
        p = p[:, :, [cat_idx]]
    v = p[p > -1]
    return float(v.mean()) if v.size else float("nan")


def _ar(E, area="all"):
    r = E.eval["recall"][:, :, list(E.params.areaRngLbl).index(area), -1]
    v = r[r > -1]
    return float(v.mean()) if v.size else float("nan")


def _iou_xywh(d, g, crowd):
    """pycocotools-compatible IoU: for crowd GT, IoU = inter / area(det)."""
    dx2, dy2 = d[:, 0] + d[:, 2], d[:, 1] + d[:, 3]
    gx2, gy2 = g[:, 0] + g[:, 2], g[:, 1] + g[:, 3]
    iw = np.clip(np.minimum(dx2[:, None], gx2[None]) - np.maximum(d[:, None, 0], g[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(dy2[:, None], gy2[None]) - np.maximum(d[:, None, 1], g[None, :, 1]), 0, None)
    inter = iw * ih
    da = (d[:, 2] * d[:, 3])[:, None]
    union = np.where(crowd[None, :], da, da + (g[:, 2] * g[:, 3])[None, :] - inter)
    return inter / np.maximum(union, 1e-12)


def pr_at_score(coco_gt, dets, score_thr=0.5, iou_thr=0.5, max_dets=300, img_ids=None):
    """Precision/recall at a score threshold (area=all), replicating pycocotools greedy matching:
    dets sorted by score, top max_dets per (image, category); non-crowd GT matched first; a det
    matched only to a crowd (ignore) region is ignored (neither TP nor FP)."""
    by = {}
    for x in dets:
        if img_ids is None or x["image_id"] in img_ids:
            by.setdefault((x["image_id"], x["category_id"]), []).append(x)
    gts = {}
    for a in coco_gt.dataset["annotations"]:
        if img_ids is None or a["image_id"] in img_ids:
            gts.setdefault((a["image_id"], a["category_id"]), []).append(a)
    tp = fp = ngt = 0
    for key in set(by) | set(gts):
        g = gts.get(key, [])
        crowd = np.array([bool(a.get("iscrowd", 0)) for a in g], bool)
        ngt += int((~crowd).sum())
        d = sorted(by.get(key, []), key=lambda z: -z["score"])[:max_dets]
        d = [z for z in d if z["score"] >= score_thr]
        if not d:
            continue
        if not g:
            fp += len(d)
            continue
        order = np.argsort(crowd, kind="stable")  # non-crowd first, like pycocotools
        gb = np.array([g[i]["bbox"] for i in order], float)
        gc = crowd[order]
        ious = _iou_xywh(np.array([z["bbox"] for z in d], float), gb, gc)
        used = np.zeros(len(gb), bool)
        for i in range(len(d)):
            best, m = min(iou_thr, 1 - 1e-10), -1
            for j in range(len(gb)):
                if used[j] and not gc[j]:
                    continue
                if m > -1 and not gc[m] and gc[j]:
                    break
                if ious[i, j] < best:
                    continue
                best, m = ious[i, j], j
            if m == -1:
                fp += 1
            elif gc[m]:
                pass  # ignored
            else:
                used[m] = True
                tp += 1
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / ngt if ngt else float("nan")
    return prec, rec


def _run(coco_gt, dets, img_ids=None):
    dets = [d for d in dets if img_ids is None or d["image_id"] in img_ids]
    if not dets:
        return None
    if BACKEND == "faster_coco_eval":
        if not hasattr(coco_gt, "_fast"):
            coco_gt._fast = _quiet(_FastCOCO, copy.deepcopy(coco_gt.dataset))
        coco_dt = _quiet(coco_gt._fast.loadRes, copy.deepcopy(dets))
        E = _FastEval(coco_gt._fast, coco_dt, "bbox", print_function=lambda *a, **k: None)
    else:
        coco_dt = _quiet(coco_gt.loadRes, copy.deepcopy(dets))
        E = _PyEval(coco_gt, coco_dt, "bbox")
    E.params.maxDets = MAX_DETS
    E.params.areaRng = list(AREA_RNG.values())
    E.params.areaRngLbl = list(AREA_RNG.keys())
    if img_ids is not None:
        E.params.imgIds = sorted(img_ids)
    _quiet(E.evaluate)
    _quiet(E.accumulate)
    return E


def summarize(E, cat_names=None, pr=None):
    out = {"AP": _ap(E), "AP50": _ap(E, 0.5), "AP70": _ap(E, 0.7), "AP75": _ap(E, 0.75)}
    for a in ("tiny", "small", "medium", "large"):
        out[f"AP_{a}"] = _ap(E, area=a)
        out[f"AP50_{a}"] = _ap(E, 0.5, area=a)
    out["AR300"] = _ar(E)
    if pr is not None:
        out["P@0.5"], out["R@0.5"] = pr
    if cat_names:
        for k, name in enumerate(cat_names):
            out[f"AP50_{name}"] = _ap(E, 0.5, cat_idx=k)
    return {k: round(100 * v, 2) if v == v else None for k, v in out.items()}


def evaluate(gt, dets, per_sequence=True, modes=("3class", "agnostic")):
    """gt: path/dict of a COCO json produced by tools/prepare_uavdt.py.

    Returns {"3class": {...}, "agnostic": {...}, "per_seq": {seq: {...}}, "seq_mean": {...}}
    """
    d = json.load(open(gt)) if isinstance(gt, str) else gt
    seq_of = {im["id"]: im["video"] for im in d["images"]}
    res = {}
    gts = {}
    for mode in modes:
        ag = mode == "agnostic"
        coco_gt = load_gt(d, class_agnostic=ag)
        gts[mode] = coco_gt
        dd = [dict(x, category_id=1) for x in dets] if ag else dets
        E = _run(coco_gt, dd)
        names = None if ag else [c["name"] for c in d["categories"]]
        res[mode] = summarize(E, names, pr_at_score(coco_gt, dd)) if E else None
    if per_sequence:
        res["per_seq"] = {}
        seqs = sorted(set(seq_of.values()))
        for s in seqs:
            ids = {i for i, v in seq_of.items() if v == s}
            row = {}
            for mode in modes:
                ag = mode == "agnostic"
                dd = [dict(x, category_id=1) for x in dets] if ag else dets
                E = _run(gts[mode], dd, ids)
                if E is None:
                    row[f"{mode}_AP50"] = 0.0
                    continue
                row[f"{mode}_AP50"] = round(100 * _ap(E, 0.5), 2)
                row[f"{mode}_AP"] = round(100 * _ap(E), 2)
                row[f"{mode}_AP50_small"] = round(100 * _ap(E, 0.5, area="small"), 2)
            res["per_seq"][s] = row
        keys = sorted({k for r in res["per_seq"].values() for k in r})
        res["seq_mean"] = {k: round(float(np.nanmean([r.get(k, np.nan) for r in res["per_seq"].values()])), 2)
                           for k in keys}
    return res
