# SPDX-License-Identifier: Apache-2.0
"""Record full engine traces of the Detectron2 methods on short CPU fixtures, audit them, and compare them field by
field with reference traces recorded earlier (regression check for code changes; run both on the same CPU type).

  python3 -u tools/trace_capture.py --tag ref --threads 4                    # record reference traces
  python3 -u tools/trace_capture.py --tag new --threads 4 --ref ref          # after a code change: must be exact
  python3 -u tools/trace_capture.py --tag new --ref ref --rebuild            # recompute the comparison from disk

Fixtures (target-DEV data only, never test splits; COCO json with `video` and `frame_id` per image):
  GB  30 consecutive frames of dev_night M0202 (adaptation runs: EMA, pseudo-labels, updates)
  GC  last 8 frames of M0202 + first 8 frames of M0204 (sequence boundary inside the stream)
Output: golden/<tag>/<level>/<method>/engine_trace.json.gz and golden/<tag>/trace_compare.json (merged).
Run on CPU (CUDA_VISIBLE_DEVICES=""): GPU kernels are not bitwise deterministic.
"""
import argparse
import gzip
import json
import os
import sys

sys.dont_write_bytecode = True
SX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SX, "src"))

import torch  # noqa: E402

from uav_shiftlab.backends.detectron2.common import read_bgr  # noqa: E402
from uav_shiftlab.backends.detectron2.plugins import PROFILES, build_method  # noqa: E402
from uav_shiftlab.core.runner import OnlineRunner  # noqa: E402
from uav_shiftlab.core.stream import StreamManifest  # noqa: E402
from uav_shiftlab.core.types import FrameRef  # noqa: E402
from uav_shiftlab.protocols.audit import audit_trace  # noqa: E402
from uav_shiftlab.reproducibility.trace_compare import compare  # noqa: E402

LEVELS = {"GB": [("M0202", 0, 30)], "GC": [("M0202", -8, None), ("M0204", 0, 8)]}
ASSETS = os.environ.get("UAVSL_ASSETS_ROOT", os.path.dirname(SX))


def fixture_stream(coco_path, image_root, level):
    gt = json.load(open(coco_path))
    frames = []
    for seq, a, b in LEVELS[level]:
        ims = sorted((im for im in gt["images"] if im["video"] == seq), key=lambda im: im["frame_id"])
        frames += ims[a:b] if b is not None else ims[a:]
    return StreamManifest([FrameRef(int(im["id"]), im["video"], int(im["frame_id"]),
                                    os.path.join(image_root, im["file_name"])) for im in frames], name=level)


def load_trace(tag, level, m):
    p = os.path.join(SX, "golden", tag, level, m, "engine_trace.json.gz")
    if not os.path.exists(p):
        return None
    with gzip.open(p, "rt") as f:
        return json.load(f)


def evaluate(trace, level, m, ref):
    audit = audit_trace(trace)
    row = {"frames": len(trace["frames"]), "sec": round(sum(f["sec"] for f in trace["frames"]), 1),
           "audit_pass": audit["pass"], "audit_failed": sorted(k for k, v in audit.items() if k != "pass" and not v["pass"]),
           "decisions": {k: sum(f["decision"] == k for f in trace["frames"])
                         for k in ("update", "skip_gate", "skip_when", "no_adaptation")}}
    gold = load_trace(ref, level, m) if ref else None
    if gold is not None:
        d = compare(trace, gold)
        row.update({"ref": ref, "exact": d["exact"], "first_divergent_frame": d["first_divergent_frame"], "diff": d})
    return row


def save_row(res_path, key, row):
    """Merge one row into the results file (re-read first: several processes may write it)."""
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}
    results[key] = row
    tmp = res_path + f".tmp{os.getpid()}"
    json.dump(results, open(tmp, "w"), indent=1)
    os.replace(tmp, res_path)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ref", default="", help="tag of the reference traces to compare with")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--levels", nargs="+", default=list(LEVELS), choices=list(LEVELS))
    ap.add_argument("--methods", nargs="+", default=list(PROFILES), choices=list(PROFILES))
    ap.add_argument("--assets-root", default=ASSETS, help="directory with checkpoints/ (source model, WHW stats)")
    ap.add_argument("--coco", default="", help="dev_night COCO json (default: <assets>/data/coco/dev_night.json)")
    ap.add_argument("--images", default="", help="image root (default: <assets>/data/UAV-benchmark-M)")
    ap.add_argument("--rebuild", action="store_true", help="recompute trace_compare.json from the traces on disk")
    a = ap.parse_args()
    out_root = os.path.join(SX, "golden", a.tag)
    res_path = os.path.join(out_root, "trace_compare.json")
    if a.rebuild:
        res = {}
        for level in a.levels:
            for m in a.methods:
                tr = load_trace(a.tag, level, m)
                if tr is not None:
                    res[f"{level}/{m}"] = evaluate(tr, level, m, a.ref)
        json.dump(res, open(res_path, "w"), indent=1)
        print(f"[rebuild] {len(res)} rows -> {res_path}")
        return
    torch.set_num_threads(a.threads)
    coco = a.coco or os.path.join(a.assets_root, "data", "coco", "dev_night.json")
    images = a.images or os.path.join(a.assets_root, "data", "UAV-benchmark-M")
    os.makedirs(out_root, exist_ok=True)
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}
    for level in a.levels:
        stream = fixture_stream(coco, images, level)
        for m in a.methods:
            key = f"{level}/{m}"
            if key in results and load_trace(a.tag, level, m) is not None:
                print(f"[skip] {key} already done", flush=True)
                continue
            method = build_method(m, a.assets_root, seed=a.seed)
            trace = OnlineRunner(method, stream, read_bgr).run()
            trace.update({"threads": a.threads, "seed": a.seed, "level": level})
            d = os.path.join(out_root, level, m)
            os.makedirs(d, exist_ok=True)
            with gzip.open(os.path.join(d, "engine_trace.json.gz"), "wt") as f:
                json.dump(trace, f)
            row = evaluate(trace, level, m, a.ref)
            results = save_row(res_path, key, row)
            print(f"[{key}] {row['frames']} frames {row['sec']}s audit_failed={row['audit_failed']} "
                  f"exact={row.get('exact')} first_div={row.get('first_divergent_frame')} "
                  f"decisions={row['decisions']}", flush=True)
    print(f"[capture] {a.tag} complete", flush=True)


if __name__ == "__main__":
    main()
