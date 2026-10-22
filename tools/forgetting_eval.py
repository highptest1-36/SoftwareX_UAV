# SPDX-License-Identifier: Apache-2.0
"""Fixed-reference forgetting: evaluate the source model and every adapter snapshot of a run (uavsl run
--snapshot-every N) on the SAME reference subset (default: test_day, every 5th frame of every sequence) with frozen
adapters (no adaptation), so that the curve measures what adaptation to the target stream did to source performance.

  python3 -u tools/forgetting_eval.py --run-dir RUN_DIR --assets-root ASSETS [--split test_day] [--stride 5]
Output: <run-dir>/forgetting_<split>_s<stride>.json  {source, snapshots: {frame: metrics}, curve}
Resume-safe: snapshots already evaluated are kept; the source evaluation is cached per split/stride.
"""
import argparse
import glob
import json
import os
import sys

sys.dont_write_bytecode = True
SX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.environ.get("UAVSL_ASSETS_ROOT", os.path.dirname(SX))
sys.path.insert(0, os.path.join(SX, "src"))

import torch  # noqa: E402

from uav_shiftlab.backends.detectron2.common import build_cfg, read_bgr  # noqa: E402
from uav_shiftlab.backends.detectron2.runners import AdapterRunner, SourceOnlyRunner  # noqa: E402
from uav_shiftlab.backends.detectron2.plugins import D2CFG, WEIGHTS, profile_path  # noqa: E402
from uav_shiftlab.evaluation.coco_eval import evaluate  # noqa: E402
from uav_shiftlab.reproducibility.manifest import write_json  # noqa: E402

KEYS = ("AP50", "AP50_tiny", "AP50_small", "AP50_medium", "P@0.5", "R@0.5")


def predict_all(r, ims, image_root):
    dets = []
    for im in ims:
        dets += r.step(read_bgr(os.path.join(image_root, im["file_name"])), im["id"])
    return dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--split", default="test_day")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--assets-root", default=ASSETS, help="directory with checkpoints/ (source model)")
    ap.add_argument("--coco-dir", default="", help="directory with <split>.json (default: <assets>/data/coco)")
    ap.add_argument("--images", default="", help="image root (default: <assets>/data/UAV-benchmark-M)")
    a = ap.parse_args()
    coco_dir = a.coco_dir or os.path.join(a.assets_root, "data", "coco")
    image_root = a.images or os.path.join(a.assets_root, "data", "UAV-benchmark-M")
    if a.threads:
        torch.set_num_threads(a.threads)
    run_dir = a.run_dir if os.path.isabs(a.run_dir) else os.path.join(SX, a.run_dir)
    out = os.path.join(run_dir, f"forgetting_{a.split}_s{a.stride}.json")
    res = json.load(open(out)) if os.path.exists(out) else {"split": a.split, "stride": a.stride, "snapshots": {}}
    gt = json.load(open(os.path.join(coco_dir, a.split + ".json")))
    ims = sorted((im for im in gt["images"] if (im["frame_id"] - 1) % a.stride == 0),
                 key=lambda im: (im["video"], im["frame_id"]))
    keep = {im["id"] for im in ims}
    sub = dict(gt, images=ims, annotations=[x for x in gt["annotations"] if x["image_id"] in keep])
    cfg = build_cfg(profile_path(D2CFG))
    weights = os.path.join(a.assets_root, WEIGHTS)
    cache = os.path.join(SX, "experiments", "cache", f"source_{a.split}_s{a.stride}.json")
    if "source" not in res:
        if os.path.exists(cache):
            res["source"] = json.load(open(cache))
        else:
            r = SourceOnlyRunner(cfg, weights, {"method": "source_only", "score_thresh": 0.05})
            m = evaluate(sub, predict_all(r, ims, image_root), per_sequence=True)
            res["source"] = {"agnostic": m["agnostic"], "3class": m["3class"], "seq_mean": m["seq_mean"]}
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            write_json(cache, res["source"])
        write_json(out, res)
    snaps = sorted(glob.glob(os.path.join(run_dir, "snapshots", "teacher_*.pth")))
    for p in snaps:
        frame = os.path.basename(p)[8:14].lstrip("0") or "0"
        if frame in res["snapshots"]:
            continue
        r = AdapterRunner(cfg, weights, {"method": "frozen", "score_thresh": 0.05})   # predicts with the teacher
        r.frozen = True
        r.teacher.load_state_dict(torch.load(p, map_location=r.device))
        m = evaluate(sub, predict_all(r, ims, image_root), per_sequence=True)
        res["snapshots"][frame] = {"agnostic": m["agnostic"], "3class": m["3class"], "seq_mean": m["seq_mean"]}
        write_json(out, res)
        print(f"[forgetting] {os.path.basename(run_dir)} frame {frame}: AP50 agn {m['agnostic']['AP50']} "
              f"(source {res['source']['agnostic']['AP50']})", flush=True)
    src = res["source"]["agnostic"]
    res["curve"] = [{"frame": int(f), **{k: v["agnostic"][k] for k in KEYS},
                     **{f"delta_{k}": round(v["agnostic"][k] - src[k], 2) for k in KEYS}}
                    for f, v in sorted(res["snapshots"].items(), key=lambda kv: int(kv[0]))]
    res["n_frames"] = len(ims)
    write_json(out, res)
    print(f"[forgetting] {len(res['curve'])} snapshots -> {out}", flush=True)


if __name__ == "__main__":
    main()
