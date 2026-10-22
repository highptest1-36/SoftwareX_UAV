# SPDX-License-Identifier: Apache-2.0
"""Mid-stream checkpoint/resume of the Detectron2 plugins, with a REAL process boundary.

  python3 -u tools/resume_check.py --phase interrupt --level GC --method mean_teacher --every 5 --stop 12
  python3 -u tools/resume_check.py --phase resume    --level GC --method mean_teacher --every 5 --ref ref
  (tools/run_resume_check.sh runs both phases as separate processes for several methods)

The interrupted run stops after `--stop` frames (frames after the last checkpoint are lost, as in a disconnect); a
new process resumes from the checkpoint. The full resumed trace is audited and compared field by field (predictions,
decisions, losses, every state hash) with the uninterrupted trace recorded by tools/trace_capture.py --tag <ref>.
Result rows: golden/resume/resume_results.json.
"""
import argparse
import json
import os
import sys

sys.dont_write_bytecode = True
SX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SX, "src"))
sys.path.insert(0, os.path.join(SX, "tools"))

import torch  # noqa: E402

import trace_capture as tc  # noqa: E402
from uav_shiftlab.backends.detectron2.common import read_bgr  # noqa: E402
from uav_shiftlab.backends.detectron2.plugins import PROFILES, build_method  # noqa: E402
from uav_shiftlab.core.runner import OnlineRunner, RunStore  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("interrupt", "resume"), required=True)
    ap.add_argument("--level", required=True, choices=list(tc.LEVELS))
    ap.add_argument("--method", required=True, choices=list(PROFILES))
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--stop", type=int, default=12)
    ap.add_argument("--ref", default="ref", help="tag of the uninterrupted reference traces (trace_capture.py)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--assets-root", default=tc.ASSETS)
    ap.add_argument("--coco", default="")
    ap.add_argument("--images", default="")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    coco = a.coco or os.path.join(a.assets_root, "data", "coco", "dev_night.json")
    images = a.images or os.path.join(a.assets_root, "data", "UAV-benchmark-M")
    stream = tc.fixture_stream(coco, images, a.level)
    method = build_method(a.method, a.assets_root, seed=0)
    run_dir = os.path.join(SX, "golden", "resume", a.level, a.method)
    runner = OnlineRunner(method, stream, read_bgr)
    if a.phase == "interrupt":
        RunStore(run_dir).reset()
        runner.run(run_dir=run_dir, checkpoint_every=a.every, stop_after=a.stop)
        print(f"[interrupt] {a.level}/{a.method}: stopped after {a.stop} frames "
              f"(checkpoint every {a.every})", flush=True)
        return
    trace = runner.run(run_dir=run_dir, checkpoint_every=a.every)
    row = tc.evaluate(trace, a.level, a.method, a.ref)
    row["resume_log"] = json.load(open(os.path.join(run_dir, RunStore.RESUMES)))
    row.update({"checkpoint_every": a.every, "interrupted_after": a.stop})
    tc.save_row(os.path.join(SX, "golden", "resume", "resume_results.json"), f"{a.level}/{a.method}", row)
    print(f"[resume] {a.level}/{a.method}: resumed at {row['resume_log'][0]['resume_at_frame']} "
          f"(recomputed {row['resume_log'][0]['frames_recomputed']}) audit_failed={row['audit_failed']} "
          f"exact={row.get('exact')} first_div={row.get('first_divergent_frame')}", flush=True)


if __name__ == "__main__":
    main()
