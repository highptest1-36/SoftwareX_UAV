# SPDX-License-Identifier: Apache-2.0
"""Command-line interface.

  uavsl demo     [--out DIR]                       synthetic day->night demo on CPU (no dataset, no Detectron2)
  uavsl run      --method M --coco SPLIT.json --images ROOT --assets-root ROOT --out RUN_DIR [...]
                                                   online run (resumes automatically after an interruption)
  uavsl audit    RUN_DIR                           protocol audit P1-P5 of a run
  uavsl eval     RUN_DIR --coco SPLIT.json          (re)compute metrics from the recorded online predictions
  uavsl compare  RUN_A RUN_B                       frame-by-frame comparison of two runs (A/A, A/B)
  uavsl validate RUN_DIR                           re-hash the inputs recorded in the run manifest
"""
import argparse
import json
import os
import queue
import sys
import threading
import time

from .core.runner import OnlineRunner, RunStore
from .core.stream import StreamManifest
from .protocols.audit import audit_trace
from .reproducibility.manifest import build_manifest, verify_manifest, write_json


# ---------------------------------------------------------------------------------------------------- helpers
class PrefetchLoader:
    """Reads the stream's images ahead in a background thread, starting at the first requested frame."""

    def __init__(self, paths, read, depth=8):
        self.paths, self.read, self.depth = list(paths), read, depth
        self.q, self.thread, self.pos = None, None, None

    def _work(self, start):
        for p in self.paths[start:]:
            self.q.put((p, self.read(p)))

    def __call__(self, path):
        if self.thread is None:
            self.q = queue.Queue(maxsize=self.depth)
            self.thread = threading.Thread(target=self._work, args=(self.paths.index(path),), daemon=True)
            self.thread.start()
        p, img = self.q.get()
        if p != path:
            raise RuntimeError(f"prefetch order mismatch: expected {path}, got {p}")
        return img


def thirds(gt, dets, image_ids):
    from .evaluation.coco_eval import evaluate
    n, out = len(image_ids), {}
    for name, sl in (("early", slice(0, n // 3)), ("mid", slice(n // 3, 2 * n // 3)), ("late", slice(2 * n // 3, n))):
        ids = set(image_ids[sl])
        sub = dict(gt, images=[im for im in gt["images"] if im["id"] in ids],
                   annotations=[a for a in gt["annotations"] if a["image_id"] in ids])
        r = evaluate(sub, [d for d in dets if d["image_id"] in ids], per_sequence=False)
        out[name] = {"mAP50": r["3class"]["AP50"] if r.get("3class") else None,
                     "AP50_agn": r["agnostic"]["AP50"] if r.get("agnostic") else None}
    return out


def evaluate_run(run_dir, coco_path, stream_ids=None):
    from .evaluation.coco_eval import evaluate
    gt = json.load(open(coco_path))
    st = RunStore(run_dir)
    dets = [d for r in st.records(RunStore.PREDS) for d in r["detections"]]
    ids = stream_ids or [r["image_id"] for r in st.records(RunStore.PREDS)]
    keep = set(ids)
    if len(keep) < len(gt["images"]):   # partial stream (e.g. --limit-per-seq): evaluate the processed frames only
        gt = dict(gt, images=[im for im in gt["images"] if im["id"] in keep],
                  annotations=[a for a in gt["annotations"] if a["image_id"] in keep])
    res = evaluate(gt, dets, per_sequence=True)
    res["thirds"] = thirds(gt, dets, ids)
    return res


def _split_name(coco_path):
    return os.path.splitext(os.path.basename(coco_path))[0]


# ---------------------------------------------------------------------------------------------------- commands
def cmd_demo(a):
    from .backends.mock.methods import AdaptiveThreshold, SourceOnly
    from .backends.mock.synthetic import make_dataset
    from .evaluation.coco_eval import evaluate
    coco, images = make_dataset()
    stream = StreamManifest.from_coco(coco, "")
    os.makedirs(a.out, exist_ok=True)
    print(f"synthetic stream: {len(stream)} frames, sequences {stream.sequence_order()} (day, night, night)")
    for m in (SourceOnly(), AdaptiveThreshold()):
        d = os.path.join(a.out, m.name)
        RunStore(d).reset()
        r = OnlineRunner(m, stream, lambda p: images[p])
        tr = r.run(run_dir=d, checkpoint_every=10)
        res = evaluate(coco, r.coco_detections(), modes=("agnostic",))
        rep = audit_trace(tr)
        write_json(os.path.join(d, "metrics.json"), res)
        write_json(os.path.join(d, "audit.json"), rep)
        write_json(os.path.join(d, "manifest.json"), build_manifest(method=m.name, stream=stream.fingerprint()))
        per = {s: v["agnostic_AP50"] for s, v in res["per_seq"].items()}
        print(f"  {m.name:20s} AP50 {res['agnostic']['AP50']:6.2f}  per sequence {per}  protocol "
              f"{'PASS' if rep['pass'] else 'FAIL'}  -> {d}")


def cmd_run(a):
    import torch
    import yaml
    from .backends.detectron2.common import read_bgr
    from .backends.detectron2.plugins import D2CFG, PROFILES, WEIGHTS, build_method, profile_path
    if a.threads:
        torch.set_num_threads(a.threads)
    split = _split_name(a.coco)
    if "test" in split and not a.allow_test:
        raise SystemExit(f"{split}: test split - pass --allow-test only for frozen configurations (never tune on test)")
    coco = json.load(open(a.coco))
    stream = StreamManifest.from_coco(coco, a.images, order=a.order, name=split, limit_per_seq=a.limit_per_seq)
    del coco  # labels are not needed (and not available) during the online run
    st = RunStore(a.out)
    if st.completed() and os.path.exists(os.path.join(a.out, "metrics.json")):
        print(f"[skip] {a.out} complete")
        return
    over = json.loads(a.overrides) if a.overrides else None
    init = os.path.abspath(a.init_adapters) if a.init_adapters else None
    method = build_method(a.method, a.assets_root, seed=a.seed, init_adapters=init, extra_overrides=over)
    r = method.r
    if a.frozen:   # inference only with the given adapters (e.g. --init-adapters); no adaptation
        r.frozen = True
    from .protocols.violations import apply as apply_violation
    method, runner_cls = apply_violation(a.violation, method)
    if a.violation != "none":
        print(f"[run] WARNING: deliberate protocol violation '{a.violation}' (impact study only)", flush=True)
    files = {"weights": os.path.join(a.assets_root, WEIGHTS), "tta_profile": profile_path(PROFILES[a.method]),
             "d2_config": profile_path(D2CFG), "source_stats": r.t.get("source_stats"), "split_json": a.coco,
             "init_adapters": init}
    mpath = os.path.join(a.out, "manifest.json")
    if os.path.exists(mpath) and st.has_checkpoint():
        man = json.load(open(mpath))
        bad = verify_manifest(man, files)
        if bad or man["stream_fingerprint"] != stream.fingerprint():
            raise SystemExit(f"refusing to resume: inputs changed since the run started: {bad or 'stream'}")
        man.setdefault("resumed_environments", []).append(build_manifest()["environment"])
    else:
        man = build_manifest(files, method=a.method, runner=type(r).__name__, seed=a.seed, split=split,
                             order=a.order, stream_fingerprint=stream.fingerprint(), n_frames=len(stream),
                             sequences=stream.sequence_order(), declared_stream=stream.image_ids(), overrides=over,
                             checkpoint_every=a.checkpoint_every, inputs={k: v for k, v in files.items() if v},
                             violation=a.violation, frozen=a.frozen,
                             command=" ".join(sys.argv))
        with open(os.path.join(a.out, "config.resolved.yaml"), "w") as f:
            yaml.safe_dump(r.t, f, sort_keys=False)
    write_json(mpath, man)
    loader = PrefetchLoader([f.path for f in stream.frames], read_bgr)
    t0 = time.time()
    print(f"[run] {a.method} on {split} ({a.order}): {len(stream)} frames, seed {a.seed} -> {a.out}", flush=True)
    runner = runner_cls(method, stream, loader)
    snap = None
    if a.snapshot_every:
        os.makedirs(os.path.join(a.out, "snapshots"), exist_ok=True)

        def snap(i, m):  # teacher adapters after frame i (fixed-reference forgetting, E5)
            if (i + 1) % a.snapshot_every == 0 or i + 1 == len(stream):
                torch.save(m.r.teacher.state_dict(), os.path.join(a.out, "snapshots", f"teacher_{i + 1:06d}.pth"))
    trace = runner.run(run_dir=a.out, checkpoint_every=a.checkpoint_every, on_frame_end=snap)
    wall = time.time() - t0
    rep = audit_trace(trace)
    write_json(os.path.join(a.out, "audit.json"), rep)
    res = evaluate_run(a.out, a.coco, stream.image_ids())
    nf = max(1, runner.processed)   # not r.stats: a reset_each_sequence run also resets the method's counters
    res["stream"] = {"stats": r.stats, "wall_s_this_invocation": round(wall, 1),
                     "ms_per_frame_this_invocation": round(1000 * wall / nf, 1),
                     "ms_breakdown": {k: round(1000 * v / nf, 1) for k, v in r.timing.items()},
                     "updates": sum(f["decision"] == "update" for f in trace["frames"]),
                     "trainable_params": r.student.num_params(),
                     "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
                     if torch.cuda.is_available() else None,
                     "resumes": json.load(open(st.path(RunStore.RESUMES))) if os.path.exists(st.path(RunStore.RESUMES)) else []}
    res["protocol_pass"] = rep["pass"]
    write_json(os.path.join(a.out, "metrics.json"), res)
    print(f"[done] {a.out}: AP50 agn {res['agnostic']['AP50']} tiny {res['agnostic']['AP50_tiny']} "
          f"P@.5 {res['agnostic']['P@0.5']} | protocol {'PASS' if rep['pass'] else 'FAIL'} | "
          f"{res['stream']['ms_per_frame_this_invocation']} ms/frame", flush=True)


def _trace(run_dir):
    man = json.load(open(os.path.join(run_dir, "manifest.json"))) if os.path.exists(os.path.join(run_dir, "manifest.json")) else {}
    frames = RunStore(run_dir).records()
    pol = {"student": "keep", "teacher": "keep", "optimizer": "keep", "method_state": "keep", "rng_torch": "keep",
           "rng_numpy": "keep"} if "runner" in man else {}
    declared = man.get("declared_stream") or [f["image_id"] for f in frames]
    return {"engine": True, "frames": frames, "boundary_policy": pol, "declared_stream": declared}


def cmd_audit(a):
    rep = audit_trace(_trace(a.run_dir))
    write_json(os.path.join(a.run_dir, "audit.json"), rep)
    for k, v in rep.items():
        if k != "pass":
            print(f"  {k:28s} {'PASS' if v['pass'] else 'FAIL'}  {v['violations'][:3] if v['violations'] else ''}")
    print("protocol:", "PASS" if rep["pass"] else "FAIL")
    return 0 if rep["pass"] else 1


def cmd_eval(a):
    res = evaluate_run(a.run_dir, a.coco)
    write_json(os.path.join(a.run_dir, "metrics.json"), res)
    print(json.dumps({k: res["agnostic"][k] for k in ("AP50", "AP50_tiny", "AP50_small", "P@0.5", "R@0.5")}))


def compare_runs(da, db):
    fa, fb = RunStore(da).records(), RunStore(db).records()
    out = {"frames": [len(fa), len(fb)], "decision_mismatch": 0, "emitted_count_mismatch": 0, "bbox_max_abs": 0.0,
           "score_max_abs": 0.0, "student_hash_mismatch": 0, "first_decision_divergence": None,
           "first_any_divergence": None}
    norm = lambda d: "skip" if str(d).startswith("skip") else d   # traces that record a single "skip" label
    for x, y in zip(fa, fb):
        diverged = False
        if norm(x["decision"]) != norm(y["decision"]):
            out["decision_mismatch"] += 1
            diverged = True
            if out["first_decision_divergence"] is None:
                out["first_decision_divergence"] = x["i"]
        ex, ey = x["emitted"], y["emitted"]
        if ex["n"] != ey["n"]:
            out["emitted_count_mismatch"] += 1
            diverged = True
        else:
            for k, key in (("bbox", "bbox_max_abs"), ("score", "score_max_abs")):
                import numpy as np
                if ex[k]:
                    v = float(np.abs(np.asarray(ex[k], float) - np.asarray(ey[k], float)).max())
                    out[key] = max(out[key], v)
                    diverged |= v > 0
        if x["state_post"].get("student") != y["state_post"].get("student"):
            out["student_hash_mismatch"] += 1
            diverged = True
        if diverged and out["first_any_divergence"] is None:
            out["first_any_divergence"] = x["i"]
    return out


def cmd_compare(a):
    print(json.dumps(compare_runs(a.run_a, a.run_b), indent=1))


def cmd_validate(a):
    from .backends.detectron2.plugins import PROFILE_DIR
    from .reproducibility.manifest import sha256_file
    man = json.load(open(os.path.join(a.run_dir, "manifest.json")))
    bad = verify_manifest(man, man.get("inputs", {}))
    prov = json.load(open(os.path.join(PROFILE_DIR, "PROFILE_SHA256.json")))
    bad_prof = [rel for rel, v in prov["files"].items() if sha256_file(os.path.join(PROFILE_DIR, rel)) != v["sha256"]]
    print("inputs unchanged since the run:", not bad, bad or "")
    print("profile files intact:          ", not bad_prof, bad_prof or "")
    print("code version of the run:       ", man.get("code"))
    return 0 if not (bad or bad_prof) else 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog="uavsl", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("demo")
    p.add_argument("--out", default="outputs/demo")
    p = sub.add_parser("run")
    p.add_argument("--method", required=True)
    p.add_argument("--coco", required=True, help="COCO json of the split (images need `video` and `frame_id`)")
    p.add_argument("--images", required=True, help="image root (file_name is relative to it)")
    p.add_argument("--assets-root", required=True, help="directory with checkpoints/ (source model, WHW stats)")
    p.add_argument("--out", required=True)
    p.add_argument("--order", default="original", help="original | reverse | perm:<k>")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint-every", type=int, default=250)
    p.add_argument("--snapshot-every", type=int, default=0, help="save teacher adapters every N frames (E5)")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--limit-per-seq", type=int, default=0)
    p.add_argument("--init-adapters", default="")
    p.add_argument("--overrides", default="", help="JSON dict merged into the profile (never for published runs)")
    p.add_argument("--allow-test", action="store_true")
    p.add_argument("--frozen", action="store_true", help="no adaptation (evaluate fixed adapters)")
    p.add_argument("--violation", default="none", choices=("none", "adapt_before_predict", "reset_each_sequence"),
                   help="deliberate protocol violation (Protocol-Violation Impact Study only)")
    for name in ("audit", "validate"):
        p = sub.add_parser(name)
        p.add_argument("run_dir")
    p = sub.add_parser("eval")
    p.add_argument("run_dir")
    p.add_argument("--coco", required=True)
    p = sub.add_parser("compare")
    p.add_argument("run_a")
    p.add_argument("run_b")
    a = ap.parse_args(argv)
    fn = {"demo": cmd_demo, "run": cmd_run, "audit": cmd_audit, "eval": cmd_eval, "compare": cmd_compare,
          "validate": cmd_validate}[a.cmd]
    return fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
