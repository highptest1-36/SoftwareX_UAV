# SPDX-License-Identifier: Apache-2.0
"""Collect the summaries of finished runs (metrics.json, audit.json, manifest.json, forgetting curves, run
comparisons) under a directory into one JSON file, without the large per-frame files.

  python3 tools/summarize_runs.py --root experiments/runs --out results/runs_summary.json [--methods ...]

A run directory is any directory with metrics.json and manifest.json; its key is the path relative to --root.
Run directories of the original research code (manifest field `implementation`) are labelled implementation
"original", engine runs (`uavsl run`) "engine". `compare_*.json` files (output of `uavsl compare`) are collected per
directory. Only the fields needed to rebuild the tables in docs/RESULTS.md are kept (tools/results_report.py).
"""
import argparse
import json
import os

AGN = ("AP", "AP50", "AP75", "AP50_tiny", "AP50_small", "AP50_medium", "AP50_large", "AR300", "P@0.5", "R@0.5")


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def run_summary(d):
    m, man = jl(os.path.join(d, "metrics.json")), jl(os.path.join(d, "manifest.json"))
    st, env = m.get("stream", {}), man.get("environment", {})
    audit = jl(os.path.join(d, "audit.json"))
    row = {"implementation": "original" if "implementation" in man else "engine",
           "method": man.get("method"), "seed": man.get("seed"), "split": man.get("split"),
           "order": man.get("order", "original"), "violation": man.get("violation", "none"),
           "frozen": bool(man.get("frozen")), "init_adapters": bool((man.get("inputs") or {}).get("init_adapters")),
           "n_frames": man.get("n_frames"), "stream_fingerprint": man.get("stream_fingerprint"),
           "input_sha256": {k: v for k, v in (man.get("sha256") or {}).items() if k in ("weights", "split_json")},
           "gpu": env.get("gpu") or env.get("cuda_device"), "torch": env.get("torch"),
           "agnostic": {k: m["agnostic"].get(k) for k in AGN},
           "3class_AP50": (m.get("3class") or {}).get("AP50"),
           "seq_mean_agnostic_AP50": (m.get("seq_mean") or {}).get("agnostic_AP50"),
           "ms_per_frame": st.get("ms_per_frame_this_invocation", st.get("ms_per_frame")),
           "peak_vram_gb": st.get("peak_vram_gb"),
           "updates": st.get("updates", (st.get("stats") or {}).get("updates")),
           "skipped": (st.get("stats") or {}).get("skipped"),
           "protocol_pass": audit["pass"] if audit else m.get("protocol_pass"),
           "audit_failed": sorted(k for k, v in audit.items() if k != "pass" and not v["pass"]) if audit else None}
    for f in sorted(os.listdir(d)):
        if f.startswith("forgetting_") and f.endswith(".json"):
            fg = jl(os.path.join(d, f))
            row.setdefault("forgetting", {})[f[len("forgetting_"):-5]] = {
                "source_AP50": fg["source"]["agnostic"]["AP50"], "n_frames": fg.get("n_frames"),
                "curve": [{k: c[k] for k in ("frame", "AP50", "delta_AP50", "delta_AP50_tiny", "delta_P@0.5")}
                          for c in fg.get("curve", [])]}
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", nargs="+", default=["source_only", "mean_teacher", "tent", "whw"])
    a = ap.parse_args()
    runs, compares = {}, {}
    for d, _, files in sorted(os.walk(a.root)):
        rel = os.path.relpath(d, a.root)
        if "metrics.json" in files and "manifest.json" in files:
            r = run_summary(d)
            if r["method"] in a.methods:
                runs[rel] = r
        for f in files:
            if f.startswith("compare_") and f.endswith(".json") and any(f.endswith(f"_{m}.json") for m in a.methods):
                c = jl(os.path.join(d, f))
                c.pop("restores", None)   # counter of a branch none of these methods has
                compares[os.path.join(rel, f)] = c
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"runs": runs, "compares": compares}, open(a.out, "w"), indent=1, sort_keys=True)
    print(f"[summarize] {len(runs)} runs, {len(compares)} comparisons -> {a.out}")


if __name__ == "__main__":
    main()
