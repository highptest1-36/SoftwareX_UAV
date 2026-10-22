# SPDX-License-Identifier: Apache-2.0
"""Render docs/RESULTS.md from a run summary (tools/summarize_runs.py). Every number is read from the summary.

  python3 tools/results_report.py --summary results/runs_summary.json --out docs/RESULTS.md

Expected layout of the run keys (as produced by experiments/run_experiments.sh, plus the reproduction study):
  reproduction/<study>/{orig_a,orig_b,engine}/<method>   original research code twice (A, A') and the engine once (B)
  reproduction/timing/engine/<method>                    optional re-measurement of the engine's time per frame
  e1_seeds/<method>_s<seed>                             main runs (snapshots -> forgetting curves, E5)
  e2_violations/{viol_abp,viol_reset,viol_ate}/<method>  deliberate protocol violations
  e4_order/<method>_<order>                             other stream orders (seed 0)
"""
import argparse
import json
import os
import statistics as stt
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from uav_shiftlab.diagnostics.order import order_sensitivity  # noqa: E402

METHODS = ("source_only", "mean_teacher", "tent", "whw")
NAME = {"source_only": "Source only", "mean_teacher": "Mean Teacher", "tent": "Tent", "whw": "WHW"}
VIOL = (("viol_abp", "adapt-before-predict"), ("viol_reset", "silent reset at every sequence"),
        ("viol_ate", "adapt-then-evaluate (final adapters re-run frozen)"))
ORDERS = ("original", "reverse", "perm1", "perm2", "perm3", "perm4")


def f2(x):
    return "–" if x is None else f"{x:.2f}"


def reproduction(S):
    runs, cmp = S["runs"], S["compares"]
    L = ["## 1. Engine vs. original research code (test_night, seed 0, one GPU)", "",
         "The original research code was run twice (A, A') and the UAV-ShiftLab engine once (B) with the same inputs. "
         "A/A' shows the run-to-run variation of the GPU itself; B is expected to differ from A by no more than A' does. "
         "Decisions = per-frame update/skip decisions. The last column of the second table gives the first frame "
         "whose emitted boxes, scores or adapter state differ: without adaptation all runs are identical; with "
         "adaptation, A and A' already differ after the first update, because GPU kernels are not bitwise "
         "deterministic.", "",
         "| method | AP50 A | AP50 A' | AP50 B | B − A | A' − A | decision mismatches A/A' · A/B | audit B | ms/frame A, A' → B (ratio) |",
         "|---|---:|---:|---:|---:|---:|---|:-:|---|"]
    for m in METHODS:
        study = next((s for s in ("pilot", "baselines") if f"reproduction/{s}/engine/{m}" in runs), None)
        if study is None:
            continue
        g = f"reproduction/{study}"
        a, a2, b = (runs.get(f"{g}/{k}/{m}") for k in ("orig_a", "orig_b", "engine"))
        if not (a and a2 and b):
            continue
        A, A2, B = a["agnostic"]["AP50"], a2["agnostic"]["AP50"], b["agnostic"]["AP50"]
        dm = [cmp.get(f"{g}/compare_{p}_{m}.json", {}).get("decision_mismatch") for p in ("AA", "AB")]
        t = runs.get(f"reproduction/timing/engine/{m}", b)
        t_orig = stt.mean([a["ms_per_frame"], a2["ms_per_frame"]])
        L.append(f"| {NAME[m]} | {f2(A)} | {f2(A2)} | {f2(B)} | {B - A:+.2f} | {A2 - A:+.2f} | "
                 f"{dm[0] if dm[0] is not None else '–'} · {dm[1] if dm[1] is not None else '–'} | "
                 f"{'pass' if b['protocol_pass'] else 'FAIL'} | {a['ms_per_frame']}, {a2['ms_per_frame']} → "
                 f"{t['ms_per_frame']} ({t['ms_per_frame'] / t_orig:.2f}×) |")
    L += ["", "| pair | method | decision mismatches | frames with a different number of boxes | first differing frame |",
          "|---|---|---:|---:|---:|"]
    for k, c in sorted(cmp.items()):
        name = k.rsplit("/", 1)[-1][len("compare_"):-len(".json")]
        pair, m = name.split("_", 1)
        if pair in ("AA", "AB", "AprimeB") and m in METHODS:
            pl = {"AA": "A / A'", "AB": "A / B", "AprimeB": "A' / B"}[pair]
            fd = c["first_any_divergence"]
            L.append(f"| {pl} | {NAME[m]} | {c['decision_mismatch']} | {c['emitted_count_mismatch']} | "
                     f"{'none' if fd is None else fd} |")
    L += ["", "The engine's extra time per frame is the price of the per-frame state fingerprints (SHA-256 of adapters, "
          "optimizer state and RNG states before and after each frame) that the protocol audit needs.", ""]
    return L


def e1(S):
    runs = S["runs"]
    L = ["## 2. Main runs (test_night, seeds 0–2) and forgetting", "",
         "Mean ± standard deviation over seeds (source only is deterministic given the stream: one run). "
         "Class-agnostic AP50, AP50 on tiny objects (< 16 px), precision at IoU 0.5, 3-class mAP50.", "",
         "| method | runs | AP50 | AP50 tiny | P@0.5 | mAP50 (3 classes) | audits passed | forgetting ΔAP50 on test_day (final snapshot) |",
         "|---|---:|---|---|---|---|:-:|---:|"]
    for m in METHODS:
        rs = [r for k, r in sorted(runs.items()) if k.startswith(f"e1_seeds/{m}_s")]
        if not rs:
            continue

        def ms_(get):
            v = [get(r) for r in rs]
            return f"{stt.mean(v):.2f}" + (f" ± {stt.stdev(v):.2f}" if len(v) > 1 else "")
        fg = next((r["forgetting"] for r in rs if r.get("forgetting") and r["seed"] == 0), None)
        fgs = "–"
        if fg:
            c = next(iter(fg.values()))
            fgs = f"{c['curve'][-1]['delta_AP50']:+.2f} (source {c['source_AP50']:.2f})" if c["curve"] else "–"
        L.append(f"| {NAME[m]} | {len(rs)} | {ms_(lambda r: r['agnostic']['AP50'])} | "
                 f"{ms_(lambda r: r['agnostic']['AP50_tiny'])} | {ms_(lambda r: r['agnostic']['P@0.5'])} | "
                 f"{ms_(lambda r: r['3class_AP50'])} | {sum(bool(r['protocol_pass']) for r in rs)}/{len(rs)} | {fgs} |")
    L += ["", "Forgetting: the seed-0 run's teacher adapters, saved every 1000 frames, are evaluated frozen on a fixed "
          "reference subset of the source domain (test_day, every 5th frame); Δ is relative to the source model.", ""]
    for m in METHODS:
        r = runs.get(f"e1_seeds/{m}_s0")
        if r and r.get("forgetting"):
            c = next(iter(r["forgetting"].values()))
            L.append(f"- {NAME[m]}: " + ", ".join(f"{p['frame']}: {p['delta_AP50']:+.2f}" for p in c["curve"]))
    return L + [""]


def e2(S):
    runs = S["runs"]
    L = ["## 3. Protocol-violation impact (test_night, seed 0)", "",
         "Each violation is injected deliberately into an otherwise identical run; Δ is relative to the correct seed-0 "
         "run. The audit column lists the checks that flag the run.", "",
         "| method | violation | AP50 | Δ AP50 | Δ AP50 tiny | Δ P@0.5 | flagged by |", "|---|---|---:|---:|---:|---:|---|"]
    for m in METHODS:
        base = runs.get(f"e1_seeds/{m}_s0")
        for d, label in VIOL:
            x = runs.get(f"e2_violations/{d}/{m}")
            if not (x and base):
                continue
            dd = {k: x["agnostic"][k] - base["agnostic"][k] for k in ("AP50", "AP50_tiny", "P@0.5")}
            L.append(f"| {NAME[m]} | {label} | {f2(x['agnostic']['AP50'])} | {dd['AP50']:+.2f} | "
                     f"{dd['AP50_tiny']:+.2f} | {dd['P@0.5']:+.2f} | {', '.join(x['audit_failed'] or []) or 'none'} |")
    L += ["", "Adapt-then-evaluate re-uses the adapters of the finished stream on the same frames: it is not a trace "
          "violation of the online run (no check can flag it), but it is not an online result either.", ""]
    return L


def e4(S):
    runs = S["runs"]
    scores = {}
    for m in METHODS:
        b = runs.get(f"e1_seeds/{m}_s0")
        if b:
            scores.setdefault("original", {})[m] = b["agnostic"]["AP50"]
        for o in ORDERS[1:]:
            x = runs.get(f"e4_order/{m}_{o}")
            if x:
                scores.setdefault(o, {})[m] = x["agnostic"]["AP50"]
    full = {o: v for o, v in scores.items() if len(v) == len(METHODS)}
    if len(full) < 2:
        return []
    r = order_sensitivity(full)
    orders = [o for o in ORDERS if o in r["orders"]]
    L = ["## 4. Stream-order sensitivity (test_night, seed 0, AP50)", "",
         "Same frames, sequences presented in a different order (frames inside a sequence keep their order).", "",
         "| method | " + " | ".join(orders) + " | spread |", "|---|" + "---:|" * (len(orders) + 1)]
    for m in METHODS:
        pm = r["per_method"][m]
        L.append(f"| {NAME[m]} | " + " | ".join(f2(pm["by_order"][o]) for o in orders) + f" | {pm['spread']:.2f} |")
    taus = list(r["kendall_tau"].values())
    L += ["", f"Ranking of the four methods identical across orders: {'yes' if r['ranking_identical'] else 'no'}; "
              f"Kendall τ between orders: min {min(taus):.2f}, max {max(taus):.2f}.", ""]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/runs_summary.json")
    ap.add_argument("--out", default="docs/RESULTS.md")
    a = ap.parse_args()
    S = json.load(open(a.summary))
    gpus = sorted({r["gpu"] for r in S["runs"].values() if r.get("gpu")})
    L = ["# Results", "",
         f"Generated by `tools/results_report.py` from `{a.summary}` (do not edit by hand). UAVDT, source model "
         f"trained on daylight sequences; online adaptation on the night test sequences. GPU: {', '.join(gpus)}; bf16.",
         "",
         "The runs were recorded with pre-release revisions of the engine. The released code reproduces the per-frame "
         "traces of those revisions bit for bit on CPU fixtures for all four methods (predictions, decisions, losses "
         "and every state fingerprint; `tools/trace_capture.py --ref`).", ""]
    for part in (reproduction, e1, e2, e4):
        L += part(S)
    open(a.out, "w").write("\n".join(L))
    print(f"[results] -> {a.out}")


if __name__ == "__main__":
    main()
