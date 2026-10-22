# SPDX-License-Identifier: Apache-2.0
"""Frame-by-frame comparison of two traces of the same stream (OnlineRunner schema), judged against a tolerance
profile. Discrete quantities (events, decisions, counts, state hashes) are reported as mismatch counts, continuous
ones (boxes, scores, losses) as maximum absolute / relative differences.

Typical uses: the same code twice on the same machine (A/A, expected: exact equality on CPU), two code versions on the
same machine (a refactoring must not change any field), or the same code on two machines (only the continuous
quantities may differ, within a tolerance fixed in advance).
"""
import math

import numpy as np

STATE_FIELDS = ("state_pre", "state_before", "state_after_predict", "state_post")


def _maxabs(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape:
        return math.inf
    return float(np.abs(a - b).max()) if a.size else 0.0


def compare(a, b) -> dict:
    fa, fb = a["frames"], b["frames"]
    d = {"n_frames_mismatch": int(len(fa) != len(fb)), "image_id_mismatch": 0, "events_mismatch": 0,
         "decision_mismatch": 0, "reason_mismatch": 0, "update_extra_mismatch": 0, "emitted_count_mismatch": 0,
         "category_mismatch": 0, "bbox_max_abs": 0.0, "score_max_abs": 0.0, "loss_keys_mismatch": 0,
         "loss_max_rel": 0.0, "state_hash_mismatch": {}, "first_divergent_frame": None}

    def mark(frame):
        if d["first_divergent_frame"] is None:
            d["first_divergent_frame"] = frame

    def bump(key, frame):
        d[key] += 1
        mark(frame)

    def worst(key, value, frame):
        d[key] = max(d[key], value)
        if value > 0:
            mark(frame)

    for x, y in zip(fa, fb):
        i = x["i"]
        for key in ("image_id", "events", "decision", "reason", "update_extra"):
            if x.get(key) != y.get(key):
                bump(f"{key}_mismatch", i)
        ex, ey = x["emitted"], y["emitted"]
        if ex["n"] != ey["n"]:
            bump("emitted_count_mismatch", i)
        else:
            worst("bbox_max_abs", _maxabs(ex["bbox"], ey["bbox"]), i)
            worst("score_max_abs", _maxabs(ex["score"], ey["score"]), i)
            if ex["category_id"] != ey["category_id"]:
                bump("category_mismatch", i)
        lx, ly = x.get("losses") or {}, y.get("losses") or {}
        if set(lx) != set(ly):
            bump("loss_keys_mismatch", i)
        for k in set(lx) & set(ly):
            worst("loss_max_rel", abs(lx[k] - ly[k]) / max(abs(lx[k]), 1e-12), i)
        for field in STATE_FIELDS:
            sx, sy = x.get(field) or {}, y.get(field) or {}
            for comp in set(sx) | set(sy):
                if sx.get(comp) != sy.get(comp):
                    key = f"{field}.{comp}"
                    d["state_hash_mismatch"][key] = d["state_hash_mismatch"].get(key, 0) + 1
                    mark(i)
    d["exact"] = d["first_divergent_frame"] is None and not d["n_frames_mismatch"]
    return d


def judge(diff: dict, profile: dict) -> dict:
    """profile: quantity -> maximum allowed value (missing quantities must be 0; state hashes must always match)."""
    failures = {k: v for k, v in diff.items()
                if k not in ("first_divergent_frame", "exact", "state_hash_mismatch") and v > profile.get(k, 0)}
    if diff["state_hash_mismatch"] and not profile.get("allow_state_hash_mismatch"):
        failures["state_hash_mismatch"] = diff["state_hash_mismatch"]
    return {"pass": not failures, "failures": failures}
