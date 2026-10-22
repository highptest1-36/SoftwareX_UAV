# SPDX-License-Identifier: Apache-2.0
"""Protocol audit of an online-TTA trace recorded by uav_shiftlab.core.runner.OnlineRunner.

Checks (docs/PROTOCOL_SPEC.md):
  P1 predict-before-update  the emitted predictions of frame t come from the state theta_{t-1}
  P2 state continuity       nothing changes the state between frames; boundary changes only where declared "reset"
  P3 boundary policy        a sequence_start event exactly at every sequence boundary
  P4 stream integrity       processed frames == declared stream (no duplicate, missing or reordered frame)
  P5 single emission        exactly one emission per frame, only for that frame's image id
"""
MAX_LISTED = 20


def _result(viol):
    return {"pass": not viol, "n_violations": len(viol), "violations": viol[:MAX_LISTED]}


def check_p1(trace):
    viol = []
    for f in trace["frames"]:
        ev = f["events"]
        if "adapt" in ev and ev.index("adapt") < ev.index("predict"):
            viol.append({"i": f["i"], "detail": "adapt() called before predict()"})
        elif f["state_after_predict"] != f["state_before"]:
            viol.append({"i": f["i"], "detail": "predict() changed the method state"})
    return _result(viol)


def check_p2(trace):
    viol, policy = [], trace.get("boundary_policy") or {}
    frames = trace["frames"]
    for prev, f in zip(frames, frames[1:]):
        post_prev = prev["state_post"]
        if f["state_pre"] != post_prev:
            viol.append({"i": f["i"], "detail": "state changed between frames outside the runner"})
        pre = f["state_before"]
        for comp, h in pre.items():
            if post_prev.get(comp) == h:
                continue
            if f.get("boundary") and policy.get(comp) == "reset":
                continue
            viol.append({"i": f["i"], "image_id": f["image_id"], "component": comp,
                         "detail": "state at frame start differs from the end of the previous frame"
                                   + (" (sequence boundary, policy keep)" if f.get("boundary") else "")})
    return _result(viol)


def check_p3(trace):
    viol = []
    for f in trace["frames"]:
        if bool(f["boundary"]) != ("sequence_start" in f["events"]):
            viol.append({"i": f["i"], "detail": "sequence_start event does not match the sequence boundary"})
    return _result(viol)


def check_p4(trace):
    seen = [f["image_id"] for f in trace["frames"]]
    declared = list(trace["declared_stream"])
    viol = []
    dup = sorted({x for x in seen if seen.count(x) > 1})
    if dup:
        viol.append({"detail": "duplicate frames", "image_ids": dup})
    missing = sorted(set(declared) - set(seen))
    if missing:
        viol.append({"detail": "declared frames never processed", "image_ids": missing})
    extra = sorted(set(seen) - set(declared))
    if extra:
        viol.append({"detail": "frames not in the declared stream", "image_ids": extra})
    if not viol and seen != declared:
        viol.append({"detail": "frames processed in a different order than declared"})
    return _result(viol)


def check_p5(trace):
    viol = []
    for f in trace["frames"]:
        if f["events"].count("emit") != 1:
            viol.append({"i": f["i"], "detail": f"{f['events'].count('emit')} emissions"})
        if f["emitted"]["n"] and f["emitted"]["image_ids"] != [f["image_id"]]:
            viol.append({"i": f["i"], "detail": "predictions emitted for another image id"})
    return _result(viol)


CHECKS = {"P1_predict_before_update": check_p1, "P2_state_continuity": check_p2, "P3_boundary_policy": check_p3,
          "P4_stream_integrity": check_p4, "P5_single_emission": check_p5}


def audit_trace(trace) -> dict:
    res = {name: fn(trace) for name, fn in CHECKS.items()}
    res["pass"] = all(r["pass"] for r in res.values())
    return res
