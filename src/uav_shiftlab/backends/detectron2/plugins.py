# SPDX-License-Identifier: Apache-2.0
"""Detectron2 methods as UAV-ShiftLab plugins (OnlineMethod), built from the shipped `uavdt` profiles.

    method = build_method("mean_teacher", assets_root="/path/to/assets", seed=0)
    OnlineRunner(method, stream, load=read_bgr).run()

`assets_root` holds the source checkpoint and the WHW source statistics (not redistributed with the code). The
detector and TTA configurations are the YAML files in `profiles/uavdt/` (SHA-256 in PROFILE_SHA256.json).
"""
import copy
import os
import random

import yaml

from ...core.method import KEEP, OnlineMethod
from ...core.types import Detections, UpdateEvent
from .fingerprint import runner_fingerprint

PROFILE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles", "uavdt")
D2CFG = "detector.yaml"
WEIGHTS = "checkpoints/source_day_r50_1080/model_final.pth"
# label -> TTA yaml in the profile directory (the runner class is selected by the yaml's `method`)
PROFILES = {
    "source_only": "tta/source_only.yaml",
    "mean_teacher": "tta/mean_teacher.yaml",
    "tent": "tta/tent.yaml",
    "whw": "tta/whw.yaml",
}


def profile_path(rel):
    return os.path.join(PROFILE_DIR, rel)


def load_profile(label, extra_overrides=None):
    return dict(yaml.safe_load(open(profile_path(PROFILES[label]))), **(extra_overrides or {}))


class D2Method(OnlineMethod):
    """Wraps a runner of `runners.py`. Every state component is kept across sequence boundaries (one continual
    stream); none of the shipped methods has per-sequence state."""
    boundary_policy = {"student": KEEP, "teacher": KEEP, "optimizer": KEEP, "method_state": KEEP,
                       "rng_torch": KEEP, "rng_numpy": KEEP}

    def __init__(self, runner, name):
        self.r, self.name = runner, name

    def on_sequence_start(self, seq_id):
        self.r.new_sequence()

    def predict(self, view):
        dets, cache = self.r.predict_part(view.image, view.image_id)
        return Detections.from_coco(view.image_id, dets), cache

    def adapt(self, view, cache):
        decision, reason, losses, extra = self.r.adapt_part(cache)
        return UpdateEvent(decision, reason, losses, extra)

    # ---- mid-stream checkpoint / resume: everything that evolves during adaptation
    def state_dict(self):
        import numpy as np
        import torch
        r = self.r
        sd = {"student": copy.deepcopy(r.student.state_dict()), "teacher": copy.deepcopy(r.teacher.state_dict()),
              "optimizer": copy.deepcopy(r.opt.state_dict()), "method_state": copy.deepcopy(r.method_state()),
              "stats": copy.deepcopy(r.stats), "timing": copy.deepcopy(r.timing),
              "rng_torch": torch.get_rng_state(), "rng_numpy": np.random.get_state(), "rng_python": random.getstate()}
        if torch.cuda.is_available():
            sd["rng_cuda"] = torch.cuda.get_rng_state_all()
        return sd

    def load_state_dict(self, sd):
        import numpy as np
        import torch
        r = self.r
        r.student.load_state_dict(sd["student"])
        r.teacher.load_state_dict(sd["teacher"])
        r.opt.load_state_dict(sd["optimizer"])
        if sd["method_state"] is not None:
            r.load_method_state(sd["method_state"])
        r.stats, r.timing = copy.deepcopy(sd["stats"]), copy.deepcopy(sd["timing"])
        torch.set_rng_state(sd["rng_torch"])
        np.random.set_state(sd["rng_numpy"])
        random.setstate(sd["rng_python"])
        if "rng_cuda" in sd and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(sd["rng_cuda"])

    def state_fingerprint(self):
        return runner_fingerprint(self.r)


def build_method(label, assets_root, seed=0, init_adapters=None, extra_overrides=None):
    import torch
    from .common import build_cfg
    from .runners import RUNNERS

    tcfg = load_profile(label, extra_overrides)
    if tcfg.get("source_stats") and not os.path.isabs(tcfg["source_stats"]):
        tcfg["source_stats"] = os.path.join(assets_root, tcfg["source_stats"])
    cfg = build_cfg(profile_path(D2CFG))   # weights are loaded from `assets_root` by the runner itself
    r = RUNNERS[tcfg["method"]](cfg, os.path.join(assets_root, WEIGHTS), tcfg, seed=seed)
    if init_adapters:
        sd = torch.load(init_adapters, map_location="cpu")
        r.student.load_state_dict(sd)
        r.teacher.load_state_dict(sd)
    return D2Method(r, label)
