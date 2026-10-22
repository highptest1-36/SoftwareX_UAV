# SPDX-License-Identifier: Apache-2.0
"""Toy detector and toy online methods on the synthetic streams (NumPy + SciPy only).

ThresholdDetector: connected components of pixels above a threshold.
SourceOnly:        fixed threshold calibrated on day frames (never adapts).
AdaptiveThreshold: label-free online adaptation - the threshold follows an EMA of a high intensity quantile of
                   the incoming frames (a stand-in for test-time normalisation / self-training).
"""
import hashlib
import struct

import numpy as np
from scipy import ndimage

from ...core.method import KEEP, OnlineMethod
from ...core.types import Detections, UpdateEvent


def detect(img, thr, image_id, min_area=12):
    lab, n = ndimage.label(img > thr)
    boxes, scores = [], []
    for k, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        ys, xs = sl
        area = (ys.stop - ys.start) * (xs.stop - xs.start)
        if area < min_area:
            continue
        boxes.append([xs.start, ys.start, xs.stop, ys.stop])
        inside = img[sl][lab[sl] == k]
        scores.append(float(np.clip((inside.mean() - thr) / max(1e-6, 1 - thr) + 0.5, 0, 1)))
    return Detections(image_id, np.asarray(boxes, np.float32), np.asarray(scores, np.float32),
                      np.zeros(len(boxes), np.int64))


def _h(*vals):
    return hashlib.sha256(struct.pack(f"{len(vals)}d", *vals)).hexdigest()


class SourceOnly(OnlineMethod):
    name = "source_only"

    def __init__(self, thr=0.6):
        self.thr = thr

    def predict(self, view):
        return detect(view.image, self.thr, view.image_id), None

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass

    def state_fingerprint(self):
        return {"threshold": _h(self.thr)}


class AdaptiveThreshold(OnlineMethod):
    name = "adaptive_threshold"
    boundary_policy = {"threshold": KEEP, "frames_seen": KEEP}

    def __init__(self, thr0=0.6, momentum=0.6, quantile=99.0, ratio=0.67):
        self.thr0, self.momentum, self.quantile, self.ratio = thr0, momentum, quantile, ratio
        self.thr, self.n = thr0, 0

    def on_stream_start(self):
        self.thr, self.n = self.thr0, 0

    def predict(self, view):
        return detect(view.image, self.thr, view.image_id), None

    def adapt(self, view, cache):
        target = self.ratio * float(np.percentile(view.image, self.quantile))
        self.thr = self.momentum * self.thr + (1 - self.momentum) * target
        self.n += 1
        return UpdateEvent("update", losses={"threshold": self.thr})

    def state_dict(self):
        return {"thr": self.thr, "n": self.n}

    def load_state_dict(self, state):
        self.thr, self.n = state["thr"], state["n"]

    def state_fingerprint(self):
        return {"threshold": _h(self.thr), "frames_seen": _h(float(self.n))}
