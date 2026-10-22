# SPDX-License-Identifier: Apache-2.0
"""Data types exchanged between the stream, the online runner and TTA methods.

A method only ever receives a `FrameView`: pixels plus stream position. Ground truth is not part of any type a
method can see; it is only joined with predictions by the evaluator after the stream has been processed.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass(frozen=True)
class FrameRef:
    """One entry of a stream manifest (no labels)."""
    image_id: int
    seq_id: str
    frame_id: int
    path: str


@dataclass(frozen=True)
class FrameView:
    """What a TTA method may observe at stream position `index`."""
    index: int
    image_id: int
    seq_id: str
    frame_id: int
    image: np.ndarray = field(repr=False)


@dataclass
class Detections:
    """Predictions for one frame, xyxy boxes in original image coordinates."""
    image_id: int
    boxes: np.ndarray            # [N, 4] float32 xyxy
    scores: np.ndarray           # [N] float32
    classes: np.ndarray          # [N] int64, 0-based

    raw: Optional[list] = field(default=None, repr=False)   # original COCO dicts, emitted verbatim if given

    @classmethod
    def from_coco(cls, image_id, dets, category_offset=1):
        """Wrap COCO-style dicts without altering their floating-point values."""
        b = np.asarray([d["bbox"] for d in dets], np.float64).reshape(-1, 4)
        xyxy = np.concatenate([b[:, :2], b[:, :2] + b[:, 2:]], 1)
        return cls(image_id, xyxy, [d["score"] for d in dets],
                   [d["category_id"] - category_offset for d in dets], raw=list(dets))

    def __post_init__(self):
        self.boxes = np.asarray(self.boxes, np.float32).reshape(-1, 4)
        self.scores = np.asarray(self.scores, np.float32).reshape(-1)
        self.classes = np.asarray(self.classes, np.int64).reshape(-1)
        if not (len(self.boxes) == len(self.scores) == len(self.classes)):
            raise ValueError("boxes, scores and classes must have the same length")

    def to_coco(self, category_offset=1):
        if self.raw is not None:
            return [dict(d) for d in self.raw]
        x1, y1, x2, y2 = (self.boxes[:, k] for k in range(4))
        return [{"image_id": int(self.image_id), "category_id": int(c) + category_offset,
                 "bbox": [float(a), float(b), float(c2 - a), float(d - b)], "score": float(s)}
                for a, b, c2, d, s, c in zip(x1, y1, x2, y2, self.scores, self.classes)]


@dataclass
class UpdateEvent:
    """Outcome of `adapt()` on one frame."""
    decision: str                         # "update" | "skip" | "no_adaptation"
    reason: str = ""
    losses: Dict[str, float] = field(default_factory=dict)
    extra: Optional[dict] = None
