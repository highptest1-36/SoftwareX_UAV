# SPDX-License-Identifier: Apache-2.0
"""Stream manifests: an ordered, label-free list of frames, and the sequence orders.

Orders (sequence level; frames inside a sequence always stay in temporal order):
  original   sequences sorted by id
  reverse    reversed original order
  perm:<k>   original order shuffled with random.Random(k)
"""
import hashlib
import json
import os
import random
from typing import Callable, Iterator, List, Optional

import numpy as np

from .types import FrameRef, FrameView


class StreamManifest:
    def __init__(self, frames: List[FrameRef], name: str = "", order: str = "original"):
        self.frames, self.name, self.order = list(frames), name, order

    @classmethod
    def from_coco(cls, coco: dict, image_root: str, order: str = "original", name: str = "",
                  sequences: Optional[List[str]] = None, limit_per_seq: int = 0) -> "StreamManifest":
        """Build a manifest from a COCO dict whose images carry `video` and `frame_id`. Annotations are ignored."""
        seqs = sorted({im["video"] for im in coco["images"]}) if sequences is None else list(sequences)
        if order == "reverse":
            seqs = seqs[::-1]
        elif order.startswith("perm:"):
            random.Random(int(order.split(":", 1)[1])).shuffle(seqs)
        elif order != "original":
            raise ValueError(f"unknown order {order!r}")
        frames = []
        for s in seqs:
            ims = sorted((im for im in coco["images"] if im["video"] == s), key=lambda im: im["frame_id"])
            if limit_per_seq:
                ims = ims[:limit_per_seq]
            frames += [FrameRef(int(im["id"]), s, int(im["frame_id"]), os.path.join(image_root, im["file_name"]))
                       for im in ims]
        return cls(frames, name=name, order=order)

    def sequence_order(self) -> List[str]:
        out = []
        for f in self.frames:
            if not out or out[-1] != f.seq_id:
                out.append(f.seq_id)
        return out

    def image_ids(self) -> List[int]:
        return [f.image_id for f in self.frames]

    def fingerprint(self) -> str:
        """SHA256 of the ordered (image_id, seq_id, frame_id) list - identical streams have identical fingerprints."""
        payload = json.dumps([(f.image_id, f.seq_id, f.frame_id) for f in self.frames]).encode()
        return hashlib.sha256(payload).hexdigest()

    def iter_views(self, load: Callable[[str], np.ndarray], start: int = 0) -> Iterator[FrameView]:
        for i, f in enumerate(self.frames):
            if i < start:
                continue
            yield FrameView(index=i, image_id=f.image_id, seq_id=f.seq_id, frame_id=f.frame_id, image=load(f.path))

    def __len__(self):
        return len(self.frames)
