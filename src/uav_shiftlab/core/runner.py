# SPDX-License-Identifier: Apache-2.0
"""Online runner: predict -> record -> adapt, one frame at a time, with a full protocol trace.

    y_t     = f_{theta_{t-1}}(x_t)     recorded before anything else happens on frame t
    theta_t = U(theta_{t-1}, x_t)

The trace is checked by uav_shiftlab.protocols.audit and compared by uav_shiftlab.reproducibility.trace_compare.

Persistent mode (`run(run_dir=...)`): every frame record is appended to trace.jsonl and predictions.jsonl, and a
checkpoint (complete method state + stream position) is written atomically every `checkpoint_every` frames. A run
that is interrupted (Colab disconnect, crash) resumes from the last checkpoint: both files are truncated to the
checkpointed number of frames, the method state is restored, and the frames after the checkpoint are processed again
- so no frame is duplicated or lost, and the resumed run equals an uninterrupted one.
"""
import json
import os
import pickle
import time
from typing import Callable, Dict, List

import numpy as np

from .method import OnlineMethod
from .stream import StreamManifest
from .types import Detections


class RunStore:
    """Append-only per-frame files + atomic checkpoints in a run directory."""
    TRACE, PREDS, CKPT, RESUMES, DONE = "trace.jsonl", "predictions.jsonl", "checkpoint.pkl", "resume_log.json", "COMPLETE"

    def __init__(self, run_dir):
        self.dir = run_dir
        self._fh = {}           # append handles kept open: re-opening a file per frame is slow on network drives
        os.makedirs(run_dir, exist_ok=True)

    def path(self, name):
        return os.path.join(self.dir, name)

    def completed(self):
        return os.path.exists(self.path(self.DONE))

    def has_checkpoint(self):
        return os.path.exists(self.path(self.CKPT))

    def reset(self):
        self.close()
        for n in (self.TRACE, self.PREDS, self.CKPT, self.RESUMES, self.DONE):
            if os.path.exists(self.path(n)):
                os.remove(self.path(n))

    @staticmethod
    def _complete_lines(path):
        if not os.path.exists(path):
            return []
        with open(path) as f:
            lines = f.read().split("\n")
        return lines[:-1]       # the last element is "" or an incomplete line left by a crash

    def truncate(self, n):
        self.close()
        for name in (self.TRACE, self.PREDS):
            keep = self._complete_lines(self.path(name))[:n]
            tmp = self.path(name + ".tmp")
            with open(tmp, "w") as f:
                f.write("".join(ln + "\n" for ln in keep))
            os.replace(tmp, self.path(name))

    def append(self, record, coco):
        """Flushed per frame (visible to readers); made durable by `sync()` before every checkpoint. A crash can
        only lose frames after the last checkpoint, which a resume truncates and recomputes anyway."""
        for name, obj in ((self.TRACE, record), (self.PREDS, {"i": record["i"], "image_id": record["image_id"],
                                                              "detections": coco})):
            if name not in self._fh:
                self._fh[name] = open(self.path(name), "a")
            self._fh[name].write(json.dumps(obj) + "\n")
            self._fh[name].flush()

    def sync(self):
        for f in self._fh.values():
            f.flush()
            os.fsync(f.fileno())

    def close(self):
        self.sync()
        for f in self._fh.values():
            f.close()
        self._fh = {}

    def records(self, name=None):
        return [json.loads(ln) for ln in self._complete_lines(self.path(name or self.TRACE))]

    def save_checkpoint(self, ck):
        self.sync()             # every frame the checkpoint covers is on disk before the checkpoint is
        tmp = self.path(self.CKPT + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(ck, f, protocol=4)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path(self.CKPT))

    def load_checkpoint(self):
        with open(self.path(self.CKPT), "rb") as f:
            return pickle.load(f)

    def log_resume(self, entry):
        log = json.load(open(self.path(self.RESUMES))) if os.path.exists(self.path(self.RESUMES)) else []
        log.append(entry)
        json.dump(log, open(self.path(self.RESUMES), "w"), indent=1)


class OnlineRunner:
    def __init__(self, method: OnlineMethod, stream: StreamManifest, load: Callable[[str], np.ndarray]):
        self.method, self.stream, self.load = method, stream, load
        self.predictions: List[Detections] = []
        self.store = None

    # hooks for deliberately wrong runners in negative tests; the default order is the protocol
    def _process(self, view, ev: List[str], rec: Dict):
        dets, cache = self.method.predict(view)
        ev.append("predict")
        rec["state_after_predict"] = self.method.state_fingerprint()
        ev.append("emit")
        upd = self.method.adapt(view, cache)
        ev.append("adapt")
        return dets, upd

    def _views(self, start=0):
        return self.stream.iter_views(self.load, start)

    def _header(self):
        m = self.method
        return {"method": m.name, "runner": type(self).__name__, "engine": True,
                "boundary_policy": dict(m.boundary_policy), "declared_stream": self.stream.image_ids(),
                "stream_fingerprint": self.stream.fingerprint()}

    def run(self, run_dir=None, checkpoint_every=0, stop_after=None, on_frame_end=None):
        """Process the stream. With `run_dir`, results are persisted and the run resumes from its last checkpoint.
        `stop_after` (tests) stops after that many frames of this invocation, emulating an interruption.
        `on_frame_end(index, method)` is called after each frame is recorded (e.g. to save adapter snapshots)."""
        m = self.method
        start, prev_seq, frames = 0, None, []
        if run_dir is not None:
            self.store = RunStore(run_dir)
            if self.store.completed():
                return self.trace()
            if self.store.has_checkpoint():
                ck = self.store.load_checkpoint()
                if ck["stream_fingerprint"] != self.stream.fingerprint():
                    raise ValueError("checkpoint belongs to a different stream")
                m.load_state_dict(ck["method_state"])
                start, prev_seq = ck["next_index"], ck["prev_seq"]
                lost = len(self.store.records()) - start
                self.store.truncate(start)
                self.store.log_resume({"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                       "resume_at_frame": start, "frames_recomputed": max(0, lost)})
            else:
                self.store.reset()
                m.on_stream_start()
        else:
            m.on_stream_start()
        done = self.processed = 0
        for view in self._views(start):
            ev, rec = [], {}
            boundary = view.seq_id != prev_seq
            if boundary:
                rec["state_pre"] = m.state_fingerprint()
                ev.append("sequence_start")
                m.on_sequence_start(view.seq_id)
                prev_seq = view.seq_id
            rec["state_before"] = m.state_fingerprint()
            if not boundary:    # no code runs between the two fingerprints, so they are equal by construction
                rec["state_pre"] = rec["state_before"]
            t0 = time.perf_counter()
            dets, upd = self._process(view, ev, rec)
            coco = dets.to_coco()
            record = {
                "i": view.index, "image_id": view.image_id, "seq": view.seq_id, "frame_id": view.frame_id,
                "boundary": boundary, "events": ev,
                "emitted": {"n": len(coco), "image_ids": sorted({d["image_id"] for d in coco}),
                            "bbox": [d["bbox"] for d in coco], "score": [d["score"] for d in coco],
                            "category_id": [d["category_id"] for d in coco]},
                "decision": upd.decision, "reason": upd.reason, "losses": upd.losses, "update_extra": upd.extra,
                "state_pre": rec["state_pre"], "state_before": rec["state_before"],
                "state_after_predict": rec["state_after_predict"], "state_post": m.state_fingerprint(),
                "sec": round(time.perf_counter() - t0, 6)}
            if self.store is not None:
                self.store.append(record, coco)
                if checkpoint_every and (view.index + 1) % checkpoint_every == 0:
                    self.store.save_checkpoint({"next_index": view.index + 1, "prev_seq": prev_seq,
                                                "method_state": m.state_dict(),
                                                "stream_fingerprint": self.stream.fingerprint()})
            else:
                frames.append(record)
                self.predictions.append(dets)
            if on_frame_end is not None:
                on_frame_end(view.index, m)
            done += 1
            self.processed = done
            if stop_after and done >= stop_after:
                return None
        if self.store is not None:
            self.store.close()
            with open(self.store.path(RunStore.DONE), "w") as f:
                f.write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            return self.trace()
        return dict(self._header(), frames=frames)

    def trace(self) -> dict:
        t = self._header()
        t["frames"] = self.store.records() if self.store is not None else []
        return t

    def coco_detections(self) -> list:
        if self.store is not None:
            return [d for r in self.store.records(RunStore.PREDS) for d in r["detections"]]
        return [d for dets in self.predictions for d in dets.to_coco()]
