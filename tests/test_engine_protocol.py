# SPDX-License-Identifier: Apache-2.0
"""Engine + protocol audit on synthetic streams: the correct runner passes, every deliberate violation is caught."""
import copy
import dataclasses

import numpy as np
import pytest

from uav_shiftlab.backends.mock.methods import AdaptiveThreshold, SourceOnly
from uav_shiftlab.backends.mock.synthetic import make_dataset
from uav_shiftlab.core.runner import OnlineRunner
from uav_shiftlab.core.stream import StreamManifest
from uav_shiftlab.core.types import FrameView
from uav_shiftlab.evaluation.coco_eval import evaluate
from uav_shiftlab.protocols.audit import audit_trace


@pytest.fixture(scope="module")
def data():
    coco, imgs = make_dataset(frames=8)
    return coco, imgs, StreamManifest.from_coco(coco, "")


def run(runner_cls, method, data):
    coco, imgs, st = data
    r = runner_cls(method, st, lambda p: imgs[p])
    return r, r.run()


def failed(trace):
    rep = audit_trace(trace)
    return {k for k, v in rep.items() if k != "pass" and not v["pass"]}


def test_correct_runner_passes_all_checks(data):
    for m in (SourceOnly(), AdaptiveThreshold()):
        _, tr = run(OnlineRunner, m, data)
        assert failed(tr) == set()


def test_adaptation_helps_on_shifted_sequences(data):
    coco = data[0]
    aps = {}
    for m in (SourceOnly(), AdaptiveThreshold()):
        r, _ = run(OnlineRunner, m, data)
        aps[m.name] = evaluate(coco, r.coco_detections(), modes=("agnostic",))["agnostic"]["AP50"]
    assert aps["adaptive_threshold"] > aps["source_only"] + 20


class AdaptBeforePredict(OnlineRunner):
    def _process(self, view, ev, rec):
        upd = self.method.adapt(view, None)
        ev.append("adapt")
        dets, _ = self.method.predict(view)
        ev.append("predict")
        rec["state_after_predict"] = self.method.state_fingerprint()
        ev.append("emit")
        return dets, upd


def test_adapt_before_predict_is_detected(data):
    _, tr = run(AdaptBeforePredict, AdaptiveThreshold(), data)
    assert "P1_predict_before_update" in failed(tr)


class StatefulPredict(AdaptiveThreshold):
    def predict(self, view):  # bug: updates its state while predicting
        self.thr *= 0.99
        return super().predict(view)


def test_state_change_inside_predict_is_detected(data):
    _, tr = run(OnlineRunner, StatefulPredict(), data)
    assert "P1_predict_before_update" in failed(tr)


class SilentReset(AdaptiveThreshold):
    def on_sequence_start(self, seq_id):  # bug: declared "keep" but re-initialises at every sequence
        self.thr = self.thr0


def test_undeclared_boundary_reset_is_detected(data):
    _, tr = run(OnlineRunner, SilentReset(), data)
    assert "P2_state_continuity" in failed(tr)


def test_declared_boundary_reset_is_allowed(data):
    class DeclaredReset(SilentReset):
        boundary_policy = {"threshold": "reset", "frames_seen": "keep"}
    _, tr = run(OnlineRunner, DeclaredReset(), data)
    assert failed(tr) == set()


class DuplicateFrame(OnlineRunner):
    def _views(self, start=0):
        views = list(super()._views(start))
        return views[:3] + [views[2]] + views[3:]


class DropFrame(OnlineRunner):
    def _views(self, start=0):
        views = list(super()._views(start))
        return views[:3] + views[4:]


class Reordered(OnlineRunner):
    def _views(self, start=0):
        views = list(super()._views(start))
        return views[1:2] + views[:1] + views[2:]


@pytest.mark.parametrize("runner_cls", [DuplicateFrame, DropFrame, Reordered])
def test_stream_integrity_violations_are_detected(data, runner_cls):
    _, tr = run(runner_cls, SourceOnly(), data)
    assert "P4_stream_integrity" in failed(tr)


def test_missing_sequence_start_is_detected(data):
    _, tr = run(OnlineRunner, SourceOnly(), data)
    tr = copy.deepcopy(tr)
    nxt = next(f for f in tr["frames"][1:] if f["boundary"])
    nxt["events"].remove("sequence_start")
    assert "P3_boundary_policy" in failed(tr)


def test_double_emission_is_detected(data):
    _, tr = run(OnlineRunner, SourceOnly(), data)
    tr = copy.deepcopy(tr)
    tr["frames"][2]["events"].append("emit")
    assert "P5_single_emission" in failed(tr)


def test_labels_cannot_reach_methods(data):
    """Canary: corrupting every ground-truth box must not change a single prediction."""
    coco, imgs, st = data
    r1, _ = run(OnlineRunner, AdaptiveThreshold(), data)
    bad = copy.deepcopy(coco)
    rng = np.random.default_rng(1)
    for a in bad["annotations"]:
        a["bbox"] = [float(v) for v in rng.uniform(0, 50, 4)]
    st2 = StreamManifest.from_coco(bad, "")
    r2 = OnlineRunner(AdaptiveThreshold(), st2, lambda p: imgs[p])
    r2.run()
    assert r1.coco_detections() == r2.coco_detections()
    assert st.fingerprint() == st2.fingerprint()
    fields = {f.name for f in dataclasses.fields(FrameView)}
    assert not fields & {"annotations", "boxes", "labels", "gt"}
