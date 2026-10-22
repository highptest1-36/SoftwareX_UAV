# SPDX-License-Identifier: Apache-2.0
"""Mid-stream checkpoint/resume: an interrupted + resumed run equals an uninterrupted one, frame by frame."""
import os

import pytest

from uav_shiftlab.backends.mock.methods import AdaptiveThreshold
from uav_shiftlab.backends.mock.synthetic import make_dataset
from uav_shiftlab.core.runner import OnlineRunner, RunStore
from uav_shiftlab.core.stream import StreamManifest
from uav_shiftlab.protocols.audit import audit_trace


@pytest.fixture(scope="module")
def data():
    coco, imgs = make_dataset(frames=10)
    return coco, imgs, StreamManifest.from_coco(coco, "")


def strip(trace):
    return [{k: v for k, v in f.items() if k != "sec"} for f in trace["frames"]]


def runner(data):
    coco, imgs, st = data
    return OnlineRunner(AdaptiveThreshold(), st, lambda p: imgs[p])


def test_persistent_run_equals_in_memory_run(data, tmp_path):
    a = runner(data).run()
    b = runner(data).run(run_dir=str(tmp_path / "run"), checkpoint_every=4)
    assert strip(a) == strip(b)
    assert audit_trace(b)["pass"]


@pytest.mark.parametrize("every,stop", [(1, 7), (4, 11), (5, 5), (7, 3)])
def test_interrupted_and_resumed_run_is_identical(data, tmp_path, every, stop):
    ref_r = runner(data)
    ref = ref_r.run()
    d = str(tmp_path / "run")
    assert runner(data).run(run_dir=d, checkpoint_every=every, stop_after=stop) is None   # "disconnect"
    r2 = runner(data)                                                                       # fresh method object
    res = r2.run(run_dir=d, checkpoint_every=every)
    assert strip(res) == strip(ref)
    assert r2.coco_detections() == ref_r.coco_detections()
    rep = audit_trace(res)
    assert rep["pass"], rep
    log = RunStore(d)
    if os.path.exists(log.path(RunStore.RESUMES)):
        import json
        entry = json.load(open(log.path(RunStore.RESUMES)))[0]
        assert entry["resume_at_frame"] == (stop // every) * every
        assert entry["frames_recomputed"] == stop - entry["resume_at_frame"]


def test_partial_line_from_crash_is_discarded(data, tmp_path):
    ref = runner(data).run()
    d = str(tmp_path / "run")
    runner(data).run(run_dir=d, checkpoint_every=3, stop_after=4)
    with open(os.path.join(d, RunStore.TRACE), "a") as f:
        f.write('{"i": 4, "image_id": 99, "trunc')                 # process killed mid-write
    res = runner(data).run(run_dir=d, checkpoint_every=3)
    assert strip(res) == strip(ref)


def test_completed_run_is_not_recomputed(data, tmp_path):
    d = str(tmp_path / "run")
    a = runner(data).run(run_dir=d, checkpoint_every=4)
    mtime = os.path.getmtime(os.path.join(d, RunStore.TRACE))
    b = runner(data).run(run_dir=d, checkpoint_every=4)
    assert strip(a) == strip(b) and os.path.getmtime(os.path.join(d, RunStore.TRACE)) == mtime


def test_checkpoint_of_another_stream_is_rejected(data, tmp_path):
    coco, imgs, st = data
    d = str(tmp_path / "run")
    runner(data).run(run_dir=d, checkpoint_every=2, stop_after=3)
    other = StreamManifest.from_coco(coco, "", order="reverse")
    with pytest.raises(ValueError):
        OnlineRunner(AdaptiveThreshold(), other, lambda p: imgs[p]).run(run_dir=d)
