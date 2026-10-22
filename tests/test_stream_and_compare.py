# SPDX-License-Identifier: Apache-2.0
"""Stream manifests (sequence orders) and the trace comparator."""
import copy
import random


from uav_shiftlab.backends.mock.methods import AdaptiveThreshold
from uav_shiftlab.backends.mock.synthetic import make_dataset
from uav_shiftlab.core.runner import OnlineRunner
from uav_shiftlab.core.stream import StreamManifest
from uav_shiftlab.reproducibility.trace_compare import compare, judge


def test_sequence_orders():
    coco, _ = make_dataset(seqs=[(f"S{k:02d}", "day") for k in range(5)], frames=2)
    orig = StreamManifest.from_coco(coco, "", "original").sequence_order()
    assert orig == sorted(orig)
    assert StreamManifest.from_coco(coco, "", "reverse").sequence_order() == orig[::-1]
    exp = list(orig)
    random.Random(1).shuffle(exp)
    assert StreamManifest.from_coco(coco, "", "perm:1").sequence_order() == exp
    st = StreamManifest.from_coco(coco, "")
    assert [f.frame_id for f in st.frames[:2]] == [1, 2]
    assert st.fingerprint() != StreamManifest.from_coco(coco, "", "reverse").fingerprint()


def _trace(frames=6):
    coco, imgs = make_dataset(frames=frames)
    return OnlineRunner(AdaptiveThreshold(), StreamManifest.from_coco(coco, ""), lambda p: imgs[p]).run()


def test_identical_traces_compare_exactly():
    a, b = _trace(), _trace()
    d = compare(a, b)
    assert judge(d, {})["pass"], d
    assert d["exact"] and d["first_divergent_frame"] is None


def test_comparator_detects_small_and_discrete_differences():
    a = _trace()
    b = copy.deepcopy(a)
    b["frames"][3]["emitted"]["score"][0] += 1e-7
    b["frames"][4]["decision"] = "skip"
    d = compare(a, b)
    assert d["score_max_abs"] > 0 and d["decision_mismatch"] == 1
    assert d["first_divergent_frame"] == 3
    j = judge(d, {"score_max_abs": 1e-6})
    assert not j["pass"] and set(j["failures"]) == {"decision_mismatch"}


def test_state_hashes_are_compared():
    a = _trace()
    b = copy.deepcopy(a)
    comp = next(iter(b["frames"][2]["state_post"]))
    b["frames"][2]["state_post"][comp] = "x"
    d = compare(a, b)
    assert d["state_hash_mismatch"] == {f"state_post.{comp}": 1} and d["first_divergent_frame"] == 2
    assert not judge(d, {"score_max_abs": 1.0})["pass"]
