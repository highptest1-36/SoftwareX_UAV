# SPDX-License-Identifier: Apache-2.0
"""The violations used in the impact study change the results and are flagged by the audit."""
from uav_shiftlab.backends.mock.methods import AdaptiveThreshold
from uav_shiftlab.backends.mock.synthetic import make_dataset
from uav_shiftlab.core.stream import StreamManifest
from uav_shiftlab.protocols.audit import audit_trace
from uav_shiftlab.protocols.violations import apply


def _run(violation):
    coco, imgs = make_dataset(frames=8)
    m, R = apply(violation, AdaptiveThreshold())
    r = R(m, StreamManifest.from_coco(coco, ""), lambda p: imgs[p])
    tr = r.run()
    return tr, r.coco_detections(), {k for k, v in audit_trace(tr).items() if k != "pass" and not v["pass"]}


def test_violations_are_flagged_and_change_predictions():
    ref, ref_dets, ok = _run("none")
    assert ok == set()
    _, d1, f1 = _run("adapt_before_predict")
    _, d2, f2 = _run("reset_each_sequence")
    assert "P1_predict_before_update" in f1 and d1 != ref_dets
    assert "P2_state_continuity" in f2 and d2 != ref_dets
