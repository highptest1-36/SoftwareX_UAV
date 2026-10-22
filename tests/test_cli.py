# SPDX-License-Identifier: Apache-2.0
"""CLI smoke tests (CPU, synthetic data)."""
import json
import os

from uav_shiftlab.cli import compare_runs, main


def test_demo_audit_and_compare(tmp_path):
    out = str(tmp_path / "demo")
    assert main(["demo", "--out", out]) == 0
    for m in ("source_only", "adaptive_threshold"):
        d = os.path.join(out, m)
        assert json.load(open(os.path.join(d, "audit.json")))["pass"]
        assert main(["audit", d]) == 0
    same = compare_runs(os.path.join(out, "adaptive_threshold"), os.path.join(out, "adaptive_threshold"))
    assert same["decision_mismatch"] == 0 and same["first_any_divergence"] is None
    diff = compare_runs(os.path.join(out, "source_only"), os.path.join(out, "adaptive_threshold"))
    assert diff["first_any_divergence"] is not None
