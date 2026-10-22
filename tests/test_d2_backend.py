# SPDX-License-Identifier: Apache-2.0
"""Detectron2 backend: shipped profiles, configuration checks, and (when the fixtures are available locally, i.e. not
on CI) bit-exact agreement of the current code with the reference traces recorded by tools/trace_capture.py."""
import hashlib
import json
import os

import pytest
import yaml

SX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILE_DIR = os.path.join(SX, "src", "uav_shiftlab", "backends", "detectron2", "profiles", "uavdt")


def test_profile_files_match_their_hashes():
    prov = json.load(open(os.path.join(PROFILE_DIR, "PROFILE_SHA256.json")))["files"]
    assert {"detector.yaml", "tta/source_only.yaml", "tta/mean_teacher.yaml", "tta/tent.yaml", "tta/whw.yaml"} <= set(prov)
    for rel, v in prov.items():
        assert hashlib.sha256(open(os.path.join(PROFILE_DIR, rel), "rb").read()).hexdigest() == v["sha256"], rel


def test_every_profile_names_a_runner():
    for name in ("source_only", "mean_teacher", "tent", "whw"):
        assert yaml.safe_load(open(os.path.join(PROFILE_DIR, "tta", f"{name}.yaml")))["method"] == name


def test_unknown_configuration_keys_are_rejected():
    pytest.importorskip("detectron2")
    from uav_shiftlab.backends.detectron2.runners import RUNNERS
    for name, cls in RUNNERS.items():
        tcfg = yaml.safe_load(open(os.path.join(PROFILE_DIR, "tta", f"{name}.yaml")))
        assert set(tcfg) <= cls.KEYS, name
        with pytest.raises(ValueError, match="unknown configuration keys"):
            cls(None, None, dict(tcfg, tracker=True))


RES = os.path.join(SX, "golden", "post_split", "trace_compare.json")
rows = json.load(open(RES)) if os.path.exists(RES) else {}


@pytest.mark.skipif(not rows, reason="reference traces not available (produced locally with UAVDT)")
@pytest.mark.parametrize("key", sorted(rows))
def test_current_code_reproduces_reference_traces(key):
    r = rows[key]
    assert r["audit_pass"], r
    assert r["exact"], r
