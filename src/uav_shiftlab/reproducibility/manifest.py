# SPDX-License-Identifier: Apache-2.0
"""Run manifests: environment, code version, and SHA256 of every input that can change a result."""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

from .. import __version__


def sha256_file(path, buf=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                return h.hexdigest()
            h.update(b)


def _git(repo, *args):
    try:
        out = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def environment():
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "hostname": platform.node(),
           "cpu_count": os.cpu_count(), "numpy": np.__version__, "uav_shiftlab": __version__}
    try:
        env["cpu_model"] = next(ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo")
                                if ln.startswith("model name"))
    except (OSError, StopIteration):
        pass
    try:
        import torch
        env.update(torch=torch.__version__, cuda=torch.version.cuda, cuda_available=torch.cuda.is_available(),
                   torch_threads=torch.get_num_threads(),
                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   bf16=bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()))
    except ImportError:
        env["torch"] = None
    for mod in ("detectron2", "cv2", "pycocotools", "faster_coco_eval"):
        try:
            env[mod] = getattr(__import__(mod), "__version__", "installed")
        except ImportError:
            env[mod] = None
    return env


def code_version():
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    commit = _git(repo, "rev-parse", "HEAD")
    dirty = _git(repo, "status", "--porcelain", "--untracked-files=no")
    return {"package_root": repo, "git_commit": commit, "git_dirty": bool(dirty) if dirty is not None else None}


def build_manifest(files=None, **extra):
    """files: name -> path (hashed). extra: free-form fields (method, seed, stream fingerprint, ...)."""
    return {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "environment": environment(),
            "code": code_version(),
            "sha256": {k: sha256_file(p) for k, p in (files or {}).items() if p and os.path.exists(p)}, **extra}


def verify_manifest(manifest, files):
    """Re-hash the inputs of a manifest; returns {name: (recorded, current)} for every mismatch."""
    bad = {}
    for k, p in files.items():
        rec = manifest.get("sha256", {}).get(k)
        cur = sha256_file(p) if p and os.path.exists(p) else None
        if rec != cur:
            bad[k] = (rec, cur)
    return bad


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)
