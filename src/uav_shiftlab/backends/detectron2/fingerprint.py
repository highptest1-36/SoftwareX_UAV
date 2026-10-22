# SPDX-License-Identifier: Apache-2.0
"""State fingerprints of the Detectron2 methods (SHA-256 of adapters, optimizer state, method state and RNG states),
recorded before and after every frame so that two runs can be compared frame by frame and the protocol audit can
check state continuity."""
import hashlib
import pickle

import numpy as np
import torch


def sha(b):
    return hashlib.sha256(b).hexdigest()


def flat(module):
    return torch.cat([p.detach().reshape(-1).float() for p in module.parameters()]).cpu().numpy()


def _host_arrays(tensors):
    """NumPy copies of `tensors` (same order, dtype and bytes as `t.detach().cpu().numpy()`), with one device->host
    transfer per (device, dtype) group instead of one per tensor."""
    out, groups = [None] * len(tensors), {}
    for i, t in enumerate(tensors):
        if t.device.type == "cpu":
            out[i] = t.detach().numpy()
        else:
            groups.setdefault((t.device, t.dtype), []).append(i)
    for idx in groups.values():
        flat_host = torch.cat([tensors[i].detach().reshape(-1) for i in idx]).cpu().numpy()
        o = 0
        for i in idx:
            n = tensors[i].numel()
            out[i] = flat_host[o:o + n]
            o += n
    return out


def opt_state_hash(opt):
    sd = opt.state_dict()["state"]
    items = [(n, sd[k][n]) for k in sorted(sd) for n in sorted(sd[k])]
    host = iter(_host_arrays([v for _, v in items if torch.is_tensor(v)]))
    h = hashlib.sha256()
    for n, v in items:
        h.update(n.encode())
        h.update(next(host).tobytes() if torch.is_tensor(v) else repr(v).encode())
    return h.hexdigest()


def method_state_hash(r):
    st = r.method_state()
    if st is None:
        return None
    h = hashlib.sha256()
    for k in sorted(st["mu_te"]):
        h.update(st["mu_te"][k].detach().cpu().numpy().tobytes())
    h.update(st["mu_te_obj"].detach().cpu().numpy().tobytes())
    h.update(st["n_obj"].detach().cpu().numpy().tobytes())
    h.update(repr(st["l_ema"]).encode())
    return h.hexdigest()


def runner_fingerprint(r):
    return {"student": sha(flat(r.student).tobytes()), "teacher": sha(flat(r.teacher).tobytes()),
            "optimizer": opt_state_hash(r.opt), "method_state": method_state_hash(r),
            "rng_torch": sha(torch.get_rng_state().numpy().tobytes()),
            "rng_numpy": sha(pickle.dumps(np.random.get_state(), protocol=4))}
