# SPDX-License-Identifier: Apache-2.0
"""Source feature statistics for the WHW profile (backends/detectron2/runners.py, WHWRunner).

Follows the official WHW code (github.com/natureyoo/ContinualTTA_ObjectDetection, rcnn.py / engine/defaults.py):
  image level : per FPN level, global-average-pooled feature of each image -> mean / full covariance over images;
  object level: ROI box-head features (1024-d) of ground-truth boxes -> per-class mean / full covariance;
  the regularised covariance S + I max(S) / 30 is stored as its inverse (KL with equal covariances needs only S^-1);
  D_in        : sum over levels of KL(N(mu_all, S) || N(mu_first20, S)) as in the official code ("when" criterion).
Ignore regions are mean-filled as in source training.

  python3 tools/source_stats.py --assets-root /path/assets --images /path/UAV-benchmark-M [--n 2000]
Output: <assets>/checkpoints/source_day_r50_1080/source_stats.pt (or --out).
"""
import argparse
import json
import os
import sys

SX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SX, "src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from detectron2.structures import Boxes  # noqa: E402

from uav_shiftlab.backends.detectron2.common import amp_dtype, build_cfg, read_bgr  # noqa: E402
from uav_shiftlab.backends.detectron2.plugins import D2CFG, WEIGHTS, profile_path  # noqa: E402
from uav_shiftlab.backends.detectron2.runners import SourceOnlyRunner  # noqa: E402
from uav_shiftlab.backends.detectron2.train_data import fill_ignore  # noqa: E402


def reg_inv(X):
    S = torch.cov(X.T.double())
    S = S + torch.eye(S.shape[0], dtype=S.dtype, device=S.device) * S.max() / 30
    return torch.linalg.inv(S).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets-root", required=True)
    ap.add_argument("--images", required=True, help="UAV-benchmark-M image root")
    ap.add_argument("--coco-dir", default="", help="default: <assets>/data/coco")
    ap.add_argument("--split", default="source_train_s2")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    weights = os.path.join(a.assets_root, WEIGHTS)
    out = a.out or os.path.join(os.path.dirname(weights), "source_stats.pt")
    r = SourceOnlyRunner(build_cfg(profile_path(D2CFG)), weights, {"method": "source_only"}, seed=a.seed)
    gt = json.load(open(os.path.join(a.coco_dir or os.path.join(a.assets_root, "data", "coco"), a.split + ".json")))
    rng = np.random.RandomState(a.seed)
    ims = [gt["images"][i] for i in sorted(rng.choice(len(gt["images"]), min(a.n, len(gt["images"])), replace=False))]
    anns = {}
    for x in gt["annotations"]:
        if not x.get("iscrowd"):
            anns.setdefault(x["image_id"], []).append(x)
    rh = r.model.roi_heads
    img_feats, obj_feats, obj_cls = {}, [], []
    for i, im in enumerate(ims):
        bgr = fill_ignore(read_bgr(os.path.join(a.images, im["file_name"])), im.get("ignore_regions", []))
        images, (sx, sy) = r._prep(bgr)
        with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype(), enabled=r.amp):
            feats = r.model.backbone(images.tensor)
            for k, v in feats.items():
                img_feats.setdefault(k, []).append(v.float().mean(dim=(0, 2, 3)))
            A = anns.get(im["id"], [])
            if A:
                bx = torch.tensor([[x["bbox"][0] * sx, x["bbox"][1] * sy, (x["bbox"][0] + x["bbox"][2]) * sx,
                                    (x["bbox"][1] + x["bbox"][3]) * sy] for x in A], device=r.device)
                bf = rh.box_head(rh.box_pooler([feats[f] for f in rh.box_in_features], [Boxes(bx)]))
                obj_feats.append(bf.float())
                obj_cls.append(torch.tensor([x["category_id"] - 1 for x in A], device=r.device))
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(ims)}", flush=True)

    st = {"img_mean": {}, "img_prec": {}, "n_images": len(ims), "split": a.split, "weights": WEIGHTS}
    d_in = 0.0
    for k, L in img_feats.items():
        X = torch.stack(L)
        st["img_mean"][k], st["img_prec"][k] = X.mean(0), reg_inv(X)
        d = X.mean(0) - X[:20].mean(0)
        d_in += float(0.5 * d @ st["img_prec"][k] @ d)
    st["d_in"] = d_in
    OF, OC = torch.cat(obj_feats), torch.cat(obj_cls)
    K = 3
    st["obj_mean"] = torch.stack([OF[OC == c].mean(0) for c in range(K)])
    st["obj_prec"] = torch.stack([reg_inv(OF[OC == c]) for c in range(K)])
    st["obj_count"] = [int((OC == c).sum()) for c in range(K)]
    torch.save(st, out)
    print(f"[source_stats] {out}: {len(ims)} images, objects per class {st['obj_count']}, D_in={d_in:.4f}")


if __name__ == "__main__":
    main()
