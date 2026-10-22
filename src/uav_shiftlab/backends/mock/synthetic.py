# SPDX-License-Identifier: Apache-2.0
"""Synthetic video streams with known ground truth (NumPy only) for tests and CPU demos.

Bright rectangles ("vehicles") move linearly over a noisy background. A `night` sequence scales all intensities
down and adds noise, which breaks a detector whose threshold was set on `day` frames - a minimal, fully
controlled domain shift. This demonstrates the software (protocol, state, evaluation); it says nothing about
detection performance on real UAV video.
"""
import numpy as np

DOMAINS = {"day": (1.0, 0.02), "night": (0.35, 0.03)}   # intensity gain, noise std


def make_dataset(seqs=(("S01", "day"), ("S02", "night"), ("S03", "night")), frames=20, size=(96, 160),
                 n_objects=4, seed=0):
    """Returns (coco_dict, images) where images maps file_name -> float32 HxW array in [0, 1]."""
    rng = np.random.default_rng(seed)
    H, W = size
    coco = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "car"}]}
    images, ann_id = {}, 1
    for si, (seq, domain) in enumerate(seqs):
        gain, noise = DOMAINS[domain]
        wh = rng.integers(6, 17, size=(n_objects, 2)).astype(float)
        pos = rng.uniform([0, 0], [W - 17, H - 17], size=(n_objects, 2))
        vel = rng.uniform(-1.5, 1.5, size=(n_objects, 2))
        for t in range(frames):
            img = np.full((H, W), 0.25, np.float32)
            boxes = []
            for k in range(n_objects):
                x, y = np.clip(pos[k] + vel[k] * t, 0, [W - wh[k, 0], H - wh[k, 1]])
                x0, y0, x1, y1 = int(x), int(y), int(x + wh[k, 0]), int(y + wh[k, 1])
                img[y0:y1, x0:x1] = 0.9
                boxes.append((x0, y0, x1 - x0, y1 - y0))
            img = np.clip(img * gain + rng.normal(0, noise, img.shape), 0, 1).astype(np.float32)
            image_id = (si + 1) * 1000 + t + 1
            name = f"{seq}/img{t + 1:06d}.png"
            images[name] = img
            coco["images"].append({"id": image_id, "file_name": name, "width": W, "height": H, "video": seq,
                                   "frame_id": t + 1, "attrs": [domain]})
            for b in boxes:
                coco["annotations"].append({"id": ann_id, "image_id": image_id, "category_id": 1,
                                            "bbox": [float(v) for v in b], "area": float(b[2] * b[3]),
                                            "iscrowd": 0})
                ann_id += 1
    return coco, images
