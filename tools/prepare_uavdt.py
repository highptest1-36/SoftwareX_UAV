# SPDX-License-Identifier: Apache-2.0
"""Build the split definitions and the COCO json files of UAVDT used by UAV-ShiftLab.

  python3 tools/prepare_uavdt.py --uavdt-root /path/UAVDT --out /path/assets/data
  (--images / --gt-dir / --attr-dir override the default locations below --uavdt-root)

Input (official UAVDT release, from its authors): UAV-benchmark-M/ (images), UAV-benchmark-MOTD_v1.0/GT/
(*_gt_whole.txt, *_gt_ignore.txt), M_attr/{train,test}/ (sequence attributes).
Output under --out:
  manifests/splits.json, manifests/sequences.csv, manifests/dataset_statistics.csv
  coco/<split>.json   (+ source_train_s2.json, dev_*_s2.json: every 2nd frame)
COCO conventions: bbox in native pixels (1024x540 / 960x540); category ids 1 car, 2 truck, 3 bus; ignore regions as
iscrowd=1 annotations replicated for each category and as image["ignore_regions"] (mean-filled when training the
source detector, never used during test-time adaptation); every image has `video` (sequence), `frame_id`, `attrs`.
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from uav_shiftlab.datasets import uavdt  # noqa: E402

STRIDED = {"source_train": 2, "dev_night": 2, "dev_fog": 2}


def image_size(path):
    from PIL import Image
    return Image.open(path).size


def size_bin(w, h):
    a = w * h
    return "S" if a < 32 * 32 else ("M" if a < 96 * 96 else "L")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uavdt-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--images", default="")
    ap.add_argument("--gt-dir", default="")
    ap.add_argument("--attr-dir", default="")
    a = ap.parse_args()
    images_dir = a.images or os.path.join(a.uavdt_root, "UAV-benchmark-M")
    gt_dir = a.gt_dir or os.path.join(a.uavdt_root, "UAV-benchmark-MOTD_v1.0", "GT")
    attr_dir = a.attr_dir or os.path.join(a.uavdt_root, "M_attr")
    man_dir, coco_dir = os.path.join(a.out, "manifests"), os.path.join(a.out, "coco")
    os.makedirs(man_dir, exist_ok=True)
    os.makedirs(coco_dir, exist_ok=True)

    attrs = uavdt.load_attributes(attr_dir)
    assert sum(v["split"] == "train" for v in attrs.values()) == 30
    assert sum(v["split"] == "test" for v in attrs.values()) == 20
    splits = uavdt.build_splits(attrs)
    json.dump(splits, open(os.path.join(man_dir, "splits.json"), "w"), indent=1)

    cache, rows = {}, []
    for seq in sorted(attrs):
        images, anns, (W, H) = uavdt.build_sequence(seq, images_dir, gt_dir, image_size)
        for im in images:
            im["attrs"] = [k for k, v in attrs[seq]["attrs"].items() if v]
        cache[seq] = (images, anns)
        real = [x for x in anns if not x["iscrowd"]]
        cls = [sum(x["category_id"] == c for x in real) for c in (1, 2, 3)]
        sz = [sum(size_bin(*x["bbox"][2:]) == b for x in real) for b in "SML"]
        tiny = sum((x["bbox"][2] * x["bbox"][3]) ** 0.5 < 16 for x in real)
        rows.append({"seq": seq, "official_split": attrs[seq]["split"],
                     "roles": ";".join(k for k, v in splits.items() if seq in v and k != "test_all"),
                     **attrs[seq]["attrs"], "n_images": len(images), "width": W, "height": H,
                     "n_boxes": len(real), "car": cls[0], "truck": cls[1], "bus": cls[2],
                     "small": sz[0], "medium": sz[1], "large": sz[2], "tiny_lt16": tiny,
                     "n_tracks": len({x["track_id"] for x in real}),
                     "n_ignore_regions": sum(len(i["ignore_regions"]) for i in images)})
    with open(os.path.join(man_dir, "sequences.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    stats = []
    for name, seqs in splits.items():
        for stride in sorted({1, STRIDED.get(name, 1)}):
            images, anns = [], []
            for s in seqs:
                ims, an = cache[s]
                keep = {im["id"] for im in ims if (im["frame_id"] - 1) % stride == 0}
                images += [im for im in ims if im["id"] in keep]
                anns += [x for x in an if x["image_id"] in keep]
            anns = [dict(x, id=i) for i, x in enumerate(anns, 1)]
            out = name if stride == 1 else f"{name}_s{stride}"
            json.dump({"info": {"split": out, "sequences": seqs, "stride": stride},
                       "images": images, "annotations": anns, "categories": uavdt.CATEGORIES},
                      open(os.path.join(coco_dir, out + ".json"), "w"))
            real = [x for x in anns if not x["iscrowd"]]
            n = max(1, len(real))
            stats.append({"split": out, "n_seqs": len(seqs), "n_images": len(images), "n_boxes": len(real),
                          "car": sum(x["category_id"] == 1 for x in real),
                          "truck": sum(x["category_id"] == 2 for x in real),
                          "bus": sum(x["category_id"] == 3 for x in real),
                          "pct_small": round(100 * sum(size_bin(*x["bbox"][2:]) == "S" for x in real) / n, 1),
                          "pct_tiny_lt16": round(100 * sum((x["bbox"][2] * x["bbox"][3]) ** 0.5 < 16 for x in real) / n, 1),
                          "sequences": " ".join(seqs)})
            print(f"{out:18s} {stats[-1]['n_images']:6d} images {stats[-1]['n_boxes']:7d} boxes", flush=True)
    with open(os.path.join(man_dir, "dataset_statistics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats[0]))
        w.writeheader()
        w.writerows(stats)


if __name__ == "__main__":
    main()
