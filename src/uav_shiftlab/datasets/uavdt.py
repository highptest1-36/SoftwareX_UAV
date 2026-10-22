# SPDX-License-Identifier: Apache-2.0
"""UAVDT (UAV-benchmark-M) parsing: sequence attributes, DET ground truth, ignore regions, splits.

Files of the official release:
  GT      <GT>/<seq>_gt_whole.txt   frame,target_id,left,top,w,h,out_of_view,occlusion,category (1 car 2 truck 3 bus)
  ignore  <GT>/<seq>_gt_ignore.txt  frame,id,left,top,w,h,1,-1,-1
  attrs   <M_attr>/{train,test}/<seq>_attr.txt   daylight,night,fog,low,medium,high,front,side,bird,long-term
  images  <UAV-benchmark-M>/<seq>/img<frame:06d>.jpg
"""
import collections
import glob
import os

ATTR_NAMES = ["day", "night", "fog", "low", "med", "high", "front", "side", "bird", "long"]
CATEGORIES = [{"id": 1, "name": "car"}, {"id": 2, "name": "truck"}, {"id": 3, "name": "bus"}]

# Daylight training sequences whose ground truth stops before the last image (later frames are unlabeled).
MAX_LABELED_FRAME = {"M0207": 571}
# Held-out daylight training sequences used to validate the source detector.
SOURCE_VAL = ["M0210", "M0301", "M1202"]


def load_attributes(attr_dir):
    """-> {seq: {"split": "train"|"test", "attrs": {name: 0/1}}}"""
    out = {}
    for split in ("train", "test"):
        for f in sorted(glob.glob(os.path.join(attr_dir, split, "*.txt"))):
            # 'M0701 _attr.txt' contains a space in the official archive
            seq = os.path.basename(f).split("_")[0].strip()
            vals = [int(x) for x in open(f).read().strip().split(",")]
            assert len(vals) == 10, f
            out[seq] = {"split": split, "attrs": dict(zip(ATTR_NAMES, vals))}
    return out


def has(info, *names):
    return all(info["attrs"][n] for n in names)


def build_splits(attrs):
    """Splits by official train/test partition and attributes: the source domain is daylight (official train
    sequences minus SOURCE_VAL); development splits (hyper-parameter selection) are official train sequences, test
    splits official test sequences."""
    tr = lambda *n: sorted(s for s, i in attrs.items() if i["split"] == "train" and has(i, *n))
    te = lambda *n: sorted(s for s, i in attrs.items() if i["split"] == "test" and has(i, *n))
    day_train = tr("day")
    splits = {
        "source_train": [s for s in day_train if s not in SOURCE_VAL],
        "source_val": list(SOURCE_VAL),
        "dev_night": tr("night"),
        "dev_fog": tr("fog"),
        "test_day": te("day"),
        "test_night": te("night"),
        "test_fog": te("fog"),
        "test_fogmed": te("fog", "med"),
        "test_high": te("day", "high"),
        "test_all": sorted(s for s, i in attrs.items() if i["split"] == "test"),
    }
    assert all(s in day_train for s in SOURCE_VAL)
    return splits


def read_gt(seq, gt_dir):
    """-> {frame: [dict(track_id, bbox[x,y,w,h], category_id, out_of_view, occlusion)]}"""
    per = collections.defaultdict(list)
    for line in open(os.path.join(gt_dir, f"{seq}_gt_whole.txt")):
        p = line.strip().split(",")
        if len(p) < 9:
            continue
        per[int(p[0])].append({
            "track_id": int(p[1]),
            "bbox": [float(p[2]), float(p[3]), float(p[4]), float(p[5])],
            "out_of_view": int(p[6]), "occlusion": int(p[7]), "category_id": int(p[8]),
        })
    return per


def read_ignore(seq, gt_dir):
    """-> {frame: [[x,y,w,h], ...]}"""
    per = collections.defaultdict(list)
    f = os.path.join(gt_dir, f"{seq}_gt_ignore.txt")
    if not os.path.exists(f):
        return per
    for line in open(f):
        p = line.strip().split(",")
        if len(p) < 6:
            continue
        per[int(p[0])].append([float(p[2]), float(p[3]), float(p[4]), float(p[5])])
    return per


def list_frames(seq, image_dir):
    names = sorted(n for n in os.listdir(os.path.join(image_dir, seq)) if n.endswith(".jpg"))
    frames = [int(n[3:9]) for n in names]
    lim = MAX_LABELED_FRAME.get(seq)
    if lim is not None:
        frames = [f for f in frames if f <= lim]
    return frames


def image_id(seq, frame):
    return int(seq[1:]) * 100000 + frame


def file_name(seq, frame):
    return f"{seq}/img{frame:06d}.jpg"


def build_sequence(seq, image_dir, gt_dir, image_size):
    """COCO images and annotations of one sequence. Ignore regions become iscrowd=1 annotations (one per category,
    so that detections inside them are neither true nor false positives) and image["ignore_regions"]."""
    frames = list_frames(seq, image_dir)
    gt, ign = read_gt(seq, gt_dir), read_ignore(seq, gt_dir)
    W, H = image_size(os.path.join(image_dir, file_name(seq, frames[0])))
    images, anns = [], []
    for fr in frames:
        iid = image_id(seq, fr)
        images.append({"id": iid, "file_name": file_name(seq, fr), "width": W, "height": H,
                       "video": seq, "frame_id": fr, "ignore_regions": ign.get(fr, [])})
        for o in gt.get(fr, []):
            x, y, w, h = o["bbox"]
            anns.append({"image_id": iid, "category_id": o["category_id"], "bbox": [x, y, w, h],
                         "area": w * h, "iscrowd": 0, "track_id": o["track_id"],
                         "occlusion": o["occlusion"], "out_of_view": o["out_of_view"]})
        for (x, y, w, h) in ign.get(fr, []):
            for c in (1, 2, 3):
                anns.append({"image_id": iid, "category_id": c, "bbox": [x, y, w, h],
                             "area": w * h, "iscrowd": 1, "ignore": 1})
    return images, anns, (W, H)
