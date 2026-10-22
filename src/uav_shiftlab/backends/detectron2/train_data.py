# SPDX-License-Identifier: Apache-2.0
"""Detectron2 dataset registration and training mapper for the UAVDT COCO json files (tools/prepare_uavdt.py),
used to train the source detector (tools/train_source.py) and to compute WHW source statistics (tools/source_stats.py).

Category ids 1..3 (car, truck, bus) -> contiguous 0..2. The training mapper mean-fills ignore regions, so that
unlabeled vehicles there are not learned as background.
"""
import copy
import json
import os

import numpy as np
import torch
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BoxMode

CLASS_NAMES = ["car", "truck", "bus"]
PIXEL_MEAN_BGR = (103.530, 116.280, 123.675)


def load_split(coco_json, image_root):
    d = json.load(open(coco_json))
    anns = {}
    for a in d["annotations"]:
        if not a.get("iscrowd", 0):
            anns.setdefault(a["image_id"], []).append(a)
    out = []
    for im in d["images"]:
        out.append({
            "file_name": os.path.join(image_root, im["file_name"]),
            "height": im["height"], "width": im["width"], "image_id": im["id"],
            "video": im["video"], "frame_id": im["frame_id"],
            "ignore_regions": im.get("ignore_regions", []),
            "annotations": [{"bbox": a["bbox"], "bbox_mode": BoxMode.XYWH_ABS,
                             "category_id": a["category_id"] - 1, "iscrowd": 0}
                            for a in anns.get(im["id"], [])],
        })
    return out


def register_all(coco_dir, image_root):
    """Register every <coco_dir>/<split>.json as dataset `uavdt_<split>`."""
    for f in sorted(os.listdir(coco_dir)):
        if not f.endswith(".json"):
            continue
        name = "uavdt_" + f[:-5]
        if name in DatasetCatalog.list():
            continue
        path = os.path.join(coco_dir, f)
        DatasetCatalog.register(name, lambda p=path: load_split(p, image_root))
        MetadataCatalog.get(name).set(thing_classes=CLASS_NAMES, json_file=path, evaluator_type="uavdt")


def fill_ignore(img_bgr, regions):
    for x, y, w, h in regions:
        x0, y0 = max(0, int(x)), max(0, int(y))
        img_bgr[y0:int(np.ceil(y + h)), x0:int(np.ceil(x + w))] = PIXEL_MEAN_BGR
    return img_bgr


class UAVDTTrainMapper:
    """Detectron2 DatasetMapper equivalent (boxes only) + ignore-region mean-fill."""

    def __init__(self, cfg, is_train=True):
        self.is_train = is_train
        if is_train:
            self.augs = T.AugmentationList([
                T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, "choice"),
                T.RandomFlip(horizontal=True),
            ])
        else:
            self.augs = T.AugmentationList([
                T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST, "choice")])
        self.fmt = cfg.INPUT.FORMAT

    def __call__(self, d):
        d = copy.deepcopy(d)
        img = utils.read_image(d["file_name"], format=self.fmt)
        if self.is_train and d.get("ignore_regions"):
            img = fill_ignore(np.ascontiguousarray(img), d["ignore_regions"])
        aug = T.AugInput(img)
        tfm = self.augs(aug)
        img = aug.image
        d["image"] = torch.as_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
        if not self.is_train:
            d.pop("annotations", None)
            return d
        annos = [utils.transform_instance_annotations(a, tfm, img.shape[:2]) for a in d.pop("annotations")]
        inst = utils.annotations_to_instances(annos, img.shape[:2])
        d["instances"] = utils.filter_empty_instances(inst)
        return d
