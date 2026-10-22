# SPDX-License-Identifier: Apache-2.0
"""Shared helpers of the Detectron2 backend: config building, image loading, COCO detections, autocast dtype."""
import os

import cv2


def build_cfg(config_file, opts=()):
    """Detectron2 config = model-zoo Faster R-CNN R50-FPN 3x + the profile's detector yaml (absolute path).
    Weights are loaded explicitly by the runners, so MODEL.WEIGHTS is not used."""
    import torch
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(list(opts))
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"
        cfg.SOLVER.AMP.ENABLED = False
    cfg.DATALOADER.NUM_WORKERS = min(cfg.DATALOADER.NUM_WORKERS, max(1, (os.cpu_count() or 2) - 1))
    return cfg


def read_bgr(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def to_dets(image_id, boxes_xyxy, scores, classes):
    """Boxes in original image coordinates (xyxy) -> COCO detections (category ids 1..K)."""
    return [{"image_id": int(image_id), "category_id": int(c) + 1,
             "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(s)}
            for (x1, y1, x2, y2), s, c in zip(boxes_xyxy, scores, classes)]


def amp_dtype():
    """Autocast dtype: bf16 where supported (A100/L4), fp16 otherwise (autocast is only enabled with bf16)."""
    import torch
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16
