# SPDX-License-Identifier: Apache-2.0
"""Train the source (daylight) Faster R-CNN R50-FPN of the Detectron2 profiles (GPU).

  python3 tools/train_source.py --assets-root /path/assets --images /path/UAV-benchmark-M      # full schedule
  python3 tools/train_source.py ... --bench 60                                                # throughput only
  python3 tools/train_source.py ... --smoke                                                   # 2 iterations, CPU ok
Config: profiles/uavdt/detector.yaml (10k iterations, batch 8, AMP; datasets uavdt_source_train_s2 / uavdt_source_val
from <assets>/data/coco, tools/prepare_uavdt.py). Initialisation: the COCO-pretrained model-zoo checkpoint
(<assets>/checkpoints/pretrained/faster_rcnn_R_50_FPN_3x_coco.pkl if present, otherwise downloaded by Detectron2).
Outputs in <assets>/checkpoints/source_day_r50_1080/: model_final.pth (used by the profiles), model_best.pth (by
source_val class-agnostic AP50), eval_*.json, metrics.json. Resumes from last_checkpoint unless --no-resume.
"""
import argparse
import json
import os
import sys
import time

SX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SX, "src"))
import torch  # noqa: E402
from detectron2.data import DatasetCatalog, build_detection_test_loader, build_detection_train_loader  # noqa: E402
from detectron2.engine import DefaultTrainer, hooks  # noqa: E402
from detectron2.evaluation import DatasetEvaluator  # noqa: E402
from detectron2.utils import comm  # noqa: E402
from detectron2.utils.logger import setup_logger  # noqa: E402

from uav_shiftlab.backends.detectron2.common import build_cfg  # noqa: E402
from uav_shiftlab.backends.detectron2.plugins import D2CFG, profile_path  # noqa: E402
from uav_shiftlab.backends.detectron2.train_data import UAVDTTrainMapper, register_all  # noqa: E402
from uav_shiftlab.evaluation.coco_eval import evaluate  # noqa: E402

ZOO_COCO = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"
COCO_DIR = None


def instances_to_dets(inst, image_id):
    """Detectron2 Instances (original image coordinates) -> COCO detections (category ids 1..3)."""
    b = inst.pred_boxes.tensor.detach().cpu().numpy()
    s = inst.scores.detach().cpu().numpy()
    c = inst.pred_classes.detach().cpu().numpy()
    return [{"image_id": int(image_id), "category_id": int(ci) + 1,
             "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(si)}
            for (x1, y1, x2, y2), si, ci in zip(b, s, c)]


class UAVDTEvaluator(DatasetEvaluator):
    def __init__(self, split, out_dir, trainer=None):
        self.split, self.out_dir, self.trainer = split, out_dir, trainer

    def reset(self):
        self.dets = []

    def process(self, inputs, outputs):
        for i, o in zip(inputs, outputs):
            self.dets += instances_to_dets(o["instances"].to("cpu"), i["image_id"])

    def evaluate(self):
        res = evaluate(os.path.join(COCO_DIR, self.split + ".json"), self.dets, per_sequence=True)
        it = self.trainer.iter + 1 if self.trainer is not None else -1
        json.dump(res, open(os.path.join(self.out_dir, f"eval_{self.split}_iter{it:06d}.json"), "w"), indent=1)
        c3, ag = res["3class"], res["agnostic"]
        return {"bbox": {"AP50_agnostic": ag["AP50"], "mAP50": c3["AP50"], "AP": c3["AP"],
                         "AP50_small": c3["AP50_small"], "AP50_tiny": c3["AP50_tiny"]}}


class Trainer(DefaultTrainer):
    _current = None

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=UAVDTTrainMapper(cfg, True))

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=UAVDTTrainMapper(cfg, False))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return UAVDTEvaluator(dataset_name.replace("uavdt_", ""), cfg.OUTPUT_DIR, cls._current)

    def build_hooks(self):
        Trainer._current = self
        hs = super().build_hooks()
        for h in hs:
            if isinstance(h, hooks.PeriodicCheckpointer):
                h.max_to_keep = 2
        if self.cfg.TEST.EVAL_PERIOD > 0:
            # must run after the EvalHook inserted by DefaultTrainer.build_hooks
            hs.insert(len(hs) - 1, hooks.BestCheckpointer(self.cfg.TEST.EVAL_PERIOD, self.checkpointer,
                                                          "bbox/AP50_agnostic", "max", "model_best"))
        return hs


def main():
    global COCO_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets-root", required=True)
    ap.add_argument("--images", required=True, help="UAV-benchmark-M image root")
    ap.add_argument("--coco-dir", default="", help="default: <assets>/data/coco")
    ap.add_argument("--bench", type=int, default=0, help="run N iterations only and report throughput")
    ap.add_argument("--smoke", action="store_true", help="tiny run (2 iterations, 4 images) to validate the pipeline")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("opts", nargs=argparse.REMAINDER, help="Detectron2 config overrides KEY VALUE ...")
    a = ap.parse_args()
    COCO_DIR = a.coco_dir or os.path.join(a.assets_root, "data", "coco")
    register_all(COCO_DIR, a.images)
    cfg = build_cfg(profile_path(D2CFG), a.opts)
    pre = os.path.join(a.assets_root, cfg.MODEL.WEIGHTS)
    cfg.MODEL.WEIGHTS = pre if os.path.exists(pre) else ZOO_COCO
    cfg.OUTPUT_DIR = os.path.join(a.assets_root, cfg.OUTPUT_DIR)
    full_iters = cfg.SOLVER.MAX_ITER
    if a.smoke:
        DatasetCatalog.register("uavdt_smoke", lambda: DatasetCatalog.get("uavdt_source_val")[:4])
        cfg.merge_from_list(["DATASETS.TRAIN", ("uavdt_smoke",), "DATASETS.TEST", (),
                             "SOLVER.IMS_PER_BATCH", 1, "SOLVER.MAX_ITER", 2, "SOLVER.STEPS", (),
                             "SOLVER.WARMUP_ITERS", 0, "INPUT.MIN_SIZE_TRAIN", (540,), "INPUT.MAX_SIZE_TRAIN", 1024,
                             "DATALOADER.NUM_WORKERS", 0, "TEST.EVAL_PERIOD", 0,
                             "OUTPUT_DIR", os.path.join(a.assets_root, "checkpoints", "smoke_source")])
    if a.bench:
        cfg.merge_from_list(["SOLVER.MAX_ITER", a.bench, "TEST.EVAL_PERIOD", 0, "SOLVER.CHECKPOINT_PERIOD", 10 ** 9,
                             "OUTPUT_DIR", os.path.join(a.assets_root, "checkpoints", "bench_source")])
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.freeze()
    setup_logger(cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="detectron2")
    with open(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), "w") as f:
        f.write(cfg.dump())
    tr = Trainer(cfg)
    tr.resume_or_load(resume=not (a.no_resume or a.bench or a.smoke))
    t0 = time.time()
    tr.train()
    dt_ = time.time() - t0
    if a.bench:
        warm = 20 if a.bench > 40 else 0
        times = [v for v, _ in tr.storage.history("time").values()[warm:]] if "time" in tr.storage.histories() else []
        spi = sum(times) / len(times) if times else dt_ / a.bench
        vram = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0
        print(f"\n[BENCH] {spi:.3f} s/iter (bs={cfg.SOLVER.IMS_PER_BATCH}) -> projected {spi * full_iters / 3600:.2f} h "
              f"for {full_iters} iters; peak VRAM {vram:.1f} GB")
        return
    if not a.smoke:
        print(Trainer.test(cfg, tr.model, [UAVDTEvaluator("source_val", cfg.OUTPUT_DIR, None)]))
    print(f"[DONE] training time {dt_ / 3600:.2f} h")


if __name__ == "__main__":
    main()
