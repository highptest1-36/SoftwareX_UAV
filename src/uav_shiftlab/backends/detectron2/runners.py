# SPDX-License-Identifier: Apache-2.0
"""Online test-time adaptation runners on a frozen Detectron2 Faster R-CNN with trainable FPN adapters.

Common to all methods: the detector (backbone, FPN, RPN, ROI heads) is frozen; the only trainable parameters are
residual FPN adapters (the student); predictions come from the teacher adapters (an EMA of the student, or the
student itself when ema = 0). Backbone + FPN features are computed once per frame and shared by prediction and
adaptation (the adapters sit after the FPN).

Every frame is split into
  predict_part(image)  -> detections of frame t with theta_{t-1}, plus a cache (features) for the update
  adapt_part(cache)    -> theta_t, returns (decision, reason, losses, extra)
with decision in {"update", "skip_gate", "skip_when", "no_adaptation"}; `step()` = predict_part + adapt_part.

Methods
  SourceOnlyRunner    no adaptation; predictions use the plain FPN features (no adapters)
  MeanTeacherRunner   mean teacher: teacher detections with score >= pseudo_tau are pseudo-labels for a
                      student update with the standard Faster R-CNN losses (RPN + ROI), then EMA teacher update
  TentRunner          Tent (Wang et al., ICLR 2021) for Faster R-CNN: mean entropy of the ROI classifier over the
                      RPN proposals of the frame
  WHWRunner           WHW (Yoo et al., CVPR 2024): image-level and class-wise object-level feature alignment to
                      source statistics, with the "when to update" criterion
Setting `runner.frozen = True` turns any runner into inference with fixed adapters (no adaptation).
"""
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import transforms as T
from detectron2.modeling import build_model
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import EventStorage

from . import heads  # noqa: F401  (registers AdapterROIHeads)
from .adapters import FPNAdapters
from .common import amp_dtype, to_dets


def _losses(losses):
    return {k: float(v.detach()) for k, v in losses.items()}


class AdapterRunner:
    """Frozen detector + student/teacher FPN adapters. Subclasses implement `_adapt`."""
    KEYS = {"method", "score_thresh", "lr", "ema", "amp", "adapter", "optim", "momentum"}

    def __init__(self, d2cfg, weights, tcfg, seed=0):
        unknown = set(tcfg) - self.KEYS
        if unknown:
            raise ValueError(f"{type(self).__name__}: unknown configuration keys {sorted(unknown)}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        d2cfg = d2cfg.clone()
        d2cfg.defrost()
        d2cfg.MODEL.ROI_HEADS.NAME = "AdapterROIHeads"
        d2cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = tcfg.get("score_thresh", 0.05)
        d2cfg.freeze()
        self.cfg, self.t = d2cfg, tcfg
        self.device = torch.device(d2cfg.MODEL.DEVICE)
        self.model = build_model(d2cfg).eval()
        DetectionCheckpointer(self.model).load(weights)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.resize = T.ResizeShortestEdge(d2cfg.INPUT.MIN_SIZE_TEST, d2cfg.INPUT.MAX_SIZE_TEST)
        self.method = tcfg["method"]
        self.frozen = False
        # fp16 without a GradScaler could underflow adapter gradients -> autocast only where bf16 exists
        self.amp = torch.cuda.is_available() and torch.cuda.is_bf16_supported() and tcfg.get("amp", True)
        a = tcfg.get("adapter", {})
        self.student = FPNAdapters(rank=a.get("rank", 32), feat_dropout=a.get("feat_dropout", 0.0)).to(self.device)
        self.teacher = copy.deepcopy(self.student).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        if tcfg.get("optim", "adam") == "sgd":
            self.opt = torch.optim.SGD(self.student.parameters(), lr=tcfg["lr"], momentum=tcfg.get("momentum", 0.9))
        else:
            self.opt = torch.optim.Adam(self.student.parameters(), lr=tcfg.get("lr", 1e-4))
        self.stats = {"frames": 0, "updates": 0, "skipped": 0}
        self.timing = {"fwd": 0.0, "cpu": 0.0, "upd": 0.0}

    # ------------------------------------------------------------------ helpers
    def _prep(self, bgr):
        h, w = bgr.shape[:2]
        img = self.resize.get_transform(bgr).apply_image(bgr)
        x = torch.as_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)), device=self.device)
        images = self.model.preprocess_image([{"image": x, "height": h, "width": w}])
        sy, sx = img.shape[0] / h, img.shape[1] / w
        return images, (sx, sy)

    def _heads(self, images, feats, train, gt=None):
        pg, rh = self.model.proposal_generator, self.model.roi_heads
        pg.train(train); rh.train(train)
        with EventStorage(0):  # detectron2 logs training statistics into an event storage
            props, rpn_l = pg(images, feats, gt)
            inst, roi_l = rh(images, feats, props, gt)
        pg.eval(); rh.eval()
        return inst, {**rpn_l, **roi_l}

    @torch.no_grad()
    def _ema(self):
        m = self.t.get("ema", 0.999)
        for pt, ps in zip(self.teacher.parameters(), self.student.parameters()):
            pt.mul_(m).add_(ps.detach(), alpha=1 - m)

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def new_sequence(self):
        """Sequence boundary: adapters, optimizer and method statistics are kept (no per-sequence state)."""

    # ------------------------------------------------------------------ one frame
    def _predict_feats(self, feats):
        return self.teacher(feats)

    def predict_part(self, bgr, image_id):
        t0 = time.perf_counter()
        self.stats["frames"] += 1
        images, (sx, sy) = self._prep(bgr)
        with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype(), enabled=self.amp):
            feats = self.model.backbone(images.tensor)
            inst, _ = self._heads(images, self._predict_feats(feats), train=False)
        inst = inst[0]
        b = inst.pred_boxes.tensor.float().cpu().numpy()
        b_orig = b / np.array([sx, sy, sx, sy], np.float32)
        scores = inst.scores.float().cpu().numpy()
        classes = inst.pred_classes.cpu().numpy()
        dets = to_dets(image_id, b_orig, scores, classes)
        self._sync()
        self.timing["fwd"] += time.perf_counter() - t0
        return dets, {"image_id": image_id, "images": images, "sx": sx, "sy": sy, "feats": feats,
                      "b_orig": b_orig, "scores": scores, "classes": classes}

    def adapt_part(self, c):
        if self.frozen:
            return "no_adaptation", "", {}, {}
        return self._adapt(c)

    def _adapt(self, c):
        return "no_adaptation", "", {}, {}

    def step(self, bgr, image_id):
        dets, cache = self.predict_part(bgr, image_id)
        self.adapt_part(cache)
        return dets

    # ------------------------------------------------------------------ method state (checkpoint / fingerprint)
    def method_state(self):
        """Evolving state besides adapters, optimizer and RNG (None if the method has none)."""
        return None

    def load_method_state(self, st):
        pass


class SourceOnlyRunner(AdapterRunner):
    """The source detector without adaptation (the adapters exist but are not used)."""

    def _predict_feats(self, feats):
        return feats


def select_fixed(boxes, scores, classes, tau):
    """Pseudo-labels of the mean teacher: detections with score >= tau (raw boxes, original image coordinates).
    Scores are compared in float64, as Python floats."""
    keep = scores.astype(np.float64) >= tau
    return (np.asarray(boxes[keep], np.float32).reshape(-1, 4), np.asarray(classes[keep], np.int64))


class MeanTeacherRunner(AdapterRunner):
    """Mean-teacher self-training with a fixed confidence threshold (`pseudo_tau`); a frame updates the student
    only if it yields at least `min_pos` pseudo-labels."""
    KEYS = AdapterRunner.KEYS | {"pseudo_tau", "min_pos"}

    def __init__(self, d2cfg, weights, tcfg, seed=0):
        super().__init__(d2cfg, weights, tcfg, seed)
        self.stats["n_pos"] = 0

    def _adapt(self, c):
        t1 = time.perf_counter()
        images, sx, sy, feats = c["images"], c["sx"], c["sy"], c["feats"]
        pos_boxes, pos_classes = select_fixed(c["b_orig"], c["scores"], c["classes"], self.t["pseudo_tau"])
        n_pos = len(pos_boxes)
        self.stats["n_pos"] += n_pos
        t2 = time.perf_counter()
        self.timing["cpu"] += t2 - t1
        extra = {"n_pos": n_pos}
        if n_pos < self.t.get("min_pos", 1):
            self.stats["skipped"] += 1
            return "skip_gate", f"n_pos={n_pos}", {}, extra
        scale = np.array([sx, sy, sx, sy], np.float32)
        gt = Instances(images.image_sizes[0])
        gt.gt_boxes = Boxes(torch.as_tensor(pos_boxes * scale, device=self.device))
        gt.gt_classes = torch.as_tensor(pos_classes, device=self.device)
        self.student.train()
        with torch.autocast("cuda", dtype=amp_dtype(), enabled=self.amp):
            feats_s = self.student({k: v.detach() for k, v in feats.items()})
            _, losses = self._heads(images, feats_s, train=True, gt=[gt])
        loss_log = _losses(losses)
        loss = sum(v.float() for v in losses.values())
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        self.student.eval()
        self._ema()
        self.stats["updates"] += 1
        self._sync()
        self.timing["upd"] += time.perf_counter() - t2
        return "update", "", loss_log, extra


class TentRunner(AdapterRunner):
    """Entropy minimisation of the ROI classifier (K+1 classes) over the RPN proposals of the current frame.
    With ema = 0 the teacher equals the student after every update (prediction and update use the same model)."""
    KEYS = AdapterRunner.KEYS | {"clip_value"}

    def __init__(self, d2cfg, weights, tcfg, seed=0):
        super().__init__(d2cfg, weights, tcfg, seed)
        self.clip = tcfg.get("clip_value", 0.0)

    def _roi_outputs(self, images, feats_s):
        """RPN proposals (eval mode, no grad) -> box-head features and class logits WITH grad wrt feats_s."""
        pg, rh = self.model.proposal_generator, self.model.roi_heads
        pg.eval(); rh.eval()
        with torch.no_grad():
            props, _ = pg(images, {k: v.detach() for k, v in feats_s.items()}, None)
        fl = [feats_s[f] for f in rh.box_in_features]
        bf = rh.box_head(rh.box_pooler(fl, [p.proposal_boxes for p in props]))
        scores, _ = rh.box_predictor(bf)
        return bf, scores

    def _loss(self, images, feats_s):
        _, scores = self._roi_outputs(images, feats_s)
        p = F.softmax(scores.float(), dim=1)
        return -(p * torch.log(p.clamp_min(1e-8))).sum(1).mean(), True

    def _adapt(self, c):
        t1 = time.perf_counter()
        images, feats = c["images"], c["feats"]
        self.student.train()
        with torch.autocast("cuda", dtype=amp_dtype(), enabled=self.amp):
            feats_s = self.student({k: v.detach() for k, v in feats.items()})
            loss, do_update = self._loss(images, feats_s)
        self.student.eval()
        loss_log = {"loss": float(loss.detach()) if torch.is_tensor(loss) else float(loss)}
        if not do_update:
            self.stats["skipped"] += 1
            return "skip_when", "WHW update criterion not met", loss_log, {}
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.clip > 0:
            torch.nn.utils.clip_grad_value_(self.student.parameters(), self.clip)
        self.opt.step()
        self._ema()
        self.stats["updates"] += 1
        self._sync()
        self.timing["upd"] += time.perf_counter() - t1
        return "update", "", loss_log, {}


class WHWRunner(TentRunner):
    """WHW, following the official code (github.com/natureyoo/ContinualTTA_ObjectDetection):
      - image level: per FPN level GAP feature; target mean mu_t = mu_{t-1} + (f - mu_{t-1}) / gamma (gamma 128);
        loss = KL between Gaussians with the source covariance S = Cov_src + I max(Cov_src)/30 (equal covariances
        -> 0.5 d^T S^-1 d);
      - object level: box-head features of proposals with max foreground prob >= 0.5, class = argmax; same
        EMA / KL per class, weight min(log(max_i N_i / N_k) + 0.01, 10)^2; terms >= 1e5 skipped;
      - "when": update iff L_img > 2 D_in tau1 or L_img / L_ema > tau2, L_ema = 0.99 L_ema + 0.01 L_img;
      - SGD (momentum 0.9), gradient clipping by value.
    Source statistics (`source_stats`, a torch file): img_mean / img_prec per FPN level, obj_mean / obj_prec per
    class, d_in. Differences to the official setting: adapters after the FPN (shared by all methods here) instead
    of parallel to every backbone block; batch size 1 (online video)."""
    KEYS = TentRunner.KEYS | {"source_stats", "whw"}

    def __init__(self, d2cfg, weights, tcfg, seed=0):
        super().__init__(d2cfg, weights, tcfg, seed)
        st = torch.load(tcfg["source_stats"], map_location=self.device)
        self.src = st
        self.levels = list(st["img_mean"].keys())
        self.mu_te = {k: st["img_mean"][k].clone() for k in self.levels}
        self.mu_te_obj = st["obj_mean"].clone()
        self.n_obj = torch.zeros(st["obj_mean"].shape[0], device=self.device)
        self.d_in = float(st["d_in"])
        self.l_ema = None
        w = tcfg.get("whw", {})
        self.gamma, self.tau1, self.tau2 = w.get("gamma", 128), w.get("tau1", 1.1), w.get("tau2", 1.05)
        self.use_obj, self.use_when = w.get("obj", True), w.get("when", True)

    @staticmethod
    def _maha(d, prec):
        return 0.5 * d @ prec @ d

    def _loss(self, images, feats_s):
        l_img, new_mu = 0.0, {}
        for k in self.levels:
            g = feats_s[k].float().mean(dim=(0, 2, 3))
            mu = self.mu_te[k] + (g - self.mu_te[k]) / self.gamma
            li_k = self._maha(self.src["img_mean"][k] - mu, self.src["img_prec"][k])
            if float(li_k) < 1e5:
                l_img = l_img + li_k
                new_mu[k] = mu.detach()
        li = float(l_img)
        if self.l_ema is None:
            self.l_ema = li
        do_update = (not self.use_when) or li > 2 * self.d_in * self.tau1 or li / max(self.l_ema, 1e-12) > self.tau2
        self.l_ema = 0.99 * self.l_ema + 0.01 * li
        self.mu_te.update(new_mu)
        loss = l_img
        if self.use_obj:
            bf, scores = self._roi_outputs(images, feats_s)
            p = F.softmax(scores.float(), dim=1)
            K = p.shape[1] - 1
            fg_s, fg_c = p[:, :K].max(1)
            keep = fg_s >= 0.5
            if keep.any():
                bfk, cls = bf[keep].float(), fg_c[keep]
                for c in cls.unique().tolist():
                    fc = bfk[cls == c]
                    self.n_obj[c] += fc.shape[0]
                    mu = self.mu_te_obj[c] + (fc - self.mu_te_obj[c]).sum(0) / self.gamma
                    nz = self.n_obj[self.n_obj > 0]
                    w = min(float(torch.log(nz.max() / self.n_obj[c])) + 0.01, 10.0) ** 2
                    lo = w * self._maha(self.src["obj_mean"][c] - mu, self.src["obj_prec"][c])
                    if float(lo) < 1e5:
                        loss = loss + lo
                        self.mu_te_obj[c] = mu.detach()
        return loss, do_update

    def method_state(self):
        return {"mu_te": self.mu_te, "mu_te_obj": self.mu_te_obj, "n_obj": self.n_obj, "l_ema": self.l_ema}

    def load_method_state(self, st):
        self.mu_te = {k: v.clone() for k, v in st["mu_te"].items()}
        self.mu_te_obj, self.n_obj, self.l_ema = st["mu_te_obj"].clone(), st["n_obj"].clone(), st["l_ema"]


RUNNERS = {"source_only": SourceOnlyRunner, "mean_teacher": MeanTeacherRunner, "tent": TentRunner,
           "whw": WHWRunner}
