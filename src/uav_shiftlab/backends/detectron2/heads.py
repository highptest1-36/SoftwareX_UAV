# SPDX-License-Identifier: Apache-2.0
"""ROI heads used for self-training at test time.

AdapterROIHeads = StandardROIHeads (same parameters, so source checkpoints load unchanged) whose box predictor
normalises the per-proposal losses explicitly as sum / number of sampled proposals:

  loss_cls     = sum_i CE(s_i, y_i) / N
  loss_box_reg = sum_{i in fg} smooth_l1(d_i, t_i) / N

Mathematically this equals Detectron2's default (mean cross-entropy, summed smooth-L1 / N), but the floating-point
summation order differs. The pseudo-label losses of the mean-teacher profile were produced with this exact form, so it
is kept to reproduce them bit for bit. Inference is unchanged.
"""
import torch.nn.functional as F
from detectron2.layers import cat
from detectron2.modeling import ROI_HEADS_REGISTRY
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.modeling.roi_heads.roi_heads import StandardROIHeads
from fvcore.nn import smooth_l1_loss


class SumNormFastRCNNOutputLayers(FastRCNNOutputLayers):
    def losses(self, predictions, proposals):
        if not proposals:
            return super().losses(predictions, proposals)
        scores, deltas = predictions
        gt_classes = cat([p.gt_classes for p in proposals], dim=0)
        n = max(1, gt_classes.numel())
        loss_cls = F.cross_entropy(scores, gt_classes, reduction="none").sum() / n
        prop = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        gtb = cat([(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals], dim=0)
        fg = (gt_classes >= 0) & (gt_classes < self.num_classes)
        fg = fg.nonzero().squeeze(1)
        box_dim = prop.shape[1]
        if deltas.shape[1] == box_dim:
            fg_pred = deltas[fg]
        else:
            fg_pred = deltas.view(-1, self.num_classes, box_dim)[fg, gt_classes[fg]]
        target = self.box2box_transform.get_deltas(prop[fg], gtb[fg])
        l_box = smooth_l1_loss(fg_pred, target, self.smooth_l1_beta, reduction="none").sum(1)
        loss_box = l_box.sum() / n
        return {"loss_cls": loss_cls * self.loss_weight.get("loss_cls", 1.0),
                "loss_box_reg": loss_box * self.loss_weight.get("loss_box_reg", 1.0)}


@ROI_HEADS_REGISTRY.register()
class AdapterROIHeads(StandardROIHeads):
    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        ret = super()._init_box_head(cfg, input_shape)
        ret["box_predictor"] = SumNormFastRCNNOutputLayers(cfg, ret["box_head"].output_shape)
        return ret
