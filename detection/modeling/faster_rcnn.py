import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import build_backbone
from .roi_heads import BoxHead
from .rpn import RPN


class FasterRCNN(nn.Module):
    def __init__(self, cfg):
        super(FasterRCNN, self).__init__()
        backbone = build_backbone(cfg)
        in_channels = backbone.out_channels
        self.backbone = backbone
        self.rpn = RPN(cfg, in_channels)
        self.box_head = BoxHead(cfg, in_channels)

    def forward(self, images, img_metas, targets=None, t_images=None, t_img_metas=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        outputs = dict()
        loss_dict = dict()

        features = self.backbone(images)
        proposals, rpn_losses, s_rpn_logits = self.rpn(images, features, img_metas, targets)
        dets, box_losses, s_proposals, box_features, roi_features = self.box_head(features, proposals, img_metas, targets)

        if self.training and t_images is not None:
            t_features = self.backbone(t_images)

            t_proposals, _, t_rpn_logits = self.rpn(t_images, t_features, t_img_metas, targets=None)
            _, _, t_proposals, t_box_features, t_roi_features = self.box_head(t_features, t_proposals, t_img_metas, targets=None)

            outputs['s_features'] = [features]
            outputs['t_features'] = [t_features]
            outputs['s_rpn_logits'] = s_rpn_logits
            outputs['t_rpn_logits'] = t_rpn_logits
            outputs['s_box_features'] = box_features
            outputs['t_box_features'] = t_box_features
            outputs['s_roi_features'] = roi_features
            outputs['t_roi_features'] = t_roi_features
            outputs['s_proposals'] = s_proposals
            outputs['t_proposals'] = t_proposals

        if self.training:
            loss_dict.update(rpn_losses)
            loss_dict.update(box_losses)
            return loss_dict, outputs
        return dets
