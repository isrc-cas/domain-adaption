import torch.nn as nn
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

    def forward(self, images, img_metas, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        features = self.backbone(images)
        proposals, rpn_losses = self.rpn(images, features, img_metas, targets)
        dets, box_losses = self.box_head(images, features, proposals, img_metas, targets)
        loss_dict = {}
        if self.training:
            loss_dict.update(rpn_losses)
            loss_dict.update(box_losses)
            return loss_dict
        return dets
