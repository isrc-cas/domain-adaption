import torch.nn as nn
from torchvision import models

from .roi_heads import BoxHead
from .rpn import RPN


class FasterRCNN(nn.Module):
    def __init__(self, cfg):
        super(FasterRCNN, self).__init__()
        vgg16 = models.vgg16(True)
        features = vgg16.features[:-1]

        in_channels = 512
        self.backbone = features
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
