import torch.nn as nn

from .roi_heads import BoxHead
from .rpn import RPN


class FasterRCNN(nn.Module):
    def __init__(self, features, num_classes=9, in_channels=512):
        super(FasterRCNN, self).__init__()
        self.features = features
        self.rpn = RPN(in_channels)
        self.box_head = BoxHead(in_channels, num_classes)

    def forward(self, images, img_metas, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        features = self.features(images)
        loss_dict = {}
        proposals, rpn_losses = self.rpn(images, features, img_metas, targets)
        dets, box_losses = self.box_head(images, features, proposals, img_metas, targets)
        if self.training:
            loss_dict.update(rpn_losses)
            loss_dict.update(box_losses)
            return loss_dict
        return dets
