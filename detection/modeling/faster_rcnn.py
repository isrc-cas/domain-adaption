import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import build_backbone
from .roi_heads import BoxHead
from .rpn import RPN


class LocalWindowExtractor:
    def __init__(self, window_sizes=(3, 7, 13, 21, 32)):
        assert 1 not in window_sizes, 'Not support window size 1'
        self.window_sizes = window_sizes
        self.strides = (1, 3, 6, 10, 15)

    def __call__(self, feature):
        N, C, H, W = feature.shape
        windows = []
        for i, K in enumerate(self.window_sizes):
            # stride = max(1, (K - 1) // 2)
            stride = self.strides[i]
            NEW_H, NEW_W = int((H - K) / stride + 1), int((W - K) / stride + 1)

            img_windows = F.unfold(feature, kernel_size=K, stride=stride)
            img_windows = img_windows.view(N, C, K, K, -1)
            var, mean = torch.var_mean(img_windows, dim=(2, 3), unbiased=False)  # (N, C, NEW_H * NEW_W)
            std = torch.sqrt(var + 1e-12)
            x = torch.cat((mean, std), dim=1)  # (N, C * 2, NEW_H * NEW_W)
            x = x.view(N, C * 2, NEW_H, NEW_W)
            windows.append(x)

        return windows


class FasterRCNN(nn.Module):
    def __init__(self, cfg):
        super(FasterRCNN, self).__init__()
        backbone = build_backbone(cfg)
        in_channels = backbone.out_channels
        self.backbone = backbone
        self.rpn = RPN(cfg, in_channels)
        self.box_head = BoxHead(cfg, in_channels)
        window_sizes = (3, 7, 13, 21, 32)
        self.local_window_extractor = LocalWindowExtractor(window_sizes)

    def forward(self, images, img_metas, targets=None, t_images=None, t_img_metas=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        outputs = dict()
        loss_dict = dict()

        features = self.backbone(images)
        proposals, rpn_losses, s_rpn_logits = self.rpn(images, features, img_metas, targets)
        dets, box_losses, box_features = self.box_head(features, proposals, img_metas, targets)

        if self.training and t_images is not None:
            s_windows = self.local_window_extractor(features)

            t_features = self.backbone(t_images)
            t_windows = self.local_window_extractor(t_features)

            t_proposals, _, t_rpn_logits = self.rpn(t_images, t_features, t_img_metas, targets=None)
            _, _, t_box_features = self.box_head(t_features, t_proposals, t_img_metas, targets=None)

            outputs['s_windows'] = s_windows
            outputs['t_windows'] = t_windows
            outputs['s_rpn_logits'] = s_rpn_logits
            outputs['t_rpn_logits'] = t_rpn_logits
            outputs['s_box_features'] = box_features
            outputs['t_box_features'] = t_box_features

        if self.training:
            loss_dict.update(rpn_losses)
            loss_dict.update(box_losses)
            return loss_dict, outputs
        return dets
