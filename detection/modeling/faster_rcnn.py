import numpy as np
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from terminaltables import AsciiTable

from detection.layers import grad_reverse, softmax_focal_loss, sigmoid_focal_loss, style_pool2d, l2_loss
from .backbone import build_backbone
from .roi_heads import BoxHead
from .rpn import RPN


class Dis(nn.Module):
    def __init__(self,
                 cfg,
                 in_channels,
                 embedding_kernel_size=3,
                 embedding_norm=True,
                 embedding_dropout=True,
                 func_name='focal_loss',
                 focal_loss_gamma=5,
                 pool_type='avg',
                 loss_weight=1.0,
                 window_strides=None,
                 window_sizes=(3, 9, 15, 21, -1)):
        super().__init__()
        # fmt:off
        anchor_scales       = cfg.MODEL.RPN.ANCHOR_SIZES
        anchor_ratios       = cfg.MODEL.RPN.ASPECT_RATIOS
        num_anchors         = len(anchor_scales) * len(anchor_ratios)
        # fmt:on
        self.in_channels = in_channels
        self.embedding_kernel_size = embedding_kernel_size
        self.embedding_norm = embedding_norm
        self.embedding_dropout = embedding_dropout
        self.num_windows = len(window_sizes)
        self.num_anchors = num_anchors
        self.window_sizes = window_sizes
        if window_strides is None:
            self.window_strides = [None] * len(window_sizes)
        else:
            assert len(window_strides) == len(window_sizes), 'window_strides and window_sizes should has same len'
            self.window_strides = window_strides

        if pool_type == 'avg':
            channel_multiply = 1
            pool_func = F.avg_pool2d
        elif pool_type == 'style':
            channel_multiply = 2
            pool_func = style_pool2d
        else:
            raise ValueError
        self.pool_type = pool_type
        self.pool_func = pool_func

        if func_name == 'focal_loss':
            num_domain_classes = 2
            loss_func = partial(softmax_focal_loss, gamma=focal_loss_gamma)
        elif func_name == 'cross_entropy':
            num_domain_classes = 2
            loss_func = F.cross_entropy
        elif func_name == 'l2':
            num_domain_classes = 1
            loss_func = l2_loss
        else:
            raise ValueError
        self.focal_loss_gamma = focal_loss_gamma
        self.func_name = func_name
        self.loss_func = loss_func
        self.loss_weight = loss_weight
        self.num_domain_classes = num_domain_classes

        NormModule = nn.BatchNorm2d if embedding_norm else nn.Identity
        DropoutModule = nn.Dropout if embedding_dropout else nn.Identity

        padding = (embedding_kernel_size - 1) // 2
        bias = not embedding_norm
        self.embedding = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=embedding_kernel_size, stride=1, padding=padding, bias=bias),
            NormModule(in_channels),
            nn.ReLU(True),
            DropoutModule(),

            nn.Conv2d(in_channels, 256, kernel_size=embedding_kernel_size, stride=1, padding=padding, bias=bias),
            NormModule(256),
            nn.ReLU(True),
            DropoutModule(),

            nn.Conv2d(256, 256, kernel_size=embedding_kernel_size, stride=1, padding=padding, bias=bias),
            NormModule(256),
            nn.ReLU(True),
            DropoutModule(),
        )

        self.shared_semantic = nn.Sequential(
            nn.Conv2d(in_channels + num_anchors, in_channels, kernel_size=embedding_kernel_size, stride=1, padding=padding, bias=bias),
            NormModule(in_channels),
            nn.ReLU(True),

            nn.Conv2d(in_channels, 256, kernel_size=embedding_kernel_size, stride=1, padding=padding, bias=bias),
            NormModule(256),
            nn.ReLU(True),
            DropoutModule(),

            nn.Conv2d(256, 256, kernel_size=embedding_kernel_size, stride=1, padding=padding, bias=bias),
            NormModule(256),
            nn.ReLU(True),
        )

        self.semantic_list = nn.ModuleList()

        self.inter_channels = 128
        for i in range(self.num_windows):
            self.semantic_list += [
                nn.Sequential(
                    nn.Conv2d(256, 128, 1, bias=bias),
                    NormModule(128),
                    nn.ReLU(True),
                    nn.Conv2d(128, 1, 1),
                )
            ]

        self.fc = nn.Sequential(
            nn.Conv2d(256 * channel_multiply, 128, 1, bias=False),
            NormModule(128),
            nn.ReLU(inplace=True),
        )

        self.split_fc = nn.Sequential(
            nn.Conv2d(128, self.num_windows * 256 * channel_multiply, 1, bias=False),
        )

        self.predictor = nn.Linear(256 * channel_multiply, num_domain_classes)

    def forward(self, feature, rpn_logits):
        if feature.shape != rpn_logits.shape:
            rpn_logits = F.interpolate(rpn_logits, size=(feature.size(2), feature.size(3)), mode='bilinear', align_corners=True)

        semantic_map = torch.cat((feature, rpn_logits), dim=1)
        semantic_map = self.shared_semantic(semantic_map)

        feature = self.embedding(feature)
        N, C, H, W = feature.shape

        pyramid_features = []
        domain_logits_list = []
        for i, k in enumerate(self.window_sizes):
            if k == -1:
                x = self.pool_func(feature, kernel_size=(H, W))
            elif k == 1:
                x = feature
            else:
                stride = self.window_strides[i]
                if stride is None:
                    stride = 1  # default
                x = self.pool_func(feature, kernel_size=k, stride=stride)
            _, _, h, w = x.shape
            semantic_map_per_level = F.interpolate(semantic_map, size=(h, w), mode='bilinear', align_corners=True)
            domain_logits = self.semantic_list[i](semantic_map_per_level)
            domain_logits_list.append(domain_logits)

            domain_probs = domain_logits.sigmoid()

            domain_uncertainty = - domain_probs * torch.log(domain_probs)

            w_spatial = 1 - domain_uncertainty
            x = x + x * w_spatial
            x = F.adaptive_avg_pool2d(x, output_size=1)
            pyramid_features.append(x)

        fuse = sum(pyramid_features)  # [N, 256, 1, 1]
        merge = self.fc(fuse)  # [N, 128, 1, 1]
        split = self.split_fc(merge)  # [N, num_windows * 256, 1, 1]

        split = split.view(N, self.num_windows, -1, 1, 1)

        w = F.softmax(split, dim=1)
        w = torch.unbind(w, dim=1)  # List[N, 256, 1, 1]

        pyramid_features = list(map(lambda x, y: x * y, pyramid_features, w))
        final_features = sum(pyramid_features)
        final_features = final_features.view(N, -1)

        logits = self.predictor(final_features)
        return logits, domain_logits_list

    def __repr__(self):
        attrs = {
            'in_channels': self.in_channels,
            'embedding_kernel_size': self.embedding_kernel_size,
            'embedding_norm': self.embedding_norm,
            'embedding_dropout': self.embedding_dropout,
            'num_domain_classes': self.num_domain_classes,
            'func_name': self.func_name,
            'focal_loss_gamma': self.focal_loss_gamma,
            'pool_type': self.pool_type,
            'loss_weight': self.loss_weight,
            'window_strides': self.window_strides,
            'window_sizes': self.window_sizes,
        }
        table = AsciiTable(list(zip(attrs.keys(), attrs.values())))
        table.inner_heading_row_border = False
        return self.__class__.__name__ + '\n' + table.table


class FasterRCNN(nn.Module):
    def __init__(self, cfg):
        super(FasterRCNN, self).__init__()
        self.cfg = cfg
        backbone = build_backbone(cfg)
        in_channels = backbone.out_channels

        self.backbone = backbone
        self.rpn = RPN(cfg, in_channels)
        self.box_head = BoxHead(cfg, in_channels)

        self.enable_adaptation = len(cfg.DATASETS.TARGETS) > 0
        self.ada_layers = [False] * 3
        if self.enable_adaptation:
            self.ada_layers = cfg.ADV.LAYERS
            dis_model = cfg.ADV.DIS_MODEL

            assert len(list(filter(lambda x: x, self.ada_layers))) == len(dis_model)

            # self.netD = netD()
            # self.netD = D(cfg, in_channels)

            self.dis_list = nn.ModuleList()
            for model_config in dis_model:
                dis = Dis(cfg, **model_config)
                print(dis)
                self.dis_list += [
                    dis
                ]

    def forward_vgg16(self, x):
        adaptation_feats = []
        idx = 0
        for i in range(14):
            x = self.backbone[i](x)
        if self.ada_layers[idx]:
            adaptation_feats.append(x)

        idx += 1
        for i in range(14, 21):
            x = self.backbone[i](x)
        if self.ada_layers[idx]:
            adaptation_feats.append(x)

        idx += 1
        for i in range(21, len(self.backbone)):
            x = self.backbone[i](x)
        if self.ada_layers[idx]:
            adaptation_feats.append(x)

        return x, adaptation_feats

    def forward_resnet101(self, x):
        adaptation_feats = []
        idx = 0
        x = self.backbone.stem(x)
        x = self.backbone.layer1(x)
        if self.ada_layers[idx]:
            adaptation_feats.append(x)

        idx += 1
        x = self.backbone.layer2(x)
        if self.ada_layers[idx]:
            adaptation_feats.append(x)

        idx += 1
        x = self.backbone.layer3(x)
        if self.ada_layers[idx]:
            adaptation_feats.append(x)

        return x, adaptation_feats

    def forward(self, images, img_metas, targets=None, t_images=None, t_img_metas=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        outputs = dict()
        loss_dict = dict()

        forward_func = getattr(self, 'forward_{}'.format(self.cfg.MODEL.BACKBONE.NAME))

        features, s_adaptation_feats = forward_func(images)
        proposals, rpn_losses, s_rpn_logits = self.rpn(images, features, img_metas, targets)
        dets, box_losses, s_proposals, box_features, roi_features = self.box_head(features, proposals, img_metas, targets)

        if self.enable_adaptation and self.training and t_images is not None:
            t_features, t_adaptation_feats = forward_func(t_images)

            t_proposals, _, t_rpn_logits = self.rpn(t_images, t_features, t_img_metas, targets=None)
            _, _, t_proposals, t_box_features, t_roi_features = self.box_head(t_features, t_proposals, t_img_metas, targets=None)

            device = features.device
            for i, (s_feat, t_feat, netD) in enumerate(zip(s_adaptation_feats, t_adaptation_feats, self.dis_list)):
                s_domain_logits, s_domain_logits_list = netD(grad_reverse(s_feat, 1.0), grad_reverse(s_rpn_logits, 1.0))
                t_domain_logits, t_domain_logits_list = netD(grad_reverse(t_feat, 1.0), grad_reverse(t_rpn_logits, 1.0))
                loss_func = netD.loss_func
                loss_weight = netD.loss_weight
                num_windows = netD.num_windows
                gamma = netD.focal_loss_gamma

                w = 0.5
                s_domain_loss = loss_func(s_domain_logits, torch.zeros(s_domain_logits.size(0), dtype=torch.long, device=device)) * w
                t_domain_loss = loss_func(t_domain_logits, torch.ones(t_domain_logits.size(0), dtype=torch.long, device=device)) * w

                list_weights = (1.0 / num_windows) * 0.5

                loss_dict.update({
                    's_domain_loss%d' % i: s_domain_loss * loss_weight,
                    't_domain_loss%d' % i: t_domain_loss * loss_weight,
                    's_domain_list_loss%d' % i: list_weights * sum(sigmoid_focal_loss(la, torch.zeros_like(la), gamma=gamma) for la in s_domain_logits_list) * loss_weight,
                    't_domain_list_loss%d' % i: list_weights * sum(sigmoid_focal_loss(la, torch.ones_like(la), gamma=gamma) for la in t_domain_logits_list) * loss_weight,
                })

            # outputs['s_features'] = s_adaptation_feats
            # outputs['t_features'] = t_adaptation_feats
            # outputs['s_rpn_logits'] = s_rpn_logits
            # outputs['t_rpn_logits'] = t_rpn_logits
            # outputs['s_box_features'] = box_features
            # outputs['t_box_features'] = t_box_features
            # outputs['s_roi_features'] = roi_features
            # outputs['t_roi_features'] = t_roi_features
            # outputs['s_proposals'] = s_proposals
            # outputs['t_proposals'] = t_proposals

        if self.training:
            loss_dict.update(rpn_losses)
            loss_dict.update(box_losses)
            return loss_dict, outputs
        return dets
