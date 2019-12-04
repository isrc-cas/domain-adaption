import numpy as np
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

from detection.layers import grad_reverse, softmax_focal_loss, sigmoid_focal_loss
from .backbone import build_backbone
from .roi_heads import BoxHead
from .rpn import RPN


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class netD(nn.Module):
    def __init__(self):
        super(netD, self).__init__()
        self.conv1 = conv3x3(1024, 512, stride=2)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = conv3x3(512, 128, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = conv3x3(128, 128, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.conv1(x))), training=self.training)
        x = F.dropout(F.relu(self.bn2(self.conv2(x))), training=self.training)
        x = F.dropout(F.relu(self.bn3(self.conv3(x))), training=self.training)
        x = F.avg_pool2d(x, (x.size(2), x.size(3)))
        x = x.view(-1, 128)
        x = self.fc(x)
        return x


class LocalWindowExtractor:
    def __init__(self, start_size, stop_size, num_windows):
        assert 1 != start_size, 'Not support window size 1'
        self.start_size = start_size
        self.stop_size = stop_size
        self.num_windows = num_windows

    def __call__(self, feature):
        N, C, H, W = feature.shape
        windows = []
        stop = min(H, W) if self.stop_size == -1 else self.stop_size
        start = self.start_size if self.start_size > 0 else (self.start_size + stop)
        num = self.num_windows
        window_sizes = np.linspace(start=start, stop=stop, num=num).round().astype('int')
        window_sizes = window_sizes.tolist()
        for i, K in enumerate(window_sizes):
            stride = max(1, (K - 1) // 2)
            NEW_H, NEW_W = int((H - K) / stride + 1), int((W - K) / stride + 1)

            img_windows = F.unfold(feature, kernel_size=K, stride=stride)
            img_windows = img_windows.view(N, C, K, K, -1)
            var, mean = torch.var_mean(img_windows, dim=(2, 3), unbiased=False)  # (N, C, NEW_H * NEW_W)
            std = torch.sqrt(var + 1e-12)
            x = torch.cat((mean, std), dim=1)  # (N, C * 2, NEW_H * NEW_W)
            x = x.view(N, C * 2, NEW_H, NEW_W)
            windows.append(x)

        return windows


class D(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()
        # fmt:off
        start_size          = cfg.ADV.WINDOWS.START_SIZE
        stop_size           = cfg.ADV.WINDOWS.STOP_SIZE
        num_windows         = cfg.ADV.WINDOWS.NUM_WINDOWS
        # fmt:on
        self.in_channels = in_channels
        self.num_windows = num_windows
        self.local_window_extractor = LocalWindowExtractor(start_size, stop_size, num_windows)

        self.g_list = nn.ModuleList()
        self.theta_list = nn.ModuleList()
        self.phi_list = nn.ModuleList()
        self.out_list = nn.ModuleList()
        self.model_list = nn.ModuleList()

        norm_layer = nn.BatchNorm2d
        self.inter_channels = in_channels // 2

        for i in range(num_windows):
            self.g_list += [
                nn.Conv2d(in_channels * 2, self.inter_channels, 1)
            ]
            self.theta_list += [
                nn.Conv2d(in_channels * 2, self.inter_channels, 1)
            ]
            self.phi_list += [
                nn.Conv2d(in_channels * 2, self.inter_channels, 1)
            ]
            self.out_list += [
                nn.Sequential(
                    nn.Conv2d(self.inter_channels, in_channels * 2, 1, bias=False),
                    norm_layer(in_channels * 2),
                )
            ]
            self.model_list += [
                nn.Sequential(
                    nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
                    norm_layer(in_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout(),

                    nn.Conv2d(in_channels, 128, kernel_size=1, bias=False),
                    norm_layer(128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(),

                    nn.Conv2d(128, 128, kernel_size=1, bias=False),
                    norm_layer(128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(),

                    nn.Conv2d(128, 2, kernel_size=1),
                )
            ]

    def forward(self, feature):
        window_features = self.local_window_extractor(feature)

        logits = []
        for i in range(self.num_windows):
            x = window_features[i]
            n, c, h, w = x.shape

            # g_x: [N, HxW, C]
            g_x = self.g_list[i](x).view(n, self.inter_channels, -1)
            g_x = g_x.permute(0, 2, 1)

            # theta_x: [N, HxW, C]
            theta_x = self.theta_list[i](x).view(n, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)

            # phi_x: [N, C, HxW]
            phi_x = self.phi_list[i](x).view(n, self.inter_channels, -1)

            f = torch.matmul(theta_x, phi_x)
            f = f / theta_x.shape[-1] ** 0.5
            f = f.softmax(dim=-1)

            # y: [N, HxW, C]
            y = torch.matmul(f, g_x)
            # y: [N, C, H, W]
            y = y.permute(0, 2, 1).reshape(n, self.inter_channels, h, w)
            x = x + self.out_list[i](y)

            logit = self.model_list[i](x) * 0.5  # (N, num_classes, H, W)
            logit = logit.permute(0, 2, 3, 1).reshape(-1, 2)

            logits.append(logit)

        return logits


class DisModelPerLevel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # fmt:off
        anchor_scales       = cfg.MODEL.RPN.ANCHOR_SIZES
        anchor_ratios       = cfg.MODEL.RPN.ASPECT_RATIOS
        num_anchors         = len(anchor_scales) * len(anchor_ratios)
        in_channels         = cfg.ADV.IN_CHANNELS

        pairwise_func       = cfg.ADV.NETVLAD.PAIRWISE_FUNC
        use_scale           = cfg.ADV.NETVLAD.USE_SCALE
        center_styles_mode  = cfg.ADV.NETVLAD.CENTER_STYLES_MODE
        normalize_residual  = cfg.ADV.NETVLAD.NORMALIZE_RESIDUAL
        normalize_before_mm = cfg.ADV.NETVLAD.NORMALIZE_BEFORE_MM
        use_focal_loss      = cfg.ADV.NETVLAD.USE_FOCAL_LOSS
        focal_loss_gammas   = cfg.ADV.NETVLAD.FOCAL_LOSS_GAMMAS
        weighted            = cfg.ADV.NETVLAD.WEIGHTED
        window_weights      = cfg.ADV.NETVLAD.WINDOW_WEIGHTS

        start_size          = cfg.ADV.WINDOWS.START_SIZE
        stop_size           = cfg.ADV.WINDOWS.STOP_SIZE
        num_windows         = cfg.ADV.WINDOWS.NUM_WINDOWS
        # fmt:on

        self.pairwise_func = pairwise_func
        self.use_scale = use_scale
        self.center_styles_mode = center_styles_mode
        self.in_channels = in_channels
        self.normalize_residual = normalize_residual
        self.normalize_before_mm = normalize_before_mm
        self.use_focal_loss = use_focal_loss
        self.focal_loss_gammas = focal_loss_gammas
        self.weighted = weighted
        self.window_weights = window_weights

        self.num_windows = num_windows
        self.local_window_extractor = LocalWindowExtractor(start_size, stop_size, num_windows)
        self.model_list = nn.ModuleList()
        self.phi_list = nn.ModuleList()

        self.inter_channels = in_channels
        self.theta = nn.Conv2d(in_channels * 2, self.inter_channels, 1)

        for _ in range(num_windows):
            self.model_list += [
                nn.Sequential(
                    nn.Linear(in_channels * 2, in_channels),
                    nn.ReLU(inplace=True),
                    nn.Linear(in_channels, 256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, 2),
                )
            ]

            self.phi_list += [
                nn.Conv2d(in_channels * 2, self.inter_channels, 1)
            ]

    def embedded_gaussian(self, theta_x, phi_x):
        """
        Args:
            theta_x: (n, num_windows, c)
            phi_x: (n, c, hxw)
        Returns:
        """
        if self.normalize_before_mm:
            theta_x = F.normalize(theta_x, dim=2)
            phi_x = F.normalize(phi_x, dim=1)
        # pairwise_weight: [N, num_windows, hxw]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        if self.use_scale and not self.normalize_before_mm:
            # theta_x.shape[-1] is `self.inter_channels`
            pairwise_weight /= (theta_x.shape[-1] ** 0.5)
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    def dot_product(self, theta_x, phi_x):
        """
        Args:
            theta_x: (n, num_windows, c)
            phi_x: (n, c, hxw)
        Returns:
        """
        # pairwise_weight: [N, num_windows, hxw]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight /= pairwise_weight.shape[-1]
        return pairwise_weight

    def forward(self, feature, label=0):
        window_features = self.local_window_extractor(feature)
        logits = []
        center_styles_mode = self.center_styles_mode
        inter_channels = self.inter_channels

        if center_styles_mode == 'avg':
            num_centers = self.num_windows
            center_styles_list = [torch.mean(x, dim=(2, 3), keepdim=True) for x in window_features]  # [(n, c, 1, 1)]
            center_styles = torch.stack(center_styles_list, dim=0)  # (num_windows, n, c, 1, 1)
        else:
            raise ValueError
        _, n, c, _, _ = center_styles.shape
        # (n, num_windows, c)
        theta_x = self.theta(center_styles.view(num_centers * n, c, 1, 1)).view(num_centers, n, inter_channels)
        theta_x = theta_x.permute(1, 0, 2)
        for i, x in enumerate(window_features):  # (n, c, h, w)
            n, c, h, w = x.shape
            phi = self.phi_list[i]

            # (n, c, hxw)
            phi_x = phi(x).view(n, inter_channels, -1)

            # (N, num_windows, hxw)
            pairwise_func = getattr(self, self.pairwise_func)
            pairwise_weight = pairwise_func(theta_x, phi_x)

            # (N, num_windows, 1, hxw)
            pairwise_weight = pairwise_weight.unsqueeze(dim=2)

            # (1, n, c, h, w)
            x = x.unsqueeze(0)

            # (num_windows, n, c, h, w)
            residual = (x - center_styles)

            # (n, num_windows, c, h, w)
            residual = residual.permute(1, 0, 2, 3, 4)

            # (n, num_windows, c, hxw)
            residual = residual.view(n, num_centers, c, -1)

            # (n, num_windows, c)
            residual = (pairwise_weight * residual).sum(dim=3)
            if self.normalize_residual:
                residual = F.normalize(residual, dim=2)

            x = self.model_list[i](residual.view(n * num_centers, c))
            logits.append(x)

        # window_weights = self.window_weights
        # if self.use_focal_loss:
        #     losses = sum(sigmoid_focal_loss(l, torch.full_like(l, label), gamma=gamma) * w for i, (l, gamma, w) in
        #                  enumerate(zip(logits, self.focal_loss_gammas, window_weights)))
        # else:
        #     losses = sum(F.binary_cross_entropy_with_logits(l, torch.full_like(l, label)) * w for i, (l, w) in
        #                  enumerate(zip(logits, window_weights)))

        return logits


class Dis(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()
        # fmt:off
        start_size          = cfg.ADV.WINDOWS.START_SIZE
        stop_size           = cfg.ADV.WINDOWS.STOP_SIZE
        num_windows         = cfg.ADV.WINDOWS.NUM_WINDOWS
        anchor_scales       = cfg.MODEL.RPN.ANCHOR_SIZES
        anchor_ratios       = cfg.MODEL.RPN.ASPECT_RATIOS
        num_anchors         = len(anchor_scales) * len(anchor_ratios)
        in_channels         = cfg.ADV.IN_CHANNELS

        # fmt:on
        self.in_channels = in_channels
        self.num_windows = 5
        self.num_anchors = num_anchors

        self.embedding = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Dropout(),
        )

        self.shared_semantic = nn.Sequential(
            nn.Conv2d(in_channels + num_anchors, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        self.semantic_list = nn.ModuleList()
        # self.g_list = nn.ModuleList()
        # self.theta_list = nn.ModuleList()
        # self.phi_list = nn.ModuleList()
        # self.out_list = nn.ModuleList()

        self.inter_channels = 128
        for i in range(self.num_windows):
            self.semantic_list += [
                nn.Sequential(
                    nn.Conv2d(256, 128, 1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(True),

                    nn.Conv2d(128, 1, 1),
                )
            ]
            # self.g_list += [
            #     nn.Conv2d(256, self.inter_channels, 1)
            # ]
            # self.theta_list += [
            #     nn.Conv2d(256, self.inter_channels, 1)
            # ]
            # self.phi_list += [
            #     nn.Conv2d(256, self.inter_channels, 1)
            # ]
            # self.out_list += [
            #     nn.Sequential(
            #         nn.Conv2d(self.inter_channels, 256, 1, bias=False),
            #         norm_layer(256),
            #     )
            # ]

        self.fc = nn.Sequential(
            nn.Conv2d(256, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.split_fc = nn.Sequential(
            nn.Conv2d(128, self.num_windows * 256, 1, bias=False),
        )

        self.predictor = nn.Linear(256, 2)

    def forward(self, feature, rpn_logits):
        semantic_map = torch.cat((feature, rpn_logits), dim=1)
        semantic_map = self.shared_semantic(semantic_map)

        feature = self.embedding(feature)
        N, C, H, W = feature.shape

        pyramid_features = []
        for i, k in enumerate([3, 9, 15, 21, -1]):
            if k == -1:
                x = F.avg_pool2d(feature, kernel_size=(H, W))
            else:
                x = F.avg_pool2d(feature, kernel_size=k, stride=(k - 1) // 2)
            _, _, h, w = x.shape
            semantic_map_per_level = F.interpolate(semantic_map, size=(h, w), mode='bilinear', align_corners=True)
            semantic_map_per_level = self.semantic_list[i](semantic_map_per_level)
            semantic_map_per_level = semantic_map_per_level.view(N, -1)
            semantic_map_per_level = F.softmax(semantic_map_per_level, dim=1)
            semantic_map_per_level = semantic_map_per_level.view(N, 1, h, w)

            # # g_x: [N, HxW, C]
            # g_x = self.g_list[i](x).view(N, self.inter_channels, -1)
            # g_x = g_x.permute(0, 2, 1)
            #
            # # theta_x: [N, HxW, C]
            # theta_x = self.theta_list[i](x).view(N, self.inter_channels, -1)
            # theta_x = theta_x.permute(0, 2, 1)
            #
            # # phi_x: [N, C, HxW]
            # phi_x = self.phi_list[i](x).view(N, self.inter_channels, -1)
            #
            # # weight: [N, HxW, HxW]
            # weight = torch.matmul(theta_x, phi_x)
            # weight = weight / theta_x.shape[-1] ** 0.5
            # weight = weight.softmax(dim=-1)
            #
            # # y: [N, HxW, C]
            # y = torch.matmul(weight, g_x)
            # # y: [N, C, H, W]
            # y = y.permute(0, 2, 1).reshape(N, self.inter_channels, h, w)
            # x = x + self.out_list[i](y)
            # x = F.adaptive_avg_pool2d(x, 1)  # [N, 256, 1, 1]

            x = torch.sum(x * semantic_map_per_level, dim=(2, 3), keepdim=True)
            pyramid_features.append(x)

        fuse = sum(pyramid_features)  # [N, 256, 1, 1]
        merge = self.fc(fuse)  # [N, 128, 1, 1]
        split = self.split_fc(merge)  # [N, num_windows * 256, 1, 1]

        split = split.view(N, self.num_windows, 256, 1, 1)

        w = F.softmax(split, dim=1)
        w = torch.unbind(w, dim=1)  # List[N, 256, 1, 1]

        pyramid_features = list(map(lambda x, y: x * y, pyramid_features, w))
        final_features = sum(pyramid_features)
        final_features = final_features.view(N, -1)

        logits = self.predictor(final_features)
        return logits


class FasterRCNN(nn.Module):
    def __init__(self, cfg):
        super(FasterRCNN, self).__init__()
        self.cfg = cfg
        # fmt:off
        loss_func               = cfg.ADV.LOSS_FUNC
        focal_loss_gamma        = cfg.ADV.FOCAL_LOSS_GAMMA
        loss_weight             = cfg.ADV.LOSS_WEIGHT
        # fmt:on

        self.loss_func = loss_func
        self.focal_loss_gamma = focal_loss_gamma
        self.loss_weight = loss_weight

        backbone = build_backbone(cfg)
        in_channels = backbone.out_channels

        self.backbone = backbone
        self.rpn = RPN(cfg, in_channels)
        self.box_head = BoxHead(cfg, in_channels)

        # self.netD = netD()
        # self.netD = D(cfg, in_channels)
        self.netD = Dis(cfg, in_channels)

    def forward_vgg16(self, x):
        adaptation_feats = []
        for i in range(14):
            x = self.backbone[i](x)
        # adaptation_feats.append(x)

        for i in range(14, 21):
            x = self.backbone[i](x)
        # adaptation_feats.append(x)

        for i in range(21, len(self.backbone)):
            x = self.backbone[i](x)
        adaptation_feats.append(x)

        return x, adaptation_feats

    def forward_resnet101(self, x):
        adaptation_feats = []
        x = self.backbone.stem(x)
        x = self.backbone.layer1(x)
        # adaptation_feats.append(x)
        x = self.backbone.layer2(x)
        # adaptation_feats.append(x)
        x = self.backbone.layer3(x)
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

        if self.training and t_images is not None:
            t_features, t_adaptation_feats = forward_func(t_images)

            t_proposals, _, t_rpn_logits = self.rpn(t_images, t_features, t_img_metas, targets=None)
            _, _, t_proposals, t_box_features, t_roi_features = self.box_head(t_features, t_proposals, t_img_metas, targets=None)

            device = features.device
            # s_domain_logits = self.netD(grad_reverse(features, 1.0))
            # t_domain_logits = self.netD(grad_reverse(t_features, 1.0))

            s_domain_logits = self.netD(grad_reverse(features, 1.0), grad_reverse(s_rpn_logits, 1.0))
            t_domain_logits = self.netD(grad_reverse(t_features, 1.0), grad_reverse(t_rpn_logits, 1.0))

            # func_name = 'cross_entropy'
            func_name = self.loss_func
            if func_name == 'focal_loss':
                loss_func = partial(softmax_focal_loss, gamma=self.focal_loss_gamma)
            elif func_name == 'cross_entropy':
                loss_func = F.cross_entropy
            else:
                raise ValueError

            w = 0.5
            s_domain_loss = loss_func(s_domain_logits, torch.zeros(s_domain_logits.size(0), dtype=torch.long, device=device)) * w
            t_domain_loss = loss_func(t_domain_logits, torch.ones(t_domain_logits.size(0), dtype=torch.long, device=device)) * w

            # w = 1e-2
            # s_domain_loss = sum(F.cross_entropy(l, torch.zeros(l.size(0), dtype=torch.long, device=device)) * w for l in s_domain_logits)
            # t_domain_loss = sum(F.cross_entropy(l, torch.ones(l.size(0), dtype=torch.long, device=device)) * w for l in t_domain_logits)

            loss_dict.update({
                's_domain_loss': s_domain_loss * self.loss_weight,
                't_domain_loss': t_domain_loss * self.loss_weight,
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
