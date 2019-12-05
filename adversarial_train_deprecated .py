import argparse
import datetime
import math
import os
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from detection.layers import sigmoid_focal_loss
from detection.utils import dist_utils
from detection.config import cfg
from detection.data.build import build_data_loaders
from detection.engine.eval import evaluation
from detection.modeling.build import build_detectors
from detection import utils

global_step = 0
total_steps = 0
best_mAP = -1.0


def cosine_scheduler(eta_max, eta_min, current_step):
    y = eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(current_step / total_steps * math.pi))
    return y


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def box_to_centers(boxes):
    x = boxes[:, 2] - boxes[:, 0]
    y = boxes[:, 3] - boxes[:, 1]
    centers = torch.stack((x, y), dim=1)
    return centers


def detach_features(features):
    if isinstance(features, torch.Tensor):
        return features.detach()
    return tuple([f.detach() for f in features])


def convert_sync_batchnorm(model):
    convert = False
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            convert = True
            break
    if convert:
        print('Convert to SyncBatchNorm')
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    return model


def train_one_epoch(model, optimizer, train_loader, target_loader, device, epoch, dis_model, dis_optimizer, print_freq=10, writer=None, test_func=None, save_func=None):
    global global_step
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.0e}'))
    # metric_logger.add_meter('lr_dis', utils.SmoothedValue(window_size=1, fmt='{value:.0e}'))
    # metric_logger.add_meter('gamma', utils.SmoothedValue(window_size=1, fmt='{value:.0e}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_schedulers = []
    if epoch == 0:
        warmup_factor = 1. / 500
        warmup_iters = min(500, len(train_loader) - 1)
        # lr_schedulers = [
        #     warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor),
        #     warmup_lr_scheduler(dis_optimizer, warmup_iters, warmup_factor),
        # ]

    source_label = 1
    target_label = 0
    target_loader_iter = iter(target_loader)
    for images, img_metas, targets in metric_logger.log_every(train_loader, print_freq, header):
        global_step += 1
        images = images.to(device)
        targets = [t.to(device) for t in targets]

        try:
            t_images, t_img_metas, _ = next(target_loader_iter)
        except StopIteration:
            target_loader_iter = iter(target_loader)
            t_images, t_img_metas, _ = next(target_loader_iter)

        t_images = t_images.to(device)

        loss_dict, outputs = model(images, img_metas, targets, t_images, t_img_metas)
        loss_dict_for_log = dict(loss_dict)

        # s_features = outputs['s_features']
        # t_features = outputs['t_features']
        # s_rpn_logits = outputs['s_rpn_logits']
        # t_rpn_logits = outputs['t_rpn_logits']
        # s_box_features = outputs['s_box_features']
        # t_box_features = outputs['t_box_features']
        # s_roi_features = outputs['s_roi_features']
        # t_roi_features = outputs['t_roi_features']
        # s_proposals = outputs['s_proposals']
        # t_proposals = outputs['t_proposals']

        # -------------------------------------------------------------------
        # -----------------------------1.Train D-----------------------------
        # -------------------------------------------------------------------

        # s_dis_loss = dis_model(detach_features(s_features), source_label, s_rpn_logits.detach(), s_box_features.detach(), s_roi_features.detach(), s_proposals)
        # t_dis_loss = dis_model(detach_features(t_features), target_label, t_rpn_logits.detach(), t_box_features.detach(), t_roi_features.detach(), t_proposals)
        #
        # dis_loss = s_dis_loss + t_dis_loss
        # loss_dict_for_log['s_dis_loss'] = s_dis_loss
        # loss_dict_for_log['t_dis_loss'] = t_dis_loss
        #
        # dis_optimizer.zero_grad()
        # dis_loss.backward()
        # dis_optimizer.step()

        # -------------------------------------------------------------------
        # -----------------------------2.Train G-----------------------------
        # -------------------------------------------------------------------

        # adv_loss = dis_model(t_features, source_label, t_rpn_logits, t_box_features, t_roi_features, t_proposals, adversarial=True)
        # loss_dict_for_log['adv_loss'] = adv_loss
        #
        # gamma = cosine_scheduler(cfg.ADV.GAMMA_FROM, cfg.ADV.GAMMA_TO, current_step=global_step)
        # det_loss = sum(list(loss_dict.values()))
        # losses = det_loss + adv_loss * gamma

        losses = sum(list(loss_dict.values()))
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict_for_log)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        for lr_scheduler in lr_schedulers:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # metric_logger.update(lr_dis=dis_optimizer.param_groups[0]["lr"])
        # metric_logger.update(gamma=gamma)

        if global_step % print_freq == 0:
            if writer:
                for k, v in loss_dict_reduced.items():
                    writer.add_scalar('losses/{}'.format(k), v, global_step=global_step)
                writer.add_scalar('losses/total_loss', losses_reduced, global_step=global_step)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)
                # writer.add_scalar('lr_dis', dis_optimizer.param_groups[0]['lr'], global_step=global_step)
                # writer.add_scalar('gamma', gamma, global_step=global_step)

        if global_step % (1000 // max(1, (dist_utils.get_world_size() // 2))) == 0 and test_func is not None:
            updated = test_func()
            if updated:
                save_func('best.pth', 'mAP: {:.4f}'.format(best_mAP))
            print('Best mAP: {:.4f}'.format(best_mAP))


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
        self.theta_list = nn.ModuleList()
        self.phi_list = nn.ModuleList()

        self.inter_channels = in_channels
        for _ in range(num_windows):
            self.model_list += [
                nn.Sequential(
                    # nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
                    # nn.LeakyReLU(0.2, inplace=True),
                    # nn.Conv2d(in_channels, 256, kernel_size=1),
                    # nn.LeakyReLU(0.2, inplace=True),
                    # nn.Conv2d(256, 128, kernel_size=1),
                    # nn.LeakyReLU(0.2, inplace=True),
                    # nn.Conv2d(128, 1, kernel_size=1),
                    # Linear
                    nn.Linear(in_channels * 2, in_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(in_channels, 256),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(256, 128),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(128, 1),
                )
            ]

            # self.theta_list += [
            #     nn.Conv2d(in_channels * 2, self.inter_channels, 1)
            # ]
            self.theta = nn.Conv2d(in_channels * 2, self.inter_channels, 1)
            self.phi_list += [
                nn.Conv2d(in_channels * 2, self.inter_channels, 1)
            ]

            # self.weight_list += [
            #     nn.Sequential(
            #         nn.Conv2d(in_channels * 2 + 1, 128, 1),
            #         nn.ReLU(inplace=True),
            #         nn.Conv2d(128, 1, 1),
            #         nn.Sigmoid(),
            #     )
            # ]

        if self.weighted:
            self.weight_model = nn.Sequential(
                nn.Linear(in_channels * 2 * num_windows, in_channels * 2),
                nn.ReLU(inplace=True),
                nn.Linear(in_channels * 2, in_channels * 2),
                nn.ReLU(inplace=True),
                nn.Linear(in_channels * 2, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, num_windows),
            )

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

    def forward(self, feature, label, rpn_logits, box_features, roi_features, proposals, adversarial=False):
        # rpn_semantic_map = torch.mean(rpn_logits, dim=1, keepdim=True)
        # print(roi_features.shape) # [128, 512, 7, 7]
        window_features = self.local_window_extractor(feature)

        logits = []
        center_styles_mode = self.center_styles_mode
        inter_channels = self.inter_channels

        if center_styles_mode == 'avg':
            num_centers = self.num_windows
            center_styles_list = [torch.mean(x, dim=(2, 3), keepdim=True) for x in window_features]  # [(n, c, 1, 1)]
            center_styles = torch.stack(center_styles_list, dim=0)  # (num_windows, n, c, 1, 1)
        elif center_styles_mode == 'roi':
            # TODO: batch_size maybe not 1
            raise NotImplemented
            assert len(proposals) == 1
            roi_features = roi_features[torch.randperm(roi_features.shape[0])[:16]]
            num_centers = roi_features.shape[0]
            var, mean = torch.var_mean(roi_features, dim=(2, 3), keepdim=True, unbiased=False)
            std = torch.sqrt(var + 1e-12)
            center_styles = torch.cat((mean, std), dim=1)  # (num_roi, c, 1, 1)
            center_styles = center_styles.unsqueeze(1)  # (num_roi, n, c, 1, 1)
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

        dynamic_weights = [1] * self.num_windows
        if self.weighted:
            center_styles_list = torch.cat(center_styles_list, dim=1).view(n, c * self.num_windows)
            dynamic_weights = self.weight_model(center_styles_list)
            dynamic_weights = (F.softmax(dynamic_weights, dim=1) * self.num_windows)[0]  # (num_windows, )

        window_weights = self.window_weights if adversarial else [1] * self.num_windows
        if self.use_focal_loss and adversarial:
            losses = sum(dynamic_weights[i] * sigmoid_focal_loss(l, torch.full_like(l, label), gamma=gamma) * w for i, (l, gamma, w) in
                         enumerate(zip(logits, self.focal_loss_gammas, window_weights)))
        else:
            losses = sum(dynamic_weights[i] * F.binary_cross_entropy_with_logits(l, torch.full_like(l, label)) * w for i, (l, w) in
                         enumerate(zip(logits, window_weights)))

        return losses


class MetaConv(nn.Module):
    def __init__(self, in_channels, out_channels=(32, 1), context_dim=4096):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        r = 16

        self.meta1 = nn.Linear(context_dim, out_channels[0] * in_channels * 1 * 1 + out_channels[0])
        self.meta2 = nn.Linear(context_dim, out_channels[1] * out_channels[0] * 1 * 1 + out_channels[1])

    def forward(self, x, context):
        """
        Args:
            x:
            context: (1, 512)
        Returns:
        """

        # meta1
        y = self.meta1(context)[0]
        num_weight = self.out_channels[0] * self.in_channels * 1 * 1
        weight = y[:num_weight]
        bias = y[num_weight:]
        weight = weight.view(self.out_channels[0], self.in_channels, 1, 1)
        x = F.conv2d(x, weight=weight, bias=bias)
        x = F.relu_(x)

        # meta1
        y = self.meta2(context)[0]
        num_weight = self.out_channels[1] * self.out_channels[0] * 1 * 1
        weight = y[:num_weight]
        bias = y[num_weight:]
        weight = weight.view(self.out_channels[1], self.out_channels[0], 1, 1)
        x = F.conv2d(x, weight=weight, bias=bias)

        return x


class SelectivePerLevel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # fmt:off
        anchor_scales       = cfg.MODEL.RPN.ANCHOR_SIZES
        anchor_ratios       = cfg.MODEL.RPN.ASPECT_RATIOS
        num_anchors         = len(anchor_scales) * len(anchor_ratios)

        in_channels         = cfg.ADV.IN_CHANNELS

        start_size          = cfg.ADV.WINDOWS.START_SIZE
        stop_size           = cfg.ADV.WINDOWS.STOP_SIZE
        num_windows         = cfg.ADV.WINDOWS.NUM_WINDOWS
        method              = cfg.ADV.SELECTIVE.METHOD
        # fmt:on

        self.in_channels = in_channels
        self.num_windows = num_windows
        self.local_window_extractor = LocalWindowExtractor(start_size, stop_size, num_windows)
        self.method = method

        if method == 'weighted':
            r = 16
            in_channels *= 2
            self.fc = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // r, kernel_size=1, bias=False),
                nn.ReLU(True),
            )
            self.fc_list = nn.ModuleList()
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels // 2, 256, kernel_size=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(256, 128, kernel_size=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 1, kernel_size=1),
                # # Linear
                # nn.Linear(in_channels * 2, in_channels),
                # nn.LeakyReLU(0.2, inplace=True),
                # nn.Linear(in_channels, 256),
                # nn.LeakyReLU(0.2, inplace=True),
                # nn.Linear(256, 128),
                # nn.LeakyReLU(0.2, inplace=True),
                # nn.Linear(128, 1),
            )
            for i in range(num_windows):
                self.fc_list += [
                    nn.Conv2d(in_channels // r, in_channels, kernel_size=1, bias=False)
                ]
        elif self.method == 'avg':
            self.model_list = nn.ModuleList()
            in_channels *= 2
            for i in range(num_windows):
                self.model_list += [
                    nn.Sequential(
                        nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Conv2d(in_channels // 2, 256, kernel_size=1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Conv2d(256, 128, kernel_size=1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Conv2d(128, 1, kernel_size=1),
                        # # Linear
                        # nn.Linear(in_channels * 2, in_channels),
                        # nn.LeakyReLU(0.2, inplace=True),
                        # nn.Linear(in_channels, 256),
                        # nn.LeakyReLU(0.2, inplace=True),
                        # nn.Linear(256, 128),
                        # nn.LeakyReLU(0.2, inplace=True),
                        # nn.Linear(128, 1),
                    )
                ]
        elif self.method == 'meta':
            self.embedding = nn.ModuleList()
            self.metas = nn.ModuleList()
            self.model_list = nn.ModuleList()

            in_channels *= 2
            for i in range(num_windows):
                self.embedding += [
                    nn.Sequential(
                        nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(in_channels // 2, 256, kernel_size=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(256, in_channels, kernel_size=1),
                    )
                ]
                self.metas += [
                    MetaConv(in_channels=self.in_channels, context_dim=self.in_channels, out_channels=(64, 32))
                ]

                self.model_list += [
                    nn.Sequential(
                        nn.Conv2d(32, 256, kernel_size=1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Conv2d(256, 256, kernel_size=1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Conv2d(256, 128, kernel_size=1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Conv2d(128, 1, kernel_size=1),
                        # # Linear
                        # nn.Linear(in_channels * 2, in_channels),
                        # nn.LeakyReLU(0.2, inplace=True),
                        # nn.Linear(in_channels, 256),
                        # nn.LeakyReLU(0.2, inplace=True),
                        # nn.Linear(256, 128),
                        # nn.LeakyReLU(0.2, inplace=True),
                        # nn.Linear(128, 1),
                    )
                ]
        elif self.method == 'global':
            # self.bn = nn.BatchNorm2d(in_channels)
            self.bn = nn.Identity()
            self.embedding_list = nn.ModuleList()
            self.g_list = nn.ModuleList()
            self.theta_list = nn.ModuleList()
            self.phi_list = nn.ModuleList()
            self.out_list = nn.ModuleList()
            self.model_list = nn.ModuleList()

            self.inter_channels = in_channels // 2

            # bn = nn.BatchNorm2d
            bn = nn.Identity

            for i in range(num_windows):
                self.embedding_list += [
                    nn.Sequential(
                        nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
                        bn(in_channels),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(in_channels, in_channels, kernel_size=1),
                    )
                ]
                self.g_list += [
                    nn.Conv2d(in_channels, self.inter_channels, 1)
                ]
                self.theta_list += [
                    nn.Conv2d(in_channels, self.inter_channels, 1)
                ]
                self.phi_list += [
                    nn.Conv2d(in_channels, self.inter_channels, 1)
                ]
                self.out_list += [
                    nn.Sequential(
                        nn.Conv2d(self.inter_channels, in_channels, 1, bias=False),
                        bn(in_channels),
                    )
                ]
                self.model_list += [
                    nn.Sequential(
                        nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
                        bn(in_channels),
                        nn.LeakyReLU(0.2, inplace=True),

                        nn.Conv2d(in_channels, 256, kernel_size=1, bias=False),
                        bn(256),
                        nn.LeakyReLU(0.2, inplace=True),

                        nn.Conv2d(256, 128, kernel_size=1, bias=False),
                        bn(128),
                        nn.LeakyReLU(0.2, inplace=True),

                        nn.Conv2d(128, 1, kernel_size=1),
                    )
                ]
        else:
            raise ValueError

    def forward(self, feature, label, rpn_logits, box_features, roi_features, proposals, adversarial=False):
        if self.method == 'weighted':
            window_features = self.local_window_extractor(feature)
            pyramid_avg_styles = [torch.mean(x, dim=(2, 3), keepdim=True) for x in window_features]  # [(n, c, 1, 1)]

            sum_avg_styles = sum(pyramid_avg_styles)  # (n, c, 1, 1)
            z = self.fc(sum_avg_styles)  # (n, c//r, 1, 1)

            weight_list = [self.fc_list[i](z) for i in range(self.num_windows)]  # [(n, c, 1, 1)]
            weight_list = torch.stack(weight_list, dim=0)  # (num_windows, n, c, 1, 1)
            weight_list = F.softmax(weight_list, dim=0)  # (num_windows, n, c, 1, 1)
            weight_list = torch.unbind(weight_list, dim=0)  # [(n, c, 1, 1)]
            pyramid_avg_styles = [w * style for w, style in zip(weight_list, pyramid_avg_styles)]  # [(n, c, 1, 1)]
            pyramid_avg_styles = sum(pyramid_avg_styles)  # (n, c, 1, 1)
            logits = self.model(pyramid_avg_styles)
            losses = F.binary_cross_entropy_with_logits(logits, torch.full_like(logits, label))
        elif self.method == 'avg':
            window_features = self.local_window_extractor(feature)
            pyramid_avg_styles = [torch.mean(x, dim=(2, 3), keepdim=True) for x in window_features]  # [(n, c, 1, 1)]

            logits = []
            for i, x in enumerate(pyramid_avg_styles):
                x = self.model_list[i](x)
                logits.append(x)
            losses = sum(F.binary_cross_entropy_with_logits(l, torch.full_like(l, label)) for l in logits)
        elif self.method == 'meta':
            window_features = self.local_window_extractor(feature)
            pyramid_avg_styles = [torch.mean(x, dim=(2, 3), keepdim=True) for x in window_features]  # [(n, c, 1, 1)]

            logits = []
            for i, pyramid_avg_style in enumerate(pyramid_avg_styles):
                n, c, _, _ = pyramid_avg_style.shape
                embedding_style = self.embedding[i](pyramid_avg_style).view(n, c)
                mean = embedding_style[:, :self.in_channels]
                var = embedding_style[:, self.in_channels:]
                context = mean + torch.exp(var / 2)
                meta_feature = self.metas[i](feature, context)
                x = self.model_list[i](meta_feature)
                logits.append(x)
            losses = sum(F.binary_cross_entropy_with_logits(l, torch.full_like(l, label)) for l in logits)
        elif self.method == 'global':
            feature = self.bn(feature)
            window_features = self.local_window_extractor(feature)
            logits = []
            for i in range(self.num_windows):
                x = window_features[i]
                x = self.embedding_list[i](x)
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

                logit = self.model_list[i](x)
                logits.append(logit)

            # losses = sum(F.binary_cross_entropy_with_logits(logit, torch.full_like(logit, label)) for logit in logits)
            losses = sum(sigmoid_focal_loss(logit, torch.full_like(logit, label), gamma=4) for logit in logits)
        else:
            raise ValueError
        return losses


DISCRIMINATORS = {
    'netvlad': DisModelPerLevel,
    'selective': SelectivePerLevel,
}


class Discriminator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        method = cfg.ADV.METHOD
        model = DISCRIMINATORS[method]

        self.layers = nn.ModuleList([
            model(cfg),  # (1, 512, 32, 64) [(3, 7, 13, 21, 32)]
        ])

    def forward(self, features, label, rpn_logits, box_features, roi_features, proposals, adversarial=False):
        losses = []
        for i, (layer, feature) in enumerate(zip(self.layers, features)):
            loss = layer(feature, label, rpn_logits, box_features, roi_features, proposals, adversarial)
            losses.append(loss)
        losses = sum(losses)
        return losses


def main(cfg, args):
    train_loader = build_data_loaders(cfg.DATASETS.TRAINS, transforms=cfg.INPUT.TRANSFORMS_TRAIN, is_train=True, distributed=args.distributed,
                                      batch_size=cfg.SOLVER.BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS)
    target_loader = build_data_loaders(cfg.DATASETS.TARGETS, transforms=cfg.INPUT.TRANSFORMS_TRAIN, is_train=True, distributed=args.distributed,
                                       batch_size=cfg.SOLVER.BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS)
    test_loaders = build_data_loaders(cfg.DATASETS.TESTS, transforms=cfg.INPUT.TRANSFORMS_TEST, is_train=False,
                                      distributed=args.distributed, num_workers=cfg.DATALOADER.NUM_WORKERS)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_detectors(cfg)
    model.to(device)

    # dis_model = Discriminator(cfg)
    # dis_model.to(device)

    model_without_ddp = model
    # dis_model_without_ddp = dis_model
    if args.distributed:
        model = DistributedDataParallel(convert_sync_batchnorm(model), device_ids=[args.gpu])

        # dis_model = DistributedDataParallel(dis_model, device_ids=[args.gpu])
        model_without_ddp = model.module
        # dis_model_without_ddp = dis_model.module

    # optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], cfg.SOLVER.LR, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], cfg.SOLVER.LR, betas=(0.9, 0.999), weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    # dis_optimizer = torch.optim.Adam([p for p in dis_model.parameters() if p.requires_grad], cfg.SOLVER.LR, betas=(0.9, 0.999), weight_decay=cfg.SOLVER.WEIGHT_DECAY)

    schedulers = [
        torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.SOLVER.STEPS, gamma=cfg.SOLVER.GAMMA),
        # torch.optim.lr_scheduler.MultiStepLR(dis_optimizer, cfg.SOLVER.STEPS, gamma=cfg.SOLVER.GAMMA),
    ]

    current_epoch = -1
    if args.resume:
        print('Loading from {} ...'.format(args.resume))
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        if 'current_epoch' in checkpoint:
            current_epoch = int(checkpoint['current_epoch'])
        # if 'discriminator' in checkpoint:
        #     dis_model_without_ddp.load_state_dict(checkpoint['discriminator'])

    work_dir = cfg.WORK_DIR
    if args.test_only:
        evaluation(model, test_loaders, device, types=cfg.TEST.EVAL_TYPES, output_dir=work_dir)
        return

    losses_writer = None
    if dist_utils.is_main_process():
        losses_writer = SummaryWriter(os.path.join(work_dir, 'losses'))
        losses_writer.add_text('config', '{}'.format(str(cfg).replace('\n', '  \n')))
        losses_writer.add_text('args', str(args))

    metrics_writers = {}
    if dist_utils.is_main_process():
        test_dataset_names = [loader.dataset.dataset_name for loader in test_loaders]
        for dataset_name in test_dataset_names:
            metrics_writers[dataset_name] = SummaryWriter(os.path.join(work_dir, 'metrics', dataset_name))

    start_time = time.time()
    epochs = cfg.SOLVER.EPOCHS
    global total_steps
    start_epoch = current_epoch + 1
    total_steps = (epochs - start_epoch) * len(train_loader)
    print("Start training, total epochs: {} ({} - {}), total steps: {}".format(epochs - start_epoch, start_epoch, epochs - 1, total_steps))
    for epoch in range(start_epoch, epochs):
        if args.distributed:
            train_loader.batch_sampler.sampler.set_epoch(epoch)
            target_loader.batch_sampler.sampler.set_epoch(epoch)

        def test_func():
            global best_mAP
            updated = False
            metrics = evaluation(model, test_loaders, device, cfg.TEST.EVAL_TYPES, output_dir=work_dir, iteration=global_step)
            if dist_utils.is_main_process() and losses_writer:
                for dataset_name, metric in metrics.items():
                    for k, v in metric.items():
                        metrics_writers[dataset_name].add_scalar('metrics/' + k, v, global_step=global_step)
                        if k == 'mAP' and v > best_mAP:
                            best_mAP = v
                            updated = True
            model.train()

            return updated

        def save_func(filename=None, save_str=None):
            state_dict = {
                'model': model_without_ddp.state_dict(),
                # 'discriminator': dis_model_without_ddp.state_dict(),
                'current_epoch': epoch,
            }
            filename = filename if filename else 'model_epoch_{:02d}.pth'.format(epoch)
            save_path = os.path.join(work_dir, filename)
            dist_utils.save_on_master(state_dict, save_path)
            if dist_utils.is_main_process() and save_str is not None:
                with open(os.path.join(work_dir, 'best.txt'), 'w') as f:
                    f.write(save_str)

            print('Saved to {}'.format(save_path))

        epoch_start = time.time()
        train_one_epoch(model, optimizer, train_loader, target_loader, device, epoch,
                        dis_model=None, dis_optimizer=None,
                        writer=losses_writer, test_func=test_func, save_func=save_func)

        for scheduler in schedulers:
            scheduler.step()

        save_func()

        if epoch == (epochs - 1):
            test_func()

        epoch_cost = time.time() - epoch_start
        left = epochs - epoch - 1
        print('Epoch {} ended, cost {}. Left {} epochs, may cost {}'.format(epoch,
                                                                            str(datetime.timedelta(seconds=int(epoch_cost))),
                                                                            left,
                                                                            str(datetime.timedelta(seconds=int(left * epoch_cost)))))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument("--config-file", help="path to config file", type=str)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument("--test-only", help="Only test the model", action="store_true")

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    dist_utils.init_distributed_mode(args)

    print(args)
    world_size = dist_utils.get_world_size()
    if world_size != 4:
        lr = cfg.SOLVER.LR * (float(world_size) / 4)
        print('Change lr from {} to {}'.format(cfg.SOLVER.LR, lr))
        cfg.merge_from_list(['SOLVER.LR', lr])

    print(cfg)
    os.makedirs(cfg.WORK_DIR, exist_ok=True)
    if dist_utils.is_main_process():
        with open(os.path.join(cfg.WORK_DIR, 'config.yaml'), 'w') as fid:
            fid.write(str(cfg))
    main(cfg, args)
