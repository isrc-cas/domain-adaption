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

from detection.utils import dist_utils
from detection.config import cfg
from detection.data.build import build_data_loaders
from detection.engine.eval import evaluation
from detection.modeling.build import build_detectors
from detection import utils

global_step = 0
total_steps = 0


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
    center = torch.stack((x, y), dim=1)
    return center


def detach_features(features):
    if isinstance(features, torch.Tensor):
        return features.detach()
    return tuple([f.detach() for f in features])


def train_one_epoch(model, optimizer, train_loader, target_loader, device, epoch, dis_model, dis_optimizer, print_freq=10, writer=None):
    global global_step
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.0e}'))
    metric_logger.add_meter('lr_dis', utils.SmoothedValue(window_size=1, fmt='{value:.0e}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_schedulers = []
    if epoch == 0:
        warmup_factor = 1. / 500
        warmup_iters = min(500, len(train_loader) - 1)
        lr_schedulers = [
            warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor),
            warmup_lr_scheduler(dis_optimizer, warmup_iters, warmup_factor),
        ]

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

        s_features = outputs['s_features']
        t_features = outputs['t_features']
        s_rpn_logits = outputs['s_rpn_logits']
        t_rpn_logits = outputs['t_rpn_logits']
        s_box_features = outputs['s_box_features']
        t_box_features = outputs['t_box_features']
        s_roi_features = outputs['s_roi_features']
        t_roi_features = outputs['t_roi_features']
        s_proposals = outputs['s_proposals']
        t_proposals = outputs['t_proposals']

        # -------------------------------------------------------------------
        # -----------------------------1.Train D-----------------------------
        # -------------------------------------------------------------------

        s_dis_loss = dis_model(detach_features(s_features), source_label, s_rpn_logits.detach(), s_box_features.detach(), s_roi_features.detach(), s_proposals)
        t_dis_loss = dis_model(detach_features(t_features), target_label, t_rpn_logits.detach(), t_box_features.detach(), t_roi_features.detach(), t_proposals)

        dis_loss = s_dis_loss + t_dis_loss
        loss_dict_for_log['s_dis_loss'] = s_dis_loss
        loss_dict_for_log['t_dis_loss'] = t_dis_loss

        dis_optimizer.zero_grad()
        dis_loss.backward()
        dis_optimizer.step()

        # -------------------------------------------------------------------
        # -----------------------------2.Train G-----------------------------
        # -------------------------------------------------------------------

        adv_loss = dis_model(t_features, source_label, t_rpn_logits, t_box_features, t_roi_features, t_proposals)
        loss_dict_for_log['adv_loss'] = adv_loss
        gamma = 1e-2

        det_loss = sum(list(loss_dict.values()))
        losses = det_loss + adv_loss * gamma

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict_for_log)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        for lr_scheduler in lr_schedulers:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(lr_dis=dis_optimizer.param_groups[0]["lr"])
        metric_logger.update(gamma=gamma)

        if global_step % print_freq == 0:
            if writer:
                for k, v in loss_dict_reduced.items():
                    writer.add_scalar('losses/{}'.format(k), v, global_step=global_step)
                writer.add_scalar('losses/total_loss', losses_reduced, global_step=global_step)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)
                writer.add_scalar('lr_dis', dis_optimizer.param_groups[0]['lr'], global_step=global_step)


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

        pairwise_func       = cfg.ADV.PAIRWISE_FUNC
        use_scale           = cfg.ADV.USE_SCALE
        center_styles_mode  = cfg.ADV.CENTER_STYLES_MODE
        in_channels         = cfg.ADV.IN_CHANNELS
        normalize_residual  = cfg.ADV.NORMALIZE_RESIDUAL
        normalize_before_mm = cfg.ADV.NORMALIZE_BEFORE_MM

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

        self.num_windows = num_windows
        self.local_window_extractor = LocalWindowExtractor(start_size, stop_size, num_windows)
        self.model_list = nn.ModuleList()
        # self.weight_list = nn.ModuleList()
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

    def forward(self, feature, label, rpn_logits, box_features, roi_features, proposals):
        # rpn_semantic_map = torch.mean(rpn_logits, dim=1, keepdim=True)
        # print(roi_features.shape) # [128, 512, 7, 7]
        window_features = self.local_window_extractor(feature)

        logits = []
        center_styles_mode = self.center_styles_mode
        inter_channels = self.inter_channels

        if center_styles_mode == 'avg':
            num_centers = self.num_windows
            center_styles = [torch.mean(x, dim=(2, 3), keepdim=True) for x in window_features]  # [(n, c, 1, 1)]
            center_styles = torch.stack(center_styles, dim=0)  # (num_windows, n, c, 1, 1)
        elif center_styles_mode == 'roi':
            # TODO: batch_size maybe not 1
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
        theta_x = self.theta(center_styles.view(num_centers * n, c, 1, 1)).view(num_centers, n, inter_channels)
        theta_x = theta_x.permute(1, 0, 2)
        for i, x in enumerate(window_features):  # (n, c, h, w)
            n, c, h, w = x.shape
            # theta = self.theta_list[i]
            phi = self.phi_list[i]

            # (n, num_windows, c)
            # theta_x = theta(center_styles.view(num_centers * n, c, 1, 1)).view(num_centers, n, inter_channels)
            # theta_x = theta_x.permute(1, 0, 2)

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

        losses = sum(F.binary_cross_entropy_with_logits(l, torch.full_like(l, label)) for l in logits)
        return losses


class Discriminator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.ModuleList([
            DisModelPerLevel(cfg),  # (1, 512, 32, 64) [(3, 7, 13, 21, 32)]
        ])

    def forward(self, features, label, rpn_logits, box_features, roi_features, proposals):
        losses = []
        for i, (layer, feature) in enumerate(zip(self.layers, features)):
            loss = layer(feature, label, rpn_logits, box_features, roi_features, proposals)
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

    dis_model = Discriminator(cfg)
    dis_model.to(device)

    model_without_ddp = model
    dis_model_without_ddp = dis_model
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.gpu])
        dis_model = DistributedDataParallel(dis_model, device_ids=[args.gpu])
        model_without_ddp = model.module
        dis_model_without_ddp = dis_model.module

    # optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], cfg.SOLVER.LR, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], cfg.SOLVER.LR, betas=(0.9, 0.999), weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    dis_optimizer = torch.optim.Adam([p for p in dis_model.parameters() if p.requires_grad], cfg.SOLVER.LR, betas=(0.9, 0.999), weight_decay=cfg.SOLVER.WEIGHT_DECAY)

    schedulers = [
        torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.SOLVER.STEPS, gamma=cfg.SOLVER.GAMMA),
        torch.optim.lr_scheduler.MultiStepLR(dis_optimizer, cfg.SOLVER.STEPS, gamma=cfg.SOLVER.GAMMA),
    ]

    current_epoch = -1
    if args.resume:
        print('Loading from {} ...'.format(args.resume))
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if 'current_epoch' in checkpoint:
            current_epoch = int(checkpoint['current_epoch'])
        if 'discriminator' in checkpoint:
            dis_model_without_ddp.load_state_dict(checkpoint['discriminator'])

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

        epoch_start = time.time()
        train_one_epoch(model, optimizer, train_loader, target_loader, device, epoch,
                        dis_model=dis_model, dis_optimizer=dis_optimizer,
                        writer=losses_writer)

        for scheduler in schedulers:
            scheduler.step()

        state_dict = {
            'model': model_without_ddp.state_dict(),
            'discriminator': dis_model_without_ddp.state_dict(),
            'current_epoch': epoch,
        }
        save_path = os.path.join(work_dir, 'model_epoch_{:02d}.pth'.format(epoch))
        dist_utils.save_on_master(state_dict, save_path)
        print('Saved to {}.'.format(save_path))

        metrics = evaluation(model, test_loaders, device, cfg.TEST.EVAL_TYPES, output_dir=work_dir, iteration=epoch)
        if dist_utils.is_main_process() and losses_writer:
            for dataset_name, metric in metrics.items():
                for k, v in metric.items():
                    metrics_writers[dataset_name].add_scalar('metrics/' + k, v, global_step=global_step)

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
