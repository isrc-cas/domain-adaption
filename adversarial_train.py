import argparse
import datetime
import math
import os
import time

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
    target_loader = iter(target_loader)
    for images, img_metas, targets in metric_logger.log_every(train_loader, print_freq, header):
        global_step += 1
        images = images.to(device)
        targets = [t.to(device) for t in targets]

        t_images, t_img_metas, _ = next(target_loader)
        t_images = t_images.to(device)

        loss_dict, outputs = model(images, img_metas, targets, t_images, t_img_metas)
        loss_dict_for_log = dict(loss_dict)

        s_windows = outputs['s_windows']
        t_windows = outputs['t_windows']
        s_rpn_logits = outputs['s_rpn_logits']
        t_rpn_logits = outputs['t_rpn_logits']
        s_box_features = outputs['s_box_features']
        t_box_features = outputs['t_box_features']

        # -------------------------------------------------------------------
        # -----------------------------1.Train D-----------------------------
        # -------------------------------------------------------------------

        s_dis_loss = dis_model(detach_features(s_windows), source_label, s_rpn_logits.detach(), s_box_features.detach())
        t_dis_loss = dis_model(detach_features(t_windows), target_label, t_rpn_logits.detach(), t_box_features.detach())

        dis_loss = s_dis_loss + t_dis_loss
        loss_dict_for_log['s_dis_loss'] = s_dis_loss
        loss_dict_for_log['t_dis_loss'] = t_dis_loss

        dis_optimizer.zero_grad()
        dis_loss.backward()
        dis_optimizer.step()

        # -------------------------------------------------------------------
        # -----------------------------2.Train G-----------------------------
        # -------------------------------------------------------------------

        adv_loss = dis_model(t_windows, source_label, t_rpn_logits, t_box_features)
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


class DisModelPerLevel(nn.Module):
    def __init__(self, cfg, in_channels=512, window_sizes=(3, 7, 13, 21, 32)):
        super().__init__()
        # fmt:off
        anchor_scales       = cfg.MODEL.RPN.ANCHOR_SIZES
        anchor_ratios       = cfg.MODEL.RPN.ASPECT_RATIOS
        num_anchors         = len(anchor_scales) * len(anchor_ratios)
        # fmt:on

        self.window_sizes = window_sizes
        self.model_list = nn.ModuleList()
        # self.weight_list = nn.ModuleList()

        for _ in range(len(self.window_sizes)):
            self.model_list += [
                nn.Sequential(
                    nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(in_channels, 256, kernel_size=1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(256, 128, kernel_size=1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(128, 1, kernel_size=1),
                )
            ]
            # self.weight_list += [
            #     nn.Sequential(
            #         nn.Conv2d(in_channels * 2 + 1, 128, 1),
            #         nn.ReLU(inplace=True),
            #         nn.Conv2d(128, 1, 1),
            #         nn.Sigmoid(),
            #     )
            # ]

    def forward(self, window_features, label, rpn_logits, box_features):
        # rpn_semantic_map = torch.mean(rpn_logits, dim=1, keepdim=True)
        logits = []
        for i, x in enumerate(window_features):
            _, _, window_h, window_w = x.shape

            avg_x = torch.mean(x, dim=(2, 3), keepdim=True)  # (N, C * 2, 1, 1)
            avg_x_expanded = avg_x.expand(-1, -1, window_h, window_w)  # (N, C * 2, h, w)

            # rpn_semantic_map_per_level = F.interpolate(rpn_semantic_map, size=(window_h, window_w), mode='bilinear', align_corners=True)
            # rpn_semantic_map_per_level = torch.cat((x, rpn_semantic_map_per_level), dim=1)
            # weight = self.weight_list[i](rpn_semantic_map_per_level)  # (N, 1, h, w)
            # residual = weight * (x - avg_x_expanded)
            residual = (x - avg_x_expanded)

            x = self.model_list[i](residual)
            logits.append(x)

        losses = sum(F.binary_cross_entropy_with_logits(l, torch.full_like(l, label)) for l in logits)
        return losses


class Discriminator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.ModuleList([
            DisModelPerLevel(cfg, in_channels=512, window_sizes=(3, 7, 13, 21, 32)),  # (1, 512, 32, 64)
        ])

    def forward(self, window_features, label, rpn_logits, box_features):
        if isinstance(window_features[0], torch.Tensor):
            window_features = (window_features,)
        losses = []
        for i, (layer, feature) in enumerate(zip(self.layers, window_features)):
            loss = layer(feature, label, rpn_logits, box_features)
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
    main(cfg, args)
