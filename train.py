import argparse
import datetime
import os
import time

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from detection.utils import dist_utils
from detection.config import cfg
from detection.data.build import build_data_loaders
from detection.engine.eval import evaluation
from detection.modeling.build import build_detectors
from detection import utils

global_step = 0


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10, writer=None):
    global global_step
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 500
        warmup_iters = min(500, len(data_loader) - 1)
        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, img_metas, targets in metric_logger.log_every(data_loader, print_freq, header):
        global_step += 1
        images = images.to(device)
        targets = [t.to(device) for t in targets]

        loss_dict = model(images, img_metas, targets)
        losses = sum(list(loss_dict.values()))

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if global_step % print_freq == 0:
            if writer:
                for k, v in loss_dict_reduced.items():
                    writer.add_scalar('losses/{}'.format(k), v, global_step=global_step)
                writer.add_scalar('losses/total_loss', losses_reduced, global_step=global_step)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)


def main(cfg, args):
    train_loader = build_data_loaders(cfg.DATASETS.TRAINS, transforms=cfg.INPUT.TRANSFORMS_TRAIN, is_train=True, distributed=args.distributed,
                                      batch_size=cfg.SOLVER.BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS)
    test_loaders = build_data_loaders(cfg.DATASETS.TESTS, transforms=cfg.INPUT.TRANSFORMS_TEST, is_train=False,
                                      distributed=args.distributed, num_workers=cfg.DATALOADER.NUM_WORKERS)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_detectors(cfg)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], cfg.SOLVER.LR, betas=(0.9, 0.999), weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.SOLVER.STEPS, gamma=cfg.SOLVER.GAMMA)

    if args.resume:
        print('Loading from {} ...'.format(args.resume))
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])

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

    print("Start training")
    start_time = time.time()
    epochs = cfg.SOLVER.EPOCHS
    for epoch in range(epochs):
        if args.distributed:
            train_loader.batch_sampler.sampler.set_epoch(epoch)

        epoch_start = time.time()
        train_one_epoch(model, optimizer, train_loader, device, epoch, writer=losses_writer)
        scheduler.step()

        state_dict = {
            'model': model_without_ddp.state_dict(),
            'args': args
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
