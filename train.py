import argparse
import datetime
import os
import time

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

import utils
import utils.dist as dist_util
from data.build import build_data_loaders
from engine.eval import evaluation
from modeling.build import build_detectors

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

        loss_dict_reduced = dist_util.reduce_dict(loss_dict)
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


def main(args):
    dist_util.init_distributed_mode(args)
    print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Creating data loaders...")
    train_loader = build_data_loaders(args.trains, is_train=True, distributed=args.distributed, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loaders = build_data_loaders(args.tests, is_train=False, distributed=args.distributed, num_workers=args.num_workers)

    model = build_detectors('VGG16', num_classes=args.num_classes)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], args.lr, momentum=0.9, weight_decay=0.0005)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], args.lr, betas=(0.9, 0.999), weight_decay=0.0001)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_steps, gamma=0.1)

    if args.resume:
        print('Loading from {} ...'.format(args.resume))
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # scheduler.load_state_dict(checkpoint['lr_scheduler'])

    if args.test_only:
        evaluation(model, test_loaders, device, types=args.eval_types, output_dir=args.work_dir)
        return

    losses_writer = None
    if dist_util.is_main_process():
        losses_writer = SummaryWriter(os.path.join(args.work_dir, 'losses'))
    metrics_writers = {}
    if dist_util.is_main_process():
        test_dataset_names = [loader.dataset.dataset_name for loader in test_loaders]
        for dataset_name in test_dataset_names:
            metrics_writers[dataset_name] = SummaryWriter(os.path.join(args.work_dir, 'metrics', dataset_name))

    print("Start training")
    start_time = time.time()
    for epoch in range(args.epochs):
        if args.distributed:
            train_loader.batch_sampler.sampler.set_epoch(epoch)
        epoch_start = time.time()
        train_one_epoch(model, optimizer, train_loader, device, epoch, writer=losses_writer)
        scheduler.step()
        state_dict = {
            'model': model_without_ddp.state_dict(),
            # 'optimizer': optimizer.state_dict(),
            # 'lr_scheduler': scheduler.state_dict(),
            'args': args
        }
        save_path = os.path.join(args.work_dir, 'model_epoch_{:02d}.pth'.format(epoch))
        dist_util.save_on_master(state_dict, save_path)
        print('Saved to {}.'.format(save_path))

        metrics = evaluation(model, test_loaders, device, args.eval_types, output_dir=args.work_dir, iteration=epoch)
        if dist_util.is_main_process() and losses_writer:
            for dataset_name, metric in metrics.items():
                for k, v in metric.items():
                    metrics_writers[dataset_name].add_scalar('metrics/' + k, v, global_step=global_step)

        epoch_cost = time.time() - epoch_start
        left = args.epochs - epoch - 1
        print('Epoch {} ended, cost {}. Left {} epochs, may cost {}'.format(epoch,
                                                                            str(datetime.timedelta(seconds=int(epoch_cost))),
                                                                            left,
                                                                            str(datetime.timedelta(seconds=int(left * epoch_cost)))))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument("--work-dir", default="./works_dir/", type=str, help="path to work dir")
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--epochs', default=25, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr-steps', default=[16, 22], nargs='+', type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--num-classes', default=9, type=int, help='Number of classes(plus background)')
    parser.add_argument('--trains', default=['cityscapes_train'], nargs='+', type=str, help='Train datasets')
    parser.add_argument('--tests', default=['cityscapes_val', 'foggy_cityscapes_val'], nargs='+', type=str, help='Test datasets')
    parser.add_argument('--eval-types', default=['coco'], nargs='+', type=str, help='Evaluation types, like coco, voc...')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--num-workers', default=8, type=int, help='Number workers')
    parser.add_argument('--batch-size', default=2, type=int, help='batch size')
    parser.add_argument("--test-only", help="Only test the model", action="store_true")
    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    args = parser.parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    main(args)
