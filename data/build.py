import torch
from torch.utils.data import ConcatDataset, DataLoader

from . import collate_fn
from .cityscape import CityscapeDataset

cityscapes_images_dir = '/data7/lufficc/cityscapes/leftImg8bit'
foggy_cityscapes_images_dir = '/data7/lufficc/cityscapes/leftImg8bit_foggy'

DATASETS = {
    'cityscapes_train': {
        'ann_file': '/data7/lufficc/cityscapes/cityscapes_coco_train.json',
        'root': cityscapes_images_dir,
        'train': True,
    },

    'cityscapes_val': {
        'ann_file': '/data7/lufficc/cityscapes/cityscapes_coco_val.json',
        'root': cityscapes_images_dir,
        'train': False,
    },

    'cityscapes_test': {
        'ann_file': '/data7/lufficc/cityscapes/cityscapes_coco_test.json',
        'root': cityscapes_images_dir,
        'train': False,
    },

    'foggy_cityscapes_train': {
        'ann_file': '/data7/lufficc/cityscapes/foggy_cityscapes_coco_train.json',
        'root': foggy_cityscapes_images_dir,
        'train': True,
    },

    'foggy_cityscapes_val': {
        'ann_file': '/data7/lufficc/cityscapes/foggy_cityscapes_coco_val.json',
        'root': foggy_cityscapes_images_dir,
        'train': False,
    },

    'foggy_cityscapes_test': {
        'ann_file': '/data7/lufficc/cityscapes/foggy_cityscapes_coco_test.json',
        'root': foggy_cityscapes_images_dir,
        'train': False,
    },

}


def build_datasets(names, is_train=True):
    datasets = []
    for name in names:
        cfg = DATASETS[name]
        dataset = CityscapeDataset(**cfg)
        datasets.append(dataset)
    if is_train:
        return [ConcatDataset(datasets)]
    return datasets


def build_data_loaders(names, is_train=True, distributed=False, batch_size=1, num_workers=8):
    datasets = build_datasets(names, is_train)
    data_loaders = []
    for dataset in datasets:
        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        elif is_train:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
        if is_train:
            batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=False)
            loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers, collate_fn=collate_fn)
        else:
            loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=num_workers, collate_fn=collate_fn)

        data_loaders.append(loader)

    if is_train:
        assert len(data_loaders) == 1, 'When training, only support one dataset.'
        return data_loaders[0]
    return data_loaders
