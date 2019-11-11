import torch
from torch.utils.data import ConcatDataset, DataLoader

from . import collate_fn
from .datasets import *

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
    "coco_2017_train": {
        "ann_file": "/data7/lufficc/coco/annotations/instances_train2017.json",
        "root": "/data7/lufficc/coco/train2017",
        'train': True,
    },
    "coco_2017_val": {
        "ann_file": "/data7/lufficc/coco/annotations/instances_val2017.json",
        "root": "/data7/lufficc/coco/val2017",
        'train': False,
    },

    'voc_2007_trainval': {
        'root': '/data7/lufficc/voc/VOCdevkit/VOC2007',
        'split': 'trainval',
        'train': True,
    },

    'voc_2012_trainval': {
        'root': '/data7/lufficc/voc/VOCdevkit/VOC2012',
        'split': 'trainval',
        'train': True,
    },

    'voc_2007_test': {
        'root': '/data7/lufficc/voc/VOCdevkit/VOC2007',
        'split': 'test',
        'train': False,
    },

    'voc_watercolor_train': {
        'root': '/data7/lufficc/cross_domain_detection/watercolor',
        'split': 'train',
        'train': True,
    },
    'voc_watercolor_test': {
        'root': '/data7/lufficc/cross_domain_detection/watercolor',
        'split': 'test',
        'train': False,
    },

    'voc_comic_train': {
        'root': '/data7/lufficc/cross_domain_detection/comic',
        'split': 'train',
        'train': True,
    },
    'voc_comic_test': {
        'root': '/data7/lufficc/cross_domain_detection/comic',
        'split': 'test',
        'train': False,
    },
    'voc_clipart_train': {
        'root': '/data7/lufficc/cross_domain_detection/clipart',
        'split': 'train',
        'train': True,
    },
    'voc_clipart_test': {
        'root': '/data7/lufficc/cross_domain_detection/clipart',
        'split': 'test',
        'train': False,
    },
    'voc_clipart_traintest': {
        'root': '/data7/lufficc/cross_domain_detection/clipart',
        'split': 'traintest',
        'train': False,
    },
}


def build_datasets(names, is_train=True):
    assert len(names) > 0
    datasets = []
    for name in names:
        cfg = DATASETS[name].copy()
        cfg['dataset_name'] = name
        if 'cityscapes' in name:
            dataset = CityscapeDataset(**cfg)
        elif 'coco' in name:
            dataset = MSCOCODataset(**cfg)
        elif 'voc' in name:
            dataset = CustomVocDataset(**cfg)
        else:
            raise NotImplementedError
        print('{:<24}: {}'.format(dataset.dataset_name, len(dataset)))
        datasets.append(dataset)
    if is_train:
        return datasets if len(datasets) == 1 else [ConcatDataset(datasets)]
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
