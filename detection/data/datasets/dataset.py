import os
from collections import defaultdict

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET

from ..transforms import build_transforms


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    return True


class ABSDataset(Dataset):
    def __init__(self, images_dir, transforms=(), dataset_name=''):
        self.images_dir = images_dir
        self.transforms = build_transforms(transforms)
        self.dataset_name = dataset_name
        self.ids = []
        self.label2cat = {}

    def __getitem__(self, idx):
        target = self.get_annotations(idx)
        img_info = target['img_info']
        file_name = img_info['file_name']
        img = Image.open(os.path.join(self.images_dir, file_name)).convert('RGB')

        results = {
            'img': np.array(img),
            'boxes': np.array(target['boxes']),
            'labels': np.array(target['labels']),
            'img_shape': (img.width, img.height),
            'img_info': img_info,
        }
        results = self.transforms(results)
        return results

    def get_annotations_by_image_id(self, img_id):
        """
        Args:
            img_id: image id
        Returns: dict with keys (img_info, boxes, labels)
        """
        raise NotImplementedError

    def get_annotations(self, idx):
        """
        Args:
            idx: dataset index
        Returns: dict with keys (img_info, boxes, labels)
        """
        img_id = self.ids[idx]
        return self.get_annotations_by_image_id(img_id)

    def __repr__(self):
        return '{} Dataset(size: {})'.format(self.dataset_name, len(self))

    def __len__(self):
        return len(self.ids)


class COCODataset(ABSDataset):
    def __init__(self, ann_file, root, transforms=(), remove_empty=False, dataset_name=''):
        super().__init__(root, transforms, dataset_name)
        self.ann_file = ann_file

        from pycocotools.coco import COCO
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        if remove_empty:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.coco.getCatIds())
        }
        self.label2cat = {
            v: k for k, v in self.cat2label.items()
        }

    def get_annotations_by_image_id(self, img_id):
        coco = self.coco
        img_info = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        anns = [obj for obj in anns if obj["iscrowd"] == 0]

        boxes = []
        labels = []
        for obj in anns:
            x, y, w, h = obj["bbox"]
            box = [x, y, x + w - 1, y + h - 1]
            label = self.cat2label[obj["category_id"]]
            boxes.append(box)
            labels.append(label)
        boxes = np.array(boxes).reshape((-1, 4))
        labels = np.array(labels)

        return {'img_info': img_info, 'boxes': boxes, 'labels': labels}


class VOCDataset(ABSDataset):
    CLASSES = ('__background__',
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')

    def __init__(self, root, split='train', base_dir='.', transforms=(), keep_difficult=True, img_ext='.jpg', dataset_name=''):
        self.root = root
        self.split = split
        self.keep_difficult = keep_difficult
        self.img_ext = img_ext

        voc_root = os.path.join(self.root, base_dir)
        images_dir = os.path.join(voc_root, 'JPEGImages')
        self.annotation_dir = os.path.join(voc_root, 'Annotations')
        super().__init__(images_dir, transforms, dataset_name)

        splits_dir = os.path.join(voc_root, 'ImageSets/Main')
        split_f = os.path.join(splits_dir, split + '.txt')
        with open(os.path.join(split_f), "r") as f:
            ids = [x.strip() for x in f.readlines()]
        self.ids = ids

        cat_ids = list(range(len(VOCDataset.CLASSES)))
        self.label2cat = {
            label: cat for label, cat in enumerate(cat_ids)
        }

    def get_annotations_by_image_id(self, img_id):
        ann_path = os.path.join(self.annotation_dir, img_id + '.xml')
        target = self.parse_voc_xml(ET.parse(ann_path).getroot())['annotation']
        img_info = {
            'width': target['size']['width'],
            'height': target['size']['height'],
            'id': img_id,
            'file_name': img_id + self.img_ext,
        }

        boxes = []
        labels = []
        difficult = []
        for obj in target['object']:
            is_difficult = bool(int(obj['difficult']))
            if is_difficult and not self.keep_difficult:
                continue
            label_name = obj['name']
            if label_name not in self.CLASSES:
                continue
            difficult.append(is_difficult)
            label_id = self.CLASSES.index(label_name)
            box = obj['bndbox']
            box = list(map(float, [box['xmin'], box['ymin'], box['xmax'], box['ymax']]))
            boxes.append(box)
            labels.append(label_id)
        boxes = np.array(boxes).reshape((-1, 4))
        labels = np.array(labels)
        difficult = np.array(difficult)

        return {'img_info': img_info, 'boxes': boxes, 'labels': labels, 'difficult': difficult}

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            voc_dict = {
                node.tag: {k: v if k == 'object' else v[0] for k, v in def_dic.items()}
            }
        elif node.text:
            text = node.text.strip()
            voc_dict[node.tag] = text
        return voc_dict
