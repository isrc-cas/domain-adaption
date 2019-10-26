import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .transforms import build_transforms


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


class COCODataset(Dataset):
    def __init__(self, ann_file, root, transforms=(), remove_empty=False):
        self.ann_file = ann_file
        self.root = root
        self.transforms = build_transforms(transforms)
        self.dataset_name = os.path.basename(ann_file).split('.')[0]

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

    def __getitem__(self, idx):
        img_info, boxes, labels = self.get_annotations(idx)
        file_name = img_info['file_name']
        img = Image.open(os.path.join(self.root, file_name)).convert('RGB')

        results = {
            'img': np.array(img),
            'boxes': boxes,
            'labels': labels,
            'img_shape': (img.width, img.height),
            'img_info': img_info,
        }
        results = self.transforms(results)
        return results

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

        return img_info, boxes, labels

    def get_annotations(self, idx):
        img_id = self.ids[idx]
        return self.get_annotations_by_image_id(img_id)

    def __repr__(self):
        return '{} Dataset(size: {})'.format(self.dataset_name, len(self))

    def __len__(self):
        return len(self.ids)
