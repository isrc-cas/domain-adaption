from .dataset import COCODataset

# normalizer = {
#     'name': 'normalize',
#     'mean': [102.9801, 115.9465, 122.7717],
#     'std': [1, 1, 1],
#     'to_bgr': True,
# }

normalizer = {
    'name': 'normalize',
    'mean': [0.5, 0.5, 0.5],
    'std': [0.5, 0.5, 0.5],
    'to_01': True,
}
train_transforms = [
    {
        'name': 'random_flip'
    },
    {
        'name': 'resize',
        'min_size': (800,),
        'max_size': 1333,
    },
    {
        'name': 'pad',
        'size_divisor': 16,
    },
    normalizer,
    {
        'name': 'collect'
    }
]

test_transforms = [
    {
        'name': 'resize',
        'min_size': (800,),
        'max_size': 1333,
    },
    {
        'name': 'pad',
        'size_divisor': 16,
    },
    normalizer,
    {
        'name': 'collect'
    }
]


class MSCOCODataset(COCODataset):
    CLASSES = ('__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
               'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
               'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
               'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')

    def __init__(self, train=True, **kwargs):
        transforms = train_transforms if train else test_transforms
        super().__init__(transforms=transforms, remove_empty=train, **kwargs)
