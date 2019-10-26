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
        'min_size': (512,),
        'max_size': 1024,
    },
    normalizer,
    {
        'name': 'collect'
    }
]

test_transforms = [
    {
        'name': 'resize',
        'min_size': (512,),
        'max_size': 1024,
    },
    normalizer,
    {
        'name': 'collect'
    }
]


class CityscapeDataset(COCODataset):
    CLASSES = ('__background__', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle',)

    def __init__(self, ann_file, root, train=True):
        transforms = train_transforms if train else test_transforms
        super().__init__(ann_file, root, transforms=transforms, remove_empty=train)
