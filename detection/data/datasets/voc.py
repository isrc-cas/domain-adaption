from .dataset import VOCDataset

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
        'min_size': (600,),
    },
    normalizer,
    {
        'name': 'collect'
    }
]

test_transforms = [
    {
        'name': 'resize',
        'min_size': (600,),
    },
    normalizer,
    {
        'name': 'collect'
    }
]


class CustomVocDataset(VOCDataset):
    def __init__(self, train, **kwargs):
        transforms = train_transforms if train else test_transforms
        super().__init__(transforms=transforms, **kwargs)
