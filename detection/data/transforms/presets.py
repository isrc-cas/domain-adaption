_TRAIN_01 = [
    {
        'name': 'random_flip'
    },
    {
        'name': 'resize',
        'min_size': 600,
    },
    {
        'name': 'normalize',
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5],
        'to_01': True,
    },
    {
        'name': 'collect'
    }
]

_TEST_01 = [
    {
        'name': 'resize',
        'min_size': 600,
    },
    {
        'name': 'normalize',
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5],
        'to_01': True,
    },
    {
        'name': 'collect'
    }
]
