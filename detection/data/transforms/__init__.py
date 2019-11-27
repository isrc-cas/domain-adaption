from .transforms import *
from .presets import _TRAIN_01, _TEST_01

TRANSFORMS = {
    'random_flip': random_flip,
    'resize': resize,
    'normalize': normalize,
    'collect': collect,
    'pad': pad,
    # presets
    '_train01': lambda: compose(_TRAIN_01),
    '_test01': lambda: compose(_TEST_01),

}


def build_transforms(transforms):
    results = []
    for cfg in transforms:
        args = cfg.copy()
        name = args.pop('name')
        transform = TRANSFORMS[name](**args)
        results.append(transform)
    return compose(results)
