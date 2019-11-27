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


class compose(object):
    def __init__(self, transforms):
        self.transforms = []
        for transform in self.transforms:
            if isinstance(transform, dict):
                args = transform.copy()
                name = args.pop('name')
                transform = TRANSFORMS[name](**args)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

        self.transforms = transforms

    def __call__(self, results):
        for transform in self.transforms:
            results = transform(results)
        return results


def build_transforms(transforms):
    return compose(transforms)
