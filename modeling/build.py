from torchvision import models

from .faster_rcnn import FasterRCNN


def build_detectors(name, num_classes):
    if name == 'VGG16':
        vgg16 = models.vgg16(True)
        model = FasterRCNN(vgg16.features[:-1], num_classes=num_classes)
    else:
        raise NotImplementedError
    return model
