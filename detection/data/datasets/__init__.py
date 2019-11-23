from .coco import MSCOCODataset
from .cityscape import CityscapeDataset, CityscapeCarDataset
from .voc import CustomVocDataset, WatercolorDataset, Sim10kDataset
from .dataset import COCODataset, VOCDataset

__all__ = ['MSCOCODataset', 'CityscapeDataset', 'CityscapeCarDataset',
           'CustomVocDataset', 'WatercolorDataset', 'Sim10kDataset', 'COCODataset', 'VOCDataset']
