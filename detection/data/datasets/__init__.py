from .coco import MSCOCODataset
from .cityscape import CityscapeDataset
from .voc import CustomVocDataset, WatercolorDataset, Sim10kDataset
from .dataset import COCODataset, VOCDataset

__all__ = ['MSCOCODataset', 'CityscapeDataset', 'CustomVocDataset', 'COCODataset', 'VOCDataset']
