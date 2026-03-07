from .builder import DATASETS
from .mojow_rocks_base import MojowRocksBase


@DATASETS.register_module()
class MojowRocksFPsToGT(MojowRocksBase):
    CLASSES = ['rock', 'FP-rock']
    PALETTE = [[0, 255, 0], [255, 0, 0]]
