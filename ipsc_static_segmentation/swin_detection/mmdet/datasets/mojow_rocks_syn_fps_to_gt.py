from .builder import DATASETS
from .mojow_rocks_base import MojowRocksBase


@DATASETS.register_module()
class MojowRocksSynFPsToGT(MojowRocksBase):
    CLASSES = ['rock', 'syn', 'FP-rock', 'FP-syn']
    PALETTE = [[0, 255, 0], [255, 0, 0], [0, 255, 255], [255, 0, 255]]
