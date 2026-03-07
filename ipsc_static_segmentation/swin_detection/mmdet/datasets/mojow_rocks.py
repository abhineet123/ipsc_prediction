from .builder import DATASETS
from .mojow_rocks_base import MojowRocksBase


@DATASETS.register_module()
class MojowRocks(MojowRocksBase):
    CLASSES = ['rock', ]
    PALETTE = [[0, 255, 0], ]
