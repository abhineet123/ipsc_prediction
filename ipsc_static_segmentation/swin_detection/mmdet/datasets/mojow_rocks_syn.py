from .builder import DATASETS
from .mojow_rocks_base import MojowRocksBase

@DATASETS.register_module()
class MojowRocksSyn(MojowRocksBase):
    CLASSES = ['rock', 'syn']
    PALETTE = [[0, 255, 0], [255, 0, 0]]
