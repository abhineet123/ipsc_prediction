import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm


@DATASETS.register_module()
class MojowRocksWithSyn(CustomDataset):
    """PascalContext dataset.

    In segmentation map annotation for PascalContext, 0 stands for background,
    which is included in 60 categories. ``reduce_zero_label`` is fixed to
    False. The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is
    fixed to '.png'.

    Args:
        split (str): Split txt file for PascalContext.
    """

    CLASSES = ('background', 'rock', 'syn')

    PALETTE = [[0, 0, 0], [0, 255, 0], [255, 0, 0]]

    def __init__(self, split, img_dir, **kwargs):
        super(MojowRocksWithSyn, self).__init__(
            img_dir=img_dir,
            img_suffix='.jpg',
            seg_map_suffix='.png',
            split=split,
            reduce_zero_label=False,
            **kwargs)
        # self.resave_masks = resave_masks
        assert self.split is not None, "split cannot be none"
        assert self.img_dir is not None, "img_dir cannot be none"

    def load_annotations(self, img_dir_name, img_suffix, mask_dir_name, mask_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir_name (str): Path to image directory
            img_suffix (str): Suffix of images.
            mask_dir_name (str|None): Path to annotation directory.
            mask_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_dir_name = osp.basename(img_dir_name)
        mask_dir_name = osp.basename(mask_dir_name)

        img_infos = []
        assert split is not None, "split must be provided"
        # if self.resave_masks:

        # print('\n\n\nre-saving masks as grayscale\n\n\n')

        with open(split) as f:
            for line in tqdm(f):
                img_path = line.strip()
                # if self.data_root is not None:
                #     img_path = osp.join(self.data_root, img_rel_path)
                # else:
                #     img_path = img_rel_path

                mask_path = img_path.replace(img_dir_name, mask_dir_name).replace(img_suffix, mask_suffix)

                assert osp.exists(img_path), f"img does not exist: {img_path}"

                if not osp.exists(mask_path):
                    print(f'img_path: {img_path}')
                    print(f'img_dir_name: {img_dir_name}')
                    print(f'mask_dir_name: {mask_dir_name}')
                    print(f'img_suffix: {img_suffix}')
                    print(f'mask_suffix: {mask_suffix}')
                    raise AssertionError(f"mask does not exist: {mask_path}")

                # if self.resave_masks:
                # seg_map = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                # seg_img = Image.fromarray(seg_map).convert('P')
                # seg_img.putpalette(np.array(self.PALETTE, dtype=np.uint8))
                # seg_img.save(mask_path)

                img_info = dict(
                    filename=img_path,
                )
                img_info['ann'] = dict(seg_map=mask_path)

                img_infos.append(img_info)

        print(f'Loaded {len(img_infos)} images')
        return img_infos
