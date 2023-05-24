import os
import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import imagesize


@DATASETS.register_module()
class IPSC2Class(CustomDataset):
    """PascalContext dataset.

    In segmentation map annotation for PascalContext, 0 stands for background,
    which is included in 60 categories. ``reduce_zero_label`` is fixed to
    False. The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is
    fixed to '.png'.

    Args:
        split (str): Split txt file for PascalContext.
    """

    CLASSES = ('background', 'ipsc', 'diff')

    PALETTE = [[0, 0, 0], [0, 255, 0], [255, 0, 0]]

    def __init__(self, split, img_dir, **kwargs):
        super(IPSC2Class, self).__init__(
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

        mask_dir_name = osp.basename(mask_dir_name)

        img_infos = []
        assert split is not None, "split must be provided"
        # if self.resave_masks:

        # print('\n\n\nre-saving masks as grayscale\n\n\n')

        img_paths = open(split, "r").read().splitlines()

        n_img_paths = len(img_paths)

        print(f'getting info for {n_img_paths} images')
        empty_img_paths = []

        for img_path in tqdm(img_paths):
            # if self.data_root is not None:
            #     img_path = osp.join(self.data_root, img_rel_path)
            # else:
            #     img_path = img_rel_path

            img_dir_path = osp.dirname(img_path)
            img_name = osp.basename(img_path)
            mask_name = img_name.replace(img_suffix, mask_suffix)
            mask_path = osp.join(img_dir_path, mask_dir_name, mask_name)

            assert osp.exists(img_path), f"img does not exist: {img_path}"
            assert osp.exists(mask_path), f"mask does not exist: {mask_path}"

            img_info = dict(
                filename=img_path,
            )

            # if not osp.exists(mask_path):
            #     if self.allow_missing_mask:
            #         w, h = imagesize.get(img_path)
            #
            #         assert w > 0 and h > 0, f"invalid size {(w, h)} found for {img_path}"
            #
            #         img_path_elemn = os.path.normpath(img_path).split(os.sep)
            #         img_root_idx = img_path_elemn.index(img_dir_name)
            #
            #         img_root_path = osp.join(*img_path_elemn[:img_root_idx + 1])
            #         if img_path.startswith(os.sep):
            #             img_root_path = os.sep + img_root_path
            #
            #         mask_root_path = img_root_path.replace(img_dir_name, mask_dir_name)
            #
            #         mask_fname = f'empty_{w}x{h}.png'
            #         empty_mask_path = osp.join(mask_root_path, mask_fname)
            #         mask_path = empty_mask_path
            #         if not osp.exists(empty_mask_path):
            #             print(f'img_path: {img_path}')
            #
            #             raise AssertionError(f"empty mask does not exist: {empty_mask_path}")
            #         empty_img_paths.append(img_path)
            #
            #     else:
            #         print(f'img_path: {img_path}')
            #         print(f'img_dir_name: {img_dir_name}')
            #         print(f'mask_dir_name: {mask_dir_name}')
            #         print(f'img_suffix: {img_suffix}')
            #         print(f'mask_suffix: {mask_suffix}')
            #         raise AssertionError(f"mask does not exist: {mask_path}")

            img_info['ann'] = dict(seg_map=mask_path)

            img_infos.append(img_info)

        n_empty_img_paths = len(empty_img_paths)
        print(f'Loaded {len(img_infos)} images with {n_empty_img_paths} empty images')

        return img_infos
