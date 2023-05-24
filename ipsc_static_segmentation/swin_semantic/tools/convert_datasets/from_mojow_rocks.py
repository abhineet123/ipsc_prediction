"""
Script for converting CSV annotations to Yolo bounding box labels.

Creates a folder called labels containing a .txt label file for each image. The generated label files follow the Yolo
dataset format (i.e. each row contains `class x_center y_center width height`).

The CSV file should have the following column names (any extra columns are not used):

ImageID, LabelName,	XMin, XMax,	YMin, YMax, Synthetic


Requirements:

    pip install tqdm imagesize

"""

import os
import cv2
import PIL
import random
import pandas as pd
import numpy as np

import paramparse

from tqdm import tqdm
import imagesize

from PIL import Image

col_bgr = {
    'snow': (250, 250, 255),
    'snow_2': (233, 233, 238),
    'snow_3': (201, 201, 205),
    'snow_4': (137, 137, 139),
    'ghost_white': (255, 248, 248),
    'white_smoke': (245, 245, 245),
    'gainsboro': (220, 220, 220),
    'floral_white': (240, 250, 255),
    'old_lace': (230, 245, 253),
    'linen': (230, 240, 240),
    'antique_white': (215, 235, 250),
    'antique_white_2': (204, 223, 238),
    'antique_white_3': (176, 192, 205),
    'antique_white_4': (120, 131, 139),
    'papaya_whip': (213, 239, 255),
    'blanched_almond': (205, 235, 255),
    'bisque': (196, 228, 255),
    'bisque_2': (183, 213, 238),
    'bisque_3': (158, 183, 205),
    'bisque_4': (107, 125, 139),
    'peach_puff': (185, 218, 255),
    'peach_puff_2': (173, 203, 238),
    'peach_puff_3': (149, 175, 205),
    'peach_puff_4': (101, 119, 139),
    'navajo_white': (173, 222, 255),
    'moccasin': (181, 228, 255),
    'cornsilk': (220, 248, 255),
    'cornsilk_2': (205, 232, 238),
    'cornsilk_3': (177, 200, 205),
    'cornsilk_4': (120, 136, 139),
    'ivory': (240, 255, 255),
    'ivory_2': (224, 238, 238),
    'ivory_3': (193, 205, 205),
    'ivory_4': (131, 139, 139),
    'lemon_chiffon': (205, 250, 255),
    'seashell': (238, 245, 255),
    'seashell_2': (222, 229, 238),
    'seashell_3': (191, 197, 205),
    'seashell_4': (130, 134, 139),
    'honeydew': (240, 255, 240),
    'honeydew_2': (224, 238, 244),
    'honeydew_3': (193, 205, 193),
    'honeydew_4': (131, 139, 131),
    'mint_cream': (250, 255, 245),
    'azure': (255, 255, 240),
    'alice_blue': (255, 248, 240),
    'lavender': (250, 230, 230),
    'lavender_blush': (245, 240, 255),
    'misty_rose': (225, 228, 255),
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'dark_slate_gray': (79, 79, 49),
    'dim_gray': (105, 105, 105),
    'slate_gray': (144, 138, 112),
    'light_slate_gray': (153, 136, 119),
    'gray': (190, 190, 190),
    'light_gray': (211, 211, 211),
    'midnight_blue': (112, 25, 25),
    'navy': (128, 0, 0),
    'cornflower_blue': (237, 149, 100),
    'dark_slate_blue': (139, 61, 72),
    'slate_blue': (205, 90, 106),
    'medium_slate_blue': (238, 104, 123),
    'light_slate_blue': (255, 112, 132),
    'medium_blue': (205, 0, 0),
    'royal_blue': (225, 105, 65),
    'blue': (255, 0, 0),
    'dodger_blue': (255, 144, 30),
    'deep_sky_blue': (255, 191, 0),
    'sky_blue': (250, 206, 135),
    'light_sky_blue': (250, 206, 135),
    'steel_blue': (180, 130, 70),
    'light_steel_blue': (222, 196, 176),
    'light_blue': (230, 216, 173),
    'powder_blue': (230, 224, 176),
    'pale_turquoise': (238, 238, 175),
    'dark_turquoise': (209, 206, 0),
    'medium_turquoise': (204, 209, 72),
    'turquoise': (208, 224, 64),
    'cyan': (255, 255, 0),
    'light_cyan': (255, 255, 224),
    'cadet_blue': (160, 158, 95),
    'medium_aquamarine': (170, 205, 102),
    'aquamarine': (212, 255, 127),
    'dark_green': (0, 100, 0),
    'dark_olive_green': (47, 107, 85),
    'dark_sea_green': (143, 188, 143),
    'sea_green': (87, 139, 46),
    'medium_sea_green': (113, 179, 60),
    'light_sea_green': (170, 178, 32),
    'pale_green': (152, 251, 152),
    'spring_green': (127, 255, 0),
    'lawn_green': (0, 252, 124),
    'chartreuse': (0, 255, 127),
    'medium_spring_green': (154, 250, 0),
    'green_yellow': (47, 255, 173),
    'lime_green': (50, 205, 50),
    'yellow_green': (50, 205, 154),
    'forest_green': (34, 139, 34),
    'olive_drab': (35, 142, 107),
    'dark_khaki': (107, 183, 189),
    'khaki': (140, 230, 240),
    'pale_goldenrod': (170, 232, 238),
    'light_goldenrod_yellow': (210, 250, 250),
    'light_yellow': (224, 255, 255),
    'yellow': (0, 255, 255),
    'gold': (0, 215, 255),
    'light_goldenrod': (130, 221, 238),
    'goldenrod': (32, 165, 218),
    'dark_goldenrod': (11, 134, 184),
    'rosy_brown': (143, 143, 188),
    'indian_red': (92, 92, 205),
    'saddle_brown': (19, 69, 139),
    'sienna': (45, 82, 160),
    'peru': (63, 133, 205),
    'burlywood': (135, 184, 222),
    'beige': (220, 245, 245),
    'wheat': (179, 222, 245),
    'sandy_brown': (96, 164, 244),
    'tan': (140, 180, 210),
    'chocolate': (30, 105, 210),
    'firebrick': (34, 34, 178),
    'brown': (42, 42, 165),
    'dark_salmon': (122, 150, 233),
    'salmon': (114, 128, 250),
    'light_salmon': (122, 160, 255),
    'orange': (0, 165, 255),
    'dark_orange': (0, 140, 255),
    'coral': (80, 127, 255),
    'light_coral': (128, 128, 240),
    'tomato': (71, 99, 255),
    'orange_red': (0, 69, 255),
    'red': (0, 0, 255),
    'hot_pink': (180, 105, 255),
    'deep_pink': (147, 20, 255),
    'pink': (203, 192, 255),
    'light_pink': (193, 182, 255),
    'pale_violet_red': (147, 112, 219),
    'maroon': (96, 48, 176),
    'medium_violet_red': (133, 21, 199),
    'violet_red': (144, 32, 208),
    'violet': (238, 130, 238),
    'plum': (221, 160, 221),
    'orchid': (214, 112, 218),
    'medium_orchid': (211, 85, 186),
    'dark_orchid': (204, 50, 153),
    'dark_violet': (211, 0, 148),
    'blue_violet': (226, 43, 138),
    'purple': (240, 32, 160),
    'medium_purple': (219, 112, 147),
    'thistle': (216, 191, 216),
    'green': (0, 255, 0),
    'magenta': (255, 0, 255)
}


class Params:
    """
    Convert CSV annotations to mmseg compatible masks. Creates a folder named labels (by default) at the same path as
    the CSV file

    :ivar masks_dir: name of the directory in which to write the output label files; if min_area >0, it will be
    suffixed to this name; for example, if labels_dir="labels" and min_area=200, labels directory will be named
    "labels_min_200"
    :type masks_dir: str

    :ivar min_area: minimum bounding box area in pixels for the corresponding rock to be included in the output
    :type min_area: int

    """

    def __init__(self):
        self.cfg = ()
        self.csv_paths = []
        self.root_dir = ''
        self.csv_name = 'annotations.csv'
        self.val_ratio = ''
        self.img_dir = 'images'
        self.masks_dir = 'masks'
        self.masks_ext = 'png'
        self.strip_image_id = 1
        self.min_area = 0
        self.n_seq = 0
        self.img_list_name = ''
        self.val_ratio = 0.2
        self.write_masks = 1
        self.write_empty = 0
        self.shuffle = 0
        self.show = 1
        self.sizes = []
        self.class_info = ''


def linux_path(*args, **kwargs):
    return os.path.join(*args, **kwargs).replace(os.sep, '/')


def add_suffix(filename, suffix, sep='-'):
    return "{0}{3}{2}{1}".format(*os.path.splitext(filename) + (suffix, sep))


def main():
    params = Params()
    paramparse.process(params)

    class_info = [k.strip() for k in open(params.class_info, 'r').readlines() if k.strip()]
    class_names, class_cols = zip(*[k.split('\t') for k in class_info])

    class_to_id = {
        class_name: class_id + 1 for class_id, class_name in enumerate(class_names)
    }

    # class_to_cols = {
    #     class_name: col_bgr[class_col] for class_name, class_col in zip(class_names, class_cols)
    # }
    # n_classes = len(class_names)

    # class_dict = {x.strip(): i for (i, x) in enumerate(class_names)}

    csv_paths = params.csv_paths

    assert csv_paths, "csv_paths must be provided"

    if len(csv_paths) == 1:
        csv_list_path = csv_paths[0]

        if params.root_dir:
            csv_list_path = linux_path(params.root_dir, csv_list_path)

        if os.path.isdir(csv_list_path):
            print(f'looking for csv files in {csv_list_path}')
            csv_paths_gen = [[linux_path(dirpath, f).replace(os.sep, '/') for f in filenames if
                              any(f.lower().endswith(ext) for ext in ['.csv'])]
                             for (dirpath, dirnames, filenames) in os.walk(csv_list_path, followlinks=True)]
            csv_paths = [item for sublist in csv_paths_gen for item in sublist]
        elif os.path.isfile(csv_paths[0]) and not csv_paths[0].endswith('.csv'):
            """read list of csv files"""

            if not params.img_list_name:
                params.img_list_name = os.path.basename(csv_paths[0])

            csv_paths = open(csv_paths[0], 'r').readlines()
            csv_paths = [k.strip() for k in csv_paths if k.strip() and not k.startswith('#')]

            if params.root_dir:
                csv_paths = [linux_path(params.root_dir, k) for k in csv_paths]

            if params.csv_name:
                csv_paths = [linux_path(k, params.csv_name) for k in csv_paths]
        else:
            raise AssertionError('invalid csv paths: {}'.format(csv_paths))

    assert params.img_list_name, "img_list_name must be provided"

    csv_paths.sort()

    n_csv_paths = len(csv_paths)
    print(f'found {n_csv_paths} csv files')

    print(f'params.n_seq: {params.n_seq}')

    if 0 < params.n_seq < n_csv_paths:
        print(f'using only first {params.n_seq} csv files')

        csv_paths = csv_paths[:params.n_seq]
        n_csv_paths = params.n_seq

    # pbar = tqdm(csv_paths)
    n_skipped_boxes = 0
    n_skipped_images = 0
    total_images = 0
    total_boxes = 0
    _pause = 1

    assert params.masks_dir, "masks_dir must be provided"
    assert params.img_list_name, "img_list_name must be provided"

    # print('reading all csv files')
    # all_csv_data = list(pd.read_csv(csv_path) for csv_path in tqdm(csv_paths, position=0, leave=True))
    # cmb_csv_data = pd.concat(all_csv_data)

    train_img_list_name = add_suffix(params.img_list_name, 'train')
    val_img_list_name = add_suffix(params.img_list_name, 'val')

    # print(f'train_img_list_name: {train_img_list_name}')
    # print(f'val_img_list_name: {val_img_list_name}')

    train_img_list_path = train_img_list_name
    val_img_list_path = val_img_list_name

    if params.root_dir:
        train_img_list_path = os.path.join(params.root_dir, train_img_list_path)
        val_img_list_path = os.path.join(params.root_dir, val_img_list_path)

    print(f'train_img_list_path: {train_img_list_path}')
    print(f'val_img_list_path: {val_img_list_path}')

    train_img_paths = []
    val_img_paths = []

    if not params.write_masks:
        print('not writing masks')

    for csv_path_id, csv_path in enumerate(tqdm(csv_paths, position=0, leave=True)):
        assert os.path.exists(csv_path), "csv_path does not exist: {}".format(csv_path)
        assert os.path.isfile(csv_path), "csv_path is not a file: {}".format(csv_path)

        masks_dir = params.masks_dir

        if not masks_dir:
            masks_dir = "masks"

        if params.sizes:
            print(f'including only rocks with sizes: {params.sizes}')
            masks_dir += '_' + '_'.join(params.sizes)

        if params.min_area > 0:
            print(f'excluding rocks with area < {params.min_area}')
            masks_dir += '_min_{:d}'.format(params.min_area)

        seq_root_dir = os.path.dirname(csv_path)
        img_dir_path = linux_path(seq_root_dir, params.img_dir)
        masks_dir_path = linux_path(seq_root_dir, masks_dir)

        if params.write_masks:
            print(f"\n{csv_path_id + 1} / {n_csv_paths} :: {csv_path} --> {masks_dir_path}")

        os.makedirs(masks_dir_path, exist_ok=True)

        # vis_masks_dir_path = linux_path(seq_root_dir, masks_dir + '_vis')
        # os.makedirs(vis_masks_dir_path, exist_ok=True)

        csv_data = pd.read_csv(csv_path)

        empty_csv_data = csv_data.loc[csv_data['LabelName'].isna()]
        empty_image_ids = list(empty_csv_data['ImageID'].unique())
        n_empty_image_ids = len(empty_image_ids)
        print(f'found  {n_empty_image_ids} empty images')

        csv_data = csv_data.loc[~csv_data['LabelName'].isna()]
        image_ids = list(csv_data['ImageID'].unique())

        pbar = tqdm(image_ids, position=0, leave=True)

        valid_img_paths = []

        n_classes = len(class_cols)
        """background is class ID 0 with color black"""
        palette = [[0, 0, 0], ]
        for class_id in range(n_classes):
            col = class_cols[class_id]

            col_rgb = col_bgr[col][::-1]

            palette.append(col_rgb)

        palette_flat = [value for color in palette for value in color]
        # scale_factor = int(255 / (n_classes + 1))

        empty_image_sizes = []

        for image_id in pbar:

            if params.strip_image_id:
                img_name = os.path.basename(image_id)
            else:
                img_name = image_id

            total_images += 1
            skipped_percent = n_skipped_images / total_images * 100
            desc1 = f'skipped img: {n_skipped_images} / {total_images} ({skipped_percent:.2f}%)'

            img_path = linux_path(img_dir_path, img_name)

            assert os.path.isfile(img_path), f"img does not exist: {img_path}"

            img_fname_noext = os.path.splitext(img_name)[0]

            mask_img = None

            width, height = imagesize.get(str(img_path))

            if params.write_masks or params.show:
                mask_img = np.zeros((height, width), dtype=np.uint8)

            image_csv_data = csv_data.loc[csv_data['ImageID'] == image_id]

            if image_csv_data.empty:
                msg = f"no boxes found for {img_path}"

                # if params.wtite_empty:
                #     print(msg)
                #
                #     img_size = (width, height)
                #
                #     if img_size not in empty_image_sizes:
                #         empty_image_sizes.append(img_size)
                #
                #     valid_img_paths.append(img_path)
                #     continue
                # else:
                raise AssertionError(msg)

            n_valid_bboxes = 0

            for _, row in image_csv_data.iterrows():

                total_boxes += 1

                skipped_percent = n_skipped_boxes / total_boxes * 100
                desc2 = f' obj: {n_skipped_boxes} / {total_boxes} ({skipped_percent:.2f}%)'

                desc = desc1 + desc2

                pbar.set_description(desc)

                size = row["Size"]
                class_name = 'rock'

                try:
                    is_syn = row["Synthetic"]
                except KeyError:
                    pass
                else:
                    if is_syn:
                        class_name = 'syn'

                # col = class_to_cols[class_name]

                if params.sizes and size not in params.sizes:
                    n_skipped_boxes += 1
                    continue

                x_max = float(row["XMax"])
                y_max = float(row["YMax"])
                x_min = float(row["XMin"])
                y_min = float(row["YMin"])

                bbox_width = x_max - x_min
                bbox_height = y_max - y_min

                bbox_area = round(bbox_width * bbox_height)

                if (params.min_area > 0) and (bbox_area < params.min_area):
                    n_skipped_boxes += 1
                    continue

                n_valid_bboxes += 1

                if params.write_masks or params.show:
                    class_id = class_to_id[class_name]
                    mask_str = row["MaskXY"]

                    mask_pts = mask_str.split(';')
                    mask_pts_float = list(list(float(k) for k in mask_pt.split(',')) for mask_pt in mask_pts if mask_pt)

                    mask_img = cv2.fillPoly(mask_img, np.array([mask_pts_float, ], dtype=np.int32), class_id)

                    # mask_img_vis = mask_img * scale_factor

            if params.show:
                src_img = cv2.imread(img_path)
                cv2.imshow('src_img', src_img)
                cv2.imshow('mask_img', mask_img)

                # mask_img_vis = cv2.resize(mask_img_vis, (1280, 720))
                # cv2.imshow('mask_img_vis', mask_img_vis)

                k = cv2.waitKey(1 - _pause)
                if k == 27:
                    return
                elif k == 32:
                    _pause = 1 - _pause

            if n_valid_bboxes:
                valid_img_paths.append(img_path)
                if params.write_masks:
                    mask_fname = f'{img_fname_noext}.{params.masks_ext}'
                    mask_path = linux_path(masks_dir_path, mask_fname)
                    mask_parent_path = os.path.dirname(mask_path)
                    os.makedirs(mask_parent_path, exist_ok=1)

                    mask_img_pil = PIL.Image.fromarray(mask_img)
                    mask_img_pil = mask_img_pil.convert('P')
                    mask_img_pil.putpalette(palette_flat)
                    mask_img_pil.save(mask_path)

                    # vis_mask_path = linux_path(vis_masks_dir_path, mask_fname)
                    # vis_mask_parent_path = os.path.dirname(vis_mask_path)
                    # os.makedirs(vis_mask_parent_path, exist_ok=1)
                    # cv2.imwrite(vis_mask_path, mask_img_vis)
            else:
                n_skipped_images += 1
                if params.write_empty:
                    img_size = (width, height)
                    if img_size not in empty_image_sizes:
                        empty_image_sizes.append(img_size)

                    valid_img_paths.append(img_path)

        if params.write_empty and empty_image_sizes:

            print(f'getting sizes for {n_empty_image_ids} empty images')
            for image_id in tqdm(empty_image_ids):
                if params.strip_image_id:
                    img_name = os.path.basename(image_id)
                else:
                    img_name = image_id

                img_path = linux_path(img_dir_path, img_name)
                valid_img_paths.append(img_path)

                width, height = imagesize.get(str(img_path))

                img_size = (width, height)
                if img_size not in empty_image_sizes:
                    empty_image_sizes.append(img_size)

            n_empty_image_sizes = len(empty_image_sizes)
            print(f'writing empty masks for {n_empty_image_sizes} image sizes:\n{empty_image_sizes}')

            for w, h in empty_image_sizes:
                mask_fname = f'empty_{w}x{h}.{params.masks_ext}'
                mask_path = linux_path(masks_dir_path, mask_fname)
                mask_parent_path = os.path.dirname(mask_path)
                os.makedirs(mask_parent_path, exist_ok=1)
                mask_img = np.zeros((h, w), dtype=np.uint8)
                mask_img_pil = PIL.Image.fromarray(mask_img)
                mask_img_pil = mask_img_pil.convert('P')
                mask_img_pil.putpalette(palette_flat)
                mask_img_pil.save(mask_path)

        n_image_paths = len(valid_img_paths)
        n_train_img_paths = int(n_image_paths * (1 - params.val_ratio))

        if params.shuffle:
            random.shuffle(valid_img_paths)
        else:
            valid_img_paths.sort()

        train_img_paths += valid_img_paths[:n_train_img_paths]
        val_img_paths += valid_img_paths[n_train_img_paths:]

    # if params.shuffle:
    #     random.shuffle(train_img_paths)
    #     random.shuffle(val_img_paths)

    with open(train_img_list_path, 'w') as fid:
        fid.write('\n'.join(train_img_paths))

    with open(val_img_list_path, 'w') as fid:
        fid.write('\n'.join(val_img_paths))


if __name__ == "__main__":
    main()
