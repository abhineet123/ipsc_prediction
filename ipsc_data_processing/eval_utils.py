import operator
import os
import sys
import json
import logging
import multiprocessing

import shutil
import random
import cv2
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
from datetime import datetime

from io import StringIO

from PIL import Image

import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from tabulate import tabulate

import pycocotools.mask as mask_util

# BGR values for different colors
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
    'magenta': (255, 0, 255),
    '133_200_166': (133, 200, 166),
    '200_100_100': (200, 100, 100),
    '200_166_100': (200, 166, 100),
    '166_200_100': (166, 200, 100),
    '200_200_200': (200, 200, 200),
    '166_133_133': (166, 133, 133),
    '133_133_133': (133, 133, 133),
    '100_100_100': (100, 100, 100),
    '100_100_133': (100, 100, 133),
    '166_100_200': (166, 100, 200),
    '200_166_133': (200, 166, 133),
    '133_200_133': (133, 200, 133),
    '166_166_166': (166, 166, 166),
    '166_133_166': (166, 133, 166),
    '133_100_100': (133, 100, 100),
    '100_200_200': (100, 200, 200),
    '166_200_166': (166, 200, 166),
    '166_200_133': (166, 200, 133),
    '100_166_166': (100, 166, 166),
    '166_133_200': (166, 133, 200),
    '133_200_100': (133, 200, 100),
    '133_166_200': (133, 166, 200),
    '100_200_166': (100, 200, 166),
    '200_133_200': (200, 133, 200),
    '133_166_133': (133, 166, 133),
    '100_100_166': (100, 100, 166),
    '166_166_100': (166, 166, 100),
    '166_100_166': (166, 100, 166),
    '200_166_200': (200, 166, 200),
    '200_100_133': (200, 100, 133),
}

bgr_col = {col_num: col_name for col_name, col_num in col_bgr.items()}

import traceback


class Process(multiprocessing.Process):
    def __init__(self, *args, **kwargs):
        multiprocessing.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = multiprocessing.Pipe()
        self._exception = None

    def run(self):
        try:
            multiprocessing.Process.run(self)
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))
            raise e

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


def list_to_str(k):
    return '\n'.join(k)


def zip_dirs(agn_root_dirs, del_src=1):
    if len(agn_root_dirs) > 1:
        agn_prefix = os.path.commonpath(agn_root_dirs)
    else:
        agn_prefix = os.path.dirname(agn_root_dirs[0])

    assert agn_prefix not in agn_root_dirs, "agn_prefix cannot be same as an agn_root_dir"

    agn_rel_dirs = [os.path.relpath(k, agn_prefix) for k in agn_root_dirs]
    agn_rel_dirs.sort()

    switches = '-r -q'
    in_paths = ' '.join(agn_rel_dirs)
    out_path = agn_prefix + '.zip'
    out_path = os.path.abspath(out_path)
    zip_cmd = f'cd {agn_prefix} && zip {switches} {out_path} {in_paths}'
    print(f'{zip_cmd}')
    os.system(zip_cmd)

    if os.path.exists(out_path):
        from zipfile import ZipFile, Path
        list_zip_file = ZipFile(out_path, 'r')
        zip_subdirs = [k.name for k in Path(list_zip_file).iterdir() if k.is_dir()]
        zip_subdirs.sort()

        assert zip_subdirs == agn_rel_dirs, "zip_subdirs mismatch"

        if del_src:
            for agn_root_dir in agn_root_dirs:
                shutil.rmtree(agn_root_dir)
        print()

    if del_src and not os.listdir(agn_prefix):
        shutil.rmtree(agn_prefix)

    return out_path


def sleep_with_pbar(sleep_mins):
    for _ in tqdm(range(sleep_mins), desc='sleeping', ncols=50):
        try:
            time.sleep(60)
        except KeyboardInterrupt:
            print('sleep interrupted')
            return False
    return True


def print_stats(stats, name='', fmt='.3f'):
    if name:
        print(f"\n{name}")
    print(tabulate(stats, headers='keys', tablefmt="orgtbl", floatfmt=fmt))


def put_text_with_background(img, text, loc, col, bkg_col, **kwargs):
    text_offset_x, text_offset_y = loc

    (text_width, text_height) = cv2.getTextSize(text, **kwargs)[0]
    box_coords = ((text_offset_x, text_offset_y + 5), (text_offset_x + text_width, text_offset_y - text_height))
    cv2.rectangle(img, box_coords[0], box_coords[1], bkg_col, cv2.FILLED)

    cv2.putText(img, text, org=loc, color=col, **kwargs)

    return text_width, text_height


def show_labels(image, labels, cols, vert=1, offset=5):
    x, y = 5, 15

    font_args = dict(
        fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
        fontScale=1,
        thickness=1
    )
    for label, col in zip(labels, cols):
        txt = f'{label}'

        (_text_width, _text_height) = put_text_with_background(image, txt, (x, y), col, (0, 0, 0), **font_args)

        if vert:
            y += _text_height + offset
        else:
            x += _text_width + offset


def clamp(_vals, min_val, max_val):
    return [max(min(_val, max_val), min_val) for _val in _vals]


def drawBox(image, xmin, ymin, xmax, ymax, box_color=(0, 255, 0), label=None, font_size=0.3, mask=None, thickness=2):
    image_float = image.astype(np.float32)

    if isinstance(box_color, str):
        box_color = col_bgr[box_color]

    if mask is not None:
        mask_pts = np.asarray(mask).reshape((-1, 1, 2)).astype(np.int32)
        cv2.drawContours(image_float, mask_pts, -1, box_color, thickness=thickness, lineType=cv2.LINE_AA)
    else:
        cv2.rectangle(image_float, (int(xmin), int(ymin)), (int(xmax), int(ymax)), box_color, thickness=thickness)

    image[:] = image_float.astype(image.dtype)

    _bb = [xmin, ymin, xmax, ymax]
    if _bb[1] > 10:
        y_loc = int(_bb[1] - 5)
    else:
        y_loc = int(_bb[3] + 5)
    if label is not None:
        cv2.putText(image, label, (int(_bb[0] - 1), y_loc), cv2.FONT_HERSHEY_SIMPLEX,
                    font_size, box_color, 1, cv2.LINE_AA)


def resize_ar_tf_api(src_img, width=0, height=0, return_factors=0, add_border=1, crop=0):
    src_height, src_width, n_channels = src_img.shape

    src_aspect_ratio = float(src_width) / float(src_height)

    if width <= 0 and height <= 0:
        raise AssertionError(
            'Both width and height cannot be 0 when resize_factor is 0 too')
    elif height <= 0:
        height = int(width / src_aspect_ratio)
    elif width <= 0:
        width = int(height * src_aspect_ratio)

    aspect_ratio = float(width) / float(height)

    if add_border == 2:
        assert src_height <= height and src_width <= width, \
            f"border only mode :: source size {src_width} x {src_height} > target size {width} x {height}"

        dst_img = np.zeros((height, width, n_channels), dtype=np.uint8)
        start_row = int((height - src_height) / 2.0)
        start_col = int((width - src_width) / 2.0)
        end_row = start_row + src_height
        end_col = start_col + src_width
        dst_img[start_row:end_row, start_col:end_col, :] = src_img
        if return_factors:
            return dst_img, 1, start_row, start_col
        else:
            return dst_img

    if add_border:
        if src_aspect_ratio == aspect_ratio:
            dst_width = src_width
            dst_height = src_height
            start_row = start_col = 0
        elif src_aspect_ratio > aspect_ratio:
            dst_width = src_width
            dst_height = int(src_width / aspect_ratio)
            start_row = int((dst_height - src_height) / 2.0)
            start_col = 0
        else:
            dst_height = src_height
            dst_width = int(src_height * aspect_ratio)
            start_col = int((dst_width - src_width) / 2.0)
            start_row = 0

        end_row = start_row + src_height
        end_col = start_col + src_width
        dst_img = np.zeros((dst_height, dst_width, n_channels), dtype=np.uint8)
        dst_img[start_row:end_row, start_col:end_col, :] = src_img
        dst_img = cv2.resize(dst_img, (width, height))

        resize_factor = float(height) / float(dst_height)

        if crop:
            start_col_res = int(start_col * resize_factor)
            start_row_res = int(start_row * resize_factor)
            end_row_res = int(end_row * resize_factor)
            end_col_res = int(end_col * resize_factor)
            dst_img = dst_img[start_row_res:end_row_res, start_col_res:end_col_res, :]

            start_col = start_row = 0

        if return_factors:
            return dst_img, resize_factor, start_row, start_col
        else:
            return dst_img
    else:
        if src_aspect_ratio < aspect_ratio:
            dst_width = width
            dst_height = int(dst_width / src_aspect_ratio)
        else:
            dst_height = height
            dst_width = int(dst_height * src_aspect_ratio)
        dst_img = cv2.resize(src_img, (dst_width, dst_height))
        start_row = start_col = 0
        if return_factors:
            resize_factor = float(src_height) / float(dst_height)
            return dst_img, resize_factor, start_row, start_col
        else:
            return dst_img


def annotate(
        img_list,
        img_labels,
        text=None,
        fmt=None,
        grid_size=(-1, 1),
):
    """

    :param np.ndarray | list | tuple img_list:
    :param str text:
    :param CVText fmt:
    :param tuple(int) grid_size:
    :return:
    """

    if not isinstance(img_list, (list, tuple)):
        img_list = [img_list, ]

    if not isinstance(img_labels, (list, tuple)):
        img_labels = [img_labels, ]

    assert len(img_labels) == len(img_list), "img_labels and img_list must have same length"

    if fmt is None:
        """use default format"""
        fmt = CVText()

    size = fmt.size

    color = col_bgr[fmt.color]
    font = CVConstants.fonts[fmt.font]
    line_type = CVConstants.line_types[fmt.line_type]

    out_img_list = []

    for _id, _img in enumerate(img_list):
        if len(_img.shape) == 2:
            _img = np.stack([_img, ] * 3, axis=2)

        img_label = img_labels[_id]
        (text_width, text_height) = cv2.getTextSize(
            img_label, font,
            fontScale=fmt.size,
            thickness=fmt.thickness)[0]

        text_height += fmt.offset[1]
        text_width += fmt.offset[0]
        label_img = np.zeros((text_height, text_width), dtype=np.uint8)
        cv2.putText(label_img, img_label, tuple(fmt.offset),
                    font, size, color, fmt.thickness, line_type)

        if len(_img.shape) == 3:
            label_img = np.stack([label_img, ] * 3, axis=2)

        if text_width < _img.shape[1]:
            label_img = resize_ar(label_img, width=_img.shape[1], height=text_height,
                                  only_border=2, placement_type=1)

        # border_img = np.full((5, _img.shape[0], 3), 255, dtype=np.uint8)
        img_list_label = [label_img,
                          # border_img,
                          _img]

        _img = stack_images(img_list_label, grid_size=(-1, 1), preserve_order=1)

        # border_img = np.full((_img.shape[1], 5, 3), 255, dtype=np.uint8)
        # _img = stack_images([_img, border_img], grid_size=(1, -1), preserve_order=1)

        out_img_list.append(_img)

    img_stacked = stack_images(out_img_list, grid_size=grid_size, preserve_order=1)

    if text is not None:
        if '\n' in text:
            text_list = text.split('\n')
        else:
            text_list = [text, ]

        max_text_width = 0
        text_height = 0
        text_heights = []

        for _text in text_list:
            (_text_width, _text_height) = cv2.getTextSize(_text, font, fontScale=fmt.size, thickness=fmt.thickness)[0]
            if _text_width > max_text_width:
                max_text_width = _text_width
            text_height += _text_height + 5
            text_heights.append(_text_height)

        text_width = max_text_width + 10
        text_height += 30

        text_img = np.zeros((text_height, text_width, 3), dtype=np.uint8)
        location = list(fmt.offset)

        for _id, _text in enumerate(text_list):
            cv2.putText(text_img, _text, tuple(location), font, size, color, fmt.thickness, line_type)
            location[1] += text_heights[_id] + 5

        if text_width < img_stacked.shape[1]:
            text_img = resize_ar(text_img, width=img_stacked.shape[1], height=text_height,
                                 only_border=2, placement_type=1)

        border_img = np.full((5, img_stacked.shape[1], 3), 255, dtype=np.uint8)

        img_list_txt = [text_img, border_img, img_stacked]

        img_stacked = stack_images_with_resize(img_list_txt, grid_size=(-1, 1), preserve_order=1)

    return img_stacked


def get_video_out(video_out_dict, vis_out_fnames, vis_type, vis_video, save_h, save_w, fourcc, fps):
    all_video_out = video_out_dict[vis_type]
    if all_video_out is None:
        vis_out_fname = vis_out_fnames[vis_type]
        _save_dir = os.path.dirname(vis_out_fname)

        if _save_dir and not os.path.isdir(_save_dir):
            os.makedirs(_save_dir)

        if vis_video:
            video_h, video_w = save_h, save_w
            all_video_out = cv2.VideoWriter(vis_out_fname, fourcc, fps, (video_w, video_h))
        else:
            all_video_out = ImageSequenceWriter(vis_out_fname, verbose=0)

        if not all_video_out:
            raise AssertionError(
                f'video file: {vis_out_fname} could not be opened for writing')

        video_out_dict[vis_type] = all_video_out
    return all_video_out


def draw_and_concat(src_img, frame_det_data, frame_gt_data, class_name_to_col, vis_alpha, vis_w, vis_h,
                    vert_stack, check_det, img_id, mask=True, return_list=False, cls_cat_to_col=None):
    dets_vis_img, resize_factor, _, _ = resize_ar_tf_api(src_img, vis_w, vis_h, crop=1, return_factors=1)
    gt_vis_img = resize_ar_tf_api(src_img, vis_w, vis_h, crop=1, return_factors=0)

    dets_vis_img = draw_objs(dets_vis_img, frame_det_data, vis_alpha, class_name_to_col, check_bb=check_det,
                             thickness=1, mask=mask, bb_resize=resize_factor, cls_cat_to_col=cls_cat_to_col)
    gt_vis_img = draw_objs(gt_vis_img, frame_gt_data, vis_alpha, class_name_to_col, thickness=1, mask=mask,
                           bb_resize=resize_factor, cls_cat_to_col=cls_cat_to_col)

    if return_list:
        return [gt_vis_img, dets_vis_img]

    cat_img_vis = annotate((gt_vis_img, dets_vis_img), text=f'{img_id}', img_labels=['GT', 'Detections'],
                           grid_size=(-1, 1) if vert_stack else (1, -1))

    # cv2.imshow('dets_vis_img', dets_vis_img)
    # cv2.imshow('gt_vis_img', gt_vis_img)
    # cv2.imshow('cat_img_vis', cat_img_vis)
    # cv2.waitKey(0)

    # cat_img_vis = np.concatenate((gt_vis_img, dets_vis_img), axis=0 if vert_stack else 1)

    return cat_img_vis


def stack_images(img_list, grid_size=None, stack_order=0, borderless=1,
                 preserve_order=0, return_idx=0,
                 only_height=0, placement_type=0):
    n_images = len(img_list)

    if grid_size is None or not grid_size:
        n_cols = n_rows = int(np.ceil(np.sqrt(n_images)))
    else:
        n_rows, n_cols = grid_size

        if n_rows < 0:
            n_rows = int(np.ceil(n_images / n_cols))
        elif n_cols < 0:
            n_cols = int(np.ceil(n_images / n_rows))

    target_ar = 1920.0 / 1080.0
    if n_cols <= n_rows:
        target_ar /= 2.0
    shape_img_id = 0
    min_ar_diff = np.inf
    img_heights = np.zeros((n_images,), dtype=np.int32)
    for _img_id in range(n_images):
        height, width = img_list[_img_id].shape[:2]
        img_heights[_img_id] = height
        img_ar = float(n_cols * width) / float(n_rows * height)
        ar_diff = abs(img_ar - target_ar)
        if ar_diff < min_ar_diff:
            min_ar_diff = ar_diff
            shape_img_id = _img_id

    img_heights_sort_idx = np.argsort(-img_heights)
    row_start_idx = img_heights_sort_idx[:n_rows]
    img_idx = img_heights_sort_idx[n_rows:]
    img_size = img_list[shape_img_id].shape
    height, width = img_size[:2]

    if only_height:
        width = 0

    stacked_img = None
    list_ended = False
    img_idx_id = 0
    inner_axis = 1 - stack_order
    stack_idx = []
    stack_locations = []
    start_row = 0
    # curr_ann = ''
    for row_id in range(n_rows):
        start_id = n_cols * row_id
        curr_row = None
        start_col = 0
        for col_id in range(n_cols):
            img_id = start_id + col_id
            if img_id >= n_images:
                curr_img = np.zeros(img_size, dtype=np.uint8)
                list_ended = True
            else:
                if preserve_order:
                    _curr_img_id = img_id
                elif col_id == 0:
                    _curr_img_id = row_start_idx[row_id]
                else:
                    _curr_img_id = img_idx[img_idx_id]
                    img_idx_id += 1

                curr_img = img_list[_curr_img_id]
                stack_idx.append(_curr_img_id)
                if not borderless:
                    curr_img = resize_ar(curr_img, width, height)
                if img_id == n_images - 1:
                    list_ended = True
            if curr_row is None:
                curr_row = curr_img
            else:
                if borderless:
                    if curr_row.shape[0] < curr_img.shape[0]:
                        curr_row = resize_ar(curr_row, 0, curr_img.shape[0])
                    elif curr_img.shape[0] < curr_row.shape[0]:
                        curr_img = resize_ar(curr_img, 0, curr_row.shape[0])
                curr_row = np.concatenate((curr_row, curr_img), axis=inner_axis)

            curr_h, curr_w = curr_img.shape[:2]
            stack_locations.append((start_row, start_col, start_row + curr_h, start_col + curr_w))
            start_col += curr_w

        if stacked_img is None:
            stacked_img = curr_row
        else:
            if borderless:
                resize_factor = float(curr_row.shape[1]) / float(stacked_img.shape[1])
                if curr_row.shape[1] < stacked_img.shape[1]:
                    curr_row = resize_ar(curr_row, stacked_img.shape[1], 0, placement_type=placement_type)
                elif curr_row.shape[1] > stacked_img.shape[1]:
                    stacked_img = resize_ar(stacked_img, curr_row.shape[1], 0)

                new_start_col = 0
                for _i in range(n_cols):
                    _start_row, _start_col, _end_row, _end_col = stack_locations[_i - n_cols]
                    _w, _h = _end_col - _start_col, _end_row - _start_row
                    w_resized, h_resized = _w / resize_factor, _h / resize_factor
                    stack_locations[_i - n_cols] = (
                        _start_row, new_start_col, _start_row + h_resized, new_start_col + w_resized)
                    new_start_col += w_resized
            stacked_img = np.concatenate((stacked_img, curr_row), axis=stack_order)

        curr_h, curr_w = curr_row.shape[:2]
        start_row += curr_h

        if list_ended:
            break
    if return_idx:
        return stacked_img, stack_idx, stack_locations
    else:
        return stacked_img


def resize_ar_video(src_vid, **kwargs):
    out_imgs = [resize_ar(img, **kwargs) for img in src_vid]
    out_vid = np.stack(out_imgs, axis=0)
    return out_vid


def resize_ar(src_img, width=0, height=0, return_factors=False,
              placement_type=1, only_border=0, only_shrink=0, strict=False, white_bkg=0):
    src_height, src_width = src_img.shape[:2]
    src_aspect_ratio = float(src_width) / float(src_height)

    if len(src_img.shape) == 3:
        n_channels = src_img.shape[2]
    else:
        n_channels = 1

    if width <= 0 and height <= 0:
        raise AssertionError('Both width and height cannot be zero')
    elif height <= 0:
        if only_shrink and width > src_width:
            width = src_width
        if only_border:
            height = src_height
        else:
            height = int(width / src_aspect_ratio)
    elif width <= 0:
        if only_shrink and height > src_height:
            height = src_height
        if only_border:
            width = src_width
        else:
            width = int(height * src_aspect_ratio)

    aspect_ratio = float(width) / float(height)

    if strict:
        assert aspect_ratio == src_aspect_ratio, "aspect_ratio mismatch"

    if only_border:
        dst_width = width
        dst_height = height
        if placement_type == 0:
            start_row = start_col = 0
        elif placement_type == 1:
            start_row = int((dst_height - src_height) / 2.0)
            start_col = int((dst_width - src_width) / 2.0)
        elif placement_type == 2:
            start_row = int(dst_height - src_height)
            start_col = int(dst_width - src_width)
        else:
            raise AssertionError('Invalid placement_type: {}'.format(placement_type))
    else:

        if src_aspect_ratio == aspect_ratio:
            dst_width = src_width
            dst_height = src_height
            start_row = start_col = 0
        elif src_aspect_ratio > aspect_ratio:
            dst_width = src_width
            dst_height = int(src_width / aspect_ratio)
            start_row = int((dst_height - src_height) / 2.0)
            if placement_type == 0:
                start_row = 0
            elif placement_type == 1:
                start_row = int((dst_height - src_height) / 2.0)
            elif placement_type == 2:
                start_row = int(dst_height - src_height)
            else:
                raise AssertionError('Invalid placement_type: {}'.format(placement_type))
            start_col = 0
        else:
            dst_height = src_height
            dst_width = int(src_height * aspect_ratio)
            start_col = int((dst_width - src_width) / 2.0)
            if placement_type == 0:
                start_col = 0
            elif placement_type == 1:
                start_col = int((dst_width - src_width) / 2.0)
            elif placement_type == 2:
                start_col = int(dst_width - src_width)
            else:
                raise AssertionError('Invalid placement_type: {}'.format(placement_type))
            start_row = 0

    if white_bkg:
        dst_img = np.full((dst_height, dst_width, n_channels),
                          255 if src_img.dtype == np.uint8 else 1.0,
                          dtype=src_img.dtype)
    else:
        dst_img = np.zeros((dst_height, dst_width, n_channels), dtype=src_img.dtype)
    dst_img = dst_img.squeeze()

    dst_img[start_row:start_row + src_height, start_col:start_col + src_width, ...] = src_img
    if not only_border:
        dst_img = cv2.resize(dst_img, (width, height))

    if return_factors:
        resize_factor = float(height) / float(dst_height)
        return dst_img, resize_factor, start_row, start_col
    else:
        return dst_img


class ImageSequenceWriter:
    def __init__(self, file_path, fmt='image%06d', ext='jpg', logger=None, height=0, width=0, verbose=1):
        self._file_path = file_path
        self._logger = logger
        self._fmt = fmt
        self._ext = ext
        self._height = height
        self._width = width
        self._verbose = verbose

        split_path = os.path.splitext(file_path)
        self._save_dir = split_path[0]

        if not self._ext:
            try:
                self._ext = split_path[1][1:]
            except IndexError:
                self._ext = 'jpg'
            if not self._ext:
                self._ext = 'jpg'

        if not os.path.isdir(self._save_dir):
            os.makedirs(self._save_dir)
        self.frame_id = 0
        self.filename = self._fmt % self.frame_id + '.{}'.format(self._ext)
        if self._logger is None:
            self._logger = print
        else:
            self._logger = self._logger.info
        if self._verbose:
            self._logger('Saving images of type {:s} to {:s}\n'.format(self._ext, self._save_dir))

    def write(self, frame, frame_id=None, prefix=''):
        if self._height or self._width:
            frame = resize_ar(frame, height=self._height, width=self._width)

        if frame_id is None:
            self.frame_id += 1
        else:
            self.frame_id = frame_id

        if prefix:
            self.filename = '{:s}.{:s}'.format(prefix, self._ext)
        else:
            self.filename = self._fmt % self.frame_id + '.{}'.format(self._ext)

        self.curr_file_path = os.path.join(self._save_dir, self.filename)

        cv2.imwrite(self.curr_file_path, frame, (cv2.IMWRITE_JPEG_QUALITY, 100))

    def release(self):
        pass


class CVText:
    def __init__(self, color='white', bkg_color='black', location=0, font=5,
                 size=0.8, thickness=1, line_type=2, offset=(5, 25)):
        self.color = color
        self.bkg_color = bkg_color
        self.location = location
        self.font = font
        self.size = size
        self.thickness = thickness
        self.line_type = line_type
        self.offset = offset

        self.help = {
            'font': 'Available fonts: '
                    '0: cv2.FONT_HERSHEY_SIMPLEX, '
                    '1: cv2.FONT_HERSHEY_PLAIN, '
                    '2: cv2.FONT_HERSHEY_DUPLEX, '
                    '3: cv2.FONT_HERSHEY_COMPLEX, '
                    '4: cv2.FONT_HERSHEY_TRIPLEX, '
                    '5: cv2.FONT_HERSHEY_COMPLEX_SMALL, '
                    '6: cv2.FONT_HERSHEY_SCRIPT_SIMPLEX ,'
                    '7: cv2.FONT_HERSHEY_SCRIPT_COMPLEX; ',
            'location': '0: top left, 1: top right, 2: bottom right, 3: bottom left; ',
            'bkg_color': 'should be empty for no background',
        }


def modules_from_trace(call_stack, n_modules, start_module=1):
    """

    :param list[traceback.FrameSummary] call_stack:
    :param int n_modules:
    :param int start_module:
    :return:
    """
    call_stack = call_stack[::-1]

    modules = []

    for module_id in range(start_module, start_module + n_modules):
        module_fs = call_stack[module_id]
        file = os.path.splitext(os.path.basename(module_fs.filename))[0]
        line = module_fs.lineno
        func = module_fs.name

        modules.append('{}:{}:{}'.format(file, func, line))

    modules_str = '<'.join(modules)
    return modules_str


class CVConstants:
    similarity_types = {
        0: cv2.TM_CCOEFF_NORMED,
        1: cv2.TM_SQDIFF_NORMED,
        2: cv2.TM_CCORR_NORMED,
        3: cv2.TM_CCOEFF,
        4: cv2.TM_SQDIFF,
        5: cv2.TM_CCORR
    }
    interp_types = {
        0: cv2.INTER_NEAREST,
        1: cv2.INTER_LINEAR,
        2: cv2.INTER_AREA,
        3: cv2.INTER_CUBIC,
        4: cv2.INTER_LANCZOS4
    }
    fonts = {
        0: cv2.FONT_HERSHEY_SIMPLEX,
        1: cv2.FONT_HERSHEY_PLAIN,
        2: cv2.FONT_HERSHEY_DUPLEX,
        3: cv2.FONT_HERSHEY_COMPLEX,
        4: cv2.FONT_HERSHEY_TRIPLEX,
        5: cv2.FONT_HERSHEY_COMPLEX_SMALL,
        6: cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        7: cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    }
    line_types = {
        0: cv2.LINE_4,
        1: cv2.LINE_8,
        2: cv2.LINE_AA,
    }


def stack_images_with_resize(img_list, grid_size=None, stack_order=0, borderless=1,
                             preserve_order=0, return_idx=0,
                             # annotations=None,
                             # ann_fmt=(0, 5, 15, 1, 1, 255, 255, 255, 0, 0, 0),
                             only_height=0, only_border=1):
    n_images = len(img_list)
    # print('grid_size: {}'.format(grid_size))

    if grid_size is None:
        n_cols = n_rows = int(np.ceil(np.sqrt(n_images)))
    else:
        n_rows, n_cols = grid_size

        if n_rows < 0:
            n_rows = int(np.ceil(n_images / n_cols))
        elif n_cols < 0:
            n_cols = int(np.ceil(n_images / n_rows))

    target_ar = 1920.0 / 1080.0
    if n_cols <= n_rows:
        target_ar /= 2.0
    shape_img_id = 0
    min_ar_diff = np.inf
    img_heights = np.zeros((n_images,), dtype=np.int32)
    for _img_id in range(n_images):
        height, width = img_list[_img_id].shape[:2]
        img_heights[_img_id] = height
        img_ar = float(n_cols * width) / float(n_rows * height)
        ar_diff = abs(img_ar - target_ar)
        if ar_diff < min_ar_diff:
            min_ar_diff = ar_diff
            shape_img_id = _img_id

    img_heights_sort_idx = np.argsort(-img_heights)
    row_start_idx = img_heights_sort_idx[:n_rows]
    img_idx = img_heights_sort_idx[n_rows:]
    # print('img_heights: {}'.format(img_heights))
    # print('img_heights_sort_idx: {}'.format(img_heights_sort_idx))
    # print('img_idx: {}'.format(img_idx))

    # grid_size = [n_rows, n_cols]
    img_size = img_list[shape_img_id].shape
    height, width = img_size[:2]

    if only_height:
        width = 0
    # grid_size = [n_rows, n_cols]
    # print 'img_size: ', img_size
    # print 'n_images: ', n_images
    # print 'grid_size: ', grid_size

    # print()
    stacked_img = None
    list_ended = False
    img_idx_id = 0
    inner_axis = 1 - stack_order
    stack_idx = []
    stack_locations = []
    start_row = 0
    # curr_ann = ''
    for row_id in range(n_rows):
        start_id = n_cols * row_id
        curr_row = None
        start_col = 0
        for col_id in range(n_cols):
            img_id = start_id + col_id
            if img_id >= n_images:
                curr_img = np.zeros(img_size, dtype=np.uint8)
                list_ended = True
            else:
                if preserve_order:
                    _curr_img_id = img_id
                elif col_id == 0:
                    _curr_img_id = row_start_idx[row_id]
                else:
                    _curr_img_id = img_idx[img_idx_id]
                    img_idx_id += 1

                curr_img = img_list[_curr_img_id]
                # if annotations:
                #     curr_ann = annotations[_curr_img_id]
                stack_idx.append(_curr_img_id)
                # print(curr_img.shape[:2])

                # if curr_ann:
                #     putTextWithBackground(curr_img, curr_ann, fmt=ann_fmt)

                if not borderless:
                    curr_img = resize_ar(curr_img, width, height, only_border=only_border)
                if img_id == n_images - 1:
                    list_ended = True
            if curr_row is None:
                curr_row = curr_img
            else:
                if borderless:
                    if curr_row.shape[0] < curr_img.shape[0]:
                        curr_row = resize_ar(curr_row, 0, curr_img.shape[0], only_border=only_border)
                    elif curr_img.shape[0] < curr_row.shape[0]:
                        curr_img = resize_ar(curr_img, 0, curr_row.shape[0], only_border=only_border)
                # print('curr_row.shape: ', curr_row.shape)
                # print('curr_img.shape: ', curr_img.shape)
                curr_row = np.concatenate((curr_row, curr_img), axis=inner_axis)

            curr_h, curr_w = curr_img.shape[:2]
            stack_locations.append((start_row, start_col, start_row + curr_h, start_col + curr_w))
            start_col += curr_w

        if stacked_img is None:
            stacked_img = curr_row
        else:
            if borderless:
                resize_factor = float(curr_row.shape[1]) / float(stacked_img.shape[1])
                if curr_row.shape[1] < stacked_img.shape[1]:
                    curr_row = resize_ar(curr_row, stacked_img.shape[1], 0, only_border=only_border)
                elif curr_row.shape[1] > stacked_img.shape[1]:
                    stacked_img = resize_ar(stacked_img, curr_row.shape[1], 0, only_border=only_border)

                new_start_col = 0
                for _i in range(n_cols):
                    _start_row, _start_col, _end_row, _end_col = stack_locations[_i - n_cols]
                    _w, _h = _end_col - _start_col, _end_row - _start_row
                    w_resized, h_resized = _w / resize_factor, _h / resize_factor
                    stack_locations[_i - n_cols] = (
                        _start_row, new_start_col, _start_row + h_resized, new_start_col + w_resized)
                    new_start_col += w_resized
            # print('curr_row.shape: ', curr_row.shape)
            # print('stacked_img.shape: ', stacked_img.shape)
            stacked_img = np.concatenate((stacked_img, curr_row), axis=stack_order)

        curr_h, curr_w = curr_row.shape[:2]
        start_row += curr_h

        if list_ended:
            break
    if return_idx:
        return stacked_img, stack_idx, stack_locations
    else:
        return stacked_img


def annotate_and_show(title, img_list, text=None, pause=1,
                      fmt=CVText(), no_resize=1, grid_size=(-1, 1), n_modules=3,
                      use_plt=0, max_width=0, max_height=0, only_annotate=0):
    """

    :param str title:
    :param np.ndarray | list | tuple img_list:
    :param str | logging.RootLogger | CustomLogger text:
    :param int pause:
    :param CVText fmt:
    :param int no_resize:
    :param int n_modules:
    :param int use_plt:
    :param tuple(int) grid_size:
    :return:
    """

    # call_stack = traceback.format_stack()
    # print(pformat(call_stack))
    # for line in traceback.format_stack():
    #     print(line.strip())

    if isinstance(text, logging.RootLogger):
        string_stream = [k.stream for k in text.handlers if isinstance(k.stream, StringIO)]
        assert string_stream, "No string streams in logger"
        _str = string_stream[0].getvalue()
        _str_list = _str.split('\n')[-2].split('  :::  ')

        if n_modules:
            modules_str = modules_from_trace(traceback.extract_stack(), n_modules - 1, start_module=2)
            _str_list[0] = '{} ({})'.format(_str_list[0], modules_str)
        text = '\n'.join(_str_list)
    else:
        if n_modules:
            modules_str = modules_from_trace(traceback.extract_stack(), n_modules, start_module=1)
            if text is None:
                text = modules_str
            else:
                text = '{}\n({})'.format(text, modules_str)
        else:
            if text is None:
                text = title

    if not isinstance(img_list, (list, tuple)):
        img_list = [img_list, ]

    size = fmt.size

    # print('self.size: {}'.format(self.size))

    color = col_bgr[fmt.color]
    font = CVConstants.fonts[fmt.font]
    line_type = CVConstants.line_types[fmt.line_type]

    location = list(fmt.offset)

    if '\n' in text:
        text_list = text.split('\n')
    else:
        text_list = [text, ]

    max_text_width = 0
    text_height = 0
    text_heights = []

    for _text in text_list:
        (_text_width, _text_height) = cv2.getTextSize(_text, font, fontScale=fmt.size, thickness=fmt.thickness)[0]
        if _text_width > max_text_width:
            max_text_width = _text_width
        text_height += _text_height + 5
        text_heights.append(_text_height)

    text_width = max_text_width + 10
    text_height += 30

    text_img = np.zeros((text_height, text_width), dtype=np.uint8)
    for _id, _text in enumerate(text_list):
        cv2.putText(text_img, _text, tuple(location), font, size, color, fmt.thickness, line_type)
        location[1] += text_heights[_id] + 5

    text_img = text_img.astype(np.float32) / 255.0

    text_img = np.stack([text_img, ] * 3, axis=2)

    for _id, _img in enumerate(img_list):
        if len(_img.shape) == 2:
            _img = np.stack([_img, ] * 3, axis=2)
        if _img.dtype == np.uint8:
            _img = _img.astype(np.float32) / 255.0
        img_list[_id] = _img

    img_stacked = stack_images_with_resize(img_list, grid_size=grid_size, preserve_order=1,
                                           only_border=no_resize)
    img_list_txt = [text_img, img_stacked]

    img_stacked_txt = stack_images_with_resize(img_list_txt, grid_size=(2, 1), preserve_order=1,
                                               only_border=no_resize)
    # img_stacked_txt_res = cv2.resize(img_stacked_txt, (300, 300), fx=0, fy=0)
    # img_stacked_txt_res_gs = cv2.cvtColor(img_stacked_txt_res, cv2.COLOR_BGR2GRAY)

    img_stacked_txt = (img_stacked_txt * 255).astype(np.uint8)

    if img_stacked_txt.shape[0] > max_height > 0:
        img_stacked_txt = resize_ar(img_stacked_txt, height=max_height)

    if img_stacked_txt.shape[1] > max_width > 0:
        img_stacked_txt = resize_ar(img_stacked_txt, width=max_width)

    if only_annotate:
        return img_stacked_txt

    if use_plt:
        img_stacked_txt = cv2.resize(img_stacked_txt, (300, 300), fx=0, fy=0)
        plt.imshow(img_stacked_txt)
        plt.pause(0.0001)
    else:
        cv2.imshow(title, img_stacked_txt)
        k = cv2.waitKey(pause)
        if k == 27:
            cv2.destroyWindow(title)
            exit()
        if k == 32:
            pause = 1 - pause

    return pause


from contextlib import contextmanager
import time


@contextmanager
def profile(_id, _times=None, _rel_times=None, enable=1, show=1, _fps=None):
    """

    :param _id:
    :param dict _times:
    :param int enable:
    :return:
    """
    if not enable:
        yield None

    else:
        start_t = time.time()
        yield None
        end_t = time.time()
        _time = end_t - start_t

        if show:
            print(f'{_id} :: {_time}')

        if _fps is not None:
            if _time > 0:
                _fps[_id] = 1.0 / _time
            else:
                _fps[_id] = np.inf

        if _times is not None:

            _times[_id] = _time

            total_time = np.sum(list(_times.values()))

            if _rel_times is not None:

                for __id in _times:
                    rel__time = _times[__id] / total_time
                    _rel_times[__id] = rel__time


def get_vis_size(src_img, mult, save_w, save_h, bottom_border):
    temp_vis = np.concatenate((src_img,) * mult, axis=1)
    temp_vis_res = resize_ar_tf_api(temp_vis, save_w, save_h - bottom_border, crop=1)
    temp_vis_h, temp_vis_w = temp_vis_res.shape[:2]

    vis_h, vis_w = temp_vis_h, int(temp_vis_w / mult)

    assert vis_h <= save_h and vis_w <= save_w, \
        f"vis size {vis_w} x {vis_h} > save size {save_w} x {save_h}"

    return vis_h, vis_w


def print_with_time(*argv):
    time_stamp = datetime.now().strftime("%y%m%d %H%M%S")
    print(f'{time_stamp}:', *argv)


def to_str(iter_, sep='\n'):
    return sep.join(iter_)


def num_to_words(num):
    if num >= 1e12:
        num_tril = num / 1e12
        words = f'{num_tril:.1f}T'
    elif num >= 1e9:
        num_bil = num / 1e9
        words = f'{num_bil:.1f}B'
    elif num >= 1e6:
        num_mil = num / 1e6
        words = f'{num_mil:.1f}M'
    elif num >= 1e3:
        num_th = num / 1e3
        words = f'{num_th:.1f}K'
    else:
        words = f'{num}'
    return words


def dets_to_imagenet_vid(seq_det_bboxes_list, imagenet_vid_out_path, seq_name,
                         filename_to_frame_index, class_name_to_id):
    imagenet_vid_rows = []
    for det_bbox in seq_det_bboxes_list:
        xmin_, ymin_, xmax_, ymax_ = det_bbox['bbox']
        filename_ = seq_name + '/' + os.path.splitext(os.path.basename(det_bbox['filename']))[0]
        class_name = det_bbox['class']
        confidence_ = float(det_bbox['confidence'])

        frame_index = int(filename_to_frame_index[filename_])
        class_index = int(class_name_to_id[class_name])

        obj_str = (f'{frame_index:d} {class_index:d} {confidence_:.4f} '
                   f'{xmin_:.2f} {ymin_:.2f} {xmax_:.2f} {ymax_:.2f}')

        imagenet_vid_rows.append(obj_str)

    with open(imagenet_vid_out_path, "a") as fid:
        fid.write('\n'.join(imagenet_vid_rows))


def dets_to_csv(seq_det_bboxes_list, det_path, enable_mask,
                vid_nms_thresh, nms_thresh, class_agnostic):
    out_csv_rows = []
    for det_bbox in seq_det_bboxes_list:
        xmin_, ymin_, xmax_, ymax_ = det_bbox['bbox']
        csv_row = {
            "ImageID": det_bbox['filename'],
            "LabelName": det_bbox['class'],
            "XMin": xmin_,
            "XMax": xmax_,
            "YMin": ymin_,
            "YMax": ymax_,
            "Confidence": det_bbox['confidence'],
        }
        if enable_mask:
            mask_rle = det_bbox['mask']
            mask_h_, mask_w_ = mask_rle['size']
            csv_row.update(
                {
                    "mask_w": mask_w_,
                    "mask_h": mask_h_,
                    "mask_counts": mask_rle['counts'],
                }
            )
        out_csv_rows.append(csv_row)
    det_dir, det_name = os.path.dirname(det_path), os.path.basename(det_path)

    out_suffix = []
    if vid_nms_thresh > 0:
        out_suffix.append(f'vnms_{vid_nms_thresh:02d}')

    if nms_thresh > 0:
        out_suffix.append(f'nms_{nms_thresh:02d}')

    if class_agnostic > 0:
        out_suffix.append(f'agn')

    if out_suffix:
        out_suffix_str = '_'.join(out_suffix)
        out_det_dir = add_suffix(det_dir, out_suffix_str)

    os.makedirs(out_det_dir, exist_ok=True)
    out_det_path = linux_path(out_det_dir, det_name)
    csv_columns = [
        "ImageID", "LabelName",
        "XMin", "XMax", "YMin", "YMax", "Confidence",
        'VideoID'
    ]
    if enable_mask:
        csv_columns += ['mask_w', 'mask_h', 'mask_counts']
    df = pd.DataFrame(out_csv_rows, columns=csv_columns)
    print_(f'writing postproc results to {out_det_path}')
    df.to_csv(out_det_path, index=False)


def find_matching_obj_pairs(pred_obj_pairs, enable_mask, nms_thresh,
                            # objs_to_delete=None, global_objs_to_delete=None
                            ):
    # if objs_to_delete is None:
    #     objs_to_delete = []
    #     assert global_objs_to_delete is None, "either both or neither of objs_to_delete must be None"
    #     global_objs_to_delete = []

    n_del = 0

    for pair_id, pred_obj_pair in enumerate(pred_obj_pairs):
        obj1, obj2 = pred_obj_pair

        # local_id1, bbox1, mask1, score1, label1, vid_id1, global_id1 = obj1
        # local_id2, bbox2, mask2, score2, label2, vid_id2, global_id2 = obj2

        assert obj1['local_id'] != obj2['local_id'], "invalid obj pair with identical local IDs"

        if obj1['to_delete'] or obj2['to_delete']:
            continue

        if enable_mask:
            # pred_iou = get_mask_iou(obj1['mask'], obj2['mask'], obj1['bbox'], obj2['bbox'])
            pred_iou = get_mask_rle_iou(obj1['mask'], obj2['mask']) * 100
        else:
            pred_iou = get_iou(obj1['bbox'], obj2['bbox'], xywh=False) * 100

            # mask_iou2 = get_mask_iou(mask1, mask2, bbox1, bbox2, giou=False)
            # assert pred_iou == mask_iou2, "mask_iou2 mismatch found"

        if pred_iou >= nms_thresh:
            n_del += 1
            # print(f'found matching object pair with iou {pred_iou:.3f}')
            if obj1['confidence'] > obj2['confidence']:
                obj2['to_delete'] = 1

                # objs_to_delete.append(obj2['local_id'])
                # global_objs_to_delete.append(obj2['global_id'])

                # print(f'removing obj {local_id2} with score {score2} < {score1}')
            else:
                obj1['to_delete'] = 1

                # objs_to_delete.append(obj1['local_id'])
                # global_objs_to_delete.append(obj1['global_id'])

                # print(f'removing obj {local_id1} with score {score1} < {score2}')
    # return objs_to_delete, global_objs_to_delete
    return n_del


def perform_batch_nms(objs, enable_mask, nms_thresh_all, vid_nms_thresh_all, dup, vis, **kwargs):
    assert not enable_mask, "mask IOU is currently not supported in batch_nms"
    assert nms_thresh_all or vid_nms_thresh_all, "either vid_nms_thresh_all or nms_thresh_all must be provided"

    n_objs = len(objs)
    obj_bboxes_arr = np.asarray([obj['bbox'] for obj in objs])
    iou_arr = np.empty((n_objs, n_objs))
    compute_overlaps_multi(iou_arr, None, None, obj_bboxes_arr, obj_bboxes_arr)
    iou_arr *= 100

    all_obj_ids = [obj['local_id'] for obj in objs]
    id_to_bbox = {obj['local_id']: obj for obj in objs}
    id_to_conf = {obj['local_id']: obj['confidence'] for obj in objs}

    pred_obj_pairs = list(itertools.combinations(objs, 2))
    pred_obj_pair_ids = [(obj1['local_id'], obj2['local_id']) for obj1, obj2 in pred_obj_pairs]

    vid_pred_obj_pair_ids = None
    if vid_nms_thresh_all:
        vid_pred_obj_pair_ids = [(obj1['local_id'], obj2['local_id']) for obj1, obj2 in pred_obj_pairs
                                 if obj1['video_id'] != obj2['video_id']]
        if not dup:
            pred_obj_pair_ids = [(obj1['local_id'], obj2['local_id']) for obj1, obj2 in pred_obj_pairs
                                 if obj1['video_id'] == obj2['video_id']]
    else:
        vid_nms_thresh_all = [0, ]

    if not nms_thresh_all:
        nms_thresh_all = [0, ]

    cmb_nms_thresh = list(set(vid_nms_thresh_all + nms_thresh_all))
    is_overlapping = {}
    for nms_thresh in cmb_nms_thresh:
        if nms_thresh > 0:
            is_overlapping[nms_thresh] = iou_arr >= nms_thresh

    vid_del_obj_ids_dict = {}
    for vid_nms_thresh in vid_nms_thresh_all:
        vid_del_obj_ids = []
        if vid_nms_thresh > 0:
            is_overlapping_ = is_overlapping[vid_nms_thresh]
            vid_del_obj_ids = [id2 if id_to_conf[id1] > id_to_conf[id2] else id1
                               for id1, id2 in vid_pred_obj_pair_ids if is_overlapping_[id1, id2]]
        vid_del_obj_ids_dict[vid_nms_thresh] = vid_del_obj_ids

    del_obj_ids_dict = {}
    for nms_thresh in nms_thresh_all:
        del_obj_ids = []
        if nms_thresh > 0:
            is_overlapping_ = is_overlapping[nms_thresh]
            del_obj_ids = [id2 if id_to_conf[id1] > id_to_conf[id2] else id1
                           for id1, id2 in pred_obj_pair_ids if is_overlapping_[id1, id2]]
        del_obj_ids_dict[nms_thresh] = del_obj_ids

    thresh_to_filtered_objs = {}
    nms_thresh_pairs = itertools.product(vid_nms_thresh_all, nms_thresh_all)
    for vid_nms_thresh, nms_thresh in nms_thresh_pairs:
        # if vid_nms_thresh == 0 and nms_thresh == 0:
        #     continue
        del_obj_ids = list(set(vid_del_obj_ids_dict[vid_nms_thresh] + del_obj_ids_dict[nms_thresh]))
        keep_obj_ids = list(set(all_obj_ids) - set(del_obj_ids))
        thresh_to_filtered_objs[(vid_nms_thresh, nms_thresh)] = [id_to_bbox[i] for i in keep_obj_ids]

    if vis:
        img_paths = [obj['file_path'] for obj in objs]
        img_path = list(set(img_paths))
        assert len(img_path) == 1, "multiple  img_paths"
        img_path = img_path[0]
        img = cv2.imread(img_path)

        seq_name = os.path.basename(os.path.dirname(img_path))
        img_name = os.path.basename(img_path)

        img_vis_all = draw_objs(
            img, objs,
            title=f'{seq_name}-{img_name}-{len(objs)}',
            show_class=True, **kwargs)
        img_vis_all = resize_ar(img_vis_all, width=1800, height=1000)
        cv2.imshow('img_vis_all', img_vis_all)

        for (vid_nms_thresh, nms_thresh), filtered_objs in thresh_to_filtered_objs.items():
            if vid_nms_thresh == 0 and nms_thresh == 0:
                continue
            img_vis_filtered = draw_objs(
                img, filtered_objs,
                title=f'{seq_name}-{img_name}-{vid_nms_thresh:02d}-{nms_thresh:02d}-{len(filtered_objs)}/{len(objs)}',
                show_class=True,
                **kwargs)
            img_vis_filtered = resize_ar(img_vis_filtered, width=1800, height=1000)
            cv2.imshow('img_vis_filtered', img_vis_filtered)
            cv2.waitKey(0)

    return thresh_to_filtered_objs


def box_iou_batch(
        boxes_a: np.ndarray, boxes_b: np.ndarray
) -> np.ndarray:
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area_a = box_area(boxes_a.T)
    area_b = box_area(boxes_b.T)

    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[:, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[:, 2:])

    area_inter = np.prod(
        np.clip(bottom_right - top_left, a_min=0, a_max=None), 2)

    return area_inter / (area_a[:, None] + area_b - area_inter)


def perform_nms_fast(objs, enable_mask, iou_threshold):
    assert not enable_mask, "fast nms does not support mask IOU"
    iou_threshold /= 100.

    obj_conf_arr = np.asarray([obj['confidence'] for obj in objs])
    sort_index = np.flip(obj_conf_arr.argsort())

    boxes = np.asarray([obj['bbox'] for obj in objs])
    categories = np.asarray([obj['class_id'] for obj in objs])

    n_objs = len(objs)

    boxes = boxes[sort_index]
    categories = categories[sort_index]

    ious = box_iou_batch(boxes, boxes)
    ious = ious - np.eye(n_objs)

    keep = np.ones(n_objs, dtype=bool)

    for index, (iou, category) in enumerate(zip(ious, categories, strict=True)):
        if not keep[index]:
            continue

        condition = (iou > iou_threshold) & (categories == category)
        keep = keep & ~condition

    keep = keep[sort_index.argsort()]
    n_del = 0
    for (obj, keep_) in enumerate(zip(objs, keep, strict=True)):
        if not keep_:
            obj['to_delete'] = 1
            n_del += 1
    return n_del


def perform_nms(objs, enable_mask, nms_thresh, vid_nms_thresh, dup):
    pred_obj_pairs = list(itertools.combinations(objs, 2))

    # objs_to_delete = []
    # global_objs_to_delete = []

    n_vid_pairs = 0
    n_del = 0

    if vid_nms_thresh > 0:
        vid_pred_obj_pairs = [(obj1, obj2) for obj1, obj2 in pred_obj_pairs
                              if obj1['video_id'] != obj2['video_id']]
        n_vid_pairs = len(vid_pred_obj_pairs)

        n_match = find_matching_obj_pairs(
            vid_pred_obj_pairs, enable_mask, vid_nms_thresh,
            # objs_to_delete=objs_to_delete,
            # global_objs_to_delete=global_objs_to_delete,
        )
        n_del += n_match
        if not dup:
            pred_obj_pairs = [(obj1, obj2) for obj1, obj2 in pred_obj_pairs
                              if obj1['video_id'] == obj2['video_id']]

    n_pairs = len(pred_obj_pairs)

    if nms_thresh > 0:
        n_match = find_matching_obj_pairs(
            pred_obj_pairs, enable_mask, nms_thresh,
            # objs_to_delete=objs_to_delete,
            # global_objs_to_delete=global_objs_to_delete,
        )
        n_del += n_match

    return n_del, n_pairs, n_vid_pairs


def print_(*args, **kwargs):
    sys.stdout.write(*args, **kwargs)
    sys.stdout.write('\n')


def linux_path(*args, **kwargs):
    return os.path.join(*args, **kwargs).replace(os.sep, '/')


def load_samples_from_txt(load_paths, xml_dir_name, load_path_root='', verbose=True,
                          xml_root_dir='', root_dir=''
                          ):
    from pprint import pformat
    from collections import OrderedDict
    import ast

    seq_to_samples = OrderedDict()

    # if load_samples == '1':
    #     load_samples = 'seq_to_samples.txt'

    if load_path_root:
        load_paths = [linux_path(load_path_root, k) for k in load_paths]

    if verbose:
        print(f'\n\nLoading samples from : {load_paths}\n\n')

    for _f in load_paths:
        if os.path.isdir(_f):
            _f = linux_path(_f, 'seq_to_samples.txt')
        try:
            with open(_f, 'r') as fid:
                curr_seq_to_samples = json.load(fid)
        except json.decoder.JSONDecodeError:
            with open(_f, 'r') as fid:
                curr_seq_to_samples = ast.literal_eval(fid.read())
        for _seq in curr_seq_to_samples:
            if xml_dir_name is not None:
                _dir_img_names = [(os.path.dirname(_sample), os.path.splitext(os.path.basename(_sample))[0])
                                  for _sample in curr_seq_to_samples[_seq]]
                if xml_root_dir:
                    assert root_dir, "root_dir must be provided with xml_root_dir"
                    xml_dir_paths = [_dir_name.replace(root_dir, xml_root_dir) for _dir_name, _ in _dir_img_names]
                else:
                    xml_dir_paths = [linux_path(_dir_name, xml_dir_name) for _dir_name, _ in _dir_img_names]

                curr_seq_to_samples[_seq] = [linux_path(xml_dir_path, f'{_img_name}.xml')
                                             for xml_dir_path, (_, _img_name) in zip(
                        xml_dir_paths, _dir_img_names, strict=True)]

            if _seq in seq_to_samples:
                seq_to_samples[_seq] += curr_seq_to_samples[_seq]
            else:
                seq_to_samples[_seq] = curr_seq_to_samples[_seq]
    seq_paths = [_seq for _seq in seq_to_samples if seq_to_samples[_seq]]
    seq_to_samples = {_seq: seq_to_samples[_seq] for _seq in seq_paths}

    return seq_paths, seq_to_samples


def add_suffix_to_path(src_path, suffix):
    # abs_src_path = os.path.abspath(src_path)
    src_dir = os.path.dirname(src_path)
    src_name = os.path.basename(src_path)
    dst_path = linux_path(src_dir, suffix, src_name)
    return dst_path


def add_suffix_to_dir(src_path, suffix, sep='_'):
    # abs_src_path = os.path.abspath(src_path)
    src_dir = os.path.dirname(src_path)
    src_name = os.path.basename(src_path)
    dst_path = linux_path(src_dir + sep + suffix, src_name)
    return dst_path


def add_suffix(src_path, suffix, dst_ext='', sep='_'):
    # abs_src_path = os.path.abspath(src_path)
    src_dir = os.path.dirname(src_path)
    src_name, src_ext = os.path.splitext(os.path.basename(src_path))
    if not dst_ext:
        dst_ext = src_ext

    dst_path = linux_path(src_dir, src_name + sep + suffix + dst_ext)
    return dst_path


def compute_binary_cls_metrics(
        thresholds, probs, labels, class_names,
        fp_thresholds=(0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5),
        show_pbar=False
):
    if len(probs.shape) == 1:
        """assume only class 0 probs are provided"""
        probs = np.stack((probs, 1 - probs), axis=1)

    n_val, n_classes = probs.shape

    assert n_classes == 2, "n_classes must be 2 for binary_cls_metrics"
    assert len(class_names) == 2, "n_classes must be 2 for binary_cls_metrics"
    assert len(labels) == n_val, "n_val mismatch"

    # assert n_val > 0, "no labels found"

    n_conf_thresholds = len(thresholds)
    class_tp = np.zeros((n_conf_thresholds, 2), dtype=np.float32)
    class_fp = np.zeros((n_conf_thresholds, 2), dtype=np.float32)

    conf_to_acc = np.zeros((n_conf_thresholds, 4), dtype=np.float32)
    conf_to_acc[:, 0] = thresholds.squeeze()

    fp_tp = np.zeros((n_conf_thresholds, 3), dtype=np.float32)
    fp_tp[:, 0] = thresholds.squeeze()

    n_fp_thresholds = len(fp_thresholds) + 1
    roc_aucs = np.zeros((n_fp_thresholds, 2), dtype=np.float32)
    roc_aucs[:, 0] = list(fp_thresholds) + [1, ]

    if n_val == 0:
        return class_tp, class_fp, conf_to_acc, roc_aucs, fp_tp

    labels = labels.reshape((n_val, 1))

    thresh_iter = thresholds
    if show_pbar:
        thresh_iter = tqdm(thresholds, ncols=70)

    for conf_id, conf_threshold in enumerate(thresh_iter):
        is_class_1 = probs[:, 1] >= conf_threshold
        predictions = np.zeros((n_val, 1), dtype=np.int32)
        predictions[is_class_1] = 1

        # predictions = np.argmax(probabilities, axis=1)
        predictions = predictions.reshape((n_val, 1))

        correct_preds = np.equal(labels, predictions)
        correct_preds = correct_preds.reshape((n_val, 1)).astype(np.int32)
        n_correct = np.count_nonzero(correct_preds)

        try:
            accuracy = (n_correct / n_val)
        except ZeroDivisionError:
            accuracy = 0

        # print(f'\n\nconf_threshold: {conf_threshold:.4f}')
        # print(f'overall accuracy: {n_correct} / {n_val} ({accuracy:.4f}%)')

        conf_to_acc[conf_id, 1] = accuracy

        for class_id, class_name in enumerate(class_names):
            class_labels_mask = (labels == class_id)
            class_val_labels = labels[class_labels_mask]
            class_val_preds = predictions[class_labels_mask]

            n_class_val = len(class_val_labels)

            class_correct_preds = np.equal(class_val_labels, class_val_preds)
            class_correct_preds = class_correct_preds.reshape((n_class_val, 1)).astype(np.int32)
            n_class_correct = np.count_nonzero(class_correct_preds)
            n_class_incorrect = n_class_val - n_class_correct

            try:
                class_tp_ = class_accuracy = (n_class_correct / n_class_val)
            except ZeroDivisionError:
                class_tp_ = class_accuracy = 0

            try:
                non_class_fp_ = (n_class_incorrect / n_class_val)
            except ZeroDivisionError:
                non_class_fp_ = 0

            # print(f'\tclass {class_name}: {n_class_correct} / {n_class_val} ({class_accuracy:.4f}%)')

            conf_to_acc[conf_id, class_id + 2] = class_accuracy
            class_tp[conf_id, class_id] = class_tp_
            class_fp[conf_id, 1 - class_id] = non_class_fp_

    y_score = probs[:, 0].reshape((n_val,))
    y_true = (1 - labels).reshape((n_val,))

    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    fp_tp = np.stack((thresholds, tpr, fpr), axis=1)

    # roc_auc3 = auc(class_fp[:, 0], class_tp[:, 0])

    roc_aucs = []

    tpr_interp = np.interp(fp_thresholds, fpr, tpr)

    for fp_id, fp_threshold in enumerate(fp_thresholds):
        min_fp_idx = np.nonzero(fpr <= fp_threshold)[0]
        if len(min_fp_idx) == 0:
            roc_auc = 0
        else:
            _fpr = list(fpr[min_fp_idx])
            _tpr = list(tpr[min_fp_idx])

            max_fp = np.amax(_fpr)

            if max_fp < fp_threshold:
                _fpr.append(fp_threshold)
                _tpr.append(tpr_interp[fp_id])

            _fpr = np.asarray(_fpr)
            _tpr = np.asarray(_tpr)

            max_fp = np.amax(_fpr)

            assert max_fp == fp_threshold, "fp_threshold does not match max_fp"

            if len(_fpr) < 2:
                roc_auc = 0
            else:
                if np.all(_fpr == 0):
                    roc_auc = np.amax(_tpr).item()
                    print()
                else:
                    roc_auc = auc(_fpr, _tpr)
                    roc_auc /= fp_threshold

                    if fp_id > 0:
                        roc_auc_prev = np.round(roc_aucs[fp_id - 1][1], 3)
                        roc_auc_curr = np.round(roc_auc, 3)

                        if roc_auc_prev > roc_auc_curr:
                            print('decreasing roc_auc')

            # if roc_auc == 0:
            #     print('zero roc_auc')

            # print(f'fp_threshold {fp_threshold * 100:.2f} : {roc_auc * 100:.2f}')

        roc_aucs.append((fp_threshold, roc_auc))

    if np.isnan(fpr).all() or np.isnan(tpr).all():
        roc_auc = 0
    else:
        roc_auc = auc(fpr, tpr)
    # print(f'fp_threshold {100:.2f} : {roc_auc * 100:.2f}')

    roc_aucs.append((1, roc_auc))
    roc_aucs = np.asarray(roc_aucs, dtype=np.float32)

    return class_tp, class_fp, conf_to_acc, roc_aucs, fp_tp


def get_max_tp(tp, fp, min_fp=0, max_fp=0.01):
    fp_thresholds = np.linspace(min_fp, max_fp, 101)

    n_fp_thresholds = len(fp_thresholds)
    max_tp = np.zeros((n_fp_thresholds, 2), dtype=np.float32)
    for fp_id, fp_threshold in enumerate(fp_thresholds):
        min_fp_idx = np.nonzero(fp <= fp_threshold)[0]
        if min_fp_idx.size > 0:
            max_tp_at_min_fp = np.amax(tp[min_fp_idx])
        else:
            max_tp_at_min_fp = 0
        max_tp[fp_id, :] = (fp_threshold, max_tp_at_min_fp)

    return max_tp


def norm_auc(x_list, y_list):
    nolist = 0
    if not isinstance(x_list, list):
        x_list = [x_list, ]
        y_list = [y_list, ]
        nolist = 1

    norm_aucs = []
    for x, y in zip(x_list, y_list):
        max_x = np.amax(x)

        if max_x > 0:
            norm_factor = 1.0 / max_x
        else:
            norm_factor = 1.0

        raw_auc = auc(x, y)
        norm_auc = raw_auc * norm_factor

        norm_aucs.append(norm_auc)

    if nolist:
        return norm_aucs[0]

    return norm_aucs


def sortKey(fname):
    fname = os.path.splitext(os.path.basename(fname))[0]
    # print('fname: ', fname)
    # split_fname = fname.split('_')
    # print('split_fname: ', split_fname)

    # nums = [int(s) for s in fname.split('_') if s.isdigit()]
    # non_nums = [s for s in fname.split('_') if not s.isdigit()]

    split_list = fname.split('_')
    key = ''

    for s in split_list:
        if s.isdigit():
            if not key:
                key = '{:08d}'.format(int(s))
            else:
                key = '{}_{:08d}'.format(key, int(s))
        else:
            if not key:
                key = s
            else:
                key = '{}_{}'.format(key, s)

    return key


def arr_to_csv(arr, csv_columns, out_dir, out_fname):
    csv_df = pd.DataFrame(data=arr, columns=csv_columns)
    out_fname_csv = linux_path(out_dir, out_fname)
    csv_df.to_csv(out_fname_csv, columns=csv_columns, index=False, sep='\t')


def binary_cls_metrics(
        class_stats,
        tp_sum_thresh_all,
        fp_sum_thresh_all,
        fp_cls_sum_thresh_all,
        score_thresholds,
        gt_classes,
        out_root_dir,
        misc_out_root_dir,
        eval_result_dict,
        enable_tests=True,
        verbose=True,
):
    assert len(gt_classes) == 2, "Number of classes must be 2"

    stats_0 = class_stats[0]
    stats_1 = class_stats[1]

    n_dets_0 = stats_0['n_dets']
    n_dets_1 = stats_1['n_dets']
    n_dets = n_dets_0 + n_dets_1

    # if n_dets == 0:
    #     return

    """extract stats"""
    if True:
        csv_columns_max_tp = ['FP_threshold', 'TP']
        csv_columns_acc = ['confidence_threshold', 'Overall', gt_classes[0], gt_classes[1]]
        csv_columns_tp_fp = ['confidence_threshold', 'TP', 'FP']
        csv_columns_auc = ['FP_threshold', 'AUC']

        max_tp_out_root_dir = linux_path(out_root_dir, 'max_tp')
        roc_auc_out_root_dir = linux_path(out_root_dir, 'roc_auc')

        os.makedirs(max_tp_out_root_dir, exist_ok=True)
        os.makedirs(roc_auc_out_root_dir, exist_ok=True)

        tp_0 = stats_0['tp_class']
        tp_1 = stats_1['tp_class']

        fp_0 = stats_0['fp_class']
        fp_1 = stats_1['fp_class']

        fp_cls_0 = stats_0['fp_cls_class']
        fp_cls_1 = stats_1['fp_cls_class']

        fp_dup_0 = stats_0['fp_dup_class']
        fp_dup_1 = stats_1['fp_dup_class']

        fp_nex_0 = stats_0['fp_nex_class']
        fp_nex_1 = stats_1['fp_nex_class']

        conf_0 = np.copy(stats_0['conf_class'])
        conf_1 = np.copy(stats_1['conf_class'])

        n_fn_dets_0 = stats_0['fn_det_sum']
        n_fn_dets_1 = stats_1['fn_det_sum']

        n_gt_0 = stats_0['n_gt']
        n_gt_1 = stats_1['n_gt']
        n_gt = n_gt_0 + n_gt_1

    """get IDs from the binary arrays"""
    if True:
        tp_ids_0 = np.nonzero(tp_0)[0]
        fp_ids_0 = np.nonzero(fp_0)[0]
        fp_cls_ids_0 = np.nonzero(fp_cls_0)[0]
        fp_dup_ids_0 = np.nonzero(fp_dup_0)[0]
        fp_nex_ids_0 = np.nonzero(fp_nex_0)[0]

        tp_ids_1 = np.nonzero(tp_1)[0]
        fp_ids_1 = np.nonzero(fp_1)[0]
        fp_cls_ids_1 = np.nonzero(fp_cls_1)[0]
        fp_dup_ids_1 = np.nonzero(fp_dup_1)[0]
        fp_nex_ids_1 = np.nonzero(fp_nex_1)[0]

        if enable_tests:
            tp_fp_intersect_0 = np.intersect1d(tp_ids_0, fp_ids_0)
            tp_fp_cls_intersect_0 = np.intersect1d(tp_ids_0, fp_cls_ids_0)
            fp_cls_dup_intersect_0 = np.intersect1d(fp_dup_ids_0, fp_cls_ids_0)
            fp_cls_nex_intersect_0 = np.intersect1d(fp_cls_ids_0, fp_nex_ids_0)
            fp_dup_nex_intersect_0 = np.intersect1d(fp_dup_ids_0, fp_nex_ids_0)

            assert tp_fp_intersect_0.size == 0, "non-empty tp_fp_intersect_0"
            assert tp_fp_cls_intersect_0.size == 0, "non-empty tp_fp_cls_intersect_0"
            assert fp_cls_dup_intersect_0.size == 0, "non-empty fp_cls_dup_intersect_0"
            assert fp_cls_nex_intersect_0.size == 0, "non-empty fp_cls_nex_intersect_0"
            assert fp_dup_nex_intersect_0.size == 0, "non-empty fp_dup_nex_intersect_0"

            tp_fp_intersect_1 = np.intersect1d(tp_ids_1, fp_ids_1)
            tp_fp_cls_intersect_1 = np.intersect1d(tp_ids_1, fp_cls_ids_1)
            fp_cls_dup_intersect_1 = np.intersect1d(fp_dup_ids_1, fp_cls_ids_1)
            fp_cls_nex_intersect_1 = np.intersect1d(fp_cls_ids_1, fp_nex_ids_1)
            fp_dup_nex_intersect_1 = np.intersect1d(fp_dup_ids_1, fp_nex_ids_1)

            assert tp_fp_intersect_1.size == 0, "non-empty tp_fp_intersect_1"
            assert tp_fp_cls_intersect_1.size == 0, "non-empty tp_fp_cls_intersect_1"
            assert fp_cls_dup_intersect_1.size == 0, "non-empty fp_cls_dup_intersect_1"
            assert fp_cls_nex_intersect_1.size == 0, "non-empty fp_cls_nex_intersect_1"
            assert fp_dup_nex_intersect_1.size == 0, "non-empty fp_dup_nex_intersect_1"

    """
    accumulate GT label for all detections of both classes
    GT class for each detection can be obtained from the corresponding TP and FP flags 

    class 1 GT: class 0 detections marked as FP (or FP_CLS) and class 1 detections marked as TP
    """
    if True:
        """exclude NEX FPs to include both DUP and CLS FPs"""
        ex_ids_0 = np.sort(np.nonzero(np.logical_not(fp_nex_0))[0])
        ex_ids_1 = np.sort(np.nonzero(np.logical_not(fp_nex_1))[0])

        fp_ex_0 = fp_0[ex_ids_0]
        tp_ex_1 = tp_1[ex_ids_1]

        fp_ex_ids_0 = np.nonzero(fp_ex_0)[0]
        tp_ex_ids_1 = np.nonzero(tp_ex_1)[0]

        n_ex_0 = len(fp_ex_0)
        n_ex_1 = len(tp_ex_1)

        """class 0 dets classified as FP have GT label 1"""
        gt_ex_0 = np.zeros((n_ex_0,), dtype=np.uint8)
        gt_ex_0[fp_ex_ids_0] = 1

        """class 1 dets classified as TP have GT label 1"""
        gt_ex_1 = np.zeros((n_ex_1,), dtype=np.uint8)
        gt_ex_1[tp_ex_ids_1] = 1

        conf_ex_0 = conf_0[ex_ids_0]
        conf_ex_1 = conf_1[ex_ids_1]

        gt_ex_all = np.concatenate((gt_ex_0, gt_ex_1), axis=0)
        conf_ex_all = np.concatenate((conf_ex_0, 1 - conf_ex_1), axis=0)

        assert len(gt_ex_all) == len(conf_ex_all), "gt_ex_all-conf_ex_all size mismatch"

        """include only UEX (unique and existent) dets by removing dets corresponding to DUP and NEX FPs"""
        uex_0 = np.logical_not(np.logical_or(fp_dup_0, fp_nex_0))
        uex_ids_0 = np.sort(np.nonzero(uex_0)[0])

        uex_1 = np.logical_not(np.logical_or(fp_dup_1, fp_nex_1))
        uex_ids_1 = np.sort(np.nonzero(uex_1)[0])

        n_uex_0 = len(uex_ids_0)
        n_uex_1 = len(uex_ids_1)

        fp_uex_ids_0 = np.nonzero(fp_0[uex_ids_0])[0]
        tp_uex_ids_1 = np.nonzero(tp_1[uex_ids_1])[0]

        gt_uex_0 = np.zeros((n_uex_0,), dtype=np.uint8)
        gt_uex_0[fp_uex_ids_0] = 1

        gt_uex_1 = np.zeros((n_uex_1,), dtype=np.uint8)
        gt_uex_1[tp_uex_ids_1] = 1

        gt_uex_all = np.concatenate((gt_uex_0, gt_uex_1), axis=0)

        conf_uex_0 = conf_0[uex_ids_0]
        conf_uex_1 = conf_1[uex_ids_1]

        conf_uex_all = np.concatenate((conf_uex_0, 1 - conf_uex_1), axis=0)

        if enable_tests:
            tp_fp_cls_ids_0 = np.sort(np.concatenate((tp_ids_0, fp_cls_ids_0), axis=0))
            assert (uex_ids_0 == tp_fp_cls_ids_0).all(), \
                "mismatch between uex_ids_0 and tp_fp_cls_ids_0"

            tp_fp_cls_ids_1 = np.sort(np.concatenate((tp_ids_1, fp_cls_ids_1), axis=0))
            assert (uex_ids_1 == tp_fp_cls_ids_1).all(), \
                "mismatch between non_fp_det_ids_0 and tp_fp_cls_ids_0"

    """incorporate FN dets"""
    if True:
        """each FN det can be treated as an FP of the other class with conf=1 for that class or conf=0 for current class
        we are assuming that only class 0 objects (i.e. IPSCs) matter in that class 1 objects that are not detected 
        at all
         can be ignored just like FP dets
        """
        gt_ex_fn_0 = np.zeros((n_fn_dets_0,), dtype=np.uint8)
        conf_ex_fn_0 = np.zeros((n_fn_dets_0,), dtype=np.float32)

        gt_ex_fn_all = np.concatenate((gt_ex_0, gt_ex_1, gt_ex_fn_0), axis=0)
        conf_ex_fn_all = np.concatenate((conf_ex_0, 1 - conf_ex_1, conf_ex_fn_0), axis=0)

        gt_uex_fn_all = np.concatenate((gt_uex_0, gt_uex_1, gt_ex_fn_0), axis=0)
        conf_uex_fn_all = np.concatenate((conf_uex_0, 1 - conf_uex_1, conf_ex_fn_0), axis=0)

        if enable_tests:
            fp_cls_uex_ids_0 = np.nonzero(stats_0['fp_cls_class'][uex_ids_0])[0]
            assert (fp_uex_ids_0 == fp_cls_uex_ids_0).all(), \
                "mismatch between fp_uex_ids_0 and fp_cls_uex_ids_0"

            fp_dup_uex_ids_0 = np.nonzero(fp_dup_0[uex_ids_0])[0]
            assert fp_dup_uex_ids_0.size == 0, "non-empty fp_dup_uex_ids_0"

            fp_nex_uex_ids_0 = np.nonzero(fp_nex_0[uex_ids_0])[0]
            assert fp_nex_uex_ids_0.size == 0, "non-empty fp_nex_uex_ids_0"

            fp_uex_ids_1 = np.nonzero(fp_1[uex_ids_1])[0]

            fp_cls_uex_ids_1 = np.nonzero(stats_1['fp_cls_class'][uex_ids_1])[0]
            assert (fp_uex_ids_1 == fp_cls_uex_ids_1).all(), \
                "mismatch between fp_uex_ids_1 and fp_cls_uex_ids_1"

            fp_dup_uex_ids_1 = np.nonzero(fp_dup_1[uex_ids_1])[0]
            assert fp_dup_uex_ids_1.size == 0, "non-empty fp_dup_uex_ids_1"

            fp_nex_uex_ids_1 = np.nonzero(fp_nex_1[uex_ids_1])[0]
            assert fp_nex_uex_ids_1.size == 0, "non-empty fp_nex_uex_ids_1"

            n_gt_uex_0 = gt_uex_0.size
            n_gt_uex_1 = gt_uex_1.size
            n_gt_uex_all = gt_uex_all.size

            n_gt_ex_0 = gt_ex_0.size
            n_gt_ex_1 = gt_ex_1.size
            n_gt_ex_all = gt_ex_all.size

            n_gt_ex = [n_gt_ex_0, n_gt_ex_1]
            n_gt_uex = [n_gt_uex_0, n_gt_uex_1]

            if n_gt_0 > 0:
                gt_ex_pc_0 = n_gt_ex_0 / n_gt_0 * 100
                gt_uex_pc_0 = n_gt_uex_0 / n_gt_0 * 100
            else:
                gt_ex_pc_0 = gt_uex_pc_0 = 0

            if n_gt_1 > 0:
                gt_ex_pc_1 = n_gt_ex_1 / n_gt_1 * 100
                gt_uex_pc_1 = n_gt_uex_1 / n_gt_1 * 100
            else:
                gt_ex_pc_1 = gt_uex_pc_1 = 0

            gt_ex_pc = [gt_ex_pc_0, gt_ex_pc_1]
            gt_uex_pc = [gt_uex_pc_0, gt_uex_pc_1]

            if verbose:
                print(f'\nn_gt_ex_all: {n_gt_ex_all} :: '
                      f'{gt_classes[0]}: {n_gt_ex_0} ({gt_ex_pc_0:.2f} %), '
                      f'{gt_classes[1]}: {n_gt_ex_1} ({gt_ex_pc_1:.2f} %)\n')

                print(f'\nn_gt_uex_all: {n_gt_uex_all} :: '
                      f'{gt_classes[0]}: {n_gt_uex_0} ({gt_uex_pc_0:.2f} %), '
                      f'{gt_classes[1]}: {n_gt_uex_1} ({gt_uex_pc_1:.2f} %)\n')

                print(f'\nn_fn_dets_0: {n_fn_dets_0}')
                print(f'n_fn_dets_1: {n_fn_dets_1}')

                print(f'\nn_gt: {n_gt} :: '
                      f'{gt_classes[0]}: {n_gt_0}, '
                      f'{gt_classes[1]}: {n_gt_1}\n')

    """compute and save binary cls metrics"""
    if True:
        """Existent"""
        tpr_ex, fpr_ex, acc_ex, roc_auc_ex, tp_fp_ex = compute_binary_cls_metrics(
            score_thresholds, conf_ex_all, gt_ex_all, gt_classes)

        """Unique and Existent"""
        tpr_uex, fpr_uex, acc_uex, roc_auc_uex, tp_fp_uex = compute_binary_cls_metrics(
            score_thresholds, conf_uex_all, gt_uex_all, gt_classes)

        """Existent + FN"""
        tpr_ex_fn, fpr_ex_fn, acc_ex_fn, roc_auc_ex_fn, tp_fp_ex_fn = compute_binary_cls_metrics(
            score_thresholds, conf_ex_fn_all, gt_ex_fn_all, gt_classes)

        """Unique and Existent + FN"""
        tpr_uex_fn, fpr_uex_fn, acc_uex_fn, roc_auc_uex_fn, tp_fp_uex_fn = compute_binary_cls_metrics(
            score_thresholds, conf_uex_fn_all, gt_uex_fn_all, gt_classes)

        roc_auc_ex *= 100
        roc_auc_uex *= 100
        roc_auc_ex_fn *= 100
        roc_auc_uex_fn *= 100

        arr_to_csv(tp_fp_ex, csv_columns_tp_fp, misc_out_root_dir, 'tp_fp_ex.csv')
        arr_to_csv(tp_fp_uex, csv_columns_tp_fp, misc_out_root_dir, 'tp_fp_uex.csv')
        arr_to_csv(tp_fp_ex_fn, csv_columns_tp_fp, misc_out_root_dir, 'tp_fp_ex_fn.csv')
        arr_to_csv(tp_fp_uex_fn, csv_columns_tp_fp, misc_out_root_dir, 'tp_fp_uex_fn.csv')

        eval_result_dict['roc_auc_ex'] = roc_auc_ex.tolist()
        eval_result_dict['roc_auc_uex'] = roc_auc_uex.tolist()
        eval_result_dict['roc_auc_ex_fn'] = roc_auc_ex_fn.tolist()
        eval_result_dict['roc_auc_uex_fn'] = roc_auc_uex_fn.tolist()

        class_dict = eval_result_dict[gt_classes[0]]
        for _id, fp_threshold in enumerate(roc_auc_uex[:, 0]):
            class_dict[f'roc_auc_uex-{fp_threshold:.1f}'] = float(roc_auc_uex[_id, 1])
        # for _id, fp_threshold in enumerate(roc_auc_ex[:, 0]):
        #     class_dict[f'roc_auc_ex-{fp_threshold:.1f}'] = float(roc_auc_ex[_id, 1])
        # for _id, fp_threshold in enumerate(roc_auc_ex_fn[:, 0]):
        #     class_dict[f'roc_auc_ex_fn-{fp_threshold:.1f}'] = float(roc_auc_ex_fn[_id, 1])
        # for _id, fp_threshold in enumerate(roc_auc_uex_fn[:, 0]):
        #     class_dict[f'roc_auc_uex_fn-{fp_threshold:.1f}'] = float(roc_auc_uex_fn[_id, 1])

        arr_to_csv(roc_auc_ex, csv_columns_auc, roc_auc_out_root_dir, 'roc_auc_ex.csv')
        arr_to_csv(roc_auc_uex, csv_columns_auc, roc_auc_out_root_dir, 'roc_auc_uex.csv')
        arr_to_csv(roc_auc_ex_fn, csv_columns_auc, roc_auc_out_root_dir, 'roc_auc_fn.csv')
        arr_to_csv(roc_auc_uex_fn, csv_columns_auc, roc_auc_out_root_dir, 'roc_auc_uex_fn.csv')

        arr_to_csv(acc_ex * 100, csv_columns_acc, misc_out_root_dir, f'acc_ex.csv')
        arr_to_csv(acc_uex * 100, csv_columns_acc, misc_out_root_dir, f'acc_uex.csv')
        arr_to_csv(acc_ex_fn * 100, csv_columns_acc, misc_out_root_dir, f'acc_ex_fn.csv')
        arr_to_csv(acc_uex_fn * 100, csv_columns_acc, misc_out_root_dir, f'acc_uex_fn.csv')

    """compute and save detection-cls hybrid metrics"""
    if True:
        """
        class 0 FN = class 0 objs misclassified as class 1 =  class 1 FP
        class 1 FN = class 1 objs misclassified as class 0 =  class 0 FP
        class 0 TP + FN = total objs corresponding to class 0 GT = total class 0 objs       
        class 1 TP + FN = total objs corresponding to class 1 GT = total class 1 objs
    
        class 0 relative TP rate = (class 0 objs correctly classified as class 0) / (total class 0 objs)
        class 1 relative TP rate = (class 1 objs correctly classified as class 1) / (total class 1 objs)
        """

        """don't even remember what crap this is"""
        # n_dets_0 = stats_0['tp_sum'] + stats_1['fp_cls_sum']
        # n_dets_1 = stats_1['tp_sum'] + stats_0['fp_cls_sum']

        # total_class_dets[:, 0] = tp_sum_thresh_all[:, 0] + fp_cls_sum_thresh_all[:, 1]
        # total_class_dets[:, 1] = tp_sum_thresh_all[:, 1] + fp_cls_sum_thresh_all[:, 0]

        assert np.all(tp_sum_thresh_all[:, 0] <= n_dets_0), "TP count exceeds number of dets"
        assert np.all(tp_sum_thresh_all[:, 1] <= n_dets_1), "TP count exceeds number of dets"

        # tpr_cls_rel[:, 0] = tp_sum_thresh_all[:, 0] / n_dets_0
        # tpr_cls_rel[:, 1] = tp_sum_thresh_all[:, 1] / n_dets_1

        """
        class 0 TN = objs of class 1 correctly classified as class 1 = class 1 TP        
        class 1 TN = objs of class 0 correctly classified as class 0 = class 0 TP
    
        class 0 FP  = class 1 dets misclassified as class 0
        class 1  FP = class 0 dets misclassified as class 1
    
        class 0 relative FP rate = (class 1 objs misclassified as class 0) / (total class 1 objs)
        class 1 relative FP rate = (class 0 objs misclassified as class 1) / (total class 0 objs)        
        """

        assert np.all(fp_cls_sum_thresh_all[:, 0] <= n_dets_0), "FP cls count exceeds number of dets"
        assert np.all(fp_cls_sum_thresh_all[:, 1] <= n_dets_1), "FP cls count exceeds number of dets"

        # fpr_cls_rel[:, 0] = fp_cls_sum_thresh_all[:, 0] / n_dets_1
        # fpr_cls_rel[:, 1] = fp_cls_sum_thresh_all[:, 1] / n_dets_0

        # tpr_cls_rel[tpr_cls_rel == np.inf] = 0
        # fpr_cls_rel[fpr_cls_rel == np.inf] = 0

        n_score_thresholds = len(score_thresholds)

        """binary det-cls metrics - absolute"""
        tpr_cls_abs = np.zeros((n_score_thresholds, 2), dtype='float')
        fpr_cls_abs = np.zeros((n_score_thresholds, 2), dtype='float')

        assert np.all(tp_sum_thresh_all[:, 0] <= n_gt_0), "TP count exceeds number of GT objects"
        assert np.all(tp_sum_thresh_all[:, 1] <= n_gt_1), "TP count exceeds number of GT objects"

        assert np.all(fp_cls_sum_thresh_all[:, 0] <= n_gt_1), "FP cls count exceeds number of GT objects"
        assert np.all(fp_cls_sum_thresh_all[:, 1] <= n_gt_0), "FP cls count exceeds number of GT objects"

        if n_gt_0 > 0:
            tpr_cls_abs[:, 0] = tp_sum_thresh_all[:, 0] / n_gt_0
            fpr_cls_abs[:, 1] = fp_cls_sum_thresh_all[:, 1] / n_gt_0
        else:
            tpr_cls_abs[:, 0] = 0
            fpr_cls_abs[:, 1] = 0

        if n_gt_1 > 0:
            tpr_cls_abs[:, 1] = tp_sum_thresh_all[:, 1] / n_gt_1
            fpr_cls_abs[:, 0] = fp_cls_sum_thresh_all[:, 0] / n_gt_1
        else:
            tpr_cls_abs[:, 1] = 0
            fpr_cls_abs[:, 0] = 0

        """binary detection metrics"""
        tpr = np.zeros((n_score_thresholds, 2))
        fpr = np.zeros((n_score_thresholds, 2))

        n_fp_tn_0 = stats_0['fp_sum'] + stats_1['tp_sum']
        n_fp_tn_1 = stats_1['fp_sum'] + stats_0['tp_sum']

        if n_gt_0 > 0:
            tpr[:, 0] = tp_sum_thresh_all[:, 0] / n_gt_0

        if n_gt_1 > 0:
            tpr[:, 1] = tp_sum_thresh_all[:, 1] / n_gt_1

        assert np.all(fp_sum_thresh_all[:, 0] <= n_fp_tn_0), "FP count exceeds number of FP+TN objects"
        assert np.all(fp_sum_thresh_all[:, 1] <= n_fp_tn_1), "FP count exceeds number of FP+TN objects"

        if n_fp_tn_0 > 0:
            fpr[:, 0] = fp_sum_thresh_all[:, 0] / n_fp_tn_0
        if n_fp_tn_1 > 0:
            fpr[:, 1] = fp_sum_thresh_all[:, 1] / n_fp_tn_1

    fps = [fpr_cls_abs, fpr, fpr_ex, fpr_uex]
    tps = [tpr_cls_abs, tpr, tpr_ex, tpr_uex]

    for class_idx, class_name in enumerate(gt_classes):
        class_fps = [fp[:, class_idx] for fp in fps]
        class_tps = [tp[:, class_idx] for tp in tps]

        auc_cls, auc_overall, auc_ex, auc_uex = norm_auc(class_fps, class_tps)

        # avg_tp_cls = np.mean(tpr_cls_abs[:, i]).item()
        # avg_tp = np.mean(tpr[:, i]).item()

        max_tp_ex = get_max_tp(
            tpr_ex[:, class_idx], fpr_ex[:, class_idx])

        max_tp_uex = get_max_tp(
            tpr_uex[:, class_idx], fpr_uex[:, class_idx])

        max_tp = get_max_tp(
            tpr[:, class_idx], fpr[:, class_idx])

        max_tp_cls = get_max_tp(
            tpr_cls_abs[:, class_idx], fpr_cls_abs[:, class_idx])

        try:
            class_dict = eval_result_dict[class_name]
        except KeyError:
            class_dict = {}

        class_dict['GT_EX'] = n_gt_ex[class_idx]
        class_dict['GT_UEX'] = n_gt_uex[class_idx]
        class_dict['GT_EX_PC'] = gt_ex_pc[class_idx]
        class_dict['GT_UEX_PC'] = gt_uex_pc[class_idx]

        class_dict['auc_cls'] = auc_cls * 100
        class_dict['auc_overall'] = auc_overall * 100

        class_dict['auc_ex'] = auc_ex * 100
        class_dict['auc_uex'] = auc_uex * 100

        class_dict['max_tp_ex'] = (max_tp_ex * 100).tolist()
        class_dict['max_tp_uex'] = (max_tp_uex * 100).tolist()

        class_dict['max_tp'] = (max_tp * 100).tolist()
        class_dict['max_tp_cls'] = (max_tp_cls * 100).tolist()

        eval_result_dict[class_name] = class_dict

        """
        ******************************
        max_tp
        ******************************
        """
        arr_to_csv(max_tp_ex * 100, csv_columns_max_tp, max_tp_out_root_dir, f'{class_name}-max_tp_ex.csv')
        arr_to_csv(max_tp_uex * 100, csv_columns_max_tp, max_tp_out_root_dir, f'{class_name}-max_tp_uex.csv')
        arr_to_csv(max_tp * 100, csv_columns_max_tp, max_tp_out_root_dir, f'{class_name}-max_tp.csv')
        arr_to_csv(max_tp_cls * 100, csv_columns_max_tp, max_tp_out_root_dir, f'{class_name}-max_tp_cls.csv')
        """
        ******************************
        bin cls - tp_fp
        ******************************
        """
        arr_to_csv(np.vstack((score_thresholds, tpr_ex[:, class_idx] * 100, fpr_ex[:, class_idx] * 100)).T,
                   csv_columns_tp_fp, out_root_dir, f'{class_name}-tp_fp_ex.csv')

        arr_to_csv(np.vstack((score_thresholds, tpr_uex[:, class_idx] * 100, fpr_uex[:, class_idx] * 100)).T,
                   csv_columns_tp_fp, out_root_dir, f'{class_name}-tp_fp_uex.csv')

        arr_to_csv(np.vstack((score_thresholds, tpr_ex_fn[:, class_idx] * 100, fpr_ex_fn[:, class_idx] * 100)).T,
                   csv_columns_tp_fp, out_root_dir, f'{class_name}-tp_fp_ex_fn.csv')

        arr_to_csv(
            np.vstack((score_thresholds, tpr_uex_fn[:, class_idx] * 100, fpr_uex_fn[:, class_idx] * 100)).T,
            csv_columns_tp_fp, out_root_dir, f'{class_name}-tp_fp_uex_fn.csv')

        """
        ******************************
        det - tp_fp
        ******************************
        """
        arr_to_csv(np.vstack((score_thresholds, tpr_cls_abs[:, class_idx] * 100, fpr_cls_abs[:, class_idx] * 100)).T,
                   csv_columns_tp_fp, out_root_dir, f'{class_name}-tp_fp_cls.csv')

        arr_to_csv(np.vstack((score_thresholds, tpr[:, class_idx] * 100, fpr[:, class_idx] * 100)).T,
                   csv_columns_tp_fp, out_root_dir, f'{class_name}-tp_fp.csv')


def get_intersection(val1, val2, conf_class, score_thresh, name1, name2):
    val1 = np.asarray(val1)
    val2 = np.asarray(val2)
    conf_class = np.asarray(conf_class)

    _diff = val1 - val2

    idx = np.argwhere(np.diff(np.sign(_diff))).flatten()

    if not idx.size:
        # print('rec/prec: {}'.format(pformat(np.vstack((conf_class, rec, prec)).T)))
        if _diff.size:
            idx = np.argmin(np.abs(_diff))

        if idx.size > 1:
            idx = idx[0]

        _txt = f'No intersection between {name1} and {name2} found; ' \
               f'min_difference: {_diff[idx]} at {(val1[idx], val2[idx])} ' \
               f'for confidence: {conf_class[idx]}'
        # print(_txt)
        if not idx.size:
            _rec_prec = _score = 0
        else:
            _rec_prec = (val1[idx] + val2[idx]) / 2.0
            _score = conf_class[idx]
    else:
        _txt = f'Intersection at {val1[idx]} for confidence: {conf_class[idx]} with idx: {idx}'
        # print(_txt)
        _rec_prec = val1[idx[0]]
        _score = conf_class[idx][0]

    if _score < score_thresh:
        # print('conf_class: {}'.format(pformat(list(conf_class))))
        raise AssertionError(f'_score: {_score} < score_thresh: {score_thresh}')

    return _rec_prec, _score, _txt


def compute_thresh_rec_prec(thresh_idx, score_thresholds,
                            conf_class, fp_class,
                            fp_dup_class,
                            fp_nex_class,
                            fp_cls_class,
                            tp_class, n_gt):
    _thresh = score_thresholds[thresh_idx]
    idx_thresh = [i for i, x in enumerate(conf_class) if x >= _thresh]

    # conf_class_thresh = [conf_class[i] for i in idx_thresh]
    fp_thresh = [fp_class[i] for i in idx_thresh]
    fp_dup_thresh = [fp_dup_class[i] for i in idx_thresh]
    fp_nex_thresh = [fp_nex_class[i] for i in idx_thresh]
    fp_cls_thresh = [fp_cls_class[i] for i in idx_thresh]

    tp_thresh = [tp_class[i] for i in idx_thresh]

    tp_sum_th = np.sum(tp_thresh)
    fp_sum_th = np.sum(fp_thresh)
    fp_dup_sum_th = np.sum(fp_dup_thresh)
    fp_nex_sum_th = np.sum(fp_nex_thresh)
    fp_cls_sum_th = np.sum(fp_cls_thresh)

    try:
        _rec_th = float(tp_sum_th) / n_gt
    except ZeroDivisionError:
        _rec_th = 0
    try:
        _prec_th = float(tp_sum_th) / float(fp_sum_th + tp_sum_th)
    except ZeroDivisionError:
        _prec_th = 0

    # sys.stdout.write('\rDone {}/{}: {}'.format(thresh_idx+1, n_score_thresholds, _thresh))
    # sys.stdout.flush()

    return _rec_th, _prec_th, tp_sum_th, fp_sum_th, fp_cls_sum_th, fp_dup_sum_th, fp_nex_sum_th


def draw_objs(img, objs, alpha=0.5, class_name_to_col=None, cols=None,
              in_place=False, bbox=True, mask=False, thickness=2, check_bb=0,
              bb_resize=None, cls_cat_to_col=None, title=None, show_class=False):
    if in_place:
        vis_img = img
    else:
        vis_img = np.copy(img).astype(np.float32)

    mask_img = np.zeros_like(vis_img)

    text_args = dict(
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.8,
        thickness=2,
        # lineType=cv2.LINE_AA,
    )

    vis_h, vis_w = vis_img.shape[:2]

    resize_factor = 1.0
    n_objs = len(objs)

    if cols is not None:
        if not isinstance(cols, (list, tuple)):
            cols = [cols, ] * n_objs
    else:
        if class_name_to_col is not None:
            cols = [class_name_to_col[obj['class']] for obj in objs]
        else:
            assert cls_cat_to_col is not None, "either class_name_to_col or cls_cat_to_col must be provided"
            """
            get col based on classification status
            tp = green for both gt and det
            fp = red for det, fn=red for gt
            """
            cols = [cls_cat_to_col[obj['cls']] for obj in objs]

    for obj, obj_col in zip(objs, cols, strict=True):
        obj_col = col_bgr[obj_col]

        if mask:
            rle = obj['mask']
            mask_orig = mask_util.decode(rle).squeeze()
            mask_orig = np.ascontiguousarray(mask_orig, dtype=np.uint8)

            mask_h, mask_w = mask_orig.shape[:2]

            if (mask_h, mask_w) != (vis_h, vis_w):
                resize_factor = vis_h / mask_h
                mask_uint = cv2.resize(mask_orig, (vis_w, vis_h),
                                       interpolation=cv2.INTER_NEAREST)
            else:
                mask_uint = mask_orig

        if bbox:
            xmin, ymin, xmax, ymax = obj["bbox"]

            if check_bb:
                log_dir = 'log/masks'
                os.makedirs(log_dir, exist_ok=True)

                time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")

                bb = (xmin, ymin, xmax, ymax)

                if mask:
                    mask_pts, mask_bb, is_multi = mask_img_to_pts(mask_orig)

                    mask_xmin, mask_ymin, w, h = mask_bb
                    mask_xmax, mask_ymax = mask_xmin + w, mask_ymin + h

                    mask_bb = (mask_xmin, mask_ymin, mask_xmax, mask_ymax)
                    mask_y, mask_x = np.nonzero(mask_orig)

                    mask_rgb = np.zeros((mask_h, mask_w, 3), dtype=np.uint8)
                    mask_rgb[mask_y, mask_x, :] = 255

                    cv2.rectangle(
                        mask_rgb, (int(mask_xmin), int(mask_ymin)), (int(mask_xmax), int(mask_ymax)), (0, 255, 0),
                        thickness)

                    if is_multi:
                        msg = "annoying multi mask"
                        cv2.imwrite(f'{log_dir}/{msg} {time_stamp}.png', mask_rgb)
                        # raise AssertionError(msg)
                        continue

                    if len(mask_pts) < 4:
                        msg = 'annoying mask with too few points'
                        cv2.imwrite(f'{log_dir}/{msg} {time_stamp}.png', mask_rgb)
                        # raise AssertionError(msg)
                        continue

                    # mask_pts = contour_pts_from_mask(mask_orig, allow_multi=0)
                    # mask_pts = np.asarray(mask_pts, dtype=np.int32)

                    # cv2.rectangle(
                    #     mask_rgb, (int(mask_xmin), int(mask_ymin)), (int(mask_xmax), int(mask_ymax)), (0, 255, 0),
                    #     thickness)
                    # cv2.imwrite('mask_bb mismatch.png', mask_rgb)

                    mask_bb_area = w * h

                    if mask_bb_area < 4:
                        msg = 'annoying invalid mask with tiny area'
                        cv2.imwrite(f'{log_dir}/{msg} {time_stamp}.png', mask_rgb)
                        raise AssertionError(f'{msg} {mask_bb_area}')
                        # continue

                    if bb != mask_bb:
                        msg = "mask_bb mismatch"
                        cv2.imwrite(f'{log_dir}/{msg} {time_stamp}.png', mask_rgb)
                        # raise AssertionError(f'{msg}')
                        continue

                    bb_norm = np.linalg.norm(np.asarray(bb))
                    mask_bb_norm = np.linalg.norm(np.asarray(mask_bb))
                    bb_diff_norm = np.linalg.norm(np.asarray(bb) - np.asarray(mask_bb))
                    bb_diff_norm_ratio = bb_diff_norm / max(bb_norm, mask_bb_norm)

                    if bb_diff_norm_ratio > 0.1:
                        cv2.rectangle(
                            mask_rgb, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), thickness)

                        cv2.imwrite(f'{log_dir}/mask_bb mismatch {time_stamp}.png', mask_rgb)
                        # raise AssertionError("mask_bb mismatch")
                        continue

                    xmin, ymin, xmax, ymax = mask_bb

            if bb_resize is not None:
                if mask:
                    assert bb_resize == resize_factor, \
                        f"mismatch between bb_resize: {bb_resize} and resize_factor from mask: {resize_factor}"
                resize_factor = bb_resize

            if resize_factor != 1.0:
                xmin *= resize_factor
                ymin *= resize_factor
                xmax *= resize_factor
                ymax *= resize_factor

            cv2.rectangle(
                vis_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), obj_col, thickness)

            if show_class:
                label = obj['class']
                conf = int(obj['confidence'] * 100)

                pt = (int(xmin), int(ymin)) if xmin > 15 and ymin > 15 else (int(xmax), int(ymax))
                # cv2.putText(vis_img, f'{label} {conf:d}', pt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, obj_col, 2, cv2.LINE_AA)
                put_text_with_background(vis_img, f'{label} {conf:d}', pt, obj_col, [0, 0, 0], **text_args)

        if mask:
            mask_binary = mask_uint.astype(bool)

            mask_img[mask_binary] = obj_col
            vis_img[mask_binary] = (alpha * vis_img[mask_binary] +
                                    (1 - alpha) * mask_img[mask_binary])

    vis_img = vis_img.astype(np.uint8)

    if title is not None:
        # cv2.putText(vis_img, title, (15, 25), cv2.FONT_HERSHEY_SIMPLEX,
        #             0.8, [255, 255, 255], 2, cv2.LINE_AA)
        put_text_with_background(vis_img, title, (15, 25), [255, 255, 255], [0, 0, 0], **text_args)

    # cv2.imshow('vis_img', vis_img)
    # cv2.waitKey(0)

    return vis_img


def get_max_iou_obj(objs, matching_classes, bb, mask, enable_mask):
    ovmax = -1
    obj_match = None

    for obj in objs:
        """look for a matching gt obj of the same class"""
        if obj["class"] not in matching_classes:
            continue

        if enable_mask:
            mask_obj = obj["mask"]
            # ov = get_mask_iou(mask, mask_obj, bb, bb_obj)
            ov = get_mask_rle_iou(mask, mask_obj)
        else:
            bb_obj = obj["bbox"]
            ov = get_iou(bb, bb_obj)

        if ov == 0:
            continue

        if ov > ovmax:
            ovmax = ov
            obj_match = obj

    return ovmax, obj_match


def get_mask_rle_iou(mask_det, mask_gt):
    iscrowd = [0, ]
    iou = mask_util.iou([mask_det, ], [mask_gt, ], iscrowd)

    iou = np.asarray(iou).squeeze().item()

    # mask_det_ = mask_util.decode(mask_det).astype(bool)
    # mask_gt_ = mask_util.decode(mask_gt).astype(bool)
    #
    # iou2 = get_mask_iou(mask_det_, mask_gt_, bb_det, bb_gt)

    return iou


def get_mask_iou(mask_det, mask_gt, bb_det, bb_gt):
    x1_det, y1_det, x2_det, y2_det = bb_det
    x1_gt, y1_gt, x2_gt, y2_gt = bb_gt

    min_x, min_y = int(min(x1_det, x1_gt)), int(min(y1_det, y1_gt))
    max_x, max_y = int(max(x2_det, x2_gt)), int(max(y2_det, y2_gt))

    mask_det_ = mask_det[min_y:max_y + 1, min_x:max_x + 1]
    mask_gt_ = mask_gt[min_y:max_y + 1, min_x:max_x + 1]

    # mask_det_ = mask_det_ > 0
    # mask_gt_ = mask_gt_ > 0

    mask_union = np.logical_or(mask_det_, mask_gt_)
    n_mask_union = np.count_nonzero(mask_union)

    if n_mask_union == 0:
        return 0

    mask_inter = np.logical_and(mask_det_, mask_gt_)
    n_mask_inter = np.count_nonzero(mask_inter)

    mask_iou = n_mask_inter / n_mask_union

    return mask_iou


def compute_overlaps_multi(iou, ioa_1, ioa_2, objects_1, objects_2, xywh=False):
    """

    compute overlap between each pair of objects in two sets of objects
    can be used for computing overlap between all detections and annotations in a frame

    :type iou: np.ndarray | None
    :type ioa_1: np.ndarray | None
    :type ioa_2: np.ndarray | None
    :type objects_1: np.ndarray
    :type objects_2: np.ndarray
    :rtype: None
    """
    # handle annoying singletons
    if len(objects_1.shape) == 1:
        objects_1 = objects_1.reshape((1, 4))

    if len(objects_2.shape) == 1:
        objects_2 = objects_2.reshape((1, 4))

    n1 = objects_1.shape[0]
    n2 = objects_2.shape[0]

    ul_1 = objects_1[:, :2]  # n1 x 2
    ul_1_rep = np.tile(np.reshape(ul_1, (n1, 1, 2)), (1, n2, 1))  # np(n1 x n2 x 2) -> std(n2 x 2 x n1)
    ul_2 = objects_2[:, :2]  # n2 x 2
    ul_2_rep = np.tile(np.reshape(ul_2, (1, n2, 2)), (n1, 1, 1))  # np(n1 x n2 x 2) -> std(n2 x 2 x n1)

    if xywh:
        size_1 = objects_1[:, 2:]  # n1 x 2
        size_2 = objects_2[:, 2:]  # n2 x 2
        br_1 = ul_1 + size_1 - 1  # n1 x 2
        br_2 = ul_2 + size_2 - 1  # n2 x 2
    else:
        br_1 = objects_1[:, 2:]  # n1 x 2
        br_2 = objects_2[:, 2:]  # n2 x 2
        size_1 = br_1 - ul_1 + 1
        size_2 = br_2 - ul_2 + 1

    br_1_rep = np.tile(np.reshape(br_1, (n1, 1, 2)), (1, n2, 1))  # np(n1 x n2 x 2) -> std(n2 x 2 x n1)
    br_2_rep = np.tile(np.reshape(br_2, (1, n2, 2)), (n1, 1, 1))  # np(n1 x n2 x 2) -> std(n2 x 2 x n1)

    size_inter = np.minimum(br_1_rep, br_2_rep) - np.maximum(ul_1_rep, ul_2_rep) + 1  # n2 x 2 x n1
    size_inter[size_inter < 0] = 0
    # np(n1 x n2 x 1) -> std(n2 x 1 x n1)
    area_inter = np.multiply(size_inter[:, :, 0], size_inter[:, :, 1])

    area_1 = np.multiply(size_1[:, 0], size_1[:, 1]).reshape((n1, 1))  # n1 x 1
    area_1_rep = np.tile(area_1, (1, n2))  # np(n1 x n2 x 1) -> std(n2 x 1 x n1)
    area_2 = np.multiply(size_2[:, 0], size_2[:, 1]).reshape((n2, 1))  # n2 x 1
    area_2_rep = np.tile(area_2.transpose(), (n1, 1))  # np(n1 x n2 x 1) -> std(n2 x 1 x n1)
    area_union = area_1_rep + area_2_rep - area_inter  # n2 x 1 x n1

    if iou is not None:
        iou[:] = np.divide(area_inter, area_union)  # n1 x n2
    if ioa_1 is not None:
        ioa_1[:] = np.divide(area_inter, area_1_rep)  # n1 x n2
    if ioa_2 is not None:
        ioa_2[:] = np.divide(area_inter, area_2_rep)  # n1 x n2


def get_iou(bb_det, bb_gt, xywh=False):
    if xywh:
        det_x1, det_y1, det_w, det_h = bb_det
        gt_x1, gt_y1, gt_w, gt_h = bb_gt
        det_x2, det_y2 = det_x1 + det_w - 1, det_y1 + det_h - 1
        gt_x2, gt_y2 = gt_x1 + gt_w - 1, gt_y1 + gt_h - 1
    else:
        det_x1, det_y1, det_x2, det_y2 = bb_det
        gt_x1, gt_y1, gt_x2, gt_y2 = bb_gt

        det_w, det_h = det_x2 - det_x1 + 1, det_y2 - det_y1 + 1
        gt_w, gt_h = gt_x2 - gt_x1 + 1, gt_y2 - gt_y1 + 1

    bi = [max(det_x1, gt_x1),
          max(det_y1, gt_y1),
          min(det_x2, gt_x2),
          min(det_y2, gt_y2)]

    iw = bi[2] - bi[0] + 1
    ih = bi[3] - bi[1] + 1

    if iw <= 0 or ih <= 0:
        return 0

    ua = (det_w * det_h) + (gt_w * gt_h) - (iw * ih)

    # compute overlap (IoU) = area of intersection / area of union
    ov = iw * ih / ua

    return ov


#
# def contour_pts_from_mask(mask_img, allow_multi=1):
#     # print('Getting contour pts from mask...')
#     if len(mask_img.shape) == 3:
#         mask_img_gs = np.squeeze(mask_img[:, :, 0]).copy()
#     else:
#         mask_img_gs = mask_img.copy()
#
#     _contour_pts, _ = cv2.findContours(mask_img_gs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
#
#     _contour_pts = list(_contour_pts)
#
#     n_contours = len(_contour_pts)
#     # print('n_contours: {}'.format(n_contours))
#     # print('_contour_pts: {}'.format(_contour_pts))
#     # print('contour_pts: {}'.format(type(contour_pts)))
#
#     if allow_multi:
#         if not _contour_pts:
#             return [], []
#
#         all_mask_pts = []
#
#         for i in range(n_contours):
#             _contour_pts[i] = list(np.squeeze(_contour_pts[i]))
#             mask_pts = [[x, y, 1] for x, y in _contour_pts[i]]
#
#             all_mask_pts.append(mask_pts)
#
#         return all_mask_pts
#
#     else:
#         if not _contour_pts:
#             return []
#         contour_pts = list(np.squeeze(_contour_pts[0]))
#         # return contour_pts
#
#         if n_contours > 1:
#             """get longest contour"""
#             max_len = len(contour_pts)
#             for _pts in _contour_pts[1:]:
#                 # print('_pts: {}'.format(_pts))
#                 _pts = np.squeeze(_pts)
#                 _len = len(_pts)
#                 if max_len < _len:
#                     contour_pts = _pts
#                     max_len = _len
#         # print('contour_pts len: {}'.format(len(contour_pts)))
#         mask_pts = [[x, y, 1] for x, y in contour_pts]
#
#         return mask_pts
#
def get_class_ids_map(class_ids):
    max_class_id = max(class_ids)
    class_ids_map = {class_id: i for i, class_id in enumerate(class_ids)}
    for class_id in range(max_class_id):
        """map all missing class IDs to background"""
        try:
            _ = class_ids_map[class_id]
        except KeyError:
            class_ids_map[class_id] = 0
    return class_ids_map


def unmap_class_ids(labels_img, class_ids_map):
    out_img = np.zeros_like(labels_img)
    for class_id in class_ids_map.keys():
        out_img[labels_img == class_ids_map[class_id]] = class_id
    return out_img


def map_class_ids(labels_img, class_ids_map):
    class_ids = np.unique(labels_img, return_inverse=False)
    out_img = np.zeros_like(labels_img)
    for class_id in class_ids:
        out_img[labels_img == class_id] = class_ids_map[class_id]
    return out_img


def mask_img_to_rle_coco(binary_mask):
    binary_mask_fortran = np.asfortranarray(binary_mask, dtype="uint8")
    rle = mask_util.encode(binary_mask_fortran)
    rle["counts"] = rle["counts"].decode("utf-8")

    return rle


def mask_img_to_pts(mask_img):
    mask_img_gs = mask_img

    contour_pts, _ = cv2.findContours(mask_img_gs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
    is_multi = False

    if not contour_pts:
        return [], [], is_multi

    if len(contour_pts) > 1:
        is_multi = True
        contour_pts = max(contour_pts, key=cv2.contourArea)
    else:
        contour_pts = contour_pts[0]
        is_multi = False

    x, y, w, h = cv2.boundingRect(contour_pts)

    contour_pts = list(contour_pts.squeeze())

    return contour_pts, (x, y, w, h), is_multi


def mask_rle_to_img(rle):
    mask_img = mask_util.decode(rle).squeeze()
    mask_img = np.ascontiguousarray(mask_img, dtype=np.uint8)

    return mask_img


def resize_mask_rle_through_pts(mask_rle, det_img_w, det_img_h):
    mask_pts, bbox, is_multi = mask_rle_to_pts(mask_rle)
    mask_h, mask_w = mask_rle["size"]
    norm_h, norm_w = float(det_img_w) / float(mask_w), float(det_img_h) / float(mask_h)
    mask_pts = [(pt[0] * norm_w, pt[1] * norm_w) for pt in mask_pts]
    mask_rle = mask_pts_to_img(mask_pts, det_img_h, det_img_w, to_rle=1)
    return mask_rle


def resize_mask_rle_through_img(rle, dst_w, dst_h):
    mask_img = mask_rle_to_img(rle)
    mask_img = mask_img.astype(np.uint8)
    dst_mask_img = cv2.resize(mask_img, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)
    dst_mask_img = dst_mask_img.astype(bool)
    dst_mask_rle = mask_img_to_rle_coco(dst_mask_img)
    return dst_mask_rle


def mask_rle_to_pts(rle):
    mask_img = mask_rle_to_img(rle)

    mask_pts, bbox, is_multi = mask_img_to_pts(mask_img)
    return mask_pts, bbox, is_multi


def mask_pts_to_str(mask_pts):
    mask_str = ';'.join(f'{x},{y}' for x, y in mask_pts)
    return mask_str


def mask_str_to_pts(mask_str):
    mask_str = str(mask_str).strip('"')

    x_coordinates = []
    y_coordinates = []
    for point_string in mask_str.split(";"):
        if not point_string:
            continue
        x_coordinate, y_coordinate = point_string.split(",")
        x_coordinates.append(float(x_coordinate))
        y_coordinates.append(float(y_coordinate))

    mask_pts = [(x, y) for x, y in zip(x_coordinates, y_coordinates)]

    return mask_pts


def contour_pts_to_mask(contour_pts, patch_img, col=(255, 255, 255)):
    # np.savetxt('contourPtsToMask_mask_pts.txt', contour_pts, fmt='%.6f')

    mask_img = np.zeros_like(patch_img, dtype=np.uint8)
    # if not isinstance(contour_pts, list):
    #     raise SystemError('contour_pts must be a list rather than {}'.format(type(contour_pts)))
    if len(contour_pts) > 0:
        mask_img = cv2.fillPoly(mask_img, np.array([contour_pts, ], dtype=np.int32), col)
    blended_img = np.array(Image.blend(Image.fromarray(patch_img), Image.fromarray(mask_img), 0.5))

    return mask_img, blended_img


def mask_pts_to_img(mask_pts, img_h, img_w, to_rle):
    mask_img = np.zeros((img_h, img_w), dtype=np.uint8)
    mask_img = cv2.fillPoly(mask_img, np.array([mask_pts, ], dtype=np.int32), 1)

    # mask_img_vis = (mask_img * 255).astype(np.uint8)
    # mask_img_vis = resize_ar(mask_img_vis, 1280, 720)
    # cv2.imshow('mask_img_vis', mask_img_vis)
    # cv2.waitKey(0)

    bin_mask_img = mask_img.astype(bool)

    if to_rle:
        mask_rle = mask_img_to_rle_coco(bin_mask_img)
        return mask_rle

    return bin_mask_img


def mask_str_to_img(mask_str, img_h, img_w, to_rle):
    mask_pts = mask_str_to_pts(mask_str)

    return mask_pts_to_img(mask_pts, img_h, img_w, to_rle)


def get_csv_dict(bbox_info):
    xmin, ymin, xmax, ymax = bbox_info["bbox"]
    _class = bbox_info["class"]
    filename = bbox_info["filename"]
    img_width = bbox_info["width"]
    img_height = bbox_info["height"]
    target_id = bbox_info["target_id"]
    confidence = bbox_info["confidence"]
    raw_data = {
        'filename': filename,
        'width': img_width,
        'height': img_height,
        'class': _class,
        'xmin': xmin,
        'ymin': ymin,
        'xmax': xmax,
        'ymax': ymax,
        'confidence': confidence,
        'target_id': target_id,
    }
    return raw_data


def perform_global_association(seq_det_data, seq_gt_data_dict, gt_class, show_sim,
                               tp, fp, all_status, all_ovmax, iou_thresh,
                               count_true_positives, all_gt_match,
                               seq_name, save_sim_dets, sim_recs, sim_precs, seq_path,
                               cum_tp_sum, cum_fp_sum, enable_mask):
    def append(_dict, _key, _val, _global_idx):
        try:
            _dict[_key].append((_val, _global_idx))
        except KeyError:
            _dict[_key] = [(_val, _global_idx), ]
        return _dict[_key]

    seq_det_data_by_frame = {}

    seq_det_data_by_frame = {
        k['file_id']: append(seq_det_data_by_frame, k['file_id'], k, i)
        for i, k in enumerate(seq_det_data)
        # if k['bbox'] is not None
    }

    det_to_gt = {}
    gt_to_det = {}

    fp_dup_dets = []
    fp_dets = []
    tp_dets = []

    fp_dets_dict = {}
    fp_dup_dets_dict = {}

    n_seq_gt = 0

    try:
        fp_sum_key = next(reversed(cum_fp_sum))
    except StopIteration:
        fp_sum = 0
        tp_sum = 0
    else:
        fp_sum = cum_fp_sum[fp_sum_key]
        tp_sum = cum_tp_sum[next(reversed(cum_tp_sum))]

    for file_id in seq_det_data_by_frame:
        """all frames in the sequence"""

        fp_dets_dict[file_id] = []
        fp_dup_dets_dict[file_id] = []

        file_dets = [k for k in seq_det_data_by_frame[file_id] if k[0]['bbox'] is not None]

        try:
            file_gts = seq_gt_data_dict[file_id]
            file_gts = [obj for obj in file_gts if obj["class"] == gt_class]
        except KeyError:
            file_gts = []

        n_file_dets = len(file_dets)
        n_file_gt = len(file_gts)

        n_seq_gt += n_file_gt

        if n_file_dets != 0 and n_file_gt != 0:
            pass
        else:
            if n_file_dets == 0 and n_file_gt == 0:
                """annoying objectless frame"""
                pass
            elif n_file_dets == 0:
                """all GTs are false negatives"""
                pass
                # fn_sum += n_file_gt
            elif n_file_gt == 0:
                """all dets are false positives"""
                cum_fp_sum[file_id] = fp_sum
                for _det_idx in range(n_file_dets):
                    curr_det_data, det_global_idx = file_dets[_det_idx]
                    fp_dets.append(det_global_idx)
                    fp_dets_dict[file_id].append(_det_idx)
                    fp[det_global_idx] = 1
                    fp_sum += 1
                    all_status[det_global_idx] = 'fp_nex'
                    all_ovmax[det_global_idx] = -1
                    curr_det_data['is_duplicate'] = False
            cum_tp_sum[file_id] = tp_sum
            cum_fp_sum[file_id] = fp_sum
            continue

        det_gt_iou = np.zeros((n_file_dets, n_file_gt), dtype=np.float64)

        for _det_idx in range(n_file_dets):
            curr_det_data, det_global_idx = file_dets[_det_idx]
            bb_det = curr_det_data["bbox"]
            mask_det = curr_det_data["mask"]

            for _gt_idx in range(n_file_gt):
                curr_gt_data = file_gts[_gt_idx]

                bb_gt = curr_gt_data["bbox"]

                if enable_mask:
                    mask_gt = curr_gt_data["mask"]
                    # iou = get_mask_iou(mask_det, mask_gt, bb_det, bb_gt)
                    iou = get_mask_rle_iou(mask_det, mask_gt)
                else:
                    iou = get_iou(bb_det, bb_gt)

                det_gt_iou[_det_idx, _gt_idx] = iou

        associated_dets = []
        associated_gts = []

        unassociated_dets = list(range(n_file_dets))
        unassociated_gts = list(range(n_file_gt))

        det_gt_iou_copy = np.copy(det_gt_iou)

        """
         Assign detections to GT objects
        """
        while True:
            det_max_iou = np.argmax(det_gt_iou_copy, axis=1)
            gt_max_iou = np.argmax(det_gt_iou_copy, axis=0)

            assoc_found = 0

            for _det in unassociated_dets:
                _assoc_gt_id = det_max_iou[_det]
                _assoc_det_id = gt_max_iou[_assoc_gt_id]

                if _assoc_gt_id in associated_gts or \
                        _assoc_det_id != _det or \
                        det_gt_iou_copy[_assoc_det_id, _assoc_gt_id] < iou_thresh:
                    continue

                _, det_global_idx = file_dets[_det]

                associated_dets.append(_assoc_det_id)
                associated_gts.append(_assoc_gt_id)

                unassociated_dets.remove(_assoc_det_id)
                unassociated_gts.remove(_assoc_gt_id)

                det_to_gt[det_global_idx] = _assoc_gt_id
                gt_to_det[_assoc_gt_id] = det_global_idx

                det_gt_iou_copy[_assoc_det_id, :] = -1
                det_gt_iou_copy[:, _assoc_gt_id] = -1

                file_gts[_assoc_gt_id]['used'] = True

                # true positive
                tp_dets.append(det_global_idx)
                tp[det_global_idx] = 1
                tp_sum += 1
                count_true_positives[gt_class] += 1

                all_status[det_global_idx] = 'tp'
                all_ovmax[det_global_idx] = det_gt_iou[_assoc_det_id, _assoc_gt_id]

                all_gt_match[det_global_idx] = file_gts[_assoc_gt_id]

                assoc_found = 1

                break

            if not assoc_found:
                break

        for _det_idx in unassociated_dets:
            __det, det_global_idx = file_dets[_det_idx]
            """false positive"""
            fp[det_global_idx] = 1
            fp_sum += 1

            _gt_idx = np.argmax(det_gt_iou[_det_idx, :])
            if det_gt_iou[_det_idx, _gt_idx] > iou_thresh:
                fp_dup_dets.append(det_global_idx)
                fp_dup_dets_dict[file_id].append(_det_idx)

                all_status[det_global_idx] = 'fp_dup'
                all_ovmax[det_global_idx] = det_gt_iou[_det_idx, _gt_idx]
                all_gt_match[det_global_idx] = file_gts[_gt_idx]
                __det['is_duplicate'] = True
            else:
                fp_dets.append(det_global_idx)
                fp_dets_dict[file_id].append(_det_idx)
                if _gt_idx in unassociated_gts:
                    all_status[det_global_idx] = 'fp_nex-ovmax'
                    all_ovmax[det_global_idx] = det_gt_iou[_det_idx, _gt_idx]
                    all_gt_match[det_global_idx] = file_gts[_gt_idx]
                    __det['is_duplicate'] = False

                else:
                    all_status[det_global_idx] = 'fp_nex'
                    all_ovmax[det_global_idx] = -1
                    __det['is_duplicate'] = False

        cum_tp_sum[file_id] = tp_sum
        cum_fp_sum[file_id] = fp_sum

    n_tp = len(tp_dets)
    n_fp = len(fp_dets)
    n_fp_dup = len(fp_dup_dets)
    n_fp_total = n_fp + n_fp_dup

    print('n_tp: {}'.format(n_tp))
    print('n_fp: {}'.format(n_fp))
    print('n_fp_dup: {}'.format(n_fp_dup))
    print('n_fp_total: {}'.format(n_fp_total))

    try:
        prec = n_tp / float(n_tp + n_fp_total)
    except ZeroDivisionError:
        prec = 0

    try:
        rec = n_tp / float(n_seq_gt)
    except ZeroDivisionError:
        rec = 0

    print('seq_name: {} rec: {:.4f} prec: {:.4f}'.format(seq_name, rec, prec))
    n_frames = len(seq_det_data_by_frame)

    if save_sim_dets:
        get_simulated_detections(sim_recs, sim_precs, n_seq_gt, seq_det_data_by_frame, fp_dets_dict,
                                 n_frames, seq_gt_data_dict, gt_class, show_sim, seq_path)

    return cum_tp_sum, cum_fp_sum


def get_simulated_detections(sim_recs, sim_precs, n_seq_gt, seq_det_data_by_frame, fp_dets_dict,
                             n_frames, seq_gt_data_dict, gt_class, show_sim, seq_path):
    red = (0, 0, 255)
    blue = (255, 0, 0)
    green = (0, 255, 0)

    print('Computing simulated detections for:\n'
          'recalls: {}\n'
          'precisions: {}'.format(
        sim_recs,
        sim_precs
    ))

    # sim_precs_str = ['{:d}'.format(int(sim_prec * 100)) for sim_prec in sim_precs]
    # sim_recs_str = ['{:d}'.format(int(sim_rec * 100)) for sim_rec in sim_recs]

    sim_fp = pd.DataFrame(
        np.zeros((len(sim_recs), len(sim_precs)), dtype=np.int32),
        index=sim_recs, columns=sim_precs,
    )
    # sim_fp_probs = pd.DataFrame(
    #     np.zeros((len(sim_recs), len(sim_precs)), dtype=np.float32),
    #     index=sim_recs, columns=sim_precs,
    # )

    sim_tp = pd.DataFrame(
        np.zeros((1, len(sim_recs)), dtype=np.int32), columns=sim_recs,
    )
    # sim_tp_probs = pd.DataFrame(
    #     np.zeros((1, len(sim_recs)), dtype=np.float32), columns=sim_recs,
    # )
    for _sim_rec in sim_recs:
        sim_tp[_sim_rec] = _sim_rec * n_seq_gt
        # sim_tp_probs[_sim_rec] = _sim_rec
        for _sim_prec in sim_precs:
            sim_fp[_sim_prec][_sim_rec] = (1.0 / _sim_prec - 1.0) * sim_tp[_sim_rec]

    # sim_boxes = {
    #     _sim_prec: {
    #         _sim_rec: [] for _sim_rec in sim_recs
    #     }
    #     for _sim_prec in sim_precs
    # }

    max_sim_fp = sim_fp.to_numpy().max()

    print_stats(sim_tp, 'sim_tp', '.1f')
    print_stats(sim_fp, 'sim_fp', '.1f')

    """fill in missing boxes by drawing from a Multivariate normal distribution
    fitted to the existing boxes"""

    # n_sampled_boxes_per_frame = int(np.ceil(float(max_sim_fp - n_fp) / n_frames))
    # n_sampled_boxes = int(n_sampled_boxes_per_frame * n_frames)

    """even if total fp needed over all frames is < available fp,
    individual frames might have fewer than needed fp so we get 2 backup fp per frame"""
    # n_backup_fp = n_frames * 2
    # n_sampled_boxes = max(0, int(max_sim_fp - n_fp)) + n_backup_fp
    # print('Sampling {} FP boxes'.format(n_sampled_boxes))

    fp_boxes = [
        seq_det_data_by_frame[__file_id][__det_id][0]['norm_bbox']
        for __file_id in seq_det_data_by_frame
        for __det_id in fp_dets_dict[__file_id]
    ]

    fp_boxes_cxy_wh = np.asarray([[
        (box[0] + box[2]) / 2.0,
        (box[1] + box[3]) / 2.0,
        float(box[2] - box[0]),
        float(box[3] - box[1]),
    ] for box in fp_boxes], dtype=np.float32)

    fp_boxes_max = np.amax(fp_boxes_cxy_wh, axis=0)
    fp_boxes_min = np.amin(fp_boxes_cxy_wh, axis=0)

    fp_boxes_mean = np.mean(fp_boxes_cxy_wh, axis=0)
    fp_boxes_cov = np.cov(fp_boxes_cxy_wh, rowvar=0)

    fp_boxes_conf = np.asarray([
        seq_det_data_by_frame[__file_id][__det_id][0]['confidence']
        for __file_id in seq_det_data_by_frame
        for __det_id in fp_dets_dict[__file_id]
    ], dtype=np.float32)
    fp_boxes_conf_mean = np.mean(fp_boxes_conf)
    print('fp_boxes_conf_mean: {}'.format(fp_boxes_conf_mean))

    # n_fp += n_sampled_boxes

    # sim_fp_probs = sim_fp.divide(float(n_fp))

    _pause = 1
    for _sim_rec, _sim_prec in itertools.product(sim_recs, sim_precs):
        n_tp_needed_per_frame = int(sim_tp[_sim_rec] / n_frames)
        n_tp_boxes_per_frame = {
            ___file_id: n_tp_needed_per_frame
            for ___file_id in seq_det_data_by_frame
        }
        n_residual_tp_boxes = int(sim_tp[_sim_rec]) % n_frames
        if n_residual_tp_boxes:
            random_file_ids = np.random.permutation(list(n_tp_boxes_per_frame.keys()))
            for ___file_id in random_file_ids[:n_residual_tp_boxes]:
                n_tp_boxes_per_frame[___file_id] += 1

        n_fp_needed_per_frame = int(sim_fp[_sim_prec][_sim_rec] / n_frames)
        n_fp_boxes_per_frame = {
            ___file_id: n_fp_needed_per_frame
            for ___file_id in seq_det_data_by_frame
        }
        n_residual_fp_boxes = int(sim_fp[_sim_prec][_sim_rec]) % n_frames
        if n_residual_fp_boxes:
            random_file_ids = np.random.permutation(list(n_fp_boxes_per_frame.keys()))
            for ___file_id in random_file_ids[:n_residual_fp_boxes]:
                n_fp_boxes_per_frame[___file_id] += 1

        n_fp_boxes_all_frames = sum(list(n_fp_boxes_per_frame.values()))
        n_tp_boxes_all_frames = sum(list(n_tp_boxes_per_frame.values()))

        __fp_start = 0
        sim_boxes = []
        n_sim_fp = 0
        n_sim_tp = 0
        """evenly distributed TP might not work especially for high recall scenarios when
        individual frames have too few GT boxes so we need to redistribute these"""
        n_residual_tp = 0
        pass_id = 0
        all_file_ids = list(n_fp_boxes_per_frame.keys())
        while True:
            file_ids_to_remove = []
            for file_id in all_file_ids:
                try:
                    file_gts = seq_gt_data_dict[file_id]
                    file_gts = [obj for obj in file_gts if obj["class"] == gt_class]
                except KeyError:
                    file_gts = []

                n_tp_needed = n_tp_boxes_per_frame[file_id]
                n_actual_gt = len(file_gts)

                if n_actual_gt < n_tp_needed:
                    n_residual_tp += n_tp_needed - n_actual_gt
                    n_tp_boxes_per_frame[file_id] = n_actual_gt
                    file_ids_to_remove.append(file_id)
                elif n_actual_gt > n_tp_needed and n_residual_tp > 0:
                    redis_tp = min(n_actual_gt - n_tp_needed, n_residual_tp)
                    n_residual_tp -= redis_tp
                    n_tp_boxes_per_frame[file_id] += redis_tp

            pass_id += 1
            print('pass {} n_residual_tp: {}'.format(pass_id, n_residual_tp))

            if n_residual_tp == 0:
                break

            if file_ids_to_remove:
                all_file_ids = list(set(all_file_ids) - set(file_ids_to_remove))

        for __id, file_id in tqdm(enumerate(seq_det_data_by_frame), total=n_frames):

            fname = os.path.basename(file_id)

            img = cv2.imread(file_id)
            img_h, img_w = img.shape[:2]

            if show_sim:
                sim_img = np.copy(img)

            # sim_images = {
            #     _sim_prec: {
            #         _sim_rec: np.copy(img) for _sim_rec in sim_recs
            #     }
            #     for _sim_prec in sim_precs
            # }

            file_dets = seq_det_data_by_frame[file_id]
            n_file_dets = len(file_dets)

            file_fp_dets = fp_dets_dict[file_id]

            n_file_fp_needed = n_fp_boxes_per_frame[file_id]
            n_file_fp = len(file_fp_dets)

            n_sampled_fp_needed = n_file_fp_needed - n_file_fp

            if n_sampled_fp_needed > 0:

                while True:
                    # Generate three times the required boxes to hope that enough of them will be in the
                    # Valid range
                    sampled_fp_boxes_cxy_wh = np.random.multivariate_normal(
                        fp_boxes_mean, fp_boxes_cov, size=3 * n_sampled_fp_needed)
                    valid_idx_bool = np.all(np.logical_and(
                        np.less_equal(sampled_fp_boxes_cxy_wh, fp_boxes_max),
                        np.greater_equal(sampled_fp_boxes_cxy_wh, fp_boxes_min)
                    ), axis=1)
                    valid_idx = np.flatnonzero(valid_idx_bool)
                    if valid_idx.size < n_sampled_fp_needed:
                        print('Only {} / {} valid boxes found'.format(
                            valid_idx.size, sampled_fp_boxes_cxy_wh.shape[0]))
                        continue

                    sampled_fp_boxes_cxy_wh = sampled_fp_boxes_cxy_wh[valid_idx[:n_sampled_fp_needed],
                                              :]
                    break

                """cxy_wh to xy_min_max"""
                sampled_fp_boxes = np.asarray([[
                    box[0] - box[2] / 2.0,
                    box[1] - box[3] / 2.0,
                    box[0] + box[2] / 2.0,
                    box[1] + box[3] / 2.0,
                ] for box in sampled_fp_boxes_cxy_wh], dtype=np.float32)

                # __fp_end = int(__fp_start + n_sampled_fp_needed)
                # frame_sampled_boxes = sampled_fp_boxes[__fp_start:__fp_end, :]

                file_dets += [
                    ({
                         'class': gt_class,
                         'width': img_w,
                         'height': img_h,
                         'target_id': -1,
                         'filename': fname,
                         'file_path': file_id,
                         'file_id': file_id,
                         'confidence': fp_boxes_conf_mean,
                         'norm_bbox': list(norm_bbox),
                         'bbox': [norm_bbox[0] * img_w, norm_bbox[1] * img_h,
                                  norm_bbox[2] * img_w, norm_bbox[3] * img_h, ]

                     }, -1)
                    for norm_bbox in sampled_fp_boxes
                ]

                # __fp_start = __fp_end
                file_fp_dets += list(range(n_file_dets, n_file_dets + n_sampled_fp_needed))

            # n_file_sim_fp = len(file_fp_dets)
            n_sim_fp += n_file_fp_needed

            # dets_sample_probs = np.random.random_sample((len(file_fp_dets)))
            for _idx, _det_id in enumerate(file_fp_dets[:n_file_fp_needed]):
                # dets_sample_prob = dets_sample_probs[_idx]

                # if dets_sample_prob >= sim_fp_probs[_sim_prec][_sim_rec]:
                #     continue

                curr_det, curr_det_global_idx = file_dets[_det_id]
                sim_boxes.append(get_csv_dict(curr_det))

                bb_det = curr_det['bbox']
                if curr_det_global_idx == -1:
                    col = red
                else:
                    col = blue

                if show_sim:
                    cv2.rectangle(sim_img, (int(bb_det[0]), int(bb_det[1])),
                                  (int(bb_det[2]), int(bb_det[3])), col, 2)

            try:
                file_gts = seq_gt_data_dict[file_id]
                file_gts = [obj for obj in file_gts if obj["class"] == gt_class]
            except KeyError:
                file_gts = []

            n_tp_needed = n_tp_boxes_per_frame[file_id]

            n_actual_gt = len(file_gts)
            if n_actual_gt < n_tp_needed:
                n_residual_tp += n_tp_needed - n_actual_gt
                n_tp_needed = n_actual_gt
            elif n_actual_gt > n_tp_needed and n_residual_tp != 0:
                redis_tp = min(n_actual_gt - n_tp_needed, n_residual_tp)
                n_residual_tp -= redis_tp
                n_tp_needed += redis_tp

            file_gts_rand = np.random.permutation(file_gts)
            # gt_sample_probs = np.random.random_sample((len(file_gts)))
            n_sim_tp += n_tp_needed

            for _idx, _gt in enumerate(file_gts_rand[:n_tp_needed]):
                # gt_sample_prob = gt_sample_probs[_idx]
                # if gt_sample_prob >= sim_tp_probs[_sim_rec][0]:
                #     continue
                sim_boxes.append(get_csv_dict(_gt))
                bb_gt = _gt['bbox']
                col = green
                if show_sim:
                    cv2.rectangle(sim_img, (int(bb_gt[0]), int(bb_gt[1])),
                                  (int(bb_gt[2]), int(bb_gt[3])), col, 2)

            if show_sim:
                sim_det_out_fname = '{:s}_detections_rec_{:d}_prec_{:d}.csv'.format(
                    fname, int(_sim_rec * 100), int(_sim_prec * 100))

                _pause = annotate_and_show('simulated', sim_img, sim_det_out_fname,
                                           pause=_pause)

        sim_det_out_fname = 'detections_rec_{:d}_prec_{:d}.csv'.format(
            int(_sim_rec * 100), int(_sim_prec * 100))

        csv_file = os.path.join(seq_path, sim_det_out_fname)

        n_sim_total = len(sim_boxes)

        print('n_sim_fp: {}'.format(n_sim_fp))
        print('n_sim_tp: {}'.format(n_sim_tp))
        print('n_sim_total: {}'.format(n_sim_total))

        print('saving csv file to {}'.format(csv_file))
        df = pd.DataFrame(sim_boxes)
        df.to_csv(csv_file)


"""
 check if the number is a float between 0.0 and 1.0
"""


def is_float_between_0_and_1(value):
    try:
        val = float(value)
        if val > 0.0 and val < 1.0:
            return True
        else:
            return False
    except ValueError:
        return False


"""
 Calculate the AP given the recall and precision array
  1st) We compute a version of the measured precision/recall curve with
       precision monotonically decreasing
  2nd) We compute the AP as the area under this curve by numerical integration.
"""


def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
        mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    mrec = list(rec[:])
    mrec.insert(0, 0.0)  # insert 0.0 at begining of list
    mrec.append(1.0)  # insert 1.0 at end of list

    mpre = list(prec[:])
    mpre.insert(0, 0.0)  # insert 0.0 at begining of list
    mpre.append(0.0)  # insert 0.0 at end of list
    """
     This part makes the precision monotonically decreasing
      (goes from the end to the beginning)
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #   range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #   range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    """
     This part creates a list of indexes where the recall changes
    """
    # matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)  # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
      (numerical integration)
    """
    # matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


"""
 Convert the lines of a file to a list
"""


def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content if x.strip()]
    return content


"""
 Draws text in image
"""


def draw_text_in_image(img, text, pos, color, line_width):
    if isinstance(color, str):
        color = col_bgr[color]
    font = cv2.FONT_HERSHEY_PLAIN
    fontScale = 1
    lineType = 1
    bottomLeftCornerOfText = pos
    cv2.putText(img, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                color,
                lineType)
    text_width, _ = cv2.getTextSize(text, font, fontScale, lineType)[0]
    return img, (line_width + text_width)


"""
 Plot - adjust axes
"""


def adjust_axes(r, t, fig, axes):
    # get text width for re-scaling
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    # get axis width in inches
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    # get axis limit
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])


"""
 Draw plot using Matplotlib
"""


def draw_plot_func(dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color,
                   true_p_bar):
    # sort the dictionary by decreasing value, into a list of tuples
    sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
    # unpacking the list of tuples into two lists
    sorted_keys, sorted_values = zip(*sorted_dic_by_value)
    #
    if true_p_bar != "":
        """
         Special case to draw in (green=true predictions) & (red=false predictions)
        """
        fp_sorted = []
        tp_sorted = []
        for key in sorted_keys:
            fp_sorted.append(dictionary[key] - true_p_bar[key])
            tp_sorted.append(true_p_bar[key])
        plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Predictions')
        plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Predictions',
                 left=fp_sorted)
        # add legend
        plt.legend(loc='lower right')
        """
         Write number on side of bar
        """
        fig = plt.gcf()  # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            fp_val = fp_sorted[i]
            tp_val = tp_sorted[i]
            fp_str_val = " " + str(fp_val)
            tp_str_val = fp_str_val + " " + str(tp_val)
            # trick to paint multicolor with offset:
            #   first paint everything and then repaint the first number
            t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
            plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
            if i == (len(sorted_values) - 1):  # largest bar
                adjust_axes(r, t, fig, axes)
    else:
        plt.barh(range(n_classes), sorted_values, color=plot_color)
        """
         Write number on side of bar
        """
        fig = plt.gcf()  # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            str_val = " " + str(val)  # add a space before
            if val < 1.0:
                str_val = " {0:.2f}".format(val)
            t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
            # re-set axes to show number inside the figure
            if i == (len(sorted_values) - 1):  # largest bar
                adjust_axes(r, t, fig, axes)
    # set window title
    fig.canvas.set_window_title(window_title)
    # write classes in y-axis
    tick_font_size = 12
    plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)
    """
     Re-scale height accordingly
    """
    init_height = fig.get_figheight()
    # comput the matrix height in points and inches
    dpi = fig.dpi
    height_pt = n_classes * (tick_font_size * 1.4)  # 1.4 (some spacing)
    height_in = height_pt / dpi
    # compute the required figure height
    top_margin = 0.15  # in percentage of the figure height
    bottom_margin = 0.05  # in percentage of the figure height
    figure_height = height_in / (1 - top_margin - bottom_margin)
    # set new height
    if figure_height > init_height:
        fig.set_figheight(figure_height)

    # set plot title
    plt.title(plot_title, fontsize=14)
    # set axis titles
    # plt.xlabel('classes')
    plt.xlabel(x_label, fontsize='large')
    # adjust size of window
    fig.tight_layout()
    # save the plot
    fig.savefig(output_path)
    # show image
    if to_show:
        plt.show()
    # close the plot
    plt.close()


def draw_dotted_line(img, pt1, pt2, color, thickness=1, gap=7):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        p = (x, y)
        pts.append(p)

    for p in pts:
        cv2.circle(img, p, thickness, color, -1)


def draw_dotted_poly(img, pts, color, thickness=1):
    s = pts[0]
    e = pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s = e
        e = p
        draw_dotted_line(img, s, e, color, thickness)


def draw_dotted_rect(img, pt1, pt2, color, thickness=1):
    pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
    draw_dotted_poly(img, pts, color, thickness)


def draw_box(frame, box, _id=None, color='black', thickness=2,
             is_dotted=0, transparency=0., xywh=True, norm=False):
    """
    :type frame: np.ndarray
    :type _id: int | str | None
    :param color: indexes into col_bgr
    :type color: str
    :type thickness: int
    :type is_dotted: int
    :type transparency: float
    :rtype: None
    """
    if not isinstance(box, np.ndarray):
        box = np.asarray(box)

    if np.any(np.isnan(box)):
        print('invalid location provided: {}'.format(box))
        return

    if isinstance(color, str):
        color = col_bgr[color]

    if isinstance(box, np.ndarray):
        box = list(box.squeeze())

    if xywh:
        pt1 = (box[0], box[1])
        pt2 = (box[0] + box[2],
               box[1] + box[3])
    else:
        pt1 = (box[0], box[1])
        pt2 = (box[2], box[3])

    img_h, img_w = frame.shape[:2]

    if norm:
        pt1 = (pt1[0] * img_w, pt1[1] * img_h)
        pt2 = (pt2[0] * img_w, pt2[1] * img_h)

    pt1 = tuple(map(int, pt1))
    pt2 = tuple(map(int, pt2))

    if transparency > 0:
        _frame = np.copy(frame)
    else:
        _frame = frame

    if is_dotted:
        draw_dotted_rect(_frame, pt1, pt2, color, thickness=thickness)
    else:
        cv2.rectangle(_frame, pt1, pt2, color, thickness=thickness)

    if transparency > 0:
        frame[pt1[1]:pt2[1], pt1[0]:pt2[0], ...] = (
                frame[pt1[1]:pt2[1], pt1[0]:pt2[0], ...].astype(np.float32) * (1 - transparency) +
                _frame[pt1[1]:pt2[1], pt1[0]:pt2[0], ...].astype(np.float32) * transparency
        ).astype(frame.dtype)

    if _id is not None:
        font_line_type = cv2.LINE_AA
        cv2.putText(frame, str(_id), (int(box[0] - 1), int(box[1] - 1)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, font_line_type)


def draw_boxes(frame, boxes, **kwargs):
    if len(boxes.shape) == 1:
        boxes = np.expand_dims(boxes, axis=0)
    for box in boxes:
        draw_box(frame, box, **kwargs)


def compute_overlap(iou, ioa_1, ioa_2, object_1, objects_2):
    """

    compute overlap of a single object with one or more objects
    specialized version for greater speed

    :type iou: np.ndarray | None
    :type ioa_1: np.ndarray | None
    :type ioa_2: np.ndarray | None
    :type object_1: np.ndarray
    :type objects_2: np.ndarray
    :rtype: None
    """

    n1 = object_1.shape[0]

    assert n1 == 1, "object_1 should be a single object"

    n = objects_2.shape[0]

    ul_coord_1 = object_1[0, :2].reshape((1, 2))
    ul_coords_2 = objects_2[:, :2]  # n x 2
    ul_coords_inter = np.maximum(ul_coord_1, ul_coords_2)  # n x 2

    size_1 = object_1[0, 2:].reshape((1, 2))
    sizes_2 = objects_2[:, 2:]  # n x 2

    br_coord_1 = ul_coord_1 + size_1 - 1
    br_coords_2 = ul_coords_2 + sizes_2 - 1  # n x 2
    br_coords_inter = np.minimum(br_coord_1, br_coords_2)  # n x 2

    sizes_inter = br_coords_inter - ul_coords_inter + 1
    sizes_inter[sizes_inter < 0] = 0
    areas_inter = np.multiply(sizes_inter[:, 0], sizes_inter[:, 1]).reshape((n, 1))  # n x 1

    areas_2 = None
    if iou is not None:
        areas_2 = np.multiply(sizes_2[:, 0], sizes_2[:, 1]).reshape((n, 1))  # n x 1
        area_union = size_1[0, 0] * size_1[0, 1] + areas_2 - areas_inter
        iou[:] = np.divide(areas_inter, area_union)
    if ioa_1 is not None:
        """intersection / area of object 1"""
        ioa_1[:] = np.divide(areas_inter, size_1[0, 0] * size_1[0, 1])
    if ioa_2 is not None:
        """intersection / area of object 2"""
        if areas_2 is None:
            areas_2 = np.multiply(sizes_2[:, 0], sizes_2[:, 1])
        ioa_2[:] = np.divide(areas_inter, areas_2)


def compute_iou_single(bb1, bb2):
    x1, y1, w1, h1 = bb1
    x2, y2, w2, h2 = bb2

    # bb1 = [bb1[0], bb1[1], bb1[0] + bb1[2], bb1[1] + bb1[3]]
    # bb2 = [bb2[0], bb2[1], bb2[0] + bb2[2], bb2[1] + bb2[3]]

    bb_intersect = [max(x1, x2),
                    max(y1, y2),
                    min(x1 + w1, x2 + w2),
                    min(y1 + h1, y2 + h2)]

    iw = bb_intersect[2] - bb_intersect[0] + 1
    ih = bb_intersect[3] - bb_intersect[1] + 1

    if iw <= 0 or ih <= 0:
        return 0
    area_intersect = (iw * ih)
    area_union = (w1 + 1) * (h1 + 1) + (w2 + 1) * (h2 + 1) - area_intersect
    iou = area_intersect / area_union
    return iou


def get_shifted_boxes(
        anchor_box, img, n_samples,
        min_anchor_iou, max_anchor_iou,
        min_shift_ratio, max_shift_ratio,
        min_resize_ratio=None, max_resize_ratio=None,
        min_size=20, max_size_ratio=0.8,
        gt_boxes=None,
        max_gt_iou=None,
        sampled_boxes=None,
        max_sampled_iou=0.5,
        max_iters=100,
        name='',
        vis=0):
    if not name:
        name = 'get_shifted_boxes'

    if np.any(np.isnan(anchor_box)):
        print('invalid location provided: {}'.format(anchor_box))
        return

    assert 0 <= min_shift_ratio <= max_shift_ratio, "Invalid shift ratios provided"
    if max_gt_iou is None:
        max_gt_iou = max_anchor_iou

    if min_resize_ratio is None:
        min_resize_ratio = min_shift_ratio

    if max_resize_ratio is None:
        max_resize_ratio = max_shift_ratio

    img_h, img_w = img.shape[:2]
    x, y, w, h = anchor_box.squeeze()
    x2, y2 = x + w, y + h

    _x, _x2 = clamp((x, x2), 0, img_w)
    _y, _y2 = clamp((y, y2), 0, img_h)

    _w, _h = _x2 - _x, _y2 - _y

    if vis:
        disp_img = np.copy(img)
        if len(disp_img.shape) == 2:
            """grey scale to RGB"""
            disp_img = np.stack((disp_img,) * 3, axis=2)

    valid_boxes = []

    max_box_w, max_box_h = float(img_w) * max_size_ratio, float(img_h) * max_size_ratio

    if _w < min_size or _h < min_size or _w > img_w or _h > img_h or (_w > max_box_w and _h > max_box_h):
        if vis:
            draw_box(disp_img, [x, y, w, h], _id=None, color='blue', thickness=2,
                     is_dotted=0, transparency=0.)
            draw_box(disp_img, [_x, _y, _w, _h], _id=None, color='blue', thickness=2,
                     is_dotted=1, transparency=0.)

            annotate_and_show(name, disp_img, 'annoying crap')

        # print('\nignoring amazingly annoying vile filthy invalid box: {} clamped to: {}\n'.format(
        #     (x, y, w, h), (_x, _y, _w, _h)
        # ))
        return valid_boxes

    x, y, w, h = _x, _y, _w, _h

    # min_dx, min_dy = shift_coeff*x, shift_coeff*y
    # min_dw, min_dh= shift_coeff*w, shift_coeff*h

    if sampled_boxes is None:
        _sampled_boxes = []
    else:
        _sampled_boxes = sampled_boxes.copy()

    if vis:
        if gt_boxes is not None:
            draw_boxes(disp_img, gt_boxes, ids=None, colors='green', thickness=2,
                       is_dotted=0, transparency=0.)

        if _sampled_boxes is not None:
            draw_boxes(disp_img, _sampled_boxes, ids=None, colors='magenta', thickness=2,
                       is_dotted=1, transparency=0.)

        draw_box(disp_img, anchor_box, _id=None, color='blue', thickness=2,
                 is_dotted=0, transparency=0.)

    n_valid_boxes = 0
    txt = ''

    for iter_id in range(max_iters):

        txt = ''

        shift_x = random.uniform(min_shift_ratio, max_shift_ratio) * random.choice([1, 0, -1])
        shift_y = random.uniform(min_shift_ratio, max_shift_ratio) * random.choice([1, 0, -1])
        shift_w = random.uniform(min_resize_ratio, max_resize_ratio) * random.choice([1, 0, -1])
        shift_h = random.uniform(min_resize_ratio, max_resize_ratio) * random.choice([1, 0, -1])

        x2 = x + w * shift_x
        y2 = y + h * shift_y
        w2 = w * (1 + shift_w)
        h2 = h * (1 + shift_h)

        if w2 <= min_size:
            continue

        if h2 <= min_size:
            continue

        if x2 <= 0 or x2 + w2 >= img_w:
            continue

        if y2 <= 0 or y2 + h2 >= img_h:
            continue

        if w2 > max_box_w and h2 > max_box_h:
            continue

        shifted_box = np.asarray([x2, y2, w2, h2])

        # iou = np.empty((1, 1))
        iou = compute_iou_single(shifted_box, anchor_box)

        if iou < min_anchor_iou or iou > max_anchor_iou:
            continue

        txt = 'shift: [{:.2f},{:.2f},{:.2f},{:.2f}] iou: {:.2f}'.format(shift_x, shift_y, shift_w, shift_h, iou)

        if _sampled_boxes:
            sampled_iou = np.empty((len(_sampled_boxes), 1))
            compute_overlap(sampled_iou, None, None, shifted_box.reshape((1, 4)),
                            np.asarray(_sampled_boxes).reshape((-1, 4)))

            _max_sampled_iou = np.amax(sampled_iou)
            if _max_sampled_iou > max_sampled_iou:
                continue

            txt += ' sampled_iou: {:.2f}'.format(_max_sampled_iou)

        if gt_boxes is not None:
            gt_iou = np.empty((gt_boxes.shape[0], 1))
            compute_overlap(gt_iou, None, None, shifted_box.reshape((1, 4)), gt_boxes)

            _max_gt_iou = np.amax(gt_iou)
            if _max_gt_iou > max_gt_iou:
                continue

            txt += ' gt_iou: {:.2f}'.format(_max_gt_iou)

        # iou = iou.item()

        valid_boxes.append(shifted_box)
        _sampled_boxes.append(shifted_box)

        n_valid_boxes += 1

        if vis:
            txt += ' iters: {:d}'.format(iter_id)

            draw_box(disp_img, shifted_box, _id=None, color='red', thickness=2,
                     is_dotted=0, transparency=0.)

            annotate_and_show(name, disp_img, txt)

        if n_valid_boxes >= n_samples:
            break
    else:
        if vis:
            txt += ' iters: {:d}'.format(iter_id)

            draw_box(disp_img, shifted_box, _id=None, color='red', thickness=2,
                     is_dotted=0, transparency=0.)

            annotate_and_show(name, disp_img, txt)

        # print('max iters {} reached with only {} / {} valid sampled boxes found'.format(
        #     max_iters, n_valid_boxes, n_samples))

    # valid_boxes = np.asarray(valid_boxes).reshape((n_samples, 4))
    return valid_boxes
