import operator
import os
import sys
import traceback
import logging

import cv2
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
from datetime import datetime

from io import StringIO

from PIL import Image

import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_auc_score, roc_curve
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
    'magenta': (255, 0, 255)
}


def print_stats(stats, name='', fmt='.3f'):
    if name:
        print(f"\n{name}")
    print(tabulate(stats, headers='keys', tablefmt="orgtbl", floatfmt=fmt))


def put_text_with_background(img, text, loc, col, bgr_col, **kwargs):
    text_offset_x, text_offset_y = loc

    (text_width, text_height) = cv2.getTextSize(text, **kwargs)[0]
    box_coords = ((text_offset_x, text_offset_y + 5), (text_offset_x + text_width, text_offset_y - text_height))
    cv2.rectangle(img, box_coords[0], box_coords[1], bgr_col, cv2.FILLED)

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


def drawBox(image, xmin, ymin, xmax, ymax, box_color=(0, 255, 0), label=None, font_size=0.3, mask=None):
    # if cv2.__version__.startswith('3'):
    #     font_line_type = cv2.LINE_AA
    # else:
    #     font_line_type = cv2.CV_AA

    image_float = image.astype(np.float32)

    if mask is not None:
        mask_pts = np.asarray(mask).reshape((-1, 1, 2)).astype(np.int32)
        cv2.drawContours(image_float, mask_pts, -1, box_color, thickness=2, lineType=cv2.LINE_AA)
    else:
        cv2.rectangle(image_float, (int(xmin), int(ymin)), (int(xmax), int(ymax)), box_color)

    image[:] = image_float.astype(image.dtype)

    _bb = [xmin, ymin, xmax, ymax]
    if _bb[1] > 10:
        y_loc = int(_bb[1] - 5)
    else:
        y_loc = int(_bb[3] + 5)
    if label is not None:
        cv2.putText(image, label, (int(_bb[0] - 1), y_loc), cv2.FONT_HERSHEY_SIMPLEX,
                    font_size, box_color, 1, cv2.LINE_AA)


def resize_ar(src_img, width=0, height=0, return_factors=False,
              placement_type=0, only_border=0, only_shrink=0):
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


def linux_path(*args, **kwargs):
    return os.path.join(*args, **kwargs).replace(os.sep, '/')


def add_suffix(src_path, suffix, dst_ext=''):
    # abs_src_path = os.path.abspath(src_path)
    src_dir = os.path.dirname(src_path)
    src_name, src_ext = os.path.splitext(os.path.basename(src_path))
    if not dst_ext:
        dst_ext = src_ext

    dst_path = linux_path(src_dir, src_name + '_' + suffix + dst_ext)
    return dst_path


def compute_binary_cls_metrics(
        thresholds, probs, labels, class_names,
        fp_thresholds=(0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)):
    if len(probs.shape) == 1:
        """assume only class 0 probs are provided"""
        probs = np.stack((probs, 1 - probs), axis=1)

    n_val, n_classes = probs.shape

    assert n_classes == 2, "n_classes must be 2 for binary_cls_metrics"
    assert len(class_names) == 2, "n_classes must be 2 for binary_cls_metrics"
    assert len(labels) == n_val, "n_val mismatch"

    assert n_val > 0, "no labels found"

    n_conf_thresholds = len(thresholds)

    conf_to_acc = np.zeros((n_conf_thresholds, 4), dtype=np.float32)
    class_tp = np.zeros((n_conf_thresholds, 2), dtype=np.float32)
    class_fp = np.zeros((n_conf_thresholds, 2), dtype=np.float32)

    labels = labels.reshape((n_val, 1))

    conf_to_acc[:, 0] = thresholds.squeeze()

    for conf_id, conf_threshold in enumerate(tqdm(thresholds, ncols=70)):
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

    y_true = 1 - labels.squeeze()
    y_score = probs[:, 0].squeeze()

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
                            raise AssertionError('decreasing roc_auc')

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
        enable_tests=True
):
    assert len(gt_classes) == 2, "Number of classes must be 2"

    """extract stats"""
    if True:
        csv_columns_max_tp = ['FP_threshold', 'TP']
        csv_columns_acc = ['confidence_threshold', 'Overall', gt_classes[0], gt_classes[1]]
        csv_columns_tp_fp = ['confidence_threshold', 'TP', 'FP']
        csv_columns_auc = ['FP_threshold', 'AUC']

        max_tp_out_root_dir = linux_path(out_root_dir, 'max_tp')
        roc_auc_out_root_dir = linux_path(out_root_dir, 'roc_auc')

        os.makedirs(max_tp_out_root_dir, exist_ok=1)
        os.makedirs(roc_auc_out_root_dir, exist_ok=1)

        stats_0 = class_stats[0]
        stats_1 = class_stats[1]

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

        """don't even rememver what crap this is"""
        # n_dets_0 = stats_0['tp_sum'] + stats_1['fp_cls_sum']
        # n_dets_1 = stats_1['tp_sum'] + stats_0['fp_cls_sum']

        n_dets_0 = stats_0['n_dets']
        n_dets_1 = stats_1['n_dets']

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
        print(_txt)
        _rec_prec = (val1[idx] + val2[idx]) / 2.0
        _score = conf_class[idx]
    else:
        _txt = f'Intersection at {val1[idx]} for confidence: {conf_class[idx]} with idx: {idx}'
        print(_txt)
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


def draw_objs(img, objs, alpha=0.5, class_name_to_col=None, col=None,
              in_place=False, bbox=True, mask=True, thickness=2, check_bb=0):
    if in_place:
        vis_img = img
    else:
        vis_img = np.copy(img).astype(np.float32)

    mask_img = np.zeros_like(vis_img)

    vis_h, vis_w = vis_img.shape[:2]

    resize_factor = 1.0

    for obj in objs:
        label = obj['class']
        if col is None:
            class_col = class_name_to_col[label]
        else:
            class_col = col
        class_col = col_bgr[class_col]

        rle = obj['mask']
        mask_orig = mask_util.decode(rle).squeeze()
        mask_orig = np.ascontiguousarray(mask_orig, dtype=np.uint8)

        mask_h, mask_w = mask_orig.shape[:2]

        if (mask_h, mask_w) != (vis_h, vis_w):
            resize_factor = vis_h / mask_h
            mask_uint = cv2.resize(mask_orig, (vis_w, vis_h))
        else:
            mask_uint = mask_orig

        if bbox:
            xmin, ymin, xmax, ymax = obj["bbox"]

            if check_bb:
                log_dir = 'log/masks'
                os.makedirs(log_dir, exist_ok=1)

                time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")

                bb = (xmin, ymin, xmax, ymax)

                mask_pts, mask_bb, is_multi = contour_pts_from_mask(mask_orig)

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

            if resize_factor != 1.0:
                xmin *= resize_factor
                ymin *= resize_factor
                xmax *= resize_factor
                ymax *= resize_factor

            cv2.rectangle(
                vis_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_col, thickness)

        if mask:
            mask_binary = mask_uint.astype(bool)

            mask_img[mask_binary] = class_col
            vis_img[mask_binary] = (alpha * vis_img[mask_binary] +
                                    (1 - alpha) * mask_img[mask_binary])

    vis_img = vis_img.astype(np.uint8)

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


def binary_mask_to_rle_coco(binary_mask):
    binary_mask_fortran = np.asfortranarray(binary_mask, dtype="uint8")
    rle = mask_util.encode(binary_mask_fortran)
    rle["counts"] = rle["counts"].decode("utf-8")

    return rle


def contour_pts_to_mask(contour_pts, patch_img, col=(255, 255, 255)):
    # np.savetxt('contourPtsToMask_mask_pts.txt', contour_pts, fmt='%.6f')

    mask_img = np.zeros_like(patch_img, dtype=np.uint8)
    # if not isinstance(contour_pts, list):
    #     raise SystemError('contour_pts must be a list rather than {}'.format(type(contour_pts)))
    if len(contour_pts) > 0:
        mask_img = cv2.fillPoly(mask_img, np.array([contour_pts, ], dtype=np.int32), col)
    blended_img = np.array(Image.blend(Image.fromarray(patch_img), Image.fromarray(mask_img), 0.5))

    return mask_img, blended_img


def contour_pts_from_mask(mask_img):
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

    return contour_pts, (x, y, w, h), is_multi


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


def mask_pts_to_img(mask_pts, img_h, img_w, to_rle):
    mask_img = np.zeros((img_h, img_w), dtype=np.uint8)
    mask_img = cv2.fillPoly(mask_img, np.array([mask_pts, ], dtype=np.int32), 1)

    # mask_img_vis = (mask_img * 255).astype(np.uint8)
    # mask_img_vis = resize_ar(mask_img_vis, 1280, 720)
    # cv2.imshow('mask_img_vis', mask_img_vis)
    # cv2.waitKey(0)

    bin_mask_img = mask_img.astype(bool)

    if to_rle:
        mask_rle = binary_mask_to_rle_coco(bin_mask_img)
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
    # write classes in y axis
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


def get_shifted_boxes(anchor_box, img, n_samples,
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
    """

    :param anchor_box:
    :param img:
    :param n_samples:
    :param min_anchor_iou:
    :param max_anchor_iou:
    :param min_shift_ratio:
    :param max_shift_ratio:
    :param min_resize_ratio:
    :param max_resize_ratio:
    :param min_size:
    :param max_size_ratio:
    :param gt_boxes:
    :param max_gt_iou:
    :param sampled_boxes:
    :param max_sampled_iou:
    :param max_iters:
    :param name:
    :param vis:
    :return:
    """

    # vis = 1

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
