import glob
import os
import sys
import shutil
import pickle
import copy
import time
import multiprocessing
from multiprocessing.pool import ThreadPool
import json
import math

import functools
from contextlib import closing
from datetime import datetime

import imagesize

import paramparse

# from skimage.io import imread
# from PIL import Image


import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pprint import pformat, pprint
from tqdm import tqdm

from prettytable import PrettyTable
from tabulate import tabulate
from collections import OrderedDict
import itertools

from eval_utils import mask_str_to_img, perform_global_association, draw_text_in_image, get_max_iou_obj, \
    compute_thresh_rec_prec, voc_ap, get_intersection, binary_cls_metrics, file_lines_to_list, \
    arr_to_csv, draw_objs, norm_auc, resize_ar, sortKey, linux_path, add_suffix, ImageSequenceWriter

import paramparse


class Params:
    """

    :ivar check_seq_name: 'iou_thresh',
    :ivar delete_tmp_files: 'None',
    :ivar det_paths: 'file containing list of detection folders',
    :ivar draw_plot: 'None',
    :ivar gt_paths: 'file containing list of GT folders',
    :ivar gt_root_dir: 'folder to save the animation result in',
    :ivar img_ext: 'image extension',
    :ivar img_paths: 'file containing list of image folders',
    :ivar img_root_dir: 'folder to save the animation result in',
    :ivar iou_thresh: 'iou_thresh',
    :ivar labels_path: 'text file containing class labels',
    :ivar no_animation: 'no animation is shown.',
    :ivar no_plot: 'no plot is shown.',
    :ivar out_root_dir: 'out_fname',
    :ivar quiet: 'minimalistic console output.',
    :ivar save_vis: 'None',
    :ivar save_file_res: 'resolution of the saved video',
    :ivar score_thresholds: 'iou_thresh',
    :ivar set_class_iou: 'set IoU for a specific class.',
    :ivar show_vis: 'None',
    :ivar show_gt: 'None',
    :ivar show_tp: 'None',
    :ivar show_stats: 'None',
    :ivar show_text: 'None',
    :ivar vid_fmt: 'comma separated triple to specify the output video format: codec, FPS, extension',
    :ivar vis_out_fname: 'file to save the animation result in',
    :ivar a: 'no animation is shown.',
    :ivar p: 'no plot is shown.',
    :ivar eval_sim: evaluate already created simulated detections,
    :ivar assoc_method:
            '0: new method with det-to-GT and GT-to-det symmetrical associations;'
            '1: old method with only det-to-GT associations',
    """

    def __init__(self):
        self.cfg_ext = 'cfg'
        self.cfg_root = 'cfg'
        self.cfg = ''
        self.check_seq_name = 1
        self.delete_tmp_files = 0
        self.det_paths = ''
        self.combine_dets = 0
        self.draw_plot = 0
        self.gt_paths = ''
        self.gt_root_dir = ''

        self.img_ext = 'jpg'
        self.img_paths = ''
        self.img_root_dir = ''
        self.all_img_dirs = 1

        self.iou_thresh = 0.25
        self.labels_root = 'lists/classes/'
        self.labels_path = ''

        self.out_root_dir = f'/data/mAP'

        self.no_animation = False
        self.no_plot = False
        self.gt_pkl_dir = 'log/pkl'
        self.det_pkl_dir = ''
        self.load_gt = 1
        self.gt_pkl = ''
        self.load_det = 0
        self.det_pkl = ''
        self.quiet = False

        self.score_thresholds = [0, ]
        self.set_class_iou = [None, ]
        self.show_vis = 0
        self.show_gt = 1
        self.show_tp = 0
        self.show_stats = 1
        self.gt_check = 1
        self.assoc_method = 1

        self.fix_csv_cols = -1
        self.fix_gt_cols = 0
        self.fix_det_cols = 1

        self.ignore_invalid_class = 1
        self.enable_mask = 1

        self.start_id = 0
        self.end_id = -1

        self.img_start_id = 0
        self.img_end_id = -1

        self.write_summary = 1
        self.compute_opt = 0
        self.rec_ratios = ()
        self.wt_avg = 0
        self.n_threads = 0

        """allow_missing_gt=2 skips detections for files without GT"""
        self.allow_missing_gt = 1

        self.show_text = 1
        self.gt_csv_name = 'annotations.csv'
        self.a = False
        self.p = False
        self.detection_names = []

        self.vis_alpha = 0.5
        self.save_sim_dets = 0
        self.show_sim = 0
        self.eval_sim = 0
        self.sim_precs = (0.5, 0.60, 0.70, 0.80, 0.90, 1,)
        self.sim_recs = (0.5, 0.60, 0.70, 0.80, 0.90, 1,)

        """imagewise"""
        self.iw = 0
        self.compute_rec_prec = 1
        self.img_dir_name = ''

        self.nms_thresh = [0, ]
        self.n_proc = 1

        self.save_vis = 0
        self.save_classes = []
        self.save_cats = ['fn_det', 'fn_cls', 'fp_nex-whole', 'fp_nex-part', 'fp_cls', 'fp_dup']
        self.auto_suffix = 0
        self.save_suffix = ''
        self.save_file_res = '1920x1080'
        # self.save_file_res = '1600x900'
        # self.save_file_res = '1280x720'
        self.vid_fmt = 'mp4v,2,mp4'
        self.check_det = 0

        self.fps_to_gt = 1

        self._sweep_params = [
            'nms_thresh',
        ]


def get_vis_size(src_img, mult, save_w, save_h, bottom_border):
    temp_vis = np.concatenate((src_img,) * mult, axis=1)
    temp_vis_res = resize_ar(temp_vis, save_w, save_h - bottom_border, crop=1)
    temp_vis_h, temp_vis_w = temp_vis_res.shape[:2]

    vis_h, vis_w = temp_vis_h, int(temp_vis_w / mult)

    assert vis_h <= save_h and vis_w <= save_w, \
        f"vis size {vis_w} x {vis_h} > save size {save_w} x {save_h}"

    return vis_h, vis_w


def draw_and_concat(src_img, frame_det_data, frame_gt_data, class_name_to_col, vis_alpha, vis_w, vis_h,
                    vert_stack, check_det):
    dets_vis_img = resize_ar(src_img, vis_w, vis_h, crop=1)
    gt_vis_img = resize_ar(src_img, vis_w, vis_h, crop=1)

    dets_vis_img = draw_objs(dets_vis_img, frame_det_data, vis_alpha, class_name_to_col, check_bb=check_det,
                             thickness=1)
    gt_vis_img = draw_objs(gt_vis_img, frame_gt_data, vis_alpha, class_name_to_col, thickness=1)

    cat_img_vis = np.concatenate((gt_vis_img, dets_vis_img), axis=0 if vert_stack else 1)

    return cat_img_vis


def evaluate(params, seq_paths, gt_classes, gt_path_list, det_path_list, out_root_dir, class_name_to_col,
             img_start_id=-1, img_end_id=-1,
             _gt_data_dict=None, raw_det_data_dict=None, eval_result_dict=None, fps_to_gt=1, json_out_dir=None):
    """

    :param Params params:
    """
    """general init"""
    if True:
        assert out_root_dir, "out_root_dir must be provided"

        compute_rec_prec = params.compute_rec_prec
        assoc_method = params.assoc_method
        det_pkl_dir = params.det_pkl_dir
        gt_pkl_dir = params.gt_pkl_dir
        save_file_res = params.save_file_res
        vid_fmt = params.vid_fmt
        iou_thresh = params.iou_thresh
        save_vis = params.save_vis
        save_classes = params.save_classes
        save_cats = params.save_cats
        show_vis = params.show_vis
        show_text = params.show_text
        show_stats = params.show_stats
        show_gt = params.show_gt
        show_tp = params.show_tp
        draw_plot = params.draw_plot
        # delete_tmp_files = params.delete_tmp_files
        score_thresholds = params.score_thresholds
        check_seq_name = params.check_seq_name
        gt_check = params.gt_check
        save_sim_dets = params.save_sim_dets
        show_sim = params.show_sim
        sim_precs = params.sim_precs
        sim_recs = params.sim_recs
        enable_mask = params.enable_mask

        fix_csv_cols = params.fix_csv_cols
        if fix_csv_cols >= 0:
            params.fix_gt_cols = params.fix_det_cols = fix_csv_cols

        fix_gt_cols = params.fix_gt_cols
        fix_det_cols = params.fix_det_cols
        ignore_invalid_class = params.ignore_invalid_class
        write_summary = params.write_summary
        compute_opt = params.compute_opt
        rec_ratios = params.rec_ratios
        wt_avg = params.wt_avg
        n_threads = params.n_threads
        allow_missing_gt = params.allow_missing_gt
        img_dir_name = params.img_dir_name

        n_score_thresholds = len(score_thresholds)
        score_thresh = score_thresholds[0]

        score_thresholds = np.asarray(score_thresholds).squeeze()

        img_exts = ['jpg', 'png', 'bmp']

        save_w, save_h = [int(x) for x in save_file_res.split('x')]

        codec, fps, vid_ext = vid_fmt.split(',')
        fps = int(fps)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        # fourcc = -1

        show_pbar = not show_vis

        cls_cat_types = [
            "all",
            "tp",
            "fp_dup",
            "fp_cls",
            "fp_nex-part",
            "fp_nex-whole",
            "fn_det",
            "fn_cls",
        ]

        if save_vis:
            if not save_classes:
                """save all classes"""
                save_classes = gt_classes[:]

            if not save_cats:
                """save all status types"""
                if show_tp == 2:
                    """
                    show_tp=0: don't show TP
                    show_tp=2: show only TP
                    """
                    save_cats = ['tp', ]
                else:
                    save_cats = cls_cat_types[:]

        n_seq = len(seq_paths)

        if _gt_data_dict is None:
            if n_seq != len(seq_paths):
                raise AssertionError(
                    'Mismatch between the no. of image ({})  and GT ({}) sequences'.format(n_seq, len(seq_paths)))

        n_det_paths = len(det_path_list)
        if raw_det_data_dict is None:
            if n_seq < n_det_paths:
                print(f'Mismatch between n_seq ({n_seq}) and n_det_paths ({n_det_paths})')
                det_path_list = det_path_list[:n_seq]

        seq_root_dirs = [os.path.dirname(x) for x in seq_paths]
        seq_name_list = [os.path.basename(x) for x in seq_paths]

        if json_out_dir is None:
            json_out_dir = seq_root_dirs[0]

        # if not os.path.exists(pkl_files_path):
        #     os.makedirs(pkl_files_path)

        plots_out_dir = linux_path(out_root_dir, 'plots')
        if draw_plot:
            print('Saving plots to: {}'.format(plots_out_dir))
            os.makedirs(plots_out_dir, exist_ok=1)

        if not det_pkl_dir:
            det_pkl_dir = out_root_dir

        if not gt_pkl_dir:
            gt_pkl_dir = out_root_dir

        det_pkl = params.det_pkl
        if not det_pkl:
            det_pkl = "raw_det_data_dict.pkl"
        det_pkl = linux_path(det_pkl_dir, det_pkl)

        gt_pkl = params.gt_pkl
        if not gt_pkl:
            gt_pkl = "gt_data_dict.pkl"

        gt_pkl = linux_path(gt_pkl_dir, gt_pkl)

        if img_start_id >= 0 and img_end_id >= 0:
            assert img_end_id >= img_start_id, "img_end_id must be >= img_start_id"
            img_suffix = f'img_{img_start_id}'

            if img_start_id != img_end_id:
                img_suffix = f'{img_suffix}_{img_end_id}'

            gt_pkl = add_suffix(gt_pkl, img_suffix)
            det_pkl = add_suffix(det_pkl, img_suffix)

        gt_class_data_dict = {
            gt_class: {} for gt_class in gt_classes
        }

        if _gt_data_dict is not None:
            gt_loaded = 1
            gt_data_dict = copy.deepcopy(_gt_data_dict)
        elif params.load_gt and params.gt_pkl:
            """load GT data only if gt_pkl is explicitly provided"""
            if not os.path.exists(gt_pkl):
                msg = f"gt_pkl does not exist: {gt_pkl}"
                if params.iw:
                    print('\n' + msg + '\n')
                    return None
                raise AssertionError(msg)

            print(f'loading GT data from {gt_pkl}')
            with open(gt_pkl, 'rb') as f:
                gt_data_dict = pickle.load(f)

            gt_loaded = 1
        else:
            gt_data_dict = {}
            gt_loaded = 0
            print('Generating GT data...')

        if gt_loaded:
            for _seq_path, _seq_gt_data_dict in gt_data_dict:
                for gt_class in gt_classes:
                    gt_class_data_dict[gt_class][_seq_path] = []

                for obj in _seq_gt_data_dict:
                    gt_class_data_dict[obj['class']][_seq_path].append(obj)

        if raw_det_data_dict is not None:
            det_loaded = 1
        else:
            """det data pkl must exist if it is explicitly provided but it is loaded if it exists 
            even if not explicitly provided"""
            if params.det_pkl:
                assert os.path.exists(det_pkl), f"det_pkl does not exist: {det_pkl}"

            if params.load_det and os.path.isfile(det_pkl):
                print(f'loading detection data from {det_pkl}')
                with open(det_pkl, 'rb') as f:
                    raw_det_data_dict = pickle.load(f)
                det_loaded = 1
            else:
                raw_det_data_dict = {}
                det_loaded = 0

        det_data_dict = {}

        if not det_loaded:
            print('Generating detection data...')

        _pause = 1
        gt_counter_per_class = {}
        gt_start_t = time.time()

        gt_seq_names = [os.path.basename(_gt_path) for _gt_path in gt_path_list]
        # unique_gt_seq_names = list(set(gt_seq_names))
        if len(gt_seq_names) > 0 and gt_seq_names[0].endswith('.csv'):
            gt_seq_names = [os.path.basename(os.path.dirname(_gt_path)) for _gt_path in gt_path_list]
            # unique_gt_seq_names = list(set(gt_seq_names))
            # assert len(gt_seq_names) == len(unique_gt_seq_names), "unable to find gt_seq_names"

        csv_rename_dict = {
            'ImageID': 'filename',
            'XMin': 'xmin',
            'YMin': 'ymin',
            'XMax': 'xmax',
            'YMax': 'ymax',
            'LabelName': 'class',
            'MaskXY': 'mask',
            'Confidence': 'confidence',
        }
        all_img_paths = []

        img_path_to_size = {}

    """read gt and det"""
    for seq_idx, (_gt_path, gt_seq_name) in enumerate(zip(gt_path_list, gt_seq_names)):

        # if gt_loaded and det_loaded:
        #     break

        seq_name = seq_name_list[seq_idx]
        seq_path = seq_paths[seq_idx]

        if img_dir_name:
            seq_img_dir = linux_path(seq_path, img_dir_name)
        else:
            seq_img_dir = seq_path

        seq_img_gen = [[os.path.join(dirpath, f) for f in filenames if
                        os.path.splitext(f.lower())[1][1:] in img_exts]
                       for (dirpath, dirnames, filenames) in os.walk(seq_img_dir, followlinks=True)]
        seq_img_paths = [item for sublist in seq_img_gen for item in sublist]

        seq_img_name_to_path = {
            os.path.basename(seq_img_path): seq_img_path for seq_img_path in seq_img_paths
        }

        if gt_loaded:
            gt_img_paths = sorted(list(gt_data_dict[seq_path].keys()))
            all_img_paths += gt_img_paths

        print(f'\n\nProcessing sequence {seq_idx + 1:d}/{n_seq:d}')
        print(f'seq_path: {seq_path:s}')

        if not gt_loaded:

            gt_path = _gt_path
            if not os.path.isfile(gt_path):
                raise IOError(f'GT file: {gt_path} does not exist')

            print(f'\ngt_path: {gt_path:s}')

            seq_gt_data_dict = {}

            for gt_class in gt_classes:
                gt_class_data_dict[gt_class][seq_path] = []

            df_gt = pd.read_csv(gt_path)

            df_gt = df_gt.dropna(axis=0)

            if fix_gt_cols:
                df_gt = df_gt.rename(columns=csv_rename_dict)

            df_gt["filename"] = df_gt["filename"].apply(lambda x: seq_img_name_to_path[os.path.basename(x)])

            grouped_gt = df_gt.groupby("filename")

            n_gt = len(df_gt)
            gt_filenames = sorted(list(grouped_gt.groups.keys()))

            n_gt_filenames = len(gt_filenames)

            print(f'{gt_path} --> {n_gt} labels for {n_gt_filenames} images')

            n_gt_filenames = len(gt_filenames)

            if img_start_id >= 0 and img_end_id >= 0:
                if img_end_id >= n_gt_filenames:
                    if params.iw:
                        print('img_end_id exceeds n_gt_filenames')
                        return None
                    raise AssertionError('img_end_id exceeds n_gt_filenames')

                gt_filenames = gt_filenames[img_start_id:img_end_id + 1]
                n_gt_filenames = len(gt_filenames)
                print(f'selecting {n_gt_filenames} image(s) from ID {img_start_id} to {img_end_id}')

            # gt_file_paths = [linux_path(seq_path, gt_filename) for gt_filename in gt_filenames]
            gt_pbar = tqdm(gt_filenames, total=n_gt_filenames, ncols=100)

            valid_rows = 0
            total_rows = 0

            for gt_filename in gt_pbar:

                assert os.path.isfile(gt_filename), f"gt_filename does not exist: {gt_filename}"

                row_ids = grouped_gt.groups[gt_filename]
                img_df = df_gt.loc[row_ids]

                file_path = gt_filename

                # seq_gt_data_dict[file_path] = []
                curr_frame_gt_data = []

                gt_pbar.set_description(f'seq {seq_idx + 1} / {n_seq}: '
                                        f'valid_rows: {valid_rows} / {total_rows}')

                for _, row in img_df.iterrows():
                    total_rows += 1

                    xmin = float(row['xmin'])
                    ymin = float(row['ymin'])
                    xmax = float(row['xmax'])
                    ymax = float(row['ymax'])
                    gt_class = row['class']
                    try:
                        gt_img_w = int(row['width'])
                        gt_img_h = int(row['height'])
                    except KeyError:
                        try:
                            gt_img_w, gt_img_h = img_path_to_size[file_path]
                        except KeyError:
                            gt_img_w, gt_img_h = imagesize.get(file_path)
                    else:
                        img_path_to_size[file_path] = (gt_img_w, gt_img_h)

                    try:
                        target_id = int(row['target_id'])
                    except KeyError:
                        target_id = -1

                    valid_rows += 1
                    if gt_class not in gt_classes:
                        msg = f'{seq_name}: {gt_filename} :: invalid gt_class: {gt_class}'
                        if ignore_invalid_class:
                            print(msg)
                            continue
                        raise AssertionError(msg)

                    bbox = [xmin, ymin, xmax, ymax]

                    curr_obj_gt_data = {
                        'file_id': gt_filename,
                        'class': gt_class,
                        'bbox': bbox,
                        'width': gt_img_w,
                        'height': gt_img_h,
                        'target_id': target_id,
                        'confidence': 1.0,
                        'filename': gt_filename,
                        'used': False,
                        'used_fp': False,
                        'matched': False,
                        'mask': None
                    }
                    if enable_mask:
                        try:
                            mask_rle = mask_str_to_img(
                                row["mask"], gt_img_h, gt_img_w, to_rle=1)
                        except KeyError:
                            mask_w = int(row["mask_w"])
                            mask_h = int(row["mask_h"])
                            mask_counts = str(row["mask_counts"])

                            mask_rle = dict(
                                size=(mask_h, mask_w),
                                counts=mask_counts
                            )

                        curr_obj_gt_data['mask'] = mask_rle

                    curr_frame_gt_data.append(curr_obj_gt_data)
                    gt_class_data_dict[gt_class][seq_path].append(curr_obj_gt_data)

                    try:
                        gt_counter_per_class[gt_class] += 1
                    except KeyError:
                        gt_counter_per_class[gt_class] = 1

                if curr_frame_gt_data:
                    seq_gt_data_dict[file_path] = curr_frame_gt_data

            gt_data_dict[seq_path] = seq_gt_data_dict

            gt_img_paths = sorted(list(seq_gt_data_dict.keys()))
            all_img_paths += gt_img_paths

        if not det_loaded:
            det_paths = det_path_list[seq_idx]

            if isinstance(det_paths, str):
                det_paths = [det_paths, ]

            n_det_paths = len(det_paths)
            # if n_seq == 1:
            #     print(f'\ndet_paths: {det_paths}')

            seq_det_bounding_boxes = []

            det_pbar = None

            if n_det_paths > 1:
                det_pbar = det_paths_iter = tqdm(det_paths, ncols=100)
            else:
                det_paths_iter = det_paths

            n_invalid_dets = 0
            n_total_dets = 0

            for _det_path_id, _det_path in enumerate(det_paths_iter):
                det_seq_name = os.path.splitext(os.path.basename(_det_path))[0]
                if check_seq_name and (det_seq_name != seq_name or gt_seq_name != seq_name):
                    raise IOError(f'Mismatch between GT, detection and image sequences: '
                                  f'{gt_seq_name}, {det_seq_name}, {seq_name}')

                _det_name = os.path.basename(_det_path)

                df_det = pd.read_csv(_det_path)

                # df_dets.append(_df_det)

                # df_det = pd.concat(df_dets, axis=0)
                if fix_det_cols:
                    df_det = df_det.rename(columns=csv_rename_dict)

                df_det["filename"] = df_det["filename"].apply(lambda x: seq_img_name_to_path[os.path.basename(x)])

                grouped_dets = df_det.groupby("filename")

                n_dets = len(df_det)

                det_filenames = sorted(list(grouped_dets.groups.keys()))
                n_det_filenames = len(det_filenames)

                # print(f'{_det_name} --> {n_dets} detections for {n_det_filenames} images')

                non_gt_det_filenames = [det_filename for det_filename in det_filenames
                                        if det_filename not in gt_filenames]

                if non_gt_det_filenames:
                    if not allow_missing_gt:
                        raise AssertionError('det_filenames without corresponding gt_filenames found')

                    if allow_missing_gt == 2:
                        print(f'\n\nskipping {len(non_gt_det_filenames)} non_gt_det_filenames')
                        det_filenames = [det_filename for det_filename in det_filenames if det_filename in gt_filenames]
                        n_det_filenames = len(det_filenames)

                if n_det_paths > 1:
                    det_filename_iter = det_filenames
                else:
                    det_pbar = det_filename_iter = tqdm(det_filenames, ncols=100)

                det_pbar_base_msg = ''
                if n_seq > 1:
                    det_pbar_base_msg += f"seq {seq_idx + 1} / {n_seq} "
                if n_det_paths > 1:
                    det_pbar_base_msg += f"csv {_det_path_id + 1} / {n_det_paths} "

                for det_filename_id, det_filename in enumerate(det_filename_iter):

                    assert os.path.isfile(det_filename), f"det_filename does not exist: {det_filename}"

                    file_path = det_filename

                    row_ids = grouped_dets.groups[det_filename]
                    img_df = df_det.loc[row_ids]
                    for _, row in img_df.iterrows():

                        try:
                            target_id = int(row['target_id'])
                        except KeyError:
                            target_id = -1

                        xmin = float(row['xmin'])
                        ymin = float(row['ymin'])
                        xmax = float(row['xmax'])
                        ymax = float(row['ymax'])

                        try:
                            det_img_w = int(row['width'])
                            det_img_h = int(row['height'])
                        except KeyError:
                            try:
                                det_img_w, det_img_h = img_path_to_size[file_path]
                            except KeyError:
                                det_img_w, det_img_h = imagesize.get(file_path)
                        else:
                            img_path_to_size[file_path] = (det_img_w, det_img_h)

                        det_class = row['class']

                        try:
                            confidence = row['confidence']
                        except KeyError:
                            confidence = 1.0

                        if enable_mask:
                            try:
                                mask_rle = mask_str_to_img(row["mask"], det_img_h, det_img_w, to_rle=1)
                            except KeyError:
                                mask_w = int(row["mask_w"])
                                mask_h = int(row["mask_h"])
                                mask_counts = str(row["mask_counts"])

                                mask_rle = dict(
                                    size=(mask_h, mask_w),
                                    counts=mask_counts
                                )

                        if det_class not in gt_classes:
                            msg = f'{det_seq_name}: {det_filename} :: invalid det_class: {det_class}'
                            if ignore_invalid_class:
                                print(msg)
                                continue
                            raise AssertionError(msg)

                        det_w = xmax - xmin
                        det_h = ymax - ymin

                        n_total_dets += 1

                        if det_w <= 0 or det_h <= 0:
                            n_invalid_dets += 1
                            # print(f'ignoring invalid detection: {[xmin, ymin, xmax, ymax]}')
                            continue

                        bbox_dict = {
                            "class": det_class,
                            "width": det_img_w,
                            "height": det_img_h,
                            "target_id": target_id,
                            "confidence": confidence,
                            "filename": det_filename,
                            "file_path": file_path,
                            "bbox": [xmin, ymin, xmax, ymax],
                            'mask': None
                        }

                        if enable_mask:
                            bbox_dict["mask"] = mask_rle

                        seq_det_bounding_boxes.append(bbox_dict)

                        if det_pbar is not None:
                            det_pbar_msg = det_pbar_base_msg + f"invalid dets: {n_invalid_dets} / {n_total_dets}"

                            det_pbar.set_description(det_pbar_msg)

            raw_det_data_dict[seq_path] = seq_det_bounding_boxes

    """save pkl and detection post-proc"""
    if True:
        if not det_loaded:
            print(f'\nSaving detection data to {det_pkl}')
            os.makedirs(det_pkl_dir, exist_ok=1)
            with open(det_pkl, 'wb') as f:
                pickle.dump(raw_det_data_dict, f, pickle.HIGHEST_PROTOCOL)

        if not gt_loaded:
            gt_data_dict['counter_per_class'] = gt_counter_per_class
            os.makedirs(gt_pkl_dir, exist_ok=1)

            print(f'\nSaving GT data to {gt_pkl}')
            with open(gt_pkl, 'wb') as f:
                pickle.dump(gt_data_dict, f, pickle.HIGHEST_PROTOCOL)

        gt_end_t = time.time()
        if not (gt_loaded and det_loaded):
            print('Time taken: {} sec'.format(gt_end_t - gt_start_t))

        gt_counter_per_class = gt_data_dict['counter_per_class']

        for _class_name in gt_classes:
            if _class_name not in gt_counter_per_class.keys():
                gt_counter_per_class[_class_name] = 0

        total_gt = 0
        for _class_name in gt_counter_per_class.keys():
            total_gt += gt_counter_per_class[_class_name]

        gt_fraction_per_class = {}
        gt_fraction_per_class_list = []
        for _class_name in gt_counter_per_class.keys():
            try:
                _gt_fraction = float(gt_counter_per_class[_class_name]) / float(total_gt)
            except ZeroDivisionError:
                print('gt_counter_per_class: ', gt_counter_per_class)
                print('total_gt: ', total_gt)
                _gt_fraction = 0

            gt_fraction_per_class[_class_name] = _gt_fraction
            gt_fraction_per_class_list.append(_gt_fraction)

        # gt_classes = list(gt_counter_per_class.keys())
        # gt_classes = sorted(gt_classes)

        n_classes = len(gt_classes)

        print('gt_classes: ', gt_classes)
        print('n_classes: ', n_classes)
        print('gt_counter_per_class: ', gt_counter_per_class)

        log_dir = 'pprint_log'
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)

        if score_thresh > 0:
            print('Discarding detections with score < {}'.format(score_thresh))

        for gt_class_idx, gt_class in enumerate(gt_classes):

            class_det_data_dict = {}
            print(f'\nProcessing class {gt_class_idx + 1:d} / {n_classes:d}: {gt_class:s}')

            det_start_t = time.time()

            for seq_idx in tqdm(range(n_seq), desc="Post processing sequence"):

                # sys.stdout.write('\rPost processing sequence {:d}/{:d} '.format(
                #     seq_idx + 1, n_seq))
                # sys.stdout.flush()

                seq_name = seq_name_list[seq_idx]
                seq_path = seq_paths[seq_idx]

                seq_gt = gt_data_dict[seq_path]
                seq_det = raw_det_data_dict[seq_path]

                curr_class_det_exists = {}
                curr_class_det_bounding_boxes = []
                for _data in seq_det:
                    det_class = _data['class']
                    file_path = _data['file_path']
                    confidence = _data['confidence']
                    bbox = _data['bbox']
                    det_filename_ = _data['filename']
                    width = _data['width']
                    height = _data['height']
                    target_id = _data['target_id']
                    mask = _data['mask']

                    norm_bbox = [bbox[0] / float(width), bbox[1] / float(height),
                                 bbox[2] / float(width), bbox[3] / float(height)]

                    if file_path not in curr_class_det_exists:
                        curr_class_det_exists[file_path] = 0

                    if bbox is None or det_class != gt_class or confidence < score_thresh:
                        continue

                    curr_class_det_bounding_boxes.append(
                        {
                            "class": det_class,
                            "width": width,
                            "height": height,
                            "target_id": target_id,
                            "filename": det_filename_,
                            "file_path": file_path,
                            "confidence": confidence,
                            "file_id": file_path,
                            "bbox": bbox,
                            "norm_bbox": norm_bbox,
                            "mask": mask,
                        })
                    curr_class_det_exists[file_path] = 1

                    assert os.path.exists(file_path), f"file_path does not exist: {file_path}"

                """look for false negatives for current class by finding frames with no detections in this class 
                but with valid GT of this class;
                only works for frames that have at least one detection of some class
                """
                # for file_path in curr_class_det_exists:
                #     if curr_class_det_exists[file_path]:
                #         continue
                #
                #     try:
                #         gt_data = seq_gt[file_path]
                #     except KeyError as e:
                #         if allow_missing_gt:
                #             # print('\nNo GT found for {}\n'.format(file_path))
                #             gt_data = []
                #         else:
                #             if not os.path.isdir('map_pprint_log'):
                #                 os.makedirs('map_pprint_log')
                #             print('seq_path: {}'.format(seq_path))
                #             print('file_path: {}'.format(file_path))
                #
                #             time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
                #             log_path = linux_path('map_pprint_log', 'map_pprint_log_{}.txt'.format(time_stamp))
                #
                #             with open(log_path, 'w') as _fid:
                #                 pprint(('seq_det', seq_det), _fid)
                #                 pprint(('seq_gt', seq_gt), _fid)
                #
                #             raise KeyError(e)
                #
                #     for gt_obj in gt_data:
                #         if gt_obj['class'] == gt_class:
                #             curr_class_det_bounding_boxes.append(
                #                 {
                #                     'confidence': None,
                #                     'file_id': file_path,
                #                     'bbox': None,
                #                     'mask': None
                #                 }
                #             )
                #             break

                class_det_data_dict[seq_path] = curr_class_det_bounding_boxes

            det_end_t = time.time()
            det_data_dict[gt_class] = class_det_data_dict

            print('Time taken: {} sec'.format(det_end_t - det_start_t))

        """
         Calculate the AP for each class
        """
        wmAP = sum_AP = 0.0
        wm_prec = wm_rec = wm_rec_prec = wm_score = 0.0
        sum_prec = sum_rec = sum_rec_prec = sum_score = 0.0

        min_overlap = iou_thresh

        # ap_dictionary = {}
        print('Calculating the AP for each class...')

        # colors (OpenCV works with BGR)
        # white = (255, 255, 255)
        # light_blue = (255, 200, 100)
        # green = (0, 255, 0)
        # light_red = (30, 30, 255)
        # magenta = (255, 0, 255)
        # 1st line
        margin = 20
        # Add bottom border to image
        bottom_border = 60

        win_name = "s: next sequence c: next class q/escape: quit"

        count_true_positives = {}
        tp_sum_overall = 0

        fp_sum_overall = 0
        fp_dup_sum_overall = 0
        fp_nex_sum_overall = 0
        fp_cls_sum_overall = 0

        fn_sum_overall = 0
        fn_det_sum_overall = 0
        fn_cls_sum_overall = 0

        gt_overall = 0
        dets_overall = 0

        class_stats = [None, ] * n_classes

        if write_summary:
            summary_path = linux_path(out_root_dir, "summary.txt")
            print('Writing result summary to {}'.format(summary_path))

        out_template = linux_path(out_root_dir).replace('/', '_')
        out_text = out_template
        text = 'class\tAP(%)\tPrecision(%)\tRecall(%)\tR=P(%)\tScore(%)\t' \
               'TP\tFN\tFN_DET\tFN_CLS\tFP\tFP_DUP\tFP_NEX\tFP_CLS\tGT\tDETS'
        out_text += '\n' + text + '\n'

        text_table = PrettyTable(text.split('\t'))

        return_eval_dict = 0
        if eval_result_dict is None:
            return_eval_dict = 1
            eval_result_dict = {}

        # print(text)
        # if write_summary:
        #     out_file.write(text + '\n')

        font_line_type = cv2.LINE_AA

        rec_thresh_all = np.zeros((n_score_thresholds, n_classes))
        prec_thresh_all = np.zeros((n_score_thresholds, n_classes))
        tp_sum_thresh_all = np.zeros((n_score_thresholds, n_classes))
        fp_sum_thresh_all = np.zeros((n_score_thresholds, n_classes))
        fp_cls_sum_thresh_all = np.zeros((n_score_thresholds, n_classes))
        fp_dup_thresh_all = np.zeros((n_score_thresholds, n_classes))
        fp_nex_thresh_all = np.zeros((n_score_thresholds, n_classes))

        cmb_summary_data = {}

        class_rec_prec = np.zeros((n_score_thresholds, n_classes * 2), dtype=np.float32)
        class_rec_prec[:, 0] = score_thresholds

        frame_to_det_data = {}
        csv_columns_rec_prec = ['confidence_threshold', 'Recall', 'Precision']

        os.makedirs(out_root_dir, exist_ok=1)

        misc_out_root_dir = linux_path(out_root_dir, 'misc')
        os.makedirs(misc_out_root_dir, exist_ok=1)

        if vid_ext in img_exts:
            vis_video = 0
        else:
            vis_video = 1

    """main processing"""
    seq_name_to_csv_rows = {}
    for gt_class_idx, gt_class in enumerate(gt_classes):
        enable_vis = show_vis or (save_vis and gt_class in save_classes)

        n_class_dets = 0

        n_class_gt = gt_counter_per_class[gt_class]

        other_classes = [k for k in gt_classes if k != gt_class]

        print(f'\nProcessing class {gt_class_idx + 1:d} / {n_classes:d}: {gt_class:s}')

        count_true_positives[gt_class] = 0
        end_class = 0

        tp_class = []

        fp_class = []
        fp_dup_class = []
        fp_nex_class = []
        fp_nex_part_class = []
        fp_nex_whole_class = []
        fp_cls_class = []

        conf_class = []

        tp_sum = 0

        fp_sum = 0
        fp_cls_sum = 0
        fp_dup_sum = 0
        fp_nex_sum = 0
        fp_nex_part_sum = 0
        fp_nex_whole_sum = 0

        fn_sum = 0
        fn_cls_sum = 0
        fn_det_sum = 0

        n_used_gt = 0
        n_unused_gt = 0
        n_total_gt = 0
        all_gt_list = []

        all_considered_gt = []
        det_file_ids = []

        missing_gt_file_ids_live = []

        cum_tp_sum = OrderedDict()
        cum_fp_sum = OrderedDict()

        if save_vis:
            if not vis_video or len(save_classes) > 1:
                vis_root_dir = linux_path(out_root_dir, 'vis', gt_class)
            else:
                vis_root_dir = out_root_dir

            os.makedirs(vis_root_dir, exist_ok=1)
            print(f'\n\nsaving {save_w} x {save_h} vis videos to {vis_root_dir}\n\n')

            vis_out_fnames = {
                cat: linux_path(vis_root_dir, f'{cat}.{vid_ext}')
                for cat in cls_cat_types
            }

            video_out_dict = {
                cat: None for cat in cls_cat_types
            }

        cat_to_ids_vis_done = {k: [] for k in cls_cat_types}
        cat_to_vis_count = {k: 0 for k in cls_cat_types}

        for seq_idx in tqdm(range(n_seq), desc="sequence", ncols=70):
            seq_path = seq_paths[seq_idx]
            seq_name = seq_name_list[seq_idx]
            seq_root_dir = seq_root_dirs[seq_idx]

            seq_gt_data_dict = gt_data_dict[seq_path]
            """total number of GTs of all classes in this sequence"""
            n_seq_gts = len(seq_gt_data_dict)

            seq_class_gt_data = gt_class_data_dict[gt_class][seq_path]
            """total number of GTs of this class in this sequence"""
            n_seq_class_gts = len(seq_class_gt_data)

            seq_class_det_data = det_data_dict[gt_class][seq_path]
            """total number of detections of this class in this sequence"""
            n_seq_class_dets = len(seq_class_det_data)
            n_class_dets += n_seq_class_dets

            print(f'n_seq_gts: {n_seq_gts}')
            print(f'n_seq_class_gts: {n_seq_class_gts}')
            print(f'n_seq_class_dets: {n_seq_class_dets}')

            seq_gt_file_ids = set(seq_gt_data_dict.keys())
            n_seq_gt_file_ids = len(seq_gt_file_ids)

            seq_class_gt_file_ids = set([obj['file_id'] for obj in seq_class_gt_data])
            n_seq_class_gt_file_ids = len(seq_class_gt_file_ids)

            seq_class_det_file_ids = set([obj['file_id'] for obj in seq_class_det_data])
            n_seq_det_file_ids = len(seq_class_det_file_ids)

            print(f'n_seq_gt_file_ids: {n_seq_gt_file_ids}')
            print(f'n_seq_class_gt_file_ids: {n_seq_class_gt_file_ids}')
            print(f'n_seq_det_file_ids: {n_seq_det_file_ids}')

            """all frames with no det of this class but containing gt of this class"""
            missing_class_det_file_ids = seq_class_gt_file_ids - seq_class_det_file_ids

            """all frames with no GT of any class but containing det of this class"""
            missing_gt_file_ids = seq_class_det_file_ids - seq_gt_file_ids

            if missing_gt_file_ids:
                """
                dummy GT is only needed for images that have no GT for any class since all other images are 
                already present in the dictionary even if they have no GT of this class 
                so there is no risk of key error when trying to access GT data for those images
                """
                msg = f'{seq_name} : no GT found for {len(missing_gt_file_ids)} images ' \
                    f'with {gt_class} detections'
                if not allow_missing_gt:
                    raise AssertionError(msg)

                print('\n' + msg)

                seq_gt_data_dict.update(
                    {k: [] for k in missing_gt_file_ids}
                )
                n_seq_gts = len(seq_gt_data_dict)

            if missing_class_det_file_ids:
                """dummy detections should only be added for images that have a GT of this class 
                to avoid unnecessary processing even though the result would be the same if we were to add 
                them for all images with any class GT too"""

                n_missing_class_det_file_ids = len(missing_class_det_file_ids)
                print(f'\n{seq_name}: no {gt_class} detections found for {n_missing_class_det_file_ids} images'
                      f' with {gt_class} GT\n')

                """add one dummy detection for each missing image"""
                seq_class_det_data += [
                    {'confidence': None, 'file_id': k, 'bbox': None} for k in missing_class_det_file_ids
                ]
                n_seq_class_dets = len(seq_class_det_data)

            seq_gt_file_ids_with_dummy = set(seq_gt_data_dict.keys())
            n_seq_gt_file_ids_with_dummy = len(seq_gt_file_ids_with_dummy)

            seq_class_det_file_ids_with_dummy = set([obj['file_id'] for obj in seq_class_det_data])
            n_seq_class_det_file_ids_with_dummy = len(seq_class_det_file_ids_with_dummy)

            print(f'n_seq_gt_file_ids_with_dummy: {n_seq_gt_file_ids_with_dummy}')
            print(f'n_seq_class_det_file_ids_with_dummy: {n_seq_class_det_file_ids_with_dummy}')

            """sort detections by frame"""
            seq_class_det_data.sort(key=lambda x: x['file_id'])
            # seq_class_det_data.sort(key=lambda x: x['confidence'], reverse=True)

            """flags to mark the status of each detection"""
            tp = [0] * n_seq_class_dets

            fp = [0] * n_seq_class_dets
            fp_dup = [0] * n_seq_class_dets
            fp_nex = [0] * n_seq_class_dets
            fp_nex_whole = [0] * n_seq_class_dets
            fp_nex_part = [0] * n_seq_class_dets
            fp_cls = [0] * n_seq_class_dets

            fn_dets = [0] * n_seq_class_dets

            conf = [0] * n_seq_class_dets

            all_gt_match = [None] * n_seq_class_dets
            all_status = [''] * n_seq_class_dets
            all_ovmax = [-1] * n_seq_class_dets

            if assoc_method == 0:
                cum_tp_sum, cum_fp_sum = perform_global_association(
                    seq_class_det_data, seq_gt_data_dict, gt_class,
                    show_sim, tp, fp, all_status, all_ovmax,
                    iou_thresh, count_true_positives,
                    all_gt_match,
                    seq_name, save_sim_dets, sim_recs, sim_precs,
                    seq_path, cum_tp_sum, cum_fp_sum, enable_mask)

                if save_sim_dets:
                    continue

            fn_gts = []
            fn_cats = []

            det_idx = 0
            img = None
            prev_det_idx = None

            vis_file_id = file_id = prev_file_id = None
            text_img = None

            if show_pbar:
                pbar = tqdm(total=n_seq_class_dets, ncols=100, position=0, leave=True,
                            desc=f'seq {seq_idx + 1} / {n_seq}')

            """process all the detections in this sequence"""
            while True:
                """process one detection at a time"""
                if enable_vis and (img is not None) and (all_class_dets or all_class_gt):

                    """nothing useful to show if neither GTs nr dets of this class exist in this frame"""
                    if vis_file_id is None or file_id != vis_file_id:
                        vis_file_id = file_id
                        # vis_frames[file_id] = img

                        cat_img_vis = draw_and_concat(src_img, frame_det_data, frame_gt_data,
                                                      class_name_to_col, params.vis_alpha, vis_w, vis_h, vert_stack,
                                                      params.check_det)

                        cat_img_vis_all = draw_and_concat(src_img, frame_det_data, frame_gt_data,
                                                          class_name_to_col, params.vis_alpha, vis_w_all, vis_h_all,
                                                          vert_stack, params.check_det)

                        cat_img_h, cat_img_w = cat_img_vis_all.shape[:2]

                        all_text_img = np.zeros((bottom_border, cat_img_w, 3), dtype=np.uint8)
                        text = f"{seq_idx + 1}/{n_seq} {seq_name}: {ground_truth_img} "
                        all_text_img, _ = draw_text_in_image(all_text_img, text, (20, 20), 'white', 0)

                        all_out_img = np.concatenate((cat_img_vis_all, all_text_img), axis=0)

                        all_video_out = video_out_dict["all"]
                        if all_video_out is None:
                            vis_out_fname = vis_out_fnames["all"]
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

                            video_out_dict["all"] = all_video_out

                        if vis_video:
                            all_out_img = resize_ar(all_out_img, video_w, video_h, add_border=2)

                        # cv2.imshow('all_out_img', all_out_img)
                        # cv2.waitKey(0)

                        all_video_out.write(all_out_img)

                        # cat_img_vis = resize_ar(cat_img_vis, save_w, save_h)

                    if isinstance(img, list):
                        assert isinstance(cls_cat, list), "cls_cat must be a list"
                        assert isinstance(text_img, list), "text_img must be a list"
                        cls_cats = cls_cat
                        imgs = img
                        text_imgs = text_img
                    else:
                        imgs = [img, ]
                        text_imgs = [text_img, ]
                        cls_cats = [cls_cat, ]

                    for _cls_cat, _img, _text_img in zip(cls_cats, imgs, text_imgs):

                        if _cls_cat in ['fn_det', 'fn_cls']:
                            if vis_file_id in cat_to_ids_vis_done[_cls_cat]:
                                raise AssertionError(f'{_cls_cat} vis has already been done for {vis_file_id}')
                                # continue

                            cat_to_ids_vis_done[_cls_cat].append(vis_file_id)

                        if _img is None:
                            assert _text_img is None, "_img is None but _text_img isn't"
                            continue

                        # img = resize_ar(img, save_w, save_h)
                        out_img = np.concatenate((cat_img_vis, _img), axis=0 if vert_stack else 1)

                        # out_img = resize_ar(out_img, save_w, save_h)
                        out_img = np.concatenate((out_img, _text_img), axis=0)

                        # cv2.imshow('text_img', text_img)
                        # cv2.imshow('out_img', out_img)
                        # cv2.waitKey(1)

                        if save_vis and _cls_cat in save_cats and gt_class in save_classes:

                            video_out = video_out_dict[_cls_cat]
                            if video_out is None:
                                vis_out_fname = vis_out_fnames[_cls_cat]
                                _save_dir = os.path.dirname(vis_out_fname)

                                print(f'\n\nsaving video for {_cls_cat} at {vis_out_fname}\n\n')

                                if _save_dir and not os.path.isdir(_save_dir):
                                    os.makedirs(_save_dir)

                                if vis_video:
                                    # video_h, video_w = out_img.shape[:2]
                                    video_h, video_w = save_h, save_w
                                    # if show_text:
                                    #     video_h += bottom_border
                                    video_out = cv2.VideoWriter(vis_out_fname, fourcc, fps, (video_w, video_h))
                                else:
                                    video_out = ImageSequenceWriter(vis_out_fname, verbose=0)

                                if not video_out:
                                    raise AssertionError(
                                        f'video file: {vis_out_fname} could not be opened for writing')

                                video_out_dict[_cls_cat] = video_out

                            if vis_video:
                                out_img = resize_ar(out_img, video_w, video_h, add_border=2)

                            video_out.write(out_img)
                            cat_to_vis_count[_cls_cat] += 1

                            n_fn_cls_vis = cat_to_vis_count['fn_cls']
                            n_fn_det_vis = cat_to_vis_count['fn_det']
                            n_fp_dup_vis = cat_to_vis_count['fp_dup']
                            n_fp_nex_whole_vis = cat_to_vis_count['fp_nex-whole']
                            n_fp_cls_vis = cat_to_vis_count['fp_cls']
                            n_fp_nex_part_vis = cat_to_vis_count['fp_nex-part']

                            pbar.set_description(f'seq {seq_idx + 1} / {n_seq} '
                                                 f'fn: cls-{n_fn_cls_vis} '
                                                 f'det-{n_fn_det_vis} '
                                                 f'fp: cls-{n_fp_cls_vis} '
                                                 f'dup-{n_fp_dup_vis} '
                                                 f'nex-{n_fp_nex_whole_vis},{n_fp_nex_part_vis} '
                                                 f'')

                        if show_vis:
                            # cv2.imshow('cat_img_vis', cat_img_vis)
                            cv2.imshow(win_name, out_img)
                            k = cv2.waitKey(1 - _pause)
                            if k == ord('q') or k == 27:
                                if save_vis:
                                    for video_out in video_out_dict.values():
                                        if video_out is not None:
                                            video_out.release()
                                cv2.destroyWindow(win_name)
                                sys.exit(0)
                            elif k == ord('c'):
                                end_class = 1
                                break
                            elif k == ord('s'):
                                break
                            elif k == 32:
                                _pause = 1 - _pause

                if det_idx >= n_seq_class_dets:
                    break

                """all dets of this class in this frame"""
                curr_det_data = seq_class_det_data[det_idx]
                file_id = curr_det_data["file_id"]

                """all dets of all classes in this frame"""
                try:
                    frame_det_data = frame_to_det_data[file_id]
                except KeyError:
                    frame_det_data = []
                    for _class in gt_classes:
                        frame_det_data += [det_obj for det_obj in det_data_dict[_class][seq_path] if
                                           det_obj["file_id"] == file_id and det_obj["bbox"] is not None]
                    frame_to_det_data[file_id] = frame_det_data

                det_file_ids.append(file_id)

                n_all_frame_dets = len(frame_det_data)

                """all GTs in this frame"""
                try:
                    frame_gt_data = seq_gt_data_dict[file_id]
                except KeyError as e:
                    raise KeyError(e)
                    # print('\nno gt found for file: {}'.format(file_id))
                    # seq_gt_data_dict[file_id] = []
                    # frame_gt_data = []
                    # missing_gt_file_ids_live.append(file_id)

                n_all_frame_gt = len(frame_gt_data)

                conf[det_idx] = curr_det_data["confidence"]

                if enable_vis:
                    img_full_path = file_id
                    ground_truth_img = os.path.basename(img_full_path)

                    if prev_file_id is None or prev_file_id != file_id:
                        prev_file_id = file_id
                        src_img = cv2.imread(img_full_path)
                        # src_img = Image.open(img_full_path)
                        # src_h, src_w = src_img.shape[:2]

                        vis_h, vis_w = get_vis_size(src_img, 3, save_w, save_h, bottom_border)

                        vis_h_all, vis_w_all = get_vis_size(src_img, 2, save_w, save_h, bottom_border)

                        vert_stack = 0

                        # vert_ar = src_w / (src_h * 3)
                        # horz_ar = (src_w * 3) / src_h
                        #
                        # vis_ar = save_w / save_h
                        # if abs(vis_ar - horz_ar) < abs(vis_ar - vert_ar):
                        #     vert_stack = 0
                        # else:
                        #     vert_stack = 1

                        if src_img is None:
                            raise IOError(f'Image could not be read: {img_full_path}')

                    img, resize_factor, _, _ = resize_ar(src_img, vis_w, vis_h, crop=1, return_factors=1)

                    if vert_stack:
                        text_img_w = img.shape[1]
                    else:
                        text_img_w = int(img.shape[1] * 3)

                    if fn_gts:
                        if prev_det_idx is not None:
                            assert det_idx == prev_det_idx, "unexpected change in det_idx"

                        # print('\nfile_id: ', file_id)
                        # print('fn_gts:\n ', fn_gts)

                        color = 'magenta'
                        img_copy = img.copy()

                        text_img = []
                        img = []
                        cls_cat = []

                        # for det in fn_gts:
                        #     bb_det = [int(x) for x in det['bbox']]
                        #     cv2.rectangle(img, (bb_det[0], bb_det[1]), (bb_det[2], bb_det[3]), magenta, 2)

                        # img = cv2.copyMakeBorder(img, 0, bottom_border, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
                        # height, _ = img.shape[:2]

                        for fn_cat in ['fn_det', 'fn_cls']:

                            cat_fn_gts = [fn_gt for i, fn_gt in enumerate(fn_gts) if fn_cats[i] == fn_cat]
                            if not cat_fn_gts:
                                fn_img = fn_text_img = None
                            else:
                                fn_img = img_copy.copy()

                                fn_img = draw_objs(fn_img, cat_fn_gts, col='cyan',
                                                   in_place=True, thickness=2, mask=0)
                                fn_text_img = np.zeros((bottom_border, text_img_w, 3), dtype=np.uint8)

                                v_pos = margin
                                text = f"{seq_idx + 1}/{n_seq} {seq_name}: {ground_truth_img} "
                                fn_text_img, line_width = draw_text_in_image(fn_text_img, text, (margin, v_pos),
                                                                             'white', 0)
                                text = "Class [" + str(gt_class_idx + 1) + "/" + str(n_classes) + "]: " + gt_class + " "
                                fn_text_img, line_width = draw_text_in_image(fn_text_img, text,
                                                                             (margin + line_width, v_pos), 'cyan',
                                                                             line_width)

                                text = "Result: {}".format(fn_cat.upper())
                                fn_text_img, line_width = draw_text_in_image(fn_text_img, text,
                                                                             (margin + line_width, v_pos), color,
                                                                             line_width)

                                if show_stats:
                                    v_pos += int(bottom_border / 2)
                                    try:
                                        _recall = float(tp_sum) / float(tp_sum + fn_sum) * 100.0
                                    except ZeroDivisionError:
                                        _recall = 0
                                    try:
                                        _prec = float(tp_sum) / float(tp_sum + fp_sum) * 100.0
                                    except ZeroDivisionError:
                                        _prec = 0
                                    text = f'gts: {n_all_frame_gt:d} dets: {n_all_frame_dets:d} ' \
                                        f'tp: {tp_sum:d} fn: {fn_sum:d} ' \
                                        f'fp: {fp_sum:d} fp_dup: {fp_dup_sum:d} fp_nex: {fp_nex_sum:d} ' \
                                        f'recall: {_recall:5.2f}% prec: {_prec:5.2f} '

                                    fn_text_img, line_width = draw_text_in_image(fn_text_img, text, (margin, v_pos),
                                                                                 'white',
                                                                                 line_width)

                            cls_cat.append(fn_cat)
                            img.append(fn_img)
                            text_img.append(fn_text_img)

                        fn_gts = []
                        det_idx += 1
                        if show_pbar:
                            pbar.update(1)

                        continue
                    else:
                        text_img = np.zeros((bottom_border, text_img_w, 3), dtype=np.uint8)

                if prev_det_idx is not None:
                    assert prev_det_idx != det_idx, "repeated det_idx"

                prev_det_idx = det_idx

                all_class_gt = [obj for obj in frame_gt_data if obj['class'] == gt_class]
                all_class_dets = [obj for obj in frame_det_data if obj['class'] == gt_class]

                # dets_exist = any(obj['class'] == gt_class for obj in frame_det_data)
                # gts_exist = any(obj['class'] == gt_class for obj in frame_gt_data)

                is_last_in_frame = det_idx == n_seq_class_dets - 1 or \
                                   seq_class_det_data[det_idx + 1]['file_id'] != file_id

                if curr_det_data["bbox"] is None:
                    """no detections of this class in this frame as indicated by this dummy detection"""

                    assert is_last_in_frame, "dummy det must be the last (and only) one in frame"
                    assert not all_class_dets, "all_class_dets must be empty"

                    """all GTs are false negatives"""
                    n_all_gt = len(all_class_gt)
                    n_total_gt += n_all_gt
                    # all_gt_list += [
                    #     dict(k, **{'file_id': file_id})
                    #     for k in all_class_gt
                    # ]
                    all_gt_list += all_class_gt

                    used_gt = [obj for obj in all_class_gt if obj['used']]
                    n_used_gt += len(used_gt)

                    fn_dets[det_idx] = 1

                    fn_gts = all_class_gt[:]
                    n_fn_gts = len(fn_gts)

                    # print(f'\n\nfound {n_fn_gts} fn_gts\n\n')

                    assert n_fn_gts == n_all_gt, "n_all_gt / n_fn_gts mismatch"

                    fn_cats = [None, ] * n_fn_gts
                    """find which of the GTs have corresponding detections from other classes"""
                    for gt_obj_id, gt_obj in enumerate(all_class_gt):
                        ovmax_, det_match_ = get_max_iou_obj(frame_det_data, gt_classes,
                                                             gt_obj["bbox"], gt_obj["mask"], enable_mask)
                        if ovmax_ > min_overlap:
                            """misclassification"""
                            assert det_match_["class"] != gt_class, \
                                "unused GT object matches with a det of current class"
                            fn_cls_sum += 1
                            fn_cats[gt_obj_id] = 'fn_cls'
                        else:
                            """missing detection"""
                            fn_det_sum += 1
                            fn_cats[gt_obj_id] = 'fn_det'

                    fn_sum += n_fn_gts
                    all_considered_gt += all_class_gt

                    # print('None det {} :: {} :: n_total_gt: {} n_used_gt: {} tp_sum: {} fn_sum: {}'.format(
                    #     seq_name, file_id, n_total_gt, n_used_gt, tp_sum, fn_sum))

                    assert fn_det_sum + fn_cls_sum == fn_sum, "fn_sum mismatch"

                    if n_total_gt != n_used_gt + fn_sum:
                        print('fn_gts:\n{}'.format(pformat(fn_gts)))
                        print('all_class_gt:\n{}'.format(pformat(all_class_gt)))

                        raise AssertionError(
                            f'{gt_class} : {file_id} :: '
                            f'Mismatch between n_total_gt: {n_total_gt} and n_used_gt+fn_sum: {n_used_gt + fn_sum}'
                        )

                    if n_used_gt != tp_sum:
                        raise AssertionError(
                            f'{gt_class} : {file_id} :: Mismatch between n_used_gt: {n_used_gt} and tp_sum: {tp_sum}')

                    if enable_vis:
                        """need to show these fn_gts in the next iteration of the while loop"""
                        if fn_gts:
                            img = None
                            continue
                    else:
                        """no dets and no gt in this frame"""
                        fn_gts = []

                    det_idx += 1
                    if show_pbar:
                        pbar.update(1)

                    continue

                bb_det = curr_det_data["bbox"]
                mask_det = curr_det_data["mask"]

                if assoc_method == 1:
                    """find the maximally overlapping GT of the same class"""
                    ovmax, gt_match = get_max_iou_obj(frame_gt_data, [gt_class, ], bb_det, mask_det, enable_mask)

                    """assign curr_det_data as true positive or false positive"""
                    if ovmax >= min_overlap:
                        if not gt_match['used']:
                            # true positive
                            tp[det_idx] = 1
                            tp_sum += 1
                            gt_match['used'] = True
                            count_true_positives[gt_class] += 1

                            cls_cat = "tp"
                        else:
                            """false positive (multiple detection)"""
                            fp[det_idx] = 1
                            fp_dup[det_idx] = 1
                            fp_sum += 1
                            fp_dup_sum += 1
                            cls_cat = "fp_dup"
                    else:
                        """false positive"""
                        fp[det_idx] = 1

                        ovmax_, gt_match_ = get_max_iou_obj(frame_gt_data, other_classes, bb_det, mask_det,
                                                            enable_mask)
                        if ovmax_ >= min_overlap:
                            assert gt_match_['class'] != gt_class, "FP match has current GT class"

                            if gt_match_['used_fp']:
                                """this GT object has already been matched to a misclassified 
                                duplicate detection - the first such detection counts as an FP_CLS, 
                                all others are FP_DUPs"""
                                fp_dup[det_idx] = 1
                                fp_dup_sum += 1
                                cls_cat = "fp_dup"
                            else:
                                """Misclassification of actual object"""
                                fp_cls[det_idx] = 1
                                fp_cls_sum += 1
                                cls_cat = "fp_cls"
                                gt_match_['used_fp'] = True
                        else:
                            """det does not match any GT"""
                            fp_nex[det_idx] = 1
                            fp_nex_sum += 1
                            if ovmax > 0:
                                cls_cat = "fp_nex-part"
                                fp_nex_part[det_idx] = 1
                                fp_nex_part_sum += 1
                            else:
                                fp_nex_whole[det_idx] = 1
                                fp_nex_whole_sum += 1
                                cls_cat = "fp_nex-whole"
                        fp_sum += 1
                else:
                    gt_match = all_gt_match[det_idx]
                    cls_cat = all_status[det_idx]
                    ovmax = all_ovmax[det_idx]
                    try:
                        tp_sum = cum_tp_sum[file_id]
                    except KeyError:
                        tp_sum = 0
                    try:
                        fp_sum = cum_fp_sum[file_id]
                    except KeyError:
                        fp_sum = 0

                if enable_vis:
                    img, resize_factor, _, _ = resize_ar(src_img, vis_w, vis_h, crop=1, return_factors=1)

                    color = 'hot_pink'
                    if cls_cat == "tp":
                        color = 'lime_green'
                    elif cls_cat == "fn_det":
                        color = 'magenta'

                    # cv2.rectangle(img, (int(bb_det[0]), int(bb_det[1])), (int(bb_det[2]), int(bb_det[3])), color, 2)
                    # if there is intersections between the det and GT
                    if show_gt and cls_cat in ("fp_nex-part", "tp"):
                        img = draw_objs(img, [gt_match, ], col='cyan', in_place=True, mask=0, thickness=2)

                        # bb_gt = gt_match["bbox"]
                        # bb_gt = [float(x) for x in gt_match["bbox"].split()]
                        # cv2.rectangle(img, (int(bb_gt[0]), int(bb_gt[1])), (int(bb_gt[2]), int(bb_gt[3])),
                        # light_blue, 2)
                    img = draw_objs(img, [curr_det_data, ], col=color, in_place=True, mask=0, thickness=2)

                    if not show_text:
                        _xmin = bb_det[0] * resize_factor
                        _xmax = bb_det[2] * resize_factor
                        _ymin = bb_det[1] * resize_factor
                        _ymax = bb_det[3] * resize_factor

                        _bb = [_xmin, _ymin, _xmax, _ymax]
                        if _bb[1] > 10:
                            y_loc = int(_bb[1] - 5)
                        else:
                            y_loc = int(_bb[3] + 5)
                        box_label = '{}: {:.2f}%'.format(gt_class, float(curr_det_data["confidence"]) * 100)
                        cv2.putText(img, box_label, (int(_bb[0] - 1), y_loc),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, font_line_type)
                    else:
                        # img = cv2.copyMakeBorder(img, 0, bottom_border, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
                        # height, _ = img.shape[:2]
                        v_pos = margin
                        text = "{}: {} ".format(seq_name, ground_truth_img)
                        text_img, line_width = draw_text_in_image(text_img, text, (margin, v_pos), 'white', 0)
                        text = "Class [" + str(gt_class_idx + 1) + "/" + str(n_classes) + "]: " + gt_class + " "
                        text_img, line_width = draw_text_in_image(text_img, text, (margin + line_width, v_pos), 'cyan',
                                                                  line_width)

                        text = "Result: " + cls_cat.upper()
                        if ovmax != -1:
                            text = text + " IOU {:.2f}".format(ovmax)

                        text_img, line_width = draw_text_in_image(text_img, text, (margin + line_width, v_pos), color,
                                                                  line_width)

                        v_pos += int(bottom_border / 2)
                        # rank_pos = str(idx + 1)  # rank position (idx starts at 0)
                        # text = "Prediction #rank: " + rank_pos + " confidence: {0:.2f}% ".format(
                        #     float(curr_det_data["confidence"]) * 100)

                        text = ''
                        if show_stats:
                            try:
                                _recall = float(tp_sum) / float(tp_sum + fn_sum) * 100.0
                            except ZeroDivisionError:
                                _recall = 0
                            try:
                                _prec = float(tp_sum) / float(tp_sum + fp_sum) * 100.0
                            except ZeroDivisionError:
                                _prec = 0

                            text = f'gts: {n_all_frame_gt:d} dets: {n_all_frame_dets:d} ' \
                                f'tp: {tp_sum:d} fn: {fn_sum:d} ' \
                                f'fp: {fp_sum:d} fp_dup: {fp_dup_sum:d} fp_nex: {fp_nex_sum:d} ' \
                                f'recall: {_recall:5.2f}% prec: {_prec:5.2f} '

                        text += "conf: {0:.2f}% ".format(float(curr_det_data["confidence"]) * 100)
                        text_img, line_width = draw_text_in_image(text_img, text, (margin, v_pos), 'white', 0)

                        if ovmax != -1:
                            color = 'pale_violet_red'
                            if cls_cat == "fp_nex-part":
                                text = "IoU: {0:.2f}% ".format(ovmax * 100) + "< {0:.2f}% ".format(min_overlap * 100)
                            else:
                                text = "IoU: {0:.2f}% ".format(ovmax * 100) + ">= {0:.2f}% ".format(min_overlap * 100)
                                color = 'green'
                            text_img, line_width = draw_text_in_image(text_img, text, (margin + line_width, v_pos),
                                                                      color,
                                                                      line_width)

                if is_last_in_frame:
                    """last detection in this frame assuming detections are ordered by frame"""

                    n_total_gt += len(all_class_gt)
                    # all_gt_list += [
                    #     dict(k, **{'file_id': file_id})
                    #     for k in all_class_gt
                    # ]
                    all_gt_list += all_class_gt

                    used_gt = [obj for obj in all_class_gt if obj['used']]
                    unused_gt = [obj for obj in all_class_gt if not obj['used']]

                    n_used_gt += len(used_gt)
                    n_unused_gt += len(unused_gt)

                    assert not fn_gts, "non empty fn_gts"

                    """copy unused_gt instead of creating a reference"""
                    fn_gts = unused_gt[:]
                    n_fn_gts = len(fn_gts)

                    # print(f'\n\nfound {n_fn_gts} fn_gts\n\n')

                    fn_sum += n_fn_gts

                    fn_cats = [None, ] * n_fn_gts

                    """check which of the unused GTs have corresponding detections of other classes"""
                    for gt_obj_id, gt_obj in enumerate(unused_gt):
                        ovmax_, det_match_ = get_max_iou_obj(
                            frame_det_data, other_classes,
                            gt_obj["bbox"], gt_obj["mask"], enable_mask)

                        if ovmax_ > min_overlap:
                            """misclassification"""
                            assert det_match_["class"] != gt_class, \
                                "unused GT object matches with a det of current class"
                            fn_cls_sum += 1
                            fn_cats[gt_obj_id] = 'fn_cls'
                        else:
                            """missing detection"""
                            fn_det_sum += 1
                            fn_cats[gt_obj_id] = 'fn_det'

                    assert n_fn_gts == len(unused_gt), "n_unused_gt / n_fn_gts mismatch"

                    all_considered_gt += all_class_gt

                    assert fn_det_sum + fn_cls_sum == fn_sum, "fn_sum mismatch"

                    if n_total_gt != n_used_gt + fn_sum:
                        print('fn_gts:\n{}'.format(pformat(fn_gts)))
                        print('all_class_gt:\n{}'.format(pformat(all_class_gt)))

                        raise AssertionError(
                            f'{gt_class} : {file_id} :: '
                            f'Mismatch between n_total_gt: {n_total_gt} and n_used_gt+fn_sum: {n_used_gt + fn_sum}')

                    if n_used_gt != tp_sum:
                        raise AssertionError('{} : {} :: Mismatch between n_used_gt: {} and tp_sum: {}'.format(
                            gt_class, file_id, n_used_gt, tp_sum
                        ))

                    if enable_vis:
                        """need to show these fn_gts in the next iteration of the while loop"""
                        if fn_gts:
                            # img = None
                            continue
                    else:
                        fn_gts = []

                det_idx += 1

                if show_pbar:
                    pbar.update(1)

            """
            # -------------------------------
            # completed processing all frames for one sequence
            # -------------------------------
            """

            if fps_to_gt:
                if gt_class_idx == 0:
                    """add GT of all classes at once"""
                    assert seq_name not in seq_name_to_csv_rows, f"duplicate seq_name found: {seq_name}"
                    seq_name_to_csv_rows[seq_name] = []
                    file_id_to_img_info = {}

                    gt_csv_rows = []
                    """GT is class-agnostic so should be added to the CSV only once"""
                    for _file_id, _frame_gt_data in tqdm(
                            seq_gt_data_dict.items(),
                            desc="fps_to_gt: seq_gt_data_dict",
                            ncols=100):
                        rel_path = os.path.relpath(_file_id, json_out_dir).rstrip('.' + os.sep).replace(os.sep, '/')
                        img_id = os.path.basename(_file_id)
                        img_info = {
                            'file_name': rel_path,
                            'height': _frame_gt_datum["height"],
                            'width': _frame_gt_datum["width"],
                            'id': seq_name + '/' + img_id
                        }
                        file_id_to_img_info[_file_id] = img_info

                        for _frame_gt_datum in _frame_gt_data:
                            xmin, ymin, xmax, ymax = _frame_gt_datum["bbox"]

                            row = {
                                "filename": _file_id,
                                "class": _frame_gt_datum["class"],
                                "xmin": xmin,
                                "xmax": xmax,
                                "ymin": ymin,
                                "ymax": ymax,
                                "target_id": _frame_gt_datum["target_id"],
                                "width": _frame_gt_datum["width"],
                                "height": _frame_gt_datum["height"],
                            }
                            file_id_to_img_info = {
                                'file_name': rel_path,
                                'height': height,
                                'width': width,
                                'id': seq_name + '/' + img_id
                            }

                            if enable_mask:
                                mask_rle = _frame_gt_datum["mask"]
                                mask_h, mask_w = mask_rle['size']
                                mask_counts = mask_rle['counts']
                                row.update(
                                    {
                                        "mask_h": mask_h,
                                        "mask_w": mask_w,
                                        "mask_counts": mask_counts,
                                    }
                                )
                            gt_csv_rows.append(row)

                    n_gt_objs = len(gt_csv_rows)
                    print(f'\nfps_to_gt: n_gt_objs: {n_gt_objs}')

                    gt_csv_rows.sort(key=lambda x: x['filename'])

                    seq_name_to_csv_rows[seq_name] += gt_csv_rows

                fp_nex_whole_dets = [seq_class_det_data[i] for i in range(n_seq_class_dets) if fp_nex_whole[i]]
                n_fp_nex_whole_dets = len(fp_nex_whole_dets)
                print(f'\n{gt_class} :: n_fp_nex_whole_dets : {n_fp_nex_whole_dets}')

                det_csv_rows = []
                for _det in tqdm(fp_nex_whole_dets, desc="fps_to_gt: fp_nex_whole_dets", ncols=100):
                    xmin, ymin, xmax, ymax = _det["bbox"]
                    _det_class = _det["class"]

                    assert _det_class == gt_class, \
                        f"unexpected det_class {_det_class} while processing detections for {gt_class}"
                    row = {
                        "filename": _det["file_id"],
                        "class": f"FP-{_det_class}",
                        "xmin": xmin,
                        "xmax": xmax,
                        "ymin": ymin,
                        "ymax": ymax,
                        "target_id": _det["target_id"],
                        "width": _det["width"],
                        "height": _det["height"],
                    }
                    if enable_mask:
                        mask_rle = _det["mask"]
                        mask_h, mask_w = mask_rle['size']
                        mask_counts = mask_rle['counts']
                        row.update(
                            {
                                "mask_h": mask_h,
                                "mask_w": mask_w,
                                "mask_counts": mask_counts,
                            }
                        )
                    det_csv_rows.append(row)
                n_det_objs = len(det_csv_rows)
                print(f'fps_to_gt: n_det_objs: {n_det_objs}\n')
                det_csv_rows.sort(key=lambda x: x['filename'])

                seq_name_to_csv_rows[seq_name] += det_csv_rows

            if end_class:
                break

            tp_class += [x for i, x in enumerate(tp) if fn_dets[i] == 0]
            fp_class += [x for i, x in enumerate(fp) if fn_dets[i] == 0]
            fp_dup_class += [x for i, x in enumerate(fp_dup) if fn_dets[i] == 0]
            fp_nex_class += [x for i, x in enumerate(fp_nex) if fn_dets[i] == 0]
            fp_nex_part_class += [x for i, x in enumerate(fp_nex_part) if fn_dets[i] == 0]
            fp_nex_whole_class += [x for i, x in enumerate(fp_nex_whole) if fn_dets[i] == 0]
            fp_cls_class += [x for i, x in enumerate(fp_cls) if fn_dets[i] == 0]

            conf_class += [x for i, x in enumerate(conf) if fn_dets[i] == 0]

        if show_pbar:
            pbar.close()

        if save_vis:
            for video_out in video_out_dict.values():
                if video_out is not None:
                    video_out.release()

        if save_sim_dets:
            continue

        """
        # -------------------------------
        # completed processing all sequences for one class
        # -------------------------------
        """
        # print('Sorting by confidence...')
        sort_idx = np.argsort(conf_class)[::-1]

        fp_class = [fp_class[i] for i in sort_idx]
        fp_dup_class = [fp_dup_class[i] for i in sort_idx]
        fp_nex_class = [fp_nex_class[i] for i in sort_idx]
        fp_nex_part_class = [fp_nex_part_class[i] for i in sort_idx]
        fp_nex_whole_class = [fp_nex_whole_class[i] for i in sort_idx]
        fp_cls_class = [fp_cls_class[i] for i in sort_idx]

        tp_class = [tp_class[i] for i in sort_idx]
        conf_class = [conf_class[i] for i in sort_idx]

        ap = _prec = _rec = _rec_prec = _score = 0
        _class_auc_rec_prec = 0

        n_imgs = len(all_img_paths)

        if compute_rec_prec and n_score_thresholds > 1:
            print(
                f'\n{gt_class}: Computing recall and precision '
                f'over {n_score_thresholds} thresholds, '
                f'{n_imgs} images, '
                f'{n_class_gt} GTs, '
                f'{n_class_dets} dets'
            )

            if n_threads == 1:
                print('Not using multi threading')
                _start_t = time.time()
                _rec_prec_list = []
                for __thresh_idx in range(n_score_thresholds):
                    __temp = compute_thresh_rec_prec(
                        __thresh_idx,
                        score_thresholds=score_thresholds,
                        conf_class=conf_class,
                        fp_class=fp_class,
                        fp_dup_class=fp_dup_class,
                        fp_nex_class=fp_nex_class,
                        fp_cls_class=fp_cls_class,
                        tp_class=tp_class,
                        n_gt=n_class_gt,
                    )
                    _rec_prec_list.append(__temp)
            else:
                if n_threads == 0:
                    n_threads = multiprocessing.cpu_count()

                print(f'Using {n_threads} threads')

                _start_t = time.time()
                with closing(ThreadPool(n_threads)) as pool:
                    _rec_prec_list = pool.map(functools.partial(
                        compute_thresh_rec_prec,
                        score_thresholds=score_thresholds,
                        conf_class=conf_class,
                        fp_class=fp_class,
                        fp_dup_class=fp_dup_class,
                        fp_nex_class=fp_nex_class,
                        fp_cls_class=fp_cls_class,
                        tp_class=tp_class,
                        n_gt=n_class_gt,
                    ), range(n_score_thresholds))

            rec_thresh_all[:, gt_class_idx] = [_rec_prec[0] for _rec_prec in _rec_prec_list]
            prec_thresh_all[:, gt_class_idx] = [_rec_prec[1] for _rec_prec in _rec_prec_list]
            tp_sum_thresh_all[:, gt_class_idx] = [_rec_prec[2] for _rec_prec in _rec_prec_list]
            fp_sum_thresh_all[:, gt_class_idx] = [_rec_prec[3] for _rec_prec in _rec_prec_list]
            fp_cls_sum_thresh_all[:, gt_class_idx] = [_rec_prec[4] for _rec_prec in _rec_prec_list]
            fp_dup_thresh_all[:, gt_class_idx] = [_rec_prec[5] for _rec_prec in _rec_prec_list]
            fp_nex_thresh_all[:, gt_class_idx] = [_rec_prec[6] for _rec_prec in _rec_prec_list]

            del _rec_prec_list

            _end_t = time.time()
            print('\nTime taken: {:.4f}'.format(_end_t - _start_t))
            # print()

            tp_class_cum = tp_class.copy()
            fp_class_cum = fp_class.copy()
            # compute precision/recall
            cumsum = 0
            for det_idx, val in enumerate(fp_class_cum):
                # fp_class[det_idx] has the number of false positives encountered
                # if only the first det_idx + 1 detections are considered
                fp_class_cum[det_idx] += cumsum
                cumsum += val
            cumsum = 0
            for det_idx, val in enumerate(tp_class_cum):
                tp_class_cum[det_idx] += cumsum
                cumsum += val
            # print('tp: ', tp)

            # print('fp_class_cum: ', fp_class_cum)
            # print('tp_class_cum: ', tp_class_cum)

            rec = tp_class_cum[:]
            for det_idx, val in enumerate(tp_class_cum):
                if tp_class_cum[det_idx] > 0 and n_class_gt > 0:
                    rec[det_idx] = float(tp_class_cum[det_idx]) / n_class_gt
            # print(rec)
            prec = tp_class_cum[:]
            for det_idx, val in enumerate(tp_class_cum):
                try:
                    prec[det_idx] = float(tp_class_cum[det_idx]) / (fp_class_cum[det_idx] + tp_class_cum[det_idx])
                except ZeroDivisionError:
                    prec[det_idx] = 0

            # print(prec)

            ap, mrec, mprec = voc_ap(rec, prec)

            if draw_plot:
                fig1 = plt.figure(figsize=(18, 9), dpi=80)
                plt.subplot(1, 2, 1)
                plt.plot(rec, prec, 'b-.')
                plt.fill_between(mrec, 0, mprec, alpha=0.2, edgecolor='r')

                # set window title
                fig1.canvas.set_window_title('AP ' + gt_class)
                # set plot title
                plt.title('class: ' + text)
                plt.grid(1)
                # plt.suptitle('This is a somewhat long figure title', fontsize=16)
                # set axis titles
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                # optional - set axes
                axes = plt.gca()  # gca - get current axes
                axes.set_xlim([0.0, 1.0])
                axes.set_ylim([0.0, 1.05])  # .05 to give some extra space

                # fig2 = plt.figure()
                plt.subplot(1, 2, 2)
                plt.plot(conf_class, rec, 'r-')
                # plt.hold(1)
                plt.plot(conf_class, prec, 'g-')
                plt.title('Recall and Precision vs Confidence')
                # plt.hold(0)
                plt.grid(1)

                plt.legend(['Recall', 'Precision'])

                plt.xlabel('Confidence')
                plt.ylabel('Recall / Precision')

                axes = plt.gca()  # gca - get current axes
                axes.set_xlim([0.0, 1.0])
                axes.set_ylim([0.0, 1.0])  # .05 to give some extra space

                # Alternative option -> wait for button to be pressed
                # while not plt.waitforbuttonpress():
                #     pass

                # Alternative option -> normal display
                # plt.show()

                # save the plot
                plot_out_fname = linux_path(plots_out_dir, gt_class + ".png")
                print('Saving plot to: {}'.format(plot_out_fname))
                fig1.savefig(plot_out_fname)

                plt.close(fig1)

            _rec_prec, _score, _txt = get_intersection(rec, prec, conf_class, score_thresh,
                                                       "recall", "precision")

            if draw_plot:
                out_text_class = _txt + '\n'

                out_text_class += '{}_rec_prec\n{}\n'.format(
                    gt_class,
                    pd.DataFrame(
                        data=np.vstack((conf_class, rec * 100, prec * 100)).T,
                        columns=['score_thresh', '{}_recall'.format(gt_class),
                                 '{}_precision'.format(gt_class)]).to_csv(
                        sep='\t', index=False),
                )
                out_text_class += '\n'

                # class_summary_path = out_fname.replace('.txt', '_class.md')
                class_summary_path = summary_path + '.{}'.format(gt_class)
                with open(class_summary_path, 'w') as out_file:
                    out_file.write(out_text_class)
                print('Saved {} result summary to {}'.format(gt_class, class_summary_path))

            if tp_sum > 0 and n_class_gt > 0:
                _rec = float(tp_sum) / n_class_gt
            else:
                _rec = 0
            try:
                _prec = float(tp_sum) / (fp_sum + tp_sum)
            except ZeroDivisionError:
                _prec = 0

            wmAP += ap * gt_fraction_per_class[gt_class]
            wm_prec += _prec * gt_fraction_per_class[gt_class]
            wm_rec += _rec * gt_fraction_per_class[gt_class]
            wm_rec_prec += _rec_prec * gt_fraction_per_class[gt_class]
            wm_score += _score * gt_fraction_per_class[gt_class]

            sum_AP += ap
            sum_prec += _prec
            sum_rec += _rec
            sum_rec_prec += _rec_prec
            sum_score += _score

            """
            ******************************
            rec_prec
            ******************************
            """
            _class_rec = rec_thresh_all[:, gt_class_idx] * 100
            _class_prec = prec_thresh_all[:, gt_class_idx] * 100

            csv_df = pd.DataFrame(
                data=np.vstack((score_thresholds, _class_rec, _class_prec)).T,
                columns=csv_columns_rec_prec)
            out_fname_csv = linux_path(misc_out_root_dir, f'{gt_class}-rec_prec.csv')
            csv_df.to_csv(out_fname_csv, columns=csv_columns_rec_prec, index=False, sep='\t')

            _class_auc_rec_prec = norm_auc(_class_rec, _class_prec)

        if gt_check:
            all_class_gt = []
            for k in gt_data_dict:
                if k == 'counter_per_class':
                    continue
                for m in gt_data_dict[k]:
                    all_class_gt += [obj for obj in gt_data_dict[k][m] if obj['class'] == gt_class]

            n_all_considered_gt = len(all_considered_gt)
            # n_absolute_all_gt = len(absolute_all_gt)
            # n_duplicate_gt = len(duplicate_gt)
            n_all_class_gt = len(all_class_gt)

            if n_all_considered_gt != n_all_class_gt:
                skipped_gt = [obj for obj in all_class_gt if obj not in all_considered_gt]
                n_skipped_gt = len(skipped_gt)

                # annoying_gt = [k for k in absolute_all_gt if k not in all_considered_gt]
                # n_annoying_gt = len(annoying_gt)

                # print('annoying_gt:')
                # pprint(annoying_gt)

                # print('duplicate_gt:')
                # pprint(duplicate_gt)

                # print('skipped_gt:\n{}'.format(pformat(skipped_gt)))
                print('gt_counter_per_class:\n{}'.format(pformat(gt_counter_per_class)))

                raise AssertionError(f'{gt_class} :: Mismatch between '
                                     f'n_all_considered_gt: {n_all_considered_gt} '
                                     f'and n_all_class_gt: {n_all_class_gt}, '
                                     f'n_skipped_gt: {n_skipped_gt} ')

            if n_total_gt != n_class_gt:
                seq_gt_data_list = []

                for file_id in seq_gt_data_dict.keys():
                    seq_gt_data_list += seq_gt_data_dict[file_id]
                    # for k in seq_gt_data_dict[file_id]:
                    #     seq_gt_data_list.append(
                    #         dict(k, **{'file_id': file_id})
                    #     )

                if n_total_gt > n_class_gt:
                    missing_gt = [k for k in all_gt_list if k not in seq_gt_data_list]
                else:
                    missing_gt = [k for k in seq_gt_data_list if k not in all_gt_list]

                n_missing_gt = len(missing_gt)
                raise AssertionError(f'Mismatch between n_total_gt: {n_total_gt} and '
                                     f'gt_counter_per_class[{gt_class}]: {gt_counter_per_class[gt_class]}'
                                     f'\nn_missing_gt: {n_missing_gt}')

            assert n_total_gt == tp_sum + fn_sum, \
                f'{gt_class} :: Mismatch between n_total_gt: {n_total_gt} and tp_sum+fn_sum: {tp_sum + fn_sum}, ' \
                    f'n_used_gt: {n_used_gt}'

            assert tp_sum + fn_sum == n_class_gt, f"mismatch between tp + fn {tp_sum + fn_sum} and gt {n_class_gt}"

        assert fp_sum == fp_dup_sum + fp_nex_sum + fp_cls_sum, "fp_sum mismatch"
        assert fn_sum == fn_det_sum + fn_cls_sum, "fn_sum mismatch"
        assert n_class_gt == fn_sum + tp_sum, "n_class_gt mismatch"
        assert fp_nex_sum == fp_nex_part_sum + fp_nex_whole_sum, "fp_nex_sum mismatch"

        text = f"{gt_class:s}\t" \
            f"{ap * 100:.2f}\t" \
            f"{_prec * 100:.2f}\t{_rec * 100:.2f}\t{_rec_prec * 100:.2f}\t" \
            f"{_score * 100:.2f}" \
            f"\t{tp_sum:d}\t" \
            f"{fn_sum:d}\t{fn_det_sum:d}\t{fn_cls_sum:d}\t" \
            f"{fp_sum:d}\t{fp_dup_sum:d}\t{fp_nex_sum:d}\t{fp_cls_sum:d}\t" \
            f"{n_class_gt:d}\t{n_class_dets:d}"

        if n_class_gt == 0:
            assert fn_det_sum == 0, "fn_det_sum > 0 but n_class_gt is 0"
            fnr_det = 0
        else:
            fnr_det = fn_det_sum / n_class_gt * 100

        if n_class_dets == 0:
            assert fp_cls_sum == 0, "fp_cls_sum > 0 but n_class_dets is 0"
            assert fp_dup_sum == 0, "fp_dup_sum > 0 but n_class_dets is 0"
            assert fp_nex_part_sum == 0, "fp_nex_part_sum > 0 but n_class_dets is 0"
            assert fp_nex_whole_sum == 0, "fp_nex_whole_sum > 0 but n_class_dets is 0"
            assert fp_nex_sum == 0, "fp_nex_sum > 0 but n_class_dets is 0"

            fpr_cls = fpr_dup = fpr_nex = fpr_nex_part = fpr_nex_whole = 0
        else:
            fpr_cls = fp_cls_sum / n_class_dets * 100
            fpr_dup = fp_dup_sum / n_class_dets * 100
            fpr_nex = fp_nex_sum / n_class_dets * 100
            fpr_nex_part = fp_nex_part_sum / n_class_dets * 100
            fpr_nex_whole = fp_nex_whole_sum / n_class_dets * 100

        eval_result_dict[gt_class] = {
            'AP': ap * 100,
            'Precision': _prec * 100,
            'Recall': _rec * 100,
            'R=P': _rec_prec * 100,
            'Score': _score * 100,
            'TP': tp_sum,

            'FN': fn_sum,
            'FN_CLS': fn_cls_sum,

            'FN_DET': fn_det_sum,
            'FNR_DET': fnr_det,

            'FP': fp_sum,
            'FP_CLS': fp_cls_sum,

            'FP_DUP': fp_dup_sum,
            'FP_NEX': fp_nex_sum,
            'FP_NEX_PART': fp_nex_part_sum,
            'FP_NEX_WHOLE': fp_nex_whole_sum,

            'FPR_CLS': fpr_cls,
            'FPR_DUP': fpr_dup,
            'FPR_NEX': fpr_nex,
            'FPR_NEX_PART': fpr_nex_part,
            'FPR_NEX_WHOLE': fpr_nex_whole,

            'GT': n_class_gt,
            'DETS': n_class_dets,

            'auc_rec_prec': _class_auc_rec_prec,
        }
        text_table.add_row(text.split('\t'))

        cmb_summary_data[gt_class] = [ap * 100, _rec_prec * 100, _score * 100, gt_counter_per_class[gt_class]]

        out_text += text + '\n'

        tp_sum_overall += tp_sum

        fp_sum_overall += fp_sum
        fp_dup_sum_overall += fp_dup_sum
        fp_nex_sum_overall += fp_nex_sum
        fp_cls_sum_overall += fp_cls_sum

        class_stats[gt_class_idx] = dict(
            n_dets=n_class_dets,

            n_gt=n_class_gt,

            tp_class=np.array(tp_class, dtype=np.uint8),

            fp_class=np.array(fp_class, dtype=np.uint8),
            fp_dup_class=np.array(fp_dup_class, dtype=np.uint8),
            fp_nex_class=np.array(fp_nex_class, dtype=np.uint8),
            fp_nex_part_class=np.array(fp_nex_part_class, dtype=np.uint8),
            fp_nex_whole_class=np.array(fp_nex_whole_class, dtype=np.uint8),
            fp_cls_class=np.array(fp_cls_class, dtype=np.uint8),

            conf_class=np.array(conf_class, dtype=np.float32),

            tp_sum=tp_sum,

            fp_sum=fp_sum,
            fp_dup_sum=fp_dup_sum,
            fp_nex_sum=fp_nex_sum,
            fp_cls_sum=fp_cls_sum,

            fn_sum=fn_sum,
            fn_det_sum=fn_det_sum,
            fn_cls_sum=fn_cls_sum,
        )

        fn_sum_overall += fn_sum
        fn_det_sum_overall += fn_det_sum
        fn_cls_sum_overall += fn_cls_sum

        gt_overall += n_class_gt
        dets_overall += n_class_dets

    """**************************"""
    """**************************"""
    """completed main processing"""
    """**************************"""
    """**************************"""

    """annoyingly long bit of final postprocessing"""
    if True:
        assert fp_sum_overall == fp_dup_sum_overall + fp_nex_sum_overall + fp_cls_sum_overall, "fp_sum_overall mismatch"
        assert fn_sum_overall == fn_det_sum_overall + fn_cls_sum_overall, "fn_sum_overall mismatch"

        if n_classes == 2:
            binary_cls_metrics(
                class_stats,
                tp_sum_thresh_all,
                fp_sum_thresh_all,
                fp_cls_sum_thresh_all,
                score_thresholds,
                gt_classes,
                out_root_dir,
                misc_out_root_dir,
                eval_result_dict,
            )

        if save_sim_dets:
            return None

        if show_vis:
            cv2.destroyWindow(win_name)

        mAP = sum_AP / n_classes
        m_prec = sum_prec / n_classes
        m_rec = sum_rec / n_classes
        m_rec_prec = sum_rec_prec / n_classes
        m_score = sum_score / n_classes

        if wt_avg:
            avg_txt = 'wt_avg'
            avg_wts = gt_fraction_per_class_list
        else:
            avg_txt = 'avg'
            avg_wts = None

        # text = 'Overall\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:d}\t{:d}\t{:d}\t{:d}'.format(
        #     mAP * 100, m_prec * 100, m_rec * 100, m_rec_prec * 100, m_score * 100,
        #     tp_sum_overall, fn_sum_overall, fp_sum_overall, gt_overall)
        # text_table.add_row(text.split('\t'))
        # out_text += text + '\n'

        # text = 'mean\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}'.format(wmAP * 100, wm_prec * 100,
        #                                                          wm_rec * 100, wm_rec_prec * 100)
        # text_table.add_row(text.split('\t') + [''] * 5)

        text = f'avg\t{mAP * 100:.2f}\t' \
            f'{m_prec * 100:.2f}\t{m_rec * 100:.2f}\t{m_rec_prec * 100:.2f}\t' \
            f'{m_score * 100:.2f}\t{tp_sum_overall:d}\t' \
            f'{fn_sum_overall:d}\t{fn_det_sum_overall:d}\t{fn_cls_sum_overall:d}\t' \
            f'{fp_sum_overall:d}\t{fp_dup_sum_overall:d}\t{fp_nex_sum_overall:d}\t{fp_cls_sum_overall:d}\t' \
            f'{gt_overall:d}\t{dets_overall:d}'

        text_table.add_row(text.split('\t'))
        out_text += text + '\n'

        text = f'wt_avg\t{wmAP * 100:.2f}\t' \
            f'{wm_prec * 100:.2f}\t{wm_rec * 100:.2f}\t{wm_rec_prec * 100:.2f}\t' \
            f'{wm_score * 100:.2f}\t{tp_sum_overall:d}\t' \
            f'{fn_sum_overall:d}\t{fn_det_sum_overall:d}\t{fn_cls_sum_overall:d}\t' \
            f'{fp_sum_overall:d}\t{fp_dup_sum_overall:d}\t{fp_nex_sum_overall:d}\t{fp_cls_sum_overall:d}\t' \
            f'{gt_overall:d}\t{dets_overall:d}'

        text_table.add_row(text.split('\t'))
        out_text += text + '\n'

        eval_result_dict['overall'] = {
            '_AP': mAP * 100,
            '_Precision': m_prec * 100,
            '_Recall': m_rec * 100,
            '_R=P': m_rec_prec * 100,
            'AP': wmAP * 100,
            'Precision': wm_prec * 100,
            'Recall': wm_rec * 100,
            'R=P': wm_rec_prec * 100,
            'TP': tp_sum_overall,
            'FN': fn_sum_overall,
            'FP': fp_sum_overall,
            'GT': gt_overall,
            'DETS': dets_overall,
        }
        cmb_summary_data['avg'] = [mAP * 100, m_rec_prec * 100, m_score * 100, gt_overall]
        cmb_summary_text = ''

        if n_score_thresholds > 1:
            print('Computing combined results over {} thresholds'.format(n_score_thresholds))
            # m_rec_thresh = [0] * n_score_thresholds
            # m_prec_thresh = [0] * n_score_thresholds

            wm_rec_thresh = np.zeros((n_score_thresholds,))
            wm_prec_thresh = np.zeros((n_score_thresholds,))

            gt_fraction_per_class_list = np.asarray(gt_fraction_per_class_list).squeeze()

            for thresh_idx, _thresh in enumerate(score_thresholds):
                _rec_thresh, _prec_thresh = rec_thresh_all[thresh_idx, :].squeeze(), \
                                            prec_thresh_all[thresh_idx, :].squeeze()

                wm_rec_thresh[thresh_idx] = np.average(_rec_thresh, weights=avg_wts)
                wm_prec_thresh[thresh_idx] = np.average(_prec_thresh, weights=avg_wts)

            overall_ap_thresh, _, _ = voc_ap(wm_rec_thresh[::-1], wm_prec_thresh[::-1])
            wm_diff_thresh = wm_rec_thresh - wm_prec_thresh

            itsc_idx = np.argwhere(np.diff(np.sign(wm_rec_thresh - wm_prec_thresh))).flatten()

            if not itsc_idx.size:
                # print('rec/prec: {}'.format(pformat(np.vstack((conf_class, rec, prec)).T)))
                itsc_idx = np.argmin(np.abs(wm_diff_thresh))
                if itsc_idx.size > 1:
                    itsc_idx = itsc_idx[0]
                    print('No intersection between recall and precision found; ' \
                          'min_difference: {} at {} for confidence: {}'.format(
                        wm_diff_thresh[itsc_idx], (wm_rec_thresh[itsc_idx], wm_prec_thresh[itsc_idx]),
                        score_thresholds[itsc_idx])
                    )
            else:
                print('intersection at {} for confidence: {} with idx: {}'.format(
                    wm_rec_thresh[itsc_idx], score_thresholds[itsc_idx], itsc_idx))

            print('overall_ap: {}'.format(overall_ap_thresh))

            if draw_plot:
                fig1 = plt.figure(figsize=(18, 9), dpi=80)
                plt.subplot(1, 2, 1)
                plt.plot(wm_rec_thresh, wm_prec_thresh, 'b-.')

                # set window title
                fig1.canvas.set_window_title('AP ' + gt_class)
                # set plot title
                plt.title('class: ' + text)
                plt.grid(1)
                # plt.suptitle('This is a somewhat long figure title', fontsize=16)
                # set axis titles
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                # optional - set axes
                axes = plt.gca()  # gca - get current axes
                axes.set_xlim([0.0, 1.0])
                axes.set_ylim([0.0, 1.05])  # .05 to give some extra space

                # fig2 = plt.figure()
                plt.subplot(1, 2, 2)
                plt.plot(score_thresholds, wm_rec_thresh, 'r-')
                # plt.hold(1)
                plt.plot(score_thresholds, wm_prec_thresh, 'g-')
                plt.title('Recall and Precision vs Confidence')
                # plt.hold(0)
                plt.grid(1)

                plt.legend(['Recall', 'Precision'])

                plt.xlabel('Confidence')
                plt.ylabel('Recall / Precision')

                axes = plt.gca()  # gca - get current axes
                axes.set_xlim([0.0, 1.0])
                axes.set_ylim([0.0, 1.0])  # .05 to give some extra space

                # Alternative option -> wait for button to be pressed
                # while not plt.waitforbuttonpress():
                #     pass

                # Alternative option -> normal display
                # plt.show()

                # save the plot
                plot_out_fname = linux_path(plots_out_dir, "overall.png")
                print('Saving plot to: {}'.format(plot_out_fname))
                fig1.savefig(plot_out_fname)

                plt.close(fig1)

            try:
                itsc_idx = itsc_idx[0]
            except IndexError:
                pass

            _idx_threshs = [itsc_idx, ]

            if compute_opt:
                diff_thresh = 0.02

                opt_idx = itsc_idx
                opt_data = []

                idx_threshs = range(itsc_idx + 1)[::-1]
                itsc_wm_rec, itsc_wm_prec = wm_rec_thresh[itsc_idx], wm_prec_thresh[itsc_idx]

                for curr_idx in idx_threshs:
                    curr_wm_rec, curr_wm_prec = wm_rec_thresh[curr_idx], wm_prec_thresh[curr_idx]
                    inc_rec, dec_prec = curr_wm_rec - itsc_wm_rec, itsc_wm_prec - curr_wm_prec

                    diff_rec_prec = (curr_wm_rec - curr_wm_prec) / curr_wm_rec

                    diff_dec_rec_prec = (inc_rec - dec_prec)

                    opt_data.append([k * 100 for k in [score_thresholds[curr_idx], curr_wm_rec, curr_wm_prec,
                                                       inc_rec, dec_prec, diff_rec_prec]])

                    if inc_rec < 0 or dec_prec < 0 or diff_rec_prec < 0:
                        # raise SystemError('Something weird going on: idx_threshs:\n{}\n'
                        #                   'curr_wm_rec: {} curr_wm_prec: {} inc_rec: {}, dec_prec: {} diff_rec_prec: {
                        #                   }'.format(
                        #     idx_threshs, curr_wm_rec, curr_wm_prec, inc_rec, dec_prec, diff_rec_prec))
                        break

                    if inc_rec < dec_prec and diff_rec_prec > diff_thresh:
                        break

                    opt_idx = curr_idx

                opt_score_thresh, opt_wm_rec, opt_wm_prec = score_thresholds[opt_idx], wm_rec_thresh[opt_idx], \
                                                            wm_prec_thresh[opt_idx]

                opt_data = np.asarray(opt_data)
                opt_headers = ['score_thresh', 'recall', 'precision', 'inc_rec', 'dec_prec', 'diff_rec_prec']
                print(tabulate(opt_data, opt_headers, tablefmt="fancy_grid"))

                if opt_idx != itsc_idx:
                    _idx_threshs.append(opt_idx)

            # out_text += 'rec_ratio_data\n{}\n'.format(
            #     pd.DataFrame(data=opt_data, columns=opt_headers).to_csv(sep='\t', index=False))

            print('itsc_idx: {}'.format(itsc_idx))
            if isinstance(itsc_idx, list) and not itsc_idx:
                _score_threshold = 0
            else:
                _score_threshold = score_thresholds[itsc_idx]
                print('_score_threshold: {}'.format(_score_threshold))

            cmb_summary_text = '\tClass Specific\t\t\tmRP threshold {:.2f} %\t\t\n'.format(
                _score_threshold * 100)

            cmb_summary_text += 'class\tAP(%)\tRP(%)\tScore(%)\tRecall(%)\tPrecision(%)\tAverage(%)\tGT\n'

            for __i, _idx in enumerate(_idx_threshs):
                _score_threshold = score_thresholds[_idx] * 100

                for _class_id, _class_name in enumerate(gt_classes):
                    _header = '{:s} {:.2f}'.format(_class_name, _score_threshold)

                    class_rec = rec_thresh_all[:, _class_id].squeeze()
                    class_prec = prec_thresh_all[:, _class_id].squeeze()

                    class_tp = tp_sum_thresh_all[:, _class_id].squeeze()
                    class_fp = fp_sum_thresh_all[:, _class_id].squeeze()
                    class_fp_cls = fp_cls_sum_thresh_all[:, _class_id].squeeze()
                    class_fp_dup = fp_dup_thresh_all[:, _class_id].squeeze()
                    class_fp_nex = fp_nex_thresh_all[:, _class_id].squeeze()

                    class_ap, _, _ = voc_ap(class_rec[_idx:][::-1], class_prec[_idx:][::-1])
                    class_ap *= 100
                    # print('score_threshold {} :: {} ap: {}'.format(_score_threshold, _class_name, class_ap))

                    _curr_rec = class_rec[_idx] * 100
                    _curr_prec = class_prec[_idx] * 100

                    _curr_tp = class_tp[_idx]
                    _curr_fp = class_fp[_idx]
                    _curr_fp_cls = class_fp_cls[_idx]
                    _curr_fp_dup = class_fp_dup[_idx]
                    _curr_fp_nex = class_fp_nex[_idx]

                    _curr_rec_prec = (_curr_prec + _curr_rec) / 2.0

                    eval_result_dict[_header] = {
                        'AP': class_ap,
                        'Precision': _curr_prec,
                        'Recall': _curr_rec,
                        'TP': _curr_tp,
                        'FP': _curr_fp,
                        'FP_CLS': _curr_fp_cls,
                        'FP_DUP': _curr_fp_dup,
                        'FP_NEX': _curr_fp_nex,
                        'Score': _score_threshold,
                    }
                    text = f'{_header:s}\t' \
                        f'{class_ap:.2f}\t' \
                        f'{_curr_prec:.2f}\t' \
                        f'{_curr_rec:.2f}\t' \
                        f'{_curr_rec_prec:.2f}\t' \
                        f'{_score_threshold:.2f}'
                    out_text += text + '\n'

                    row_list = list(text.split('\t'))
                    row_list += [''] * (len(text_table.field_names) - len(row_list))

                    text_table.add_row(row_list)

                    if __i == 0:
                        __ap, __rec_prec, __score, __gt = cmb_summary_data[_class_name]

                        cmb_summary_text += '{:s}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:d}\n'.format(
                            _class_name, __ap, __rec_prec, __score,
                            _curr_rec, _curr_prec, _curr_rec_prec, __gt
                        )

                overall_ap, _, _ = voc_ap(wm_rec_thresh[_idx:][::-1], wm_prec_thresh[_idx:][::-1])
                overall_ap *= 100
                # print('score_threshold {} :: overall ap: {}'.format(_score_threshold, overall_ap))

                _wm_rec, _wm_prec = wm_rec_thresh[_idx] * 100, wm_prec_thresh[_idx] * 100
                _wm_rec_prec = (_wm_prec + _wm_rec) / 2.0

                _header = '{} {:.2f}'.format(avg_txt, _score_threshold)
                eval_result_dict[_header] = {
                    'AP': overall_ap,
                    'Precision': _wm_prec,
                    'Recall': _wm_rec,
                    'Score': _score_threshold,
                }

                text = f'{_header:s}\t' \
                    f'{overall_ap:.2f}\t' \
                    f'{_wm_prec:.2f}\t' \
                    f'{_wm_rec:.2f}\t' \
                    f'{_wm_rec_prec:.2f}\t' \
                    f'{_score_threshold:.2f}'

                out_text += text + '\n'

                row_list = list(text.split('\t'))
                row_list += [''] * (len(text_table.field_names) - len(row_list))

                text_table.add_row(row_list)

                if __i == 0:
                    __ap, __rec_prec, __score, __gt = cmb_summary_data['avg']

                    cmb_summary_text += '{:s}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:d}\n'.format(
                        'average', __ap, __rec_prec, __score,
                        _wm_rec, _wm_prec, _wm_rec_prec, __gt
                    )

            if rec_ratios:
                rec_ratio_data = np.zeros((len(rec_ratios), 5))
                for _id, rec_ratio in enumerate(rec_ratios):
                    avg_rec_prec = (wm_rec_thresh * rec_ratio + wm_prec_thresh) / (1 + rec_ratio)
                    max_id = np.argmax(avg_rec_prec)
                    rec_ratio_data[_id, :] = (rec_ratio, score_thresholds[max_id] * 100, wm_rec_thresh[max_id] * 100,
                                              wm_prec_thresh[max_id] * 100, avg_rec_prec[max_id] * 100)
                rec_ratio_headers = ['rec_ratio', 'score_thresh', 'recall', 'precision', 'average']
                print(tabulate(rec_ratio_data, rec_ratio_headers, tablefmt="fancy_grid"))
                out_text += 'rec_ratio_data\n{}\n'.format(
                    pd.DataFrame(data=rec_ratio_data, columns=rec_ratio_headers).to_csv(sep='\t', index=False))

            csv_df = pd.DataFrame(
                data=np.vstack((score_thresholds,
                                wm_rec_thresh * 100,
                                wm_prec_thresh * 100)).T,
                columns=csv_columns_rec_prec)
            out_fname_csv = linux_path(misc_out_root_dir, f'rec_prec.csv')
            csv_df.to_csv(out_fname_csv, columns=csv_columns_rec_prec, index=False, sep='\t')

            csv_txt = csv_df.to_csv(sep='\t', index=False)

            out_text += '\nrec_prec\n{}\n'.format(csv_txt)
            out_text += '\n'

        if write_summary:
            cmb_summary_text = '{}\n{}'.format(out_template, cmb_summary_text)
            print(cmb_summary_text)

            print(text_table)
            # out_file.write(text_table.get_string() + '\n')
            print(f'saving result summary to {summary_path}')

            with open(summary_path, 'w') as out_file:
                out_file.write(cmb_summary_text)
                out_file.write(out_text)

        if fps_to_gt:
            output_json_dict = {
                "images": [],
                "type": "instances",
                "annotations": [],
                "categories": []
            }

            for seq_name, out_csv_rows in seq_name_to_csv_rows.items():
                if not out_csv_rows:
                    continue

                csv_out_path = os.path.join(out_root_dir, f'{seq_name}_gt_with_fp_nex_whole.csv')
                print(f'\nsaving fps_to_gt csv to: {csv_out_path}\n')

                # out_csv_rows.sort(key=lambda x: x['filename'])
                csv_columns = ['filename', 'class', 'xmin', 'xmax', 'ymin', 'ymax', 'width', 'height']
                if enable_mask:
                    csv_columns += ['mask_h', 'mask_w', 'mask_counts']

                df = pd.DataFrame(out_csv_rows, columns=csv_columns)
                df.to_csv(csv_out_path, index=False)

        # remove the tmp_files directory
        # if delete_tmp_files:
        #     shutil.rmtree(pkl_files_path)

    if return_eval_dict:
        return eval_result_dict
    else:
        return text_table


def run(params, *argv):
    """

    :param Params params:
    :param argv:
    :return:
    """
    params = copy.deepcopy(params)

    for i, sweep_param in enumerate(params._sweep_params):

        if argv[i] is not None:
            setattr(params, sweep_param, argv[i])

        param_val = getattr(params, sweep_param)

        if isinstance(param_val, (list, tuple)):
            setattr(params, sweep_param, param_val[0])

    # print('gt_paths', params.gt_paths)
    print('det_paths', params.det_paths)
    print('img_paths', params.img_paths)
    print('labels_path', params.labels_path)

    nms_thresh = params.nms_thresh  # type: float
    labels_path = params.labels_path

    if params.labels_root:
        labels_path = linux_path(params.labels_root, labels_path)

    assert os.path.isfile(labels_path), f"invalid labels_path: {labels_path}"

    gt_paths = params.gt_paths
    img_path_list_file = params.img_paths

    _det_path_list_file = params.det_paths
    if nms_thresh > 0:
        _det_path_list_file = f'{_det_path_list_file}_nms_{int(nms_thresh * 100):02d}'

    img_root_dir = params.img_root_dir
    gt_root_dir = params.gt_root_dir

    img_start_id = params.img_start_id
    img_end_id = params.img_end_id

    start_id = params.start_id
    end_id = params.end_id

    eval_sim = params.eval_sim
    detection_names = params.detection_names

    if not gt_root_dir:
        gt_root_dir = img_root_dir

    # if there are no classes to ignore then replace None by empty list
    # if params.ignore is None:
    #     params.ignore = []

    # specific_iou_flagged = False
    # if params.set_class_iou is not None:
    #     specific_iou_flagged = True

    save_suffix = params.save_suffix
    if nms_thresh > 0:
        save_suffix = f'{save_suffix}-nms_{int(nms_thresh * 100):02d}'

    if save_suffix:
        if params.iw:
            save_suffix = f'{save_suffix}-iw'
        out_dir_name = f'{save_suffix}'
    else:
        print('Using automatically generated suffix in the absence of an custom one')
        params.auto_suffix = 1

    img_path_list_file_temp = img_path_list_file
    if params.img_root_dir:
        img_path_list_file_temp = linux_path(params.img_root_dir, img_path_list_file_temp)

    if os.path.isdir(img_path_list_file_temp):
        if params.all_img_dirs:
            img_path_list = [linux_path(img_path_list_file_temp, name) for name in os.listdir(img_path_list_file_temp)
                             if
                             os.path.isdir(linux_path(img_path_list_file_temp, name))]
            img_path_list.sort(key=sortKey)
        else:
            img_path_list = [img_path_list_file_temp, ]
        img_path_list_file_temp = os.path.abspath(img_path_list_file_temp)

        if params.auto_suffix:
            db_name = os.path.basename(img_path_list_file_temp)
            db_root_name = os.path.basename(os.path.dirname(img_path_list_file_temp))
            out_dir_name = f'{out_dir_name}_{db_root_name}_{db_name}'

    elif os.path.isfile(img_path_list_file):
        img_path_list = file_lines_to_list(img_path_list_file)
        if img_root_dir:
            img_path_list = [linux_path(img_root_dir, name) for name in img_path_list]
        if params.auto_suffix:
            db_name = os.path.splitext(os.path.basename(img_path_list_file))[0]
            out_dir_name = f'{out_dir_name}_{db_name}'

    else:
        raise IOError('invalid img_path_list_file: {}'.format(img_path_list_file))

    print(f'img_path_list: {img_path_list}')

    if not gt_paths:
        gt_path_list = [linux_path(img_path, params.gt_csv_name) for img_path in img_path_list]
    elif gt_paths.endswith('.csv'):
        if not os.path.isfile(gt_paths) and gt_root_dir:
            gt_paths = linux_path(gt_root_dir, gt_paths)

        assert os.path.isfile(gt_paths), f"invalid gt_paths csv: {gt_paths}"

        gt_path_list = [gt_paths, ]
    elif os.path.isfile(gt_paths):
        gt_path_list = file_lines_to_list(gt_paths)
        # gt_path_list = [linux_path(name, 'annotations.csv') for name in gt_path_list]
        if gt_root_dir:
            gt_path_list = [linux_path(gt_root_dir, name) for name in gt_path_list]
    else:
        if not os.path.isdir(gt_paths) and gt_root_dir:
            gt_paths = linux_path(gt_root_dir, gt_paths)

        assert os.path.isdir(gt_paths), f'invalid gt_paths: {gt_paths}'

        gt_path_list = [linux_path(gt_paths, name) for name in os.listdir(gt_paths)
                        if os.path.isdir(linux_path(gt_paths, name))]
        gt_path_list.sort(key=sortKey)

    print(f'gt_path_list: {gt_path_list}')

    if not detection_names:
        detection_names = ['detections.csv', ]
    else:
        detection_names = ['detections_{}.csv'.format(detection_name) for detection_name in detection_names]

    if not eval_sim:
        all_detection_names = [detection_names, ]
    else:
        all_detection_names = []
        for _sim_rec, _sim_prec in itertools.product(params.sim_recs, params.sim_precs):
            detection_names = ['detections_rec_{:d}_prec_{:d}.csv'.format(
                int(_sim_rec * 100), int(_sim_prec * 100)), ]
            all_detection_names.append(detection_names)
        _det_path_list_file = ''

    n_seq = len(img_path_list)
    n_gts = len(gt_path_list)

    assert n_seq == n_gts, f"mismatch between len of img_path_list: {n_seq} and gt_path_list: {n_gts}"

    if end_id < start_id:
        end_id = n_seq - 1

    if params.auto_suffix:
        if params.enable_mask:
            out_dir_name = f'{out_dir_name}_mask'

        out_dir_name = f'{out_dir_name}_assoc_{params.assoc_method}'
        out_dir_name = f'{out_dir_name}_{start_id}_{end_id}'

        if params.iw:
            out_dir_name = f'{out_dir_name}-iw'

    gt_path_list = gt_path_list[start_id:end_id + 1]

    class_info = open(labels_path, 'r').read().splitlines()
    gt_classes, gt_class_cols = zip(*[k.split('\t') for k in class_info if k])
    class_name_to_col = {
        x.strip(): col
        for x, col in zip(gt_classes, gt_class_cols)
    }

    for _detection_names in all_detection_names:

        _img_path_list = img_path_list[start_id:end_id + 1]

        det_path_list_file = _det_path_list_file

        if not det_path_list_file:
            det_path_list = [[linux_path(img_path, detection_name) for detection_name in _detection_names]
                             for img_path in img_path_list]
        elif det_path_list_file.endswith('.csv'):
            det_path_list = [det_path_list_file, ]
        elif os.path.isdir(det_path_list_file):
            det_path_list = [linux_path(det_path_list_file, name) for name in os.listdir(det_path_list_file) if
                             os.path.isfile(linux_path(det_path_list_file, name)) and name.endswith('.csv')]
            det_path_list.sort(key=sortKey)
        elif os.path.isfile(det_path_list_file):
            det_path_list = file_lines_to_list(det_path_list_file)
            # det_path_list = [name + '.csv' for name in det_path_list]
        else:
            raise IOError('invalid det_path_list_file: {}'.format(det_path_list_file))

        n_dets = len(det_path_list)
        if n_dets > 0 and params.combine_dets:
            det_path_list = [det_path_list, ]
            n_dets = 1

        assert n_seq == n_dets, f"mismatch between n_seq: {n_seq} and n_dets: {n_dets}"

        det_path_list = det_path_list[start_id:end_id + 1]
        # time_stamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")
        out_root_dir = linux_path(params.out_root_dir, f'{out_dir_name}')
        os.makedirs(out_root_dir, exist_ok=1)

        """FP threshold vs AUC for all image IDs"""
        roc_auc_metrics = [
            'roc_auc_ex',
            'roc_auc_uex',
            'roc_auc_ex_fn',
            'roc_auc_uex_fn',
        ]
        """image ID vs AUC for FP_threshold of 100"""
        class_auc_metrics = [
            'auc_ex',
            'auc_uex',
            'auc_cls',
            'auc_overall',
        ]
        max_tp_metrics = [
            'max_tp',
            'max_tp_cls',
            'max_tp_ex',
            'max_tp_uex',
        ]

        det_metrics = [
            'FNR_DET',
            'FPR_DUP',
            'FPR_NEX',
        ]

        if params.iw:
            eval_dicts = {}
            img_end_ids = []

            csv_columns_max_tp = ['Image_ID', 'FP_threshold', 'Max_TP']
            csv_columns_roc_auc = ['Image_ID', 'FP_threshold', 'AUC']

            all_metrics = class_auc_metrics + max_tp_metrics + roc_auc_metrics + det_metrics

            metrics_dict = {
                metric: [] for metric in all_metrics
            }

            img_id = -1
            while True:
                img_id += 1

                img_out_root_dir = linux_path(out_root_dir, f'img_{img_id}')

                if not params.gt_pkl:
                    params.gt_pkl_dir = out_root_dir

                img_eval_dict = evaluate(
                    params=params,
                    seq_paths=_img_path_list,
                    gt_path_list=gt_path_list,
                    det_path_list=det_path_list,
                    gt_classes=gt_classes,
                    out_root_dir=img_out_root_dir,
                    img_start_id=img_id,
                    img_end_id=img_id,
                    class_name_to_col=class_name_to_col,
                    fps_to_gt=params.fps_to_gt,
                )
                if img_eval_dict is None:
                    break

                img_end_ids.append(img_id)

                eval_dicts[img_id] = img_eval_dict

                eval_0 = img_eval_dict[gt_classes[0]]
                eval_1 = img_eval_dict[gt_classes[1]]

                for metric in roc_auc_metrics:
                    metric_arr = np.asarray(img_eval_dict[metric])
                    img_id_arr = np.full((metric_arr.shape[0], 1), img_id)

                    cmb_arr = np.concatenate((img_id_arr, metric_arr), axis=1)

                    csv_df = pd.DataFrame(
                        data=cmb_arr,
                        columns=csv_columns_roc_auc)
                    metrics_dict[metric].append(csv_df)

                    del img_eval_dict[metric]

                    # metrics_dict[metric].append(img_eval_dict[metric])

                for metric in class_auc_metrics + det_metrics:
                    metrics_dict[metric].append(eval_0[metric])

                for metric in max_tp_metrics:
                    metric_arr = np.array(eval_0[metric])
                    img_id_arr = np.full((metric_arr.shape[0], 1), img_id)
                    cmb_arr = np.concatenate((img_id_arr, metric_arr), axis=1)

                    csv_df = pd.DataFrame(data=cmb_arr, columns=csv_columns_max_tp)
                    metrics_dict[metric].append(csv_df)

                    del eval_0[metric]
                    del eval_1[metric]

                img_eval_dict_path = linux_path(img_out_root_dir, 'eval_dict.json')
                open(img_eval_dict_path, 'w').write(json.dumps(img_eval_dict, indent=4))

            eval_dicts_path = linux_path(out_root_dir, 'eval_dict.json')
            print(f'saving eval_dict to {eval_dicts_path}')
            with open(eval_dicts_path, 'w') as f:
                output_json_data = json.dumps(eval_dicts, indent=4)
                f.write(output_json_data)

            for metric in class_auc_metrics + det_metrics:
                csv_columns_metric = ['img_id', metric]
                metric_list = metrics_dict[metric]
                metric_arr = np.stack((img_end_ids, metric_list), axis=1)

                arr_to_csv(metric_arr, csv_columns_metric, out_root_dir, f'{metric}-iw.csv')

            for metric in roc_auc_metrics + max_tp_metrics:
                metric_list = metrics_dict[metric]
                metric_df = pd.concat(metric_list, axis=0)

                out_fname_csv = linux_path(out_root_dir, f'{metric}-iw.csv')
                metric_df.to_csv(out_fname_csv, index=False, sep='\t')
        else:
            # out_root_dir = linux_path(f'results', f'{out_dir_name}', time_stamp)

            eval_dict = evaluate(
                params=params,
                seq_paths=_img_path_list,
                gt_path_list=gt_path_list,
                det_path_list=det_path_list,
                gt_classes=gt_classes,
                out_root_dir=out_root_dir,
                class_name_to_col=class_name_to_col,
                img_start_id=img_start_id,
                img_end_id=img_end_id,
                fps_to_gt=params.fps_to_gt,
            )
            eval_0 = eval_dict[gt_classes[0]]
            eval_1 = eval_dict[gt_classes[1]]

            for metric in max_tp_metrics:
                del eval_0[metric]
                del eval_1[metric]

            for metric in roc_auc_metrics:
                del eval_dict[metric]

            eval_dict_path = linux_path(out_root_dir, 'eval_dict.json')
            print(f'saving eval_dict to {eval_dict_path}')
            with open(eval_dict_path, 'w') as f:
                output_json_data = json.dumps(eval_dict, indent=4)
                f.write(output_json_data)


def main():
    params = Params()
    paramparse.process(params)

    sweep_vals = []
    for i, sweep_param in enumerate(params._sweep_params):
        param_val = getattr(params, sweep_param)

        sweep_vals.append(param_val)

        print('testing over {} {}: {}'.format(len(param_val), sweep_param, param_val))

    import itertools

    sweep_val_combos = list(itertools.product(*sweep_vals))

    n_sweep_val_combos = len(sweep_val_combos)
    n_proc = min(params.n_proc, n_sweep_val_combos)

    print(f'testing over {n_sweep_val_combos} param combos')
    if n_proc > 1:

        # import multiprocessing
        import functools

        print(f'running in parallel over {n_proc} processes')
        # pool = multiprocessing.Pool(n_proc)
        pool = ThreadPool(n_proc)
        func = functools.partial(run, params)

        pool.starmap(func, sweep_val_combos)
    else:
        for sweep_val_combo in sweep_val_combos:
            run(params, *sweep_val_combo)


if __name__ == '__main__':
    main()
