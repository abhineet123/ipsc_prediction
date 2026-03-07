import os
import sys
import functools

import glob
import shutil
import sys
import pickle
import copy
import time
import multiprocessing
from multiprocessing.pool import ThreadPool
import json
import lzma

import functools
from datetime import datetime
from contextlib import closing

import imagesize

import paramparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pprint import pformat
from tqdm import tqdm

from prettytable import PrettyTable
from tabulate import tabulate
from collections import OrderedDict
import itertools

import eval_utils as utils

sys.path.append(utils.linux_path(os.path.expanduser('~'), '617', 'plotting'))

import concat_metrics


class Params(paramparse.CFG):
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
    :ivar a: 'no animation is shown.',
    :ivar p: 'no plot is shown.',
    :ivar eval_sim: evaluate already created simulated detections,
    :ivar filter_ignored: remove GT and det objects with IOA > ignore_ioa_thresh with any of the ignored regions
    supplied as
        objects with class name "ignored"; mainly for the DETRAC dataset
    :ivar assoc_method:
            '0: new method with det-to-GT and GT-to-det symmetrical associations;'
            '1: old method with only det-to-GT associations',
    """

    def __init__(self):
        paramparse.CFG.__init__(self, cfg_prefix='eval_det')

        self.check_seq_name = 1
        self.delete_tmp_files = 0

        self.verbose = 1
        self.save_dets = 0

        self.save_as_imagenet_vid = 0
        self.imagenet_vid_map_path = ''

        self.concat = 1
        self.sleep = 30

        self.det_root_dir = ''
        self.det_paths = ''

        self.allow_missing_dets = 0
        self.combine_dets = 0
        self.normalized_dets = 0

        """remove patch ID suffix from det filenames"""
        self.patch_dets = 0

        """alternative class names are used in some p2s-seg created csv files
        e.g. ips instead of ipsc and dif instead of diff
        """
        self.labels_remap = 0

        self.draw_plot = 0
        self.gt_paths = ''
        self.gt_root_dir = ''

        self.img_ext = ''
        self.img_paths = ''
        self.img_paths_suffix = []
        self.img_root_dir = ''
        self.all_img_dirs = 1

        self.load_samples = []
        self.load_samples_root = ''
        self.load_samples_suffix = ''

        self.iou_thresh = 0.25
        self.labels_root = 'lists/classes/'
        self.labels_path = ''

        self.out_root_dir = '/data/mAP'
        self.out_root_suffix = []

        self.no_animation = False
        self.no_plot = False
        self.gt_pkl_dir = 'log/pkl'
        self.det_pkl_dir = ''
        self.load_gt = 0
        self.gt_pkl = ''
        self.gt_pkl_suffix = []
        self.load_det = 0
        self.save_det_pkl = 0
        self.del_det_pkl = 0
        self.save_lzma = 0
        self.det_pkl = ''
        self.quiet = False

        self.score_thresholds = [0, ]
        self.set_class_iou = [None, ]
        self.show_vis = 0
        self.show_each = 0
        self.show_gt = 1
        self.show_tp = 0
        self.show_stats = 1
        self.gt_check = 1
        self.assoc_method = 1

        self.fix_csv_cols = -1
        self.fix_gt_cols = 0
        self.fix_det_cols = 1

        self.ignore_invalid_class = 0
        self.enable_mask = 0

        self.seq = []
        self.start_id = 0
        self.end_id = -1

        self.img_start_id = 0
        self.img_end_id = -1

        self.write_summary = 1
        self.compute_opt = 0
        self.rec_ratios = ()
        self.wt_avg = 0
        self.n_threads = 1

        """allow_missing_gt=2 skips detections for files without GT"""
        self.allow_missing_gt = 1
        """
        allow entire sequences without any gt
        usually happens only when sparsely sampling a sequence with many empty frames, 
        thereby ending up with all empty frames in the samples
        """
        self.allow_empty_gt = 0

        self.show_text = 1
        self.gt_csv_name = 'annotations.csv'
        self.gt_csv_suffix = ''
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

        self.vid_det = 0

        self.n_proc = 1

        self.save_vis = 0
        self.save_classes = []
        self.save_cats = ['fn_det', 'fn_cls', 'fp_nex-whole', 'fp_nex-part', 'fp_cls', 'fp_dup']
        self.auto_suffix = 0
        self.batch_name = ''
        self.save_suffix = ''
        self.save_file_res = '1920x1080'
        # self.save_file_res = '1600x900'
        # self.save_file_res = '1280x720'
        self.vid_fmt = 'mp4v,10,mp4'
        self.check_det = 0

        self.fps_to_gt = 0

        self.ignore_exceptions = 0
        self.ignore_inference_flag = 0
        self.ignore_eval_flag = 0

        self.show_pbar = True

        self.monitor_scale = 1.5
        self.fast_nms = 0
        self.dup_nms = 0
        self.debug = 0
        self.ckpt_iter = ''
        self.filter_ignored = 0
        self.ignore_ioa_thresh = 0.25
        self.conf_thresh = 0

        self.seq_wise = 0
        self.vid_stride = 0
        self.class_agnostic = 0
        self.det_nms = 0

        self.batch_nms = 0
        self.nms_thresh = 0
        self.vid_nms_thresh = 0

        self.sweep = Params.Sweep()
        """force sweep mode for the purpose of setting the output directory"""
        self.force_sweep = 0

        # self.sweep.nms_thresh = [0, ]
        # self.sweep.vid_nms_thresh = [0, ]
        # self.det_nms_all = [0, ]

        # self._sweep_params = [
        #     'det_nms',
        #     'nms_thresh_',
        #     'vid_nms_thresh_',
        # ]

    # @property
    # def sweep_params(self):
    #     return self._sweep_params

    class Sweep:
        def __init__(self):
            self.nms_thresh = [0, ]
            self.vid_nms_thresh = [0, ]
            self.det_nms = [0, ]


def evaluate(
        params: Params,
        seq_paths: list[str],
        gt_classes: list[str],
        gt_path_list: list[str],
        all_seq_det_paths: list[str],
        out_root_dir: str,
        class_name_to_col: dict,
        img_start_id=-1,
        img_end_id=-1,
        seq_to_samples=None,
        _gt_data_dict=None,
        raw_det_data_dict=None,
        eval_result_dict=None,
        fps_to_gt=1,
        json_out_dir=None,
        show_pbar=False,
        vid_info=None,
):
    if not params.verbose:
        print_ = dummy_print
        # tqdm = dummy_tqdm
    else:
        print_ = print

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
        show_each = params.show_each
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

        img_exts = ['jpg', 'png', 'bmp', 'jpeg']

        save_w, save_h = [int(x) for x in save_file_res.split('x')]
        video_h, video_w = save_h, save_w
        codec, fps, vid_ext = vid_fmt.split(',')
        fps = int(fps)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        # fourcc = -1

        # show_pbar = not show_vis

        cls_cat_types = [
            "raw",
            "cls",
            "tp",
            "fp_dup",
            "fp_cls",
            "fp_nex-part",
            "fp_nex-whole",
            "fn_det",
            "fn_cls",
        ]
        cls_cat_to_col = {
            "tp": "green",

            "fp_nex": "red",
            "fp_cls": "magenta",
            "fp_dup": "yellow",

            "fn_det": "red",
            "fn_cls": "magenta",
        }

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

        n_seq_det_paths = len(all_seq_det_paths)

        if raw_det_data_dict is None:
            if n_seq < n_seq_det_paths:
                print_(f'Mismatch between n_seq ({n_seq}) and n_seq_det_paths ({n_seq_det_paths})')
                all_seq_det_paths = all_seq_det_paths[:n_seq]

        seq_root_dirs = [os.path.dirname(x) for x in seq_paths]
        seq_name_list = [os.path.basename(x) for x in seq_paths]

        if json_out_dir is None:
            json_out_dir = seq_root_dirs[0]

        # if not os.path.exists(pkl_files_path):
        #     os.makedirs(pkl_files_path)

        out_root_name = os.path.basename(out_root_dir)

        plots_out_dir = utils.linux_path(out_root_dir, 'plots')
        if draw_plot:
            print_('Saving plots to: {}'.format(plots_out_dir))
            os.makedirs(plots_out_dir, exist_ok=True)

        if not det_pkl_dir:
            det_pkl_dir = out_root_dir
            # if params.seq >= 0:
            #     det_pkl_dir = os.path.dirname(out_root_dir)

        if not gt_pkl_dir:
            gt_pkl_dir = out_root_dir

        det_pkl = params.det_pkl
        if params.save_lzma:
            pkl_ext = 'xz'
        else:
            pkl_ext = 'pkl'

        if not det_pkl:
            det_pkl = f"raw_det_data_dict.{pkl_ext}"
        det_pkl = utils.linux_path(det_pkl_dir, det_pkl)

        gt_pkl = params.gt_pkl
        gt_pkl_suffix = params.gt_pkl_suffix
        if not gt_pkl:
            gt_pkl = f"{out_root_name}.{pkl_ext}"

        if gt_pkl_suffix:
            gt_pkl_suffix = '-'.join(gt_pkl_suffix)
            gt_pkl = utils.add_suffix(gt_pkl, gt_pkl_suffix, sep='-')

        gt_pkl = utils.linux_path(gt_pkl_dir, gt_pkl)

        if img_start_id >= 0 and img_end_id >= 0:
            assert img_end_id >= img_start_id, "img_end_id must be >= img_start_id"
            img_suffix = f'img_{img_start_id}'

            if img_start_id != img_end_id:
                img_suffix = f'{img_suffix}_{img_end_id}'

            gt_pkl = utils.add_suffix(gt_pkl, img_suffix)
            det_pkl = utils.add_suffix(det_pkl, img_suffix)

        pkl_suffixes = []
        if params.enable_mask:
            pkl_suffixes.append('mask')
        if params.filter_ignored:
            pkl_suffixes.append('ign')
        if params.class_agnostic:
            pkl_suffixes.append('agn')
        if pkl_suffixes:
            pkl_out_suffix = '-'.join(pkl_suffixes)
            gt_pkl = utils.add_suffix(gt_pkl, pkl_out_suffix, sep='-')

        """detection specific pkl_suffixes"""
        """added in out_root_dir instead"""

        # if params.vid_nms_thresh_ > 0:
        #     pkl_suffixes.append(f'vnms_{params.vid_nms_thresh_:02d}')
        # if params.nms_thresh_ > 0:
        #     pkl_suffixes.append(f'nms_{params.nms_thresh_:02d}')
        # if pkl_suffixes:
        #     pkl_out_suffix = '-'.join(pkl_suffixes)
        #     det_pkl = utils.add_suffix(det_pkl, pkl_out_suffix, sep='-')

        gt_class_data_dict = {
            gt_class: {} for gt_class in gt_classes
        }

        if _gt_data_dict is not None:
            gt_loaded = 1
            # if params.class_agnostic:
            #     _gt_data_dict = _gt_data_dict['agn']
            # else:
            #     _gt_data_dict = _gt_data_dict['mc']
            gt_data_dict = copy.deepcopy(_gt_data_dict)
        elif params.load_gt:
            """load GT data only if gt_pkl is explicitly provided"""
            if not os.path.exists(gt_pkl):
                msg = f"gt_pkl does not exist: {gt_pkl}"
                if params.iw:
                    print_('\n' + msg + '\n')
                    return None
                raise AssertionError(msg)

            print_(f'loading GT data from {gt_pkl}')
            if params.save_lzma:
                with lzma.open(gt_pkl, 'rb') as f:
                    gt_data_dict = pickle.load(f)
            else:
                with open(gt_pkl, 'rb') as f:
                    gt_data_dict = pickle.load(f)

            gt_loaded = 1
        else:
            gt_data_dict = {}
            if params.filter_ignored:
                gt_data_dict['ignored'] = {}

            gt_loaded = 0
            print_('Generating GT data')
        if gt_loaded:
            for _seq_path, _seq_gt_data_dict in gt_data_dict.items():
                if _seq_path in ["counter_per_class", 'ignored']:
                    continue

                for gt_class in gt_classes:
                    gt_class_data_dict[gt_class][_seq_path] = []

                for _img_path, _img_objs in _seq_gt_data_dict.items():
                    for obj in _img_objs:
                        # if params.class_agnostic:
                        #     obj['class'] = 'agnostic'
                        gt_class_data_dict[obj['class']][_seq_path].append(obj)

        if raw_det_data_dict is not None:
            if params.class_agnostic:
                raw_det_data_dict = raw_det_data_dict['agn']
            else:
                raw_det_data_dict = raw_det_data_dict['mc']
            # if params.vid_nms_thresh > 0 or params.nms_thresh > 0:
            raw_det_data_dict = raw_det_data_dict[(params.vid_nms_thresh, params.nms_thresh)]
            det_loaded = 1
        else:
            """det data pkl must exist if it is explicitly provided but it is loaded if it exists 
            even if not explicitly provided"""
            if params.det_pkl:
                assert os.path.exists(det_pkl), f"det_pkl does not exist: {det_pkl}"

            if params.load_det:
                assert os.path.isfile(det_pkl), f"nonexistent det_pkl: {det_pkl}"

                print_(f'loading detection data from {det_pkl}')
                if params.save_lzma:
                    with lzma.open(det_pkl, 'rb') as f:
                        raw_det_data_dict = pickle.load(f)
                else:
                    with open(det_pkl, 'rb') as f:
                        raw_det_data_dict = pickle.load(f)

                det_loaded = 1
                if params.del_det_pkl:
                    print_(f'deleting detection pkl: {det_pkl}')
                    os.remove(det_pkl)
            else:
                raw_det_data_dict = {}
                nms_raw_det_data_dict = {}
                det_loaded = 0

        if not det_loaded:
            print_('Generating detection data')
            if params.save_det_pkl:
                os.makedirs(det_pkl_dir, exist_ok=True)

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
            'VideoID': 'video_id',
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

        if isinstance(params.seq, int) and params.seq >= 0:
            """restrict processing to a single sequence"""
            assert params.seq <= len(gt_path_list), f"invalid seq_wise id: {params.seq}"
            gt_path_list = [gt_path_list[params.seq], ]
            gt_seq_names = [gt_seq_names[params.seq], ]
            seq_name_list = [seq_name_list[params.seq], ]
            seq_paths = [seq_paths[params.seq], ]
            all_seq_det_paths = [all_seq_det_paths[params.seq], ]
            n_seq = 1
            seq_path_ = seq_paths[0]

            # if gt_loaded:
            #     gt_data_dict_ = {
            #         seq_path_:  gt_data_dict[seq_path_]
            #     }
            #     if 'ignored' in gt_data_dict:
            #         gt_data_dict_['ignored'] = {
            #             seq_path_: gt_data_dict['ignored'][seq_path_]
            #         }
            #     gt_data_dict = gt_data_dict_
            #
            # if det_loaded:
            #     det_data_dict = {
            #         seq_name_list[0]: det_data_dict[seq_name_list[0]]
            #     }
        enable_nms = False
        if params.batch_nms:
            assert params.vid_nms_thresh == 0 and params.nms_thresh == 0, \
                "vid_nms_thresh and nms_thresh must be 0 in batch_nms mode"

            enable_nms = True
            if params.sweep.nms_thresh:
                print_(f'performing batch NMS with thresholds {params.sweep.nms_thresh}')
            if params.sweep.vid_nms_thresh:
                print_(f'performing batch video NMS with thresholds {params.sweep.vid_nms_thresh}')

        elif params.nms_thresh > 0 or params.vid_nms_thresh > 0:
            enable_nms = True
            if params.nms_thresh > 0:
                print_(f'performing NMS with threshold {params.nms_thresh:d}%')

            if params.vid_nms_thresh > 0:
                print_(f'performing video NMS with threshold {params.vid_nms_thresh:d}%')

        if params.save_as_imagenet_vid:
            imagenet_vid_out_path = utils.linux_path(out_root_dir, 'imagenet_vid.txt')
            print(f'\nimagenet_vid_out_path: {imagenet_vid_out_path}\n')
            """overwrite existing file if it exists with an empty file"""
            open(imagenet_vid_out_path, "w").close()

            assert params.imagenet_vid_map_path, "imagenet_vid_map_path must be provided"
            class_name_map_file = utils.linux_path(params.imagenet_vid_map_path, "map_vid.txt")
            filename_to_frame_map_file = utils.linux_path(params.imagenet_vid_map_path, "val.txt")

            class_name_map = open(class_name_map_file, "r").readlines()
            filename_to_frame_map = open(filename_to_frame_map_file, "r").readlines()

            class_name_map = [k.strip().split(' ') for k in class_name_map]
            class_name_to_id = {k[2]: int(k[1]) for k in class_name_map}

            filename_to_frame_map = [k.strip().split(' ') for k in filename_to_frame_map]
            filename_to_frame_index = {k[0]: int(k[1]) for k in filename_to_frame_map}

        """read gt and det including filtering and NMS"""
        read_iter = enumerate(zip(gt_path_list, gt_seq_names, strict=True))
        if params.batch_nms and not params.verbose:
            read_iter = tqdm(read_iter, total=len(gt_path_list))

    for seq_idx, (_gt_path, gt_seq_name) in read_iter:

        # if gt_loaded and det_loaded:
        #     break

        seq_name = seq_name_list[seq_idx]
        seq_path = seq_paths[seq_idx]

        if img_dir_name:
            seq_img_dir = utils.linux_path(seq_path, img_dir_name)
        else:
            seq_img_dir = seq_path

        if seq_to_samples is not None:
            seq_img_paths = seq_to_samples[seq_path]
        else:
            is_valid_img = lambda x: os.path.splitext(x.lower())[1][1:] in img_exts if not params.img_ext else \
                os.path.splitext(x)[1][1:] == params.img_ext

            seq_img_gen = [[utils.linux_path(dirpath, f) for f in filenames if is_valid_img(f)]
                           for (dirpath, dirnames, filenames) in os.walk(seq_img_dir, followlinks=True)]
            seq_img_paths = [item for sublist in seq_img_gen for item in sublist]
            assert seq_img_paths, "empty seq_img_paths"

        seq_img_name_to_path = {
            os.path.basename(seq_img_path): seq_img_path for seq_img_path in seq_img_paths
        }

        seq_gt_ignored_dict = None

        if gt_loaded:
            gt_img_paths = sorted(list(gt_data_dict[seq_path].keys()))
            gt_filenames = gt_img_paths[:]
            if params.filter_ignored:
                seq_gt_ignored_dict = gt_data_dict['ignored'][seq_path]

            all_img_paths += gt_img_paths

        if gt_loaded and det_loaded:
            continue

        print_(f'\n\nProcessing sequence {seq_idx + 1:d}/{n_seq:d}')
        print_(f'seq_path: {seq_path:s}')

        """read GT from csv"""
        if not gt_loaded:
            gt_path = _gt_path
            if not os.path.isfile(gt_path):
                # os.system('ls /data/ipsc')
                # os.system(f'ls {gt_path}')
                # os.system(f'ls {os.path.dirname(gt_path)}')

                print_(os.listdir('/'))
                print_(os.listdir(os.path.dirname(gt_path)))

                raise AssertionError(f'GT file: {gt_path} does not exist')

            print_(f'\ngt_path: {gt_path:s}')

            seq_gt_data_dict = {}
            seq_gt_ignored_dict = {}

            for gt_class in gt_classes:
                gt_class_data_dict[gt_class][seq_path] = []

            df_gt = pd.read_csv(gt_path)

            assert not df_gt.empty, f"empty gt_path: {gt_path}"

            if seq_to_samples is not None:
                sampled_filenames = [os.path.basename(seq_img_path) for seq_img_path in seq_img_paths]
                df_gt = df_gt.loc[df_gt['filename'].isin(sampled_filenames)]

            if params.class_agnostic:
                df_gt['class'] = 'agnostic'

            df_gt = df_gt.dropna(axis=0)

            if fix_gt_cols:
                df_gt = df_gt.rename(columns=csv_rename_dict)

            df_gt["filename"] = df_gt["filename"].apply(lambda x: seq_img_name_to_path[os.path.basename(x)])

            grouped_gt = df_gt.groupby("filename")

            n_gt = len(df_gt)
            gt_filenames = sorted(list(grouped_gt.groups.keys()))

            n_gt_filenames = len(gt_filenames)

            print_(f'{gt_path} --> {n_gt} labels for {n_gt_filenames} images')

            n_gt_filenames = len(gt_filenames)

            if img_start_id >= 0 and img_end_id >= 0:
                if img_end_id >= n_gt_filenames:
                    if params.iw:
                        print_('img_end_id exceeds n_gt_filenames')
                        return None
                    raise AssertionError('img_end_id exceeds n_gt_filenames')

                gt_filenames = gt_filenames[img_start_id:img_end_id + 1]
                n_gt_filenames = len(gt_filenames)
                print_(f'selecting {n_gt_filenames} image(s) from ID {img_start_id} to {img_end_id}')

            # gt_file_paths = [utils.linux_path(seq_path, gt_filename) for gt_filename in gt_filenames]

            gt_iter = gt_filenames

            if show_pbar:
                gt_iter = tqdm(gt_iter, total=n_gt_filenames, ncols=100, position=0, leave=True)
            else:
                print_('reading GT')

            valid_gts = 0
            total_rows = 0

            for gt_filename in gt_iter:

                assert os.path.isfile(gt_filename), f"gt_filename does not exist: {gt_filename}"

                row_ids = grouped_gt.groups[gt_filename]
                img_df = df_gt.loc[row_ids]
                file_path = gt_filename
                ignored_df = img_df.loc[img_df['class'] == 'ignored']
                real_df = img_df.loc[img_df['class'] != 'ignored']

                if params.filter_ignored and ignored_df.size > 0 and real_df.size > 0:
                    real_bboxes = np.asarray([[float(row['xmin']), float(row['ymin']),
                                               float(row['xmax']), float(row['ymax'])]
                                              for _, row in real_df.iterrows()
                                              ])
                    ignored_bboxes = np.asarray([[float(row['xmin']), float(row['ymin']),
                                                  float(row['xmax']), float(row['ymax'])]
                                                 for _, row in ignored_df.iterrows()
                                                 ])

                    seq_gt_ignored_dict[file_path] = ignored_bboxes
                    ioa_1 = np.empty((real_df.shape[0], ignored_df.shape[0]))
                    utils.compute_overlaps_multi(None, ioa_1, None, real_bboxes, ignored_bboxes)
                    valid_idx = np.flatnonzero(np.apply_along_axis(
                        lambda x: np.all(np.less_equal(x, params.ignore_ioa_thresh)),
                        axis=1, arr=ioa_1))
                    img_df = real_df.iloc[valid_idx]

                # seq_gt_data_dict[file_path] = []
                curr_frame_gt_data = []

                if show_pbar:
                    gt_iter.set_description(f'seq {seq_idx + 1} / {n_seq}: '
                                            f'valid_gts: {valid_gts} / {total_rows}')

                for _, row in img_df.iterrows():
                    total_rows += 1

                    xmin = float(row['xmin'])
                    ymin = float(row['ymin'])
                    xmax = float(row['xmax'])
                    ymax = float(row['ymax'])
                    gt_class = str(row['class'])
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

                    if gt_class not in gt_classes:
                        msg = f'{seq_name}: {gt_filename} :: invalid gt_class: {gt_class}'
                        if ignore_invalid_class:
                            # print(msg)
                            continue
                        raise AssertionError(msg)

                    valid_gts += 1

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
                            mask_rle = utils.mask_str_to_img(
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

            if not valid_gts:
                if params.allow_empty_gt:
                    pass
                    # print(f"no valid_gts found in {seq_name}")
                else:
                    raise AssertionError(f"no valid_gts found in {seq_name}")

            if params.filter_ignored:
                gt_data_dict['ignored'][seq_path] = seq_gt_ignored_dict

            gt_data_dict[seq_path] = seq_gt_data_dict

            gt_img_paths = sorted(list(seq_gt_data_dict.keys()))
            all_img_paths += gt_img_paths
        """read det from csv"""
        if not det_loaded:
            det_paths = all_seq_det_paths[seq_idx]

            if isinstance(det_paths, str):
                det_paths = [det_paths, ]

            n_det_paths = len(det_paths)
            # if n_seq == 1:
            #     print(f'\ndet_paths: {det_paths}')

            seq_det_bboxes_list = []
            # seq_det_bboxes_dict = {}

            from collections import defaultdict
            seq_det_file_to_bboxes = defaultdict(list)

            det_pbar = None

            if show_pbar and n_det_paths > 1:
                det_pbar = det_paths_iter = tqdm(det_paths, position=0, leave=True)
            else:
                det_paths_iter = det_paths
                print_(f'reading dets: {det_paths}')

            n_invalid_dets = 0
            n_total_dets = 0

            """load and consolidate all sets of detections for one sequence"""
            for _det_path_id, _det_path in enumerate(det_paths_iter):
                det_seq_name = os.path.splitext(os.path.basename(_det_path))[0]
                if check_seq_name and (det_seq_name != seq_name or gt_seq_name != seq_name):
                    raise AssertionError(f'Mismatch between GT, detection and image sequences: '
                                         f'{gt_seq_name}, {det_seq_name}, {seq_name}')

                _det_name = os.path.basename(_det_path)

                try:
                    df_det = pd.read_csv(_det_path)
                except pd.errors.EmptyDataError:
                    continue

                # df_det_orig = df_det.copy(deep=True)
                # df_det_orig_copy = df_det.copy(deep=True)

                # df_dets.append(_df_det)
                # df_det = pd.concat(df_dets, axis=0)

                if fix_det_cols:
                    df_det = df_det.rename(columns=csv_rename_dict)

                if params.labels_remap:
                    labels_remap_path = params.labels_remap
                    if params.labels_root:
                        labels_remap_path = utils.linux_path(params.labels_root, labels_remap_path)
                    assert os.path.isfile(labels_remap_path), f"nonexistent labels_remap_path: {labels_remap_path}"
                    labels_remap_lines = open(labels_remap_path, 'r').read().splitlines()
                    labels_remap_dict = dict([labels_remap_line.split('\t')
                                              for labels_remap_line in labels_remap_lines])
                    df_det["class"] = df_det["class"].apply(lambda x: labels_remap_dict[x])

                if vid_info is not None:
                    seq_to_video_ids, seq_to_filenames = vid_info

                    try:
                        video_ids = seq_to_video_ids[seq_name]
                    except KeyError:
                        """possibly some annoying imagenet-vid like dataset with multi-part sequence names"""
                        seq_to_video_ids.update({
                            os.path.basename(k__): v__ for k__, v__ in seq_to_video_ids.items()
                        })
                        seq_to_filenames.update({
                            os.path.basename(k__): v__ for k__, v__ in seq_to_filenames.items()
                        })
                        video_ids = seq_to_video_ids[seq_name]

                    video_ids = list(map(int, video_ids.split(',')))
                    df_det = df_det.loc[df_det['video_id'].isin(video_ids)]

                    # video_filenames = seq_to_filenames[seq_name]
                    # video_filenames = [video_filenames_.split(',') for video_filenames_ in video_filenames]

                    # vid_filtered_csv = utils.add_suffix(_det_path, 'vid_filtered')
                    # print(f'vid_filtered_csv: {vid_filtered_csv}')
                    # df_det.to_csv(vid_filtered_csv, index=False, sep=',')

                # df_det_copy = df_det.copy(deep=True)

                if params.patch_dets:
                    """remove patch ID suffix from filenames"""
                    df_det["filename"] = df_det["filename"].apply(lambda x: x.split('_')[0] + os.path.splitext(x)[1])

                if params.class_agnostic:
                    df_det['class'] = 'agnostic'

                if params.img_ext:
                    df_det["filename"] = df_det["filename"].apply(
                        lambda x: os.path.splitext(os.path.basename(x))[0] + '.' + params.img_ext)

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
                        print_(f'\n\nskipping {len(non_gt_det_filenames)} non_gt_det_filenames')
                        det_filenames = [det_filename for det_filename in det_filenames if det_filename in gt_filenames]
                        n_det_filenames = len(det_filenames)

                # det_filenames = det_filenames[:50]

                if not show_pbar or n_det_paths > 1:
                    det_filename_iter = det_filenames
                else:
                    det_pbar = det_filename_iter = tqdm(det_filenames, position=0, leave=True)

                det_pbar_base_msg = ''
                if n_seq > 1:
                    det_pbar_base_msg += f"seq {seq_idx + 1} / {n_seq} "
                if n_det_paths > 1:
                    det_pbar_base_msg += f"csv {_det_path_id + 1} / {n_det_paths} "

                valid_dets = 0
                for det_filename_id, det_filename in enumerate(det_filename_iter):

                    assert os.path.isfile(det_filename), f"det_filename does not exist: {det_filename}"

                    file_path = det_filename

                    try:
                        det_img_w, det_img_h = img_path_to_size[file_path]
                    except KeyError:
                        det_img_w, det_img_h = imagesize.get(file_path)
                        img_path_to_size[file_path] = (det_img_w, det_img_h)

                    row_ids = grouped_dets.groups[det_filename]
                    img_df = df_det.loc[row_ids]

                    if params.normalized_dets:
                        img_df['xmin'] = img_df['xmin'] * det_img_w
                        img_df['xmax'] = img_df['xmax'] * det_img_w

                        img_df['ymin'] = img_df['ymin'] * det_img_h
                        img_df['ymax'] = img_df['ymax'] * det_img_h

                    if params.filter_ignored and file_path in seq_gt_ignored_dict and img_df.size > 0:
                        det_bboxes = np.asarray([[float(row['xmin']), float(row['ymin']),
                                                  float(row['xmax']), float(row['ymax'])]
                                                 for _, row in img_df.iterrows()
                                                 ])
                        ignored_bboxes = seq_gt_ignored_dict[file_path]
                        ioa_1 = np.empty((img_df.shape[0], ignored_bboxes.shape[0]))
                        utils.compute_overlaps_multi(None, ioa_1, None, det_bboxes, ignored_bboxes)
                        valid_idx = np.flatnonzero(np.apply_along_axis(
                            lambda x: np.all(np.less_equal(x, params.ignore_ioa_thresh)),
                            axis=1, arr=ioa_1))
                        img_df = img_df.iloc[valid_idx]

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

                        det_class = str(row['class'])

                        try:
                            confidence = row['confidence']
                        except KeyError:
                            confidence = 1.0

                        if params.conf_thresh > 0 and confidence < params.conf_thresh:
                            continue

                        try:
                            det_class_id = gt_classes.index(det_class)
                        except ValueError:
                            msg = f'{det_seq_name}: {det_filename} :: invalid det_class: {det_class}'
                            if ignore_invalid_class:
                                # print(msg)
                                continue
                            raise AssertionError(msg)

                        valid_dets += 1

                        det_w = xmax - xmin
                        det_h = ymax - ymin

                        n_total_dets += 1

                        if det_w <= 0 or det_h <= 0:
                            n_invalid_dets += 1
                            # print(f'ignoring invalid detection: {[xmin, ymin, xmax, ymax]}')
                            continue

                        try:
                            video_id = int(row['video_id'])
                        except KeyError:
                            video_id = -1

                        if params.vid_det:
                            assert video_id >= 0, "vid_det csv must have valid video_id"

                        bbox_dict = {
                            "class_id": det_class_id,
                            "class": det_class,
                            "width": det_img_w,
                            "height": det_img_h,
                            "target_id": target_id,
                            "confidence": confidence,
                            "filename": det_filename,
                            "file_path": file_path,
                            "bbox": [xmin, ymin, xmax, ymax],
                            'mask': None,
                            "det_path_id": _det_path_id,
                            "video_id": video_id,
                        }

                        if enable_mask:
                            try:
                                mask_rle = utils.mask_str_to_img(row["mask"], det_img_h, det_img_w, to_rle=1)
                            except KeyError:
                                mask_w = int(row["mask_w"])
                                mask_h = int(row["mask_h"])
                                mask_counts = str(row["mask_counts"])

                                mask_rle = dict(
                                    size=(mask_h, mask_w),
                                    counts=mask_counts
                                )

                            if params.normalized_dets:
                                mask_rle = utils.resize_mask_rle_through_img(mask_rle, det_img_w, det_img_h)

                            bbox_dict["mask"] = mask_rle

                        seq_det_bboxes_list.append(bbox_dict)
                        seq_det_file_to_bboxes[det_filename].append(bbox_dict)

                        global_id = len(seq_det_bboxes_list) - 1
                        local_id = len(seq_det_file_to_bboxes[det_filename]) - 1

                        bbox_dict['local_id'] = local_id
                        bbox_dict['global_id'] = global_id
                        bbox_dict['to_delete'] = 0

                        # seq_det_bboxes_dict[local_id] = bbox_dict

                        if det_pbar is not None:
                            time_stamp = datetime.now().strftime("%y%m%d %H%M%S")
                            invalid_pc = (n_invalid_dets / n_total_dets) * 100
                            det_pbar_msg = (f"{time_stamp} {det_pbar_base_msg} "
                                            f"invalid: {utils.num_to_words(n_invalid_dets)} / "
                                            f"{utils.num_to_words(n_total_dets)} "
                                            f"({invalid_pc:.2f}%)")

                            det_pbar.set_description(det_pbar_msg)

                # assert valid_dets > 0, "no valid_dets found"

            if enable_nms:
                seq_nms_filtered_bboxes_list = defaultdict(list)
                nms_iter = seq_det_file_to_bboxes.items()
                nms_pbar = None
                if show_pbar:
                    nms_pbar = nms_iter = tqdm(nms_iter, position=0, leave=True)
                    print_('\nperforming nms')
                # seq_bbox_ids_to_delete = []
                total_bboxes = 0
                del_bboxes = 0
                for _det_filename, _bbox_info in nms_iter:
                    if params.batch_nms:
                        nms_thresh_to_filtered_objs = utils.perform_batch_nms(
                            _bbox_info,
                            enable_mask=params.enable_mask,
                            nms_thresh_all=params.sweep.nms_thresh,
                            vid_nms_thresh_all=params.sweep.vid_nms_thresh,
                            dup=params.dup_nms,
                            vis=0,
                            class_name_to_col=class_name_to_col,
                        )
                        for (vid_nms_thresh_, nms_thresh_), filtered_objs in nms_thresh_to_filtered_objs.items():
                            seq_nms_filtered_bboxes_list[(vid_nms_thresh_, nms_thresh_)] += filtered_objs
                    else:
                        if params.fast_nms:
                            assert not params.vid_nms_thresh, "fast nms does not support separate video nms"
                            n_del = utils.perform_nms_fast(_bbox_info, params.enable_mask, params.nms_thresh)
                        else:
                            n_del, n_pairs, n_vid_pairs = utils.perform_nms(
                                _bbox_info,
                                enable_mask=params.enable_mask,
                                nms_thresh=params.nms_thresh,
                                vid_nms_thresh=params.vid_nms_thresh,
                                dup=params.dup_nms,
                            )
                        # seq_bbox_ids_to_delete += img_bbox_ids_to_delete
                        del_bboxes += n_del
                        n_bboxes = len(_bbox_info)
                        total_bboxes += n_bboxes

                        if nms_pbar is not None:
                            time_stamp = datetime.now().strftime("%y%m%d %H%M%S")
                            del_pc = (del_bboxes / total_bboxes) * 100 if total_bboxes > 0 else 0
                            nms_pbar_msg = (f"{time_stamp} nms "
                                            f"del: {utils.num_to_words(del_bboxes)} / "
                                            f"{utils.num_to_words(total_bboxes)} "
                                            f"({del_pc:.2f}%) "
                                            f"boxes: {utils.num_to_words(n_bboxes)},"
                                            # f"{utils.num_to_words(n_pairs)},"
                                            # f"{utils.num_to_words(n_vid_pairs)}"
                                            )
                            nms_pbar.set_description(nms_pbar_msg)

                # n_bbox_ids_to_delete = len(bbox_ids_to_delete)
                # n_total_bbox_ids = len(seq_det_bboxes_list)
                # print(f'deleting {n_bbox_ids_to_delete} / {n_total_bbox_ids} bboxes')

                if not params.batch_nms:
                    seq_det_bboxes_list = [k for k in seq_det_bboxes_list if not k['to_delete']]
                    if params.save_as_imagenet_vid:
                        utils.dets_to_imagenet_vid(seq_det_bboxes_list, imagenet_vid_out_path, seq_name,
                                                   filename_to_frame_index, class_name_to_id)

                if n_det_paths > 1:
                    print_(f'n_det_paths: {n_det_paths}')
                elif params.save_dets:
                    if params.batch_nms:
                        for (vid_nms_thresh_, nms_thresh_), filtered_objs in seq_nms_filtered_bboxes_list.items():
                            utils.dets_to_csv(filtered_objs, det_paths[0], enable_mask,
                                              vid_nms_thresh_, nms_thresh_, params.class_agnostic)

                    else:
                        utils.dets_to_csv(seq_det_bboxes_list, det_paths[0], enable_mask,
                                          params.vid_nms_thresh, params.nms_thresh, params.class_agnostic)

            """Flat list of all the detections from all detection sets and images in this sequence"""
            raw_det_data_dict[seq_path] = seq_det_bboxes_list
            if params.batch_nms:
                for (vid_nms_thresh_, nms_thresh_), filtered_objs in seq_nms_filtered_bboxes_list.items():
                    try:
                        raw_det_data_dict_ = nms_raw_det_data_dict[(vid_nms_thresh_, nms_thresh_)]
                    except KeyError:
                        raw_det_data_dict_ = nms_raw_det_data_dict[(vid_nms_thresh_, nms_thresh_)] = {}
                    raw_det_data_dict_[seq_path] = filtered_objs
                # print("debug")

    """save pkl"""
    if True:
        if not det_loaded and params.save_det_pkl:
            if params.batch_nms:
                nms_pkl_iter = tqdm(nms_raw_det_data_dict.items())
                for (vid_nms_thresh_, nms_thresh_), raw_det_data_dict_ in nms_pkl_iter:
                    det_pkl_ = det_pkl

                    det_pkl_suffixes = []
                    if len(params.sweep.nms_thresh) > 1 or params.sweep.nms_thresh[0] != 0:
                        det_pkl_suffixes.append(f'nms_{nms_thresh_:02d}')
                    if len(params.sweep.vid_nms_thresh) > 1 or params.sweep.vid_nms_thresh[0] != 0:
                        det_pkl_suffixes.append(f'vnms_{vid_nms_thresh_:02d}')

                    if det_pkl_suffixes:
                        det_pkl_dir_suffix = '-'.join(det_pkl_suffixes)
                        det_pkl_ = utils.add_suffix_to_path(det_pkl_, det_pkl_dir_suffix)

                    det_pkl_dir = os.path.dirname(det_pkl_)
                    os.makedirs(det_pkl_dir, exist_ok=True)
                    det_pkl_name = os.path.basename(det_pkl_dir)
                    nms_pkl_iter.set_description(f'saving det: {det_pkl_name}')
                    if params.save_lzma:
                        with lzma.open(det_pkl_, 'wb') as f:
                            pickle.dump(raw_det_data_dict_, f, pickle.HIGHEST_PROTOCOL)
                    else:
                        with open(det_pkl_, 'wb') as f:
                            pickle.dump(raw_det_data_dict_, f, pickle.HIGHEST_PROTOCOL)

            else:
                print_(f'\nSaving detection data to {det_pkl}')
                if params.save_lzma:
                    with lzma.open(det_pkl, 'wb') as f:
                        pickle.dump(raw_det_data_dict, f, pickle.HIGHEST_PROTOCOL)
                else:
                    with open(det_pkl, 'wb') as f:
                        pickle.dump(raw_det_data_dict, f, pickle.HIGHEST_PROTOCOL)

        if not gt_loaded:
            gt_data_dict['counter_per_class'] = gt_counter_per_class
            os.makedirs(gt_pkl_dir, exist_ok=True)

            print_(f'\nSaving GT data to {gt_pkl}')
            if params.save_lzma:
                with lzma.open(gt_pkl, 'wb') as f:
                    pickle.dump(gt_data_dict, f, pickle.HIGHEST_PROTOCOL)
            else:
                with open(gt_pkl, 'wb') as f:
                    pickle.dump(gt_data_dict, f, pickle.HIGHEST_PROTOCOL)

        gt_end_t = time.time()
        if not (gt_loaded and det_loaded):
            print_('Time taken: {} sec'.format(gt_end_t - gt_start_t))

        if params.batch_nms:
            if params.save_det_pkl:
                return None
            return nms_raw_det_data_dict

    """detection post-proc - rearrange to have class-wise and sequence-wise lists of objects"""
    if True:
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
                print_('gt_counter_per_class: ', gt_counter_per_class)
                print_('total_gt: ', total_gt)
                _gt_fraction = 0

            gt_fraction_per_class[_class_name] = _gt_fraction
            gt_fraction_per_class_list.append(_gt_fraction)

        # gt_classes = list(gt_counter_per_class.keys())
        # gt_classes = sorted(gt_classes)

        n_classes = len(gt_classes)

        print_('gt_classes: ', gt_classes)
        print_('n_classes: ', n_classes)
        print_('gt_counter_per_class: ', gt_counter_per_class)

        # log_dir = 'pprint_log'
        # if not os.path.isdir(log_dir):
        #     os.makedirs(log_dir)

        if score_thresh > 0:
            print_('Discarding detections with score < {}'.format(score_thresh))

        det_post_proc_pbar = range(n_seq)
        if show_pbar:
            det_post_proc_pbar = tqdm(det_post_proc_pbar, position=0, leave=True)

        det_data_dict = {gt_class: {seq_path: [] for seq_path in seq_paths} for gt_class in gt_classes}
        for seq_idx in det_post_proc_pbar:
            seq_name = seq_name_list[seq_idx]
            seq_path = seq_paths[seq_idx]

            seq_det = raw_det_data_dict[seq_path]

            if show_pbar:
                det_post_proc_pbar.set_description(f"Post processing sequence {seq_name}")

            # curr_class_det_exists = {}
            # curr_class_det_objs = []
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

                # if file_path not in curr_class_det_exists:
                #     curr_class_det_exists[file_path] = 0

                if bbox is None or confidence < score_thresh:
                    continue

                obj_dict = {
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
                }

                det_data_dict[det_class][seq_path].append(obj_dict)

                # curr_class_det_exists[file_path] = 1

                # assert os.path.exists(file_path), f"file_path does not exist: {file_path}"

            if params.show_vis:
                seq_gt = gt_data_dict[seq_path]
                _pause = 1
                for file_path, gt_objs in seq_gt.items():
                    seq_name = os.path.basename(os.path.dirname(file_path))
                    img_name = os.path.basename(file_path)
                    img = cv2.imread(file_path)
                    img_gt = utils.draw_objs(img, gt_objs, title=f'{seq_name}-{img_name}-{len(gt_objs)}',
                                             show_class=True, class_name_to_col=class_name_to_col)

                    det_objs = [obj for obj in seq_det if obj['file_path'] == file_path]
                    img_det = utils.draw_objs(img, det_objs, title=f'{seq_name}-{img_name}-{len(det_objs)}',
                                              show_class=True, class_name_to_col=class_name_to_col)
                    img_gt = utils.resize_ar(img_gt, width=1200, height=700)
                    img_det = utils.resize_ar(img_det, width=1200, height=700)
                    cv2.imshow('img_gt', img_gt)
                    cv2.imshow('img_det', img_det)
                    k = cv2.waitKey(1 - _pause)
                    if k == 32:
                        _pause = 1 - _pause
                    elif k == 27:
                        exit(0)

    """set up for main processing"""
    if True:
        wmAP = sum_AP = 0.0
        wm_prec = wm_rec = wm_rec_prec = wm_score = 0.0
        sum_prec = sum_rec = sum_rec_prec = sum_score = 0.0

        min_overlap = iou_thresh

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
            summary_path = utils.linux_path(out_root_dir, "summary.txt")
            print_('Writing result summary to {}'.format(summary_path))

        out_template = utils.linux_path(out_root_dir).replace('/', '_')
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

        os.makedirs(out_root_dir, exist_ok=True)

        misc_out_root_dir = utils.linux_path(out_root_dir, 'misc')
        os.makedirs(misc_out_root_dir, exist_ok=True)

        if vid_ext in img_exts:
            vis_video = 0
        else:
            vis_video = 1

        seq_name_to_csv_rows = {}
        file_id_to_img_info = {}
        json_dict = {
            "images": [],
            "type": "instances",
            "annotations": [],
            "categories": []
        }

        bnd_id = 1
        n_all_gt_objs = n_all_fp_nex_whole_dets = 0

        json_category_name_to_id = {}
        video_out_dict = ground_truth_img = vis_out_fnames = None

        all_class_dets = all_class_gt = None
        src_img = frame_det_data = frame_gt_data = None
        vis_w = vis_h = None
        vert_stack = 0
        vis_w_all = vis_h_all = None
        cat_img_vis_list = None

    """main processing"""
    for gt_class_idx, gt_class in enumerate(gt_classes):
        """preprocessing for each class"""
        if True:
            category_info = {'supercategory': 'none', 'id': gt_class_idx, 'name': gt_class}
            json_dict['categories'].append(category_info)
            json_category_name_to_id[gt_class] = gt_class_idx

            category_info = {'supercategory': 'none', 'id': gt_class_idx + n_classes, 'name': f'FP-{gt_class}'}
            json_dict['categories'].append(category_info)
            json_category_name_to_id[f'FP-{gt_class}'] = gt_class_idx + n_classes

            enable_vis = show_vis or (save_vis and gt_class in save_classes)

            n_class_dets = 0

            n_class_gt = gt_counter_per_class[gt_class]

            other_classes = [k for k in gt_classes if k != gt_class]

            print_(f'\nProcessing class {gt_class_idx + 1:d} / {n_classes:d}: {gt_class:s}')

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
                    vis_root_dir = utils.linux_path(out_root_dir, 'vis', gt_class)
                else:
                    vis_root_dir = out_root_dir

                os.makedirs(vis_root_dir, exist_ok=True)
                print_(f'\n\nsaving {save_w} x {save_h} vis videos to {vis_root_dir}\n\n')
                video_out_dict = {
                    cat: None for cat in save_cats
                }

            cat_to_ids_vis_done = {k: [] for k in cls_cat_types}
            cat_to_vis_count = {k: 0 for k in cls_cat_types}
            seq_iter = range(n_seq)
            if show_pbar:
                seq_iter = tqdm(seq_iter, desc="sequence", ncols=70, position=0, leave=True)
            else:
                print_('computing mAP')

        for seq_idx in seq_iter:
            """preprocessing for each sequence"""
            if True:
                seq_path = seq_paths[seq_idx]
                seq_name = seq_name_list[seq_idx]
                # seq_root_dir = seq_root_dirs[seq_idx]

                if save_vis:
                    for cat, video_out in video_out_dict.items():
                        if video_out is not None:
                            video_out.release()
                            video_out_dict[cat] = None

                    vis_out_fnames = {
                        cat: utils.linux_path(vis_root_dir, cat, f'{seq_name}.{vid_ext}')
                        for cat in save_cats
                    }

                seq_gt_data_dict = gt_data_dict[seq_path]
                """total number of GTs of all classes in this sequence"""
                n_seq_gts = len(seq_gt_data_dict)

                seq_class_gt_data = gt_class_data_dict[gt_class][seq_path]
                """total number of GTs of this class in this sequence"""
                n_seq_class_gts = len(seq_class_gt_data)

                """all detections of this class in this sequence"""
                seq_class_det_data = det_data_dict[gt_class][seq_path]
                """total number of detections of this class in this sequence"""
                n_seq_class_dets = len(seq_class_det_data)

                n_class_dets += n_seq_class_dets

                seq_gt_file_ids = set(seq_gt_data_dict.keys())
                # n_seq_gt_file_ids = len(seq_gt_file_ids)

                seq_class_gt_file_ids = set([obj['file_id'] for obj in seq_class_gt_data])
                # n_seq_class_gt_file_ids = len(seq_class_gt_file_ids)

                seq_class_det_file_ids = set([obj['file_id'] for obj in seq_class_det_data])
                # n_seq_det_file_ids = len(seq_class_det_file_ids)

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

                    print_('\n' + msg)

                    seq_gt_data_dict.update(
                        {k: [] for k in missing_gt_file_ids}
                    )
                    n_seq_gts = len(seq_gt_data_dict)

                if missing_class_det_file_ids:
                    """dummy detections should only be added for images that have a GT of this class 
                    to avoid unnecessary processing even though the result would be the same if we were to add 
                    them for all images with any class GT too"""

                    n_missing_class_det_file_ids = len(missing_class_det_file_ids)
                    print_(f'\n{seq_name}: no {gt_class} detections found for {n_missing_class_det_file_ids} images'
                           f' with {gt_class} GT\n')

                    """add one dummy detection for each missing image"""
                    seq_class_det_data += [
                        {'confidence': None, 'file_id': k, 'bbox': None} for k in missing_class_det_file_ids
                    ]
                    n_seq_class_dets = len(seq_class_det_data)

                seq_gt_file_ids_with_dummy = set(seq_gt_data_dict.keys())
                # n_seq_gt_file_ids_with_dummy = len(seq_gt_file_ids_with_dummy)

                seq_class_det_file_ids_with_dummy = set([obj['file_id'] for obj in seq_class_det_data])
                # n_seq_class_det_file_ids_with_dummy = len(seq_class_det_file_ids_with_dummy)

                # if params.verbose == 2:
                #     print_(f'n_seq_gt_file_ids_with_dummy: {n_seq_gt_file_ids_with_dummy}')
                #     print_(f'n_seq_class_det_file_ids_with_dummy: {n_seq_class_det_file_ids_with_dummy}')

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
                    cum_tp_sum, cum_fp_sum = utils.perform_global_association(
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

            """process all the detections in this sequence frame-by-frame"""
            while True:
                """process one detection at a time"""
                has_objs = all_class_dets or all_class_gt
                is_first_det_in_frame = vis_file_id is None or file_id != vis_file_id

                if (enable_vis and (img is not None)
                        and has_objs  # nothing useful to show if neither GTs nr dets of this class exist in this frame
                ):
                    if is_first_det_in_frame:
                        """raw vis for previous frame"""
                        vis_file_id = file_id
                        # vis_frames[file_id] = img
                        cat_img_vis_list = utils.draw_and_concat(
                            src_img, frame_det_data, frame_gt_data, class_name_to_col, params.vis_alpha, vis_w,
                            vis_h,
                            vert_stack, params.check_det, img_id, mask=enable_mask, return_list=True)

                        if "raw" in save_cats:
                            cat_img_vis_all = utils.draw_and_concat(
                                src_img, frame_det_data, frame_gt_data, class_name_to_col, params.vis_alpha,
                                vis_w_all,
                                vis_h_all, vert_stack, params.check_det, img_id, mask=enable_mask)

                            cat_img_h, cat_img_w = cat_img_vis_all.shape[:2]
                            all_text_img = np.zeros((bottom_border, cat_img_w, 3), dtype=np.uint8)
                            text = f"{seq_idx + 1}/{n_seq} {seq_name}: {ground_truth_img} "
                            all_text_img, _ = utils.draw_text_in_image(all_text_img, text, (20, 20), 'white', 0)

                            all_out_img = np.concatenate((cat_img_vis_all, all_text_img), axis=0)

                            if save_vis:
                                all_video_out = utils.get_video_out(video_out_dict, vis_out_fnames, "raw",
                                                                    vis_video,
                                                                    save_h, save_w, fourcc, fps)
                                if show_vis:
                                    all_out_img_vis = utils.resize_ar_tf_api(all_out_img, 1280, 720)
                                    cv2.imshow('all_out_img', all_out_img_vis)

                                if vis_video:
                                    all_out_img = utils.resize_ar_tf_api(all_out_img, video_w, video_h,
                                                                         add_border=1)

                                all_video_out.write(all_out_img)

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

                            cat_to_ids_vis_done[_cls_cat].append(vis_file_id)

                        if _img is None:
                            assert _text_img is None, "_img is None but _text_img isn't"
                            continue

                        out_img = utils.annotate(cat_img_vis_list + [_img, ], text=f'{img_id}',
                                                 img_labels=['GT', 'All Detections', 'Current Detection'],
                                                 grid_size=(-1, 1) if vert_stack else (1, -1))

                        out_img = np.concatenate((out_img, _text_img), axis=0)

                        if save_vis and _cls_cat in save_cats and gt_class in save_classes:

                            video_out = video_out_dict[_cls_cat]
                            if video_out is None:
                                vis_out_fname = vis_out_fnames[_cls_cat]
                                _save_dir = os.path.dirname(vis_out_fname)

                                print_(f'\n\nsaving video for {_cls_cat} at {vis_out_fname}\n\n')

                                if _save_dir and not os.path.isdir(_save_dir):
                                    os.makedirs(_save_dir)

                                if vis_video:
                                    # video_h, video_w = out_img.shape[:2]
                                    # if show_text:
                                    #     video_h += bottom_border
                                    video_out = cv2.VideoWriter(vis_out_fname, fourcc, fps, (video_w, video_h))
                                else:
                                    video_out = utils.ImageSequenceWriter(vis_out_fname, verbose=0)

                                if not video_out:
                                    raise AssertionError(
                                        f'video file: {vis_out_fname} could not be opened for writing')

                                video_out_dict[_cls_cat] = video_out

                            if vis_video:
                                out_img = utils.resize_ar_tf_api(out_img, video_w, video_h,
                                                                 # add_border=2
                                                                 )

                            video_out.write(out_img)
                            cat_to_vis_count[_cls_cat] += 1

                            n_fn_cls_vis = cat_to_vis_count['fn_cls']
                            n_fn_det_vis = cat_to_vis_count['fn_det']
                            n_fp_dup_vis = cat_to_vis_count['fp_dup']
                            n_fp_nex_whole_vis = cat_to_vis_count['fp_nex-whole']
                            n_fp_cls_vis = cat_to_vis_count['fp_cls']
                            n_fp_nex_part_vis = cat_to_vis_count['fp_nex-part']

                            if show_pbar:
                                pbar.set_description(f'seq {seq_idx + 1} / {n_seq} '
                                                     f'fn: cls-{n_fn_cls_vis} '
                                                     f'det-{n_fn_det_vis} '
                                                     f'fp: cls-{n_fp_cls_vis} '
                                                     f'dup-{n_fp_dup_vis} '
                                                     f'nex-{n_fp_nex_whole_vis},{n_fp_nex_part_vis} '
                                                     f'')

                        if show_vis:
                            if show_each:
                                if params.monitor_scale != 1.0:
                                    out_img = utils.resize_ar(out_img, width=int(1920 / params.monitor_scale))
                                cv2.imshow(win_name, out_img)

                            k = cv2.waitKey(1 - _pause)
                            if k == ord('q') or k == 27:
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

                """current detection"""
                curr_det_data = seq_class_det_data[det_idx]
                file_id = curr_det_data["file_id"]
                img_id = os.path.basename(file_id)

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

                        if params.filter_ignored:
                            ignored_regions = gt_data_dict["ignored"][seq_path][file_id]
                            utils.draw_boxes(src_img, ignored_regions, color='black', thickness=-1, xywh=False)

                        vis_h, vis_w = utils.get_vis_size(src_img, 3, save_w, save_h, bottom_border)

                        vis_h_all, vis_w_all = utils.get_vis_size(src_img, 2, save_w, save_h, bottom_border)

                        # vert_ar = src_w / (src_h * 3)
                        # horz_ar = (src_w * 3) / src_h
                        #
                        # vis_ar = save_w / save_h
                        # if abs(vis_ar - horz_ar) < abs(vis_ar - vert_ar):
                        #     vert_stack = 0
                        # else:
                        #     vert_stack = 1

                        if src_img is None:
                            raise AssertionError(f'Image could not be read: {img_full_path}')

                    img, resize_factor, _, _ = utils.resize_ar_tf_api(src_img, vis_w, vis_h, crop=1, return_factors=1)

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

                                fn_img = utils.draw_objs(fn_img, cat_fn_gts, cols='cyan',
                                                         in_place=True, thickness=2, mask=0,
                                                         bb_resize=resize_factor)
                                fn_text_img = np.zeros((bottom_border, text_img_w, 3), dtype=np.uint8)

                                v_pos = margin
                                text = f"{seq_idx + 1}/{n_seq} {seq_name}: {ground_truth_img} "
                                fn_text_img, line_width = utils.draw_text_in_image(fn_text_img, text, (margin, v_pos),
                                                                                   'white', 0)
                                text = "Class [" + str(gt_class_idx + 1) + "/" + str(n_classes) + "]: " + gt_class + " "
                                fn_text_img, line_width = utils.draw_text_in_image(fn_text_img, text,
                                                                                   (margin + line_width, v_pos), 'cyan',
                                                                                   line_width)

                                text = "Result: {}".format(fn_cat.upper())
                                fn_text_img, line_width = utils.draw_text_in_image(fn_text_img, text,
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

                                    fn_text_img, line_width = utils.draw_text_in_image(fn_text_img, text,
                                                                                       (margin, v_pos),
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
                        ovmax_, det_match_ = utils.get_max_iou_obj(frame_det_data, gt_classes,
                                                                   gt_obj["bbox"], gt_obj["mask"], enable_mask)
                        if ovmax_ > min_overlap:
                            """misclassification"""
                            assert det_match_["class"] != gt_class, \
                                "unused GT object matches with a det of current class"
                            fn_cls_sum += 1
                            fn_cats[gt_obj_id] = 'fn_cls'
                            gt_obj['cls'] = 'fn_cls'
                        else:

                            """missing detection"""
                            fn_det_sum += 1
                            fn_cats[gt_obj_id] = 'fn_det'
                            gt_obj['cls'] = 'fn_det'

                    fn_sum += n_fn_gts
                    all_considered_gt += all_class_gt

                    # print('None det {} :: {} :: n_total_gt: {} n_used_gt: {} tp_sum: {} fn_sum: {}'.format(
                    #     seq_name, file_id, n_total_gt, n_used_gt, tp_sum, fn_sum))

                    assert fn_det_sum + fn_cls_sum == fn_sum, "fn_sum mismatch"

                    if n_total_gt != n_used_gt + fn_sum:
                        print_('fn_gts:\n{}'.format(pformat(fn_gts)))
                        print_('all_class_gt:\n{}'.format(pformat(all_class_gt)))

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

                """bounding boxes and masks of all the detections in this frame"""
                bb_det = curr_det_data["bbox"]
                mask_det = curr_det_data["mask"]

                if assoc_method == 1:
                    """find the maximally overlapping GT of the same class"""
                    ovmax, gt_match = utils.get_max_iou_obj(frame_gt_data, [gt_class, ], bb_det, mask_det, enable_mask)

                    """assign curr_det_data as true positive or false positive"""
                    if ovmax >= min_overlap:
                        if not gt_match['used']:
                            # true positive
                            tp[det_idx] = 1
                            tp_sum += 1
                            gt_match['used'] = True
                            count_true_positives[gt_class] += 1
                            gt_match['cls'] = "tp"
                            curr_det_data['cls'] = "tp"
                            cls_cat = "tp"
                        else:
                            """false positive (multiple detection)"""
                            fp[det_idx] = 1
                            fp_dup[det_idx] = 1
                            fp_sum += 1
                            fp_dup_sum += 1
                            cls_cat = "fp_dup"
                            curr_det_data['cls'] = "fp_dup"
                    else:
                        """false positive"""
                        fp[det_idx] = 1
                        ovmax_, gt_match_ = utils.get_max_iou_obj(frame_gt_data, other_classes, bb_det, mask_det,
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
                                curr_det_data['cls'] = "fp_dup"
                            else:
                                """Misclassification of actual object"""
                                fp_cls[det_idx] = 1
                                fp_cls_sum += 1
                                cls_cat = "fp_cls"
                                gt_match_['used_fp'] = True
                                curr_det_data['cls'] = "fp_cls"
                        else:
                            """det does not match any GT"""
                            fp_nex[det_idx] = 1
                            fp_nex_sum += 1
                            curr_det_data['cls'] = "fp_nex"
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
                    img, resize_factor, _, _ = utils.resize_ar_tf_api(src_img, vis_w, vis_h, crop=1, return_factors=1)

                    color = 'hot_pink'
                    if cls_cat == "tp":
                        color = 'lime_green'
                    elif cls_cat == "fn_det":
                        color = 'magenta'

                    # cv2.rectangle(img, (int(bb_det[0]), int(bb_det[1])), (int(bb_det[2]), int(bb_det[3])), color, 2)
                    # if there is intersections between the det and GT
                    if show_gt and cls_cat in ("fp_nex-part", "tp"):
                        img = utils.draw_objs(img, [gt_match, ], cols='cyan', in_place=True, mask=0, thickness=2,
                                              bb_resize=resize_factor)

                        # bb_gt = gt_match["bbox"]
                        # bb_gt = [float(x) for x in gt_match["bbox"].split()]
                        # cv2.rectangle(img, (int(bb_gt[0]), int(bb_gt[1])), (int(bb_gt[2]), int(bb_gt[3])),
                        # light_blue, 2)
                    img = utils.draw_objs(img, [curr_det_data, ], cols=color, in_place=True, mask=0, thickness=2,
                                          bb_resize=resize_factor)

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
                        text_img, line_width = utils.draw_text_in_image(text_img, text, (margin, v_pos), 'white', 0)
                        text = "Class [" + str(gt_class_idx + 1) + "/" + str(n_classes) + "]: " + gt_class + " "
                        text_img, line_width = utils.draw_text_in_image(text_img, text, (margin + line_width, v_pos),
                                                                        'cyan',
                                                                        line_width)

                        text = "Result: " + cls_cat.upper()
                        if ovmax != -1:
                            text = text + " IOU {:.2f}".format(ovmax)

                        text_img, line_width = utils.draw_text_in_image(text_img, text, (margin + line_width, v_pos),
                                                                        color,
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
                        text_img, line_width = utils.draw_text_in_image(text_img, text, (margin, v_pos), 'white', 0)

                        if ovmax != -1:
                            color = 'pale_violet_red'
                            if cls_cat == "fp_nex-part":
                                text = "IoU: {0:.2f}% ".format(ovmax * 100) + "< {0:.2f}% ".format(min_overlap * 100)
                            else:
                                text = "IoU: {0:.2f}% ".format(ovmax * 100) + ">= {0:.2f}% ".format(min_overlap * 100)
                                color = 'green'
                            text_img, line_width = utils.draw_text_in_image(text_img, text,
                                                                            (margin + line_width, v_pos),
                                                                            color,
                                                                            line_width)

                if is_last_in_frame:
                    """last detection in this frame assuming that detections are ordered by frame"""

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

                        ovmax_, det_match_ = utils.get_max_iou_obj(
                            frame_det_data, other_classes,
                            gt_obj["bbox"], gt_obj["mask"], enable_mask)

                        if ovmax_ > min_overlap:
                            """misclassification"""
                            assert det_match_["class"] != gt_class, \
                                "unused GT object matches with a det of current class"
                            fn_cls_sum += 1
                            fn_cats[gt_obj_id] = 'fn_cls'
                            gt_obj['cls'] = 'fn_cls'
                        else:
                            """missing detection"""
                            fn_det_sum += 1
                            fn_cats[gt_obj_id] = 'fn_det'
                            gt_obj['cls'] = 'fn_det'

                    assert n_fn_gts == len(unused_gt), "n_unused_gt / n_fn_gts mismatch"

                    all_considered_gt += all_class_gt

                    assert fn_det_sum + fn_cls_sum == fn_sum, "fn_sum mismatch"

                    if n_total_gt != n_used_gt + fn_sum:
                        print_('fn_gts:\n{}'.format(pformat(fn_gts)))
                        print_('all_class_gt:\n{}'.format(pformat(all_class_gt)))

                        raise AssertionError(
                            f'{gt_class} : {file_id} :: '
                            f'Mismatch between n_total_gt: {n_total_gt} and n_used_gt+fn_sum: {n_used_gt + fn_sum}')

                    if n_used_gt != tp_sum:
                        raise AssertionError('{} : {} :: Mismatch between n_used_gt: {} and tp_sum: {}'.format(
                            gt_class, file_id, n_used_gt, tp_sum
                        ))

                    if enable_vis:
                        if (
                                ("cls" in save_cats)
                                and (gt_class_idx == n_classes - 1)
                        ):
                            """all classes have been processed for this frame in this sequence so 
                            classification summary visualization can be generated
                            this shows TPs in green and FP / FN in red"""

                            cat_img_vis_cls = utils.draw_and_concat(
                                src_img,
                                # all_class_dets, all_class_gt,
                                frame_det_data, frame_gt_data,
                                None, params.vis_alpha, vis_w_all,
                                vis_h_all, vert_stack, params.check_det, img_id,
                                mask=enable_mask, cls_cat_to_col=cls_cat_to_col)

                            cat_img_h, cat_img_w = cat_img_vis_cls.shape[:2]
                            cls_text_img = np.zeros((bottom_border, cat_img_w, 3), dtype=np.uint8)
                            cls_text = ' '.join(f'{cls_cat_}:{col_}' for cls_cat_, col_ in cls_cat_to_col.items())
                            cls_text_img, _ = utils.draw_text_in_image(cls_text_img, cls_text, (20, 20), 'white', 0)
                            cls_out_img = np.concatenate((cat_img_vis_cls, cls_text_img), axis=0)

                            # cls_out_img = cat_img_vis_cls

                            if show_vis:
                                cls_out_img_vis = utils.resize_ar_tf_api(cls_out_img, 1280, 720)
                                cv2.imshow('cls_out_img_vis', cls_out_img_vis)
                                # cv2.waitKey(0)

                            if save_vis:
                                cls_video_out = utils.get_video_out(video_out_dict, vis_out_fnames, "cls",
                                                                    vis_video,
                                                                    save_h, save_w, fourcc, fps)
                                if vis_video:
                                    cls_out_img = utils.resize_ar_tf_api(
                                        cls_out_img, video_w, video_h, add_border=1)

                                cls_video_out.write(cls_out_img)

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

            """postprocessing for each sequence"""
            if True:
                if fps_to_gt:
                    if gt_class_idx == 0:
                        """add GT of all classes at once"""
                        assert seq_name not in seq_name_to_csv_rows, f"duplicate seq_name found: {seq_name}"
                        seq_name_to_csv_rows[seq_name] = []
                        gt_csv_rows = []

                        fps_to_gt_iter = seq_gt_data_dict.items()
                        if show_pbar:
                            fps_to_gt_iter = tqdm(
                                fps_to_gt_iter,
                                desc="fps_to_gt: seq_gt_data_dict",
                                ncols=100, position=0, leave=True)

                        """GT is class-agnostic so should be added to the CSV only once"""
                        for _gt_file_id, _frame_gt_data in fps_to_gt_iter:

                            if not _frame_gt_data:
                                continue

                            _frame_height = _frame_gt_data[0]["height"]
                            _frame_width = _frame_gt_data[0]["width"]
                            _frame_area = _frame_height * _frame_width

                            try:
                                _ = file_id_to_img_info[_gt_file_id]
                            except KeyError:
                                rel_path = os.path.relpath(_gt_file_id, json_out_dir).rstrip('.' + os.sep).replace(
                                    os.sep,
                                    '/')
                                img_info = {
                                    'file_name': rel_path,
                                    'height': _frame_height,
                                    'width': _frame_width,
                                    'id': seq_name + '/' + os.path.basename(_gt_file_id)
                                }
                                file_id_to_img_info[_gt_file_id] = img_info
                                json_dict['images'].append(img_info)
                            else:
                                raise AssertionError(f'_file_id: {_gt_file_id} found multiple times in GT')

                            for _frame_gt_datum in _frame_gt_data:
                                gt_xmin, gt_ymin, gt_xmax, gt_ymax = _frame_gt_datum["bbox"]

                                assert _frame_gt_datum["width"] == _frame_width, "_frame_width mismatch"
                                assert _frame_gt_datum["height"] == _frame_height, "_frame_height mismatch"

                                _gt_class = _frame_gt_datum["class"]
                                _gt_target_id = _frame_gt_datum["target_id"]
                                _gt_csv_row = {
                                    "filename": _gt_file_id,
                                    "class": _gt_class,
                                    "xmin": gt_xmin,
                                    "xmax": gt_xmax,
                                    "ymin": gt_ymin,
                                    "ymax": gt_ymax,
                                    "target_id": _gt_target_id,
                                    "width": _frame_width,
                                    "height": _frame_height,
                                }

                                gt_o_width = gt_xmax - gt_xmin
                                gt_o_height = gt_ymax - gt_ymin

                                _gt_class_id = json_category_name_to_id[_gt_class]

                                _gt_json_ann = {
                                    'image_id': img_info['id'],
                                    'id': bnd_id,
                                    'area': gt_o_width * gt_o_height,
                                    'iscrowd': 0,
                                    'bbox': [gt_xmin, gt_ymin, gt_o_width, gt_o_height],
                                    'label': _gt_class,
                                    'category_id': _gt_class_id,
                                    'ignore': 0,
                                }
                                bnd_id += 1

                                if enable_mask:
                                    mask_rle = _frame_gt_datum["mask"]
                                    mask_h, mask_w = mask_rle['size']
                                    mask_counts = mask_rle['counts']
                                    _gt_csv_row.update({
                                        "mask_h": mask_h,
                                        "mask_w": mask_w,
                                        "mask_counts": mask_counts,
                                    })
                                    mask_pts, bbox, is_multi = utils.mask_rle_to_pts(mask_rle)
                                    mask_pts_flat = [float(item) for sublist in mask_pts for item in sublist]
                                    _gt_json_ann.update({
                                        'segmentation': [mask_pts_flat, ],
                                        # 'mask_pts': mask_pts,
                                    })

                                gt_csv_rows.append(_gt_csv_row)
                                json_dict['annotations'].append(_gt_json_ann)

                        n_gt_objs = len(gt_csv_rows)
                        print_(f'\nfps_to_gt::{seq_name} n_gt_objs: {n_gt_objs}')

                        n_all_gt_objs += n_gt_objs

                        gt_csv_rows.sort(key=lambda x: x['filename'])

                        seq_name_to_csv_rows[seq_name] += gt_csv_rows

                    fp_nex_whole_dets = [seq_class_det_data[i] for i in range(n_seq_class_dets) if fp_nex_whole[i]]
                    n_fp_nex_whole_dets = len(fp_nex_whole_dets)
                    n_all_fp_nex_whole_dets += n_fp_nex_whole_dets

                    print_(f'\n{gt_class}:{seq_name} :: n_fp_nex_whole_dets : {n_fp_nex_whole_dets}')

                    det_csv_rows = []

                    fp_nex_whole_dets_iter = fp_nex_whole_dets
                    if show_pbar:
                        fp_nex_whole_dets_iter = tqdm(
                            fp_nex_whole_dets_iter, desc="fps_to_gt: fp_nex_whole_dets", ncols=100, position=0,
                            leave=True)

                    for _det in fp_nex_whole_dets_iter:
                        _det_xmin, _det_ymin, _det_xmax, _det_ymax = _det["bbox"]
                        _det_class = _det["class"]

                        assert _det_class == gt_class, \
                            f"unexpected det_class {_det_class} while processing detections for {gt_class}"

                        _det_file_id = _det["file_id"]
                        _det_frame_height = _det["height"]
                        _det_frame_width = _det["width"]
                        _det_target_id = _det["target_id"]
                        try:
                            img_info = file_id_to_img_info[_det_file_id]
                        except KeyError:
                            rel_path = os.path.relpath(_det_file_id, json_out_dir).rstrip('.' + os.sep).replace(os.sep,
                                                                                                                '/')
                            img_info = {
                                'file_name': rel_path,
                                'height': _det_frame_height,
                                'width': _det_frame_width,
                                'id': seq_name + '/' + os.path.basename(_det_file_id)
                            }
                            file_id_to_img_info[_det_file_id] = img_info
                            json_dict['images'].append(img_info)

                        assert _det_frame_height == img_info['height'], "_det_frame_height mismatch"
                        assert _det_frame_width == img_info['width'], "_det_frame_width mismatch"

                        _det_csv_row = {
                            "filename": _det_file_id,
                            "class": f"FP-{_det_class}",
                            "xmin": _det_xmin,
                            "xmax": _det_xmax,
                            "ymin": _det_ymin,
                            "ymax": _det_ymax,
                            "target_id": _det_target_id,
                            "width": _det_frame_width,
                            "height": _det_frame_height,
                        }
                        _det_o_width = _det_xmax - _det_xmin
                        _det_o_height = _det_ymax - _det_ymin

                        _det_class_id = json_category_name_to_id[f"FP-{_det_class}"]

                        _det_json_ann = {
                            'image_id': img_info['id'],
                            'id': bnd_id,
                            'area': _det_o_width * _det_o_height,
                            'bbox': [_det_xmin, _det_ymin, _det_o_width, _det_o_height],
                            'label': f"FP-{_det_class}",
                            'category_id': _det_class_id,
                            'iscrowd': 0,
                            'ignore': 0,
                        }
                        bnd_id += 1

                        if enable_mask:
                            mask_rle = _det["mask"]
                            mask_h, mask_w = mask_rle['size']
                            mask_counts = mask_rle['counts']
                            _det_csv_row.update({
                                "mask_h": mask_h,
                                "mask_w": mask_w,
                                "mask_counts": mask_counts,
                            })
                            mask_pts, bbox, is_multi = utils.mask_rle_to_pts(mask_rle)
                            mask_pts_flat = [float(item) for sublist in mask_pts for item in sublist]
                            _det_json_ann.update({
                                'segmentation': [mask_pts_flat, ],
                                # 'mask_pts': mask_pts,
                            })

                        det_csv_rows.append(_det_csv_row)
                        json_dict['annotations'].append(_det_json_ann)

                    n_det_objs = len(det_csv_rows)
                    print_(f'fps_to_gt: n_det_objs: {n_det_objs}\n')
                    det_csv_rows.sort(key=lambda x: x['filename'])

                    seq_name_to_csv_rows[seq_name] += det_csv_rows

                if save_vis:
                    for cat, video_out in video_out_dict.items():
                        if video_out is not None:
                            video_out.release()

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

        """postprocessing for each class"""
        if True:
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
            # print('Sorting by confidence')
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
                print_(
                    f'\n{gt_class}: Computing recall and precision '
                    f'over {n_score_thresholds} thresholds, '
                    f'{n_imgs} images, '
                    f'{n_class_gt} GTs, '
                    f'{n_class_dets} dets'
                )

                if n_threads == 1:
                    print_('Not using multi threading')
                    _start_t = time.time()
                    _rec_prec_list = []
                    for __thresh_idx in range(n_score_thresholds):
                        __temp = utils.compute_thresh_rec_prec(
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

                    print_(f'Using {n_threads} threads')

                    _start_t = time.time()
                    with closing(ThreadPool(n_threads)) as pool:
                        _rec_prec_list = pool.map(functools.partial(
                            utils.compute_thresh_rec_prec,
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
                print_('\nTime taken: {:.4f}'.format(_end_t - _start_t))
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

                ap, mrec, mprec = utils.voc_ap(rec, prec)

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
                    plot_out_fname = utils.linux_path(plots_out_dir, gt_class + ".png")
                    print_('Saving plot to: {}'.format(plot_out_fname))
                    fig1.savefig(plot_out_fname)

                    plt.close(fig1)

                _rec_prec, _score, _txt = utils.get_intersection(rec, prec, conf_class, score_thresh,
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
                    print_('Saved {} result summary to {}'.format(gt_class, class_summary_path))

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
                out_fname_csv = utils.linux_path(misc_out_root_dir, f'{gt_class}-rec_prec.csv')
                csv_df.to_csv(out_fname_csv, columns=csv_columns_rec_prec, index=False, sep='\t')

                _class_auc_rec_prec = utils.norm_auc(_class_rec, _class_prec)

            if gt_check:
                all_class_gt = []
                _iter = tqdm(gt_data_dict) if show_pbar else gt_data_dict

                for k in _iter:

                    if k in ["counter_per_class", 'ignored']:
                        continue

                    for m in gt_data_dict[k]:
                        all_class_gt += [obj for obj in gt_data_dict[k][m] if obj['class'] == gt_class]

                n_all_considered_gt = len(all_considered_gt)
                # n_absolute_all_gt = len(absolute_all_gt)
                # n_duplicate_gt = len(duplicate_gt)
                n_all_class_gt = len(all_class_gt)

                if n_all_considered_gt != n_all_class_gt:
                    print(f'{gt_class} :: Mismatch between '
                          f'n_all_considered_gt: {n_all_considered_gt} '
                          f'and n_all_class_gt: {n_all_class_gt}, '
                          )
                    _iter = tqdm(all_class_gt) if show_pbar else all_class_gt
                    skipped_gt = [obj for obj in _iter if obj not in all_considered_gt]
                    n_skipped_gt = len(skipped_gt)
                    print(f'n_skipped_gt: {n_skipped_gt}')

                    # annoying_gt = [k for k in absolute_all_gt if k not in all_considered_gt]
                    # n_annoying_gt = len(annoying_gt)

                    # print('annoying_gt:')
                    # pprint(annoying_gt)

                    # print('duplicate_gt:')
                    # pprint(duplicate_gt)

                    # print('skipped_gt:\n{}'.format(pformat(skipped_gt)))
                    print_('gt_counter_per_class:\n{}'.format(pformat(gt_counter_per_class)))

                    raise AssertionError()

                if n_total_gt != n_class_gt:
                    print(
                        f'Mismatch between n_total_gt: {n_total_gt} and '
                        f'gt_counter_per_class[{gt_class}]: {gt_counter_per_class[gt_class]}'
                    )
                    seq_gt_data_list = []

                    for file_id in seq_gt_data_dict.keys():
                        seq_gt_data_list += seq_gt_data_dict[file_id]
                        # for k in seq_gt_data_dict[file_id]:
                        #     seq_gt_data_list.append(
                        #         dict(k, **{'file_id': file_id})
                        #     )

                    if n_total_gt > n_class_gt:
                        _iter = tqdm(all_gt_list) if show_pbar else all_gt_list
                        missing_gt = [k for k in _iter if k not in seq_gt_data_list]
                    else:
                        _iter = tqdm(seq_gt_data_list) if show_pbar else seq_gt_data_list
                        missing_gt = [k for k in _iter if k not in all_gt_list]

                    n_missing_gt = len(missing_gt)
                    print(f'n_missing_gt: {n_missing_gt}')

                    raise AssertionError()

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
            utils.binary_cls_metrics(
                class_stats,
                tp_sum_thresh_all,
                fp_sum_thresh_all,
                fp_cls_sum_thresh_all,
                score_thresholds,
                gt_classes,
                out_root_dir,
                misc_out_root_dir,
                eval_result_dict,
                verbose=params.verbose,
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
            print_('Computing combined results over {} thresholds'.format(n_score_thresholds))
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

            overall_ap_thresh, _, _ = utils.voc_ap(wm_rec_thresh[::-1], wm_prec_thresh[::-1])
            wm_diff_thresh = wm_rec_thresh - wm_prec_thresh

            itsc_idx = np.argwhere(np.diff(np.sign(wm_rec_thresh - wm_prec_thresh))).flatten()

            if not itsc_idx.size:
                # print('rec/prec: {}'.format(pformat(np.vstack((conf_class, rec, prec)).T)))
                itsc_idx = np.argmin(np.abs(wm_diff_thresh))
                if itsc_idx.size > 1:
                    itsc_idx = itsc_idx[0]
                    print_('No intersection between recall and precision found; ' \
                           'min_difference: {} at {} for confidence: {}'.format(
                        wm_diff_thresh[itsc_idx], (wm_rec_thresh[itsc_idx], wm_prec_thresh[itsc_idx]),
                        score_thresholds[itsc_idx])
                    )
            else:
                print_('intersection at {} for confidence: {} with idx: {}'.format(
                    wm_rec_thresh[itsc_idx], score_thresholds[itsc_idx], itsc_idx))

            print_('overall_ap: {}'.format(overall_ap_thresh))

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
                plot_out_fname = utils.linux_path(plots_out_dir, "overall.png")
                print_('Saving plot to: {}'.format(plot_out_fname))
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
                print_(tabulate(opt_data, opt_headers, tablefmt="fancy_grid"))

                if opt_idx != itsc_idx:
                    _idx_threshs.append(opt_idx)

            # out_text += 'rec_ratio_data\n{}\n'.format(
            #     pd.DataFrame(data=opt_data, columns=opt_headers).to_csv(sep='\t', index=False))

            print_('itsc_idx: {}'.format(itsc_idx))
            if isinstance(itsc_idx, list) and not itsc_idx:
                _score_threshold = 0
            else:
                _score_threshold = score_thresholds[itsc_idx]
                print_('_score_threshold: {}'.format(_score_threshold))

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

                    class_ap, _, _ = utils.voc_ap(class_rec[_idx:][::-1], class_prec[_idx:][::-1])
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

                overall_ap, _, _ = utils.voc_ap(wm_rec_thresh[_idx:][::-1], wm_prec_thresh[_idx:][::-1])
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
                print_(tabulate(rec_ratio_data, rec_ratio_headers, tablefmt="fancy_grid"))
                out_text += 'rec_ratio_data\n{}\n'.format(
                    pd.DataFrame(data=rec_ratio_data, columns=rec_ratio_headers).to_csv(sep='\t', index=False))

            csv_df = pd.DataFrame(
                data=np.vstack((score_thresholds,
                                wm_rec_thresh * 100,
                                wm_prec_thresh * 100)).T,
                columns=csv_columns_rec_prec)
            out_fname_csv = utils.linux_path(misc_out_root_dir, f'rec_prec.csv')
            csv_df.to_csv(out_fname_csv, columns=csv_columns_rec_prec, index=False, sep='\t')

            csv_txt = csv_df.to_csv(sep='\t', index=False)

            out_text += '\nrec_prec\n{}\n'.format(csv_txt)
            out_text += '\n'

        if write_summary:
            cmb_summary_text = '{}\n{}'.format(out_template, cmb_summary_text)
            print_(cmb_summary_text)

            print_(text_table)
            # out_file.write(text_table.get_string() + '\n')
            print_(f'saving result summary to {summary_path}')

            with open(summary_path, 'w') as out_file:
                out_file.write(cmb_summary_text)
                out_file.write(out_text)

        if fps_to_gt:
            print_(f'\nfps_to_gt: n_all_gt_objs: {n_all_gt_objs}')
            print_(f'\nn_all_fp_nex_whole_dets : {n_all_fp_nex_whole_dets}')

            for seq_name, out_csv_rows in seq_name_to_csv_rows.items():
                if not out_csv_rows:
                    continue

                csv_out_path = utils.linux_path(out_root_dir, f'{seq_name}_gt_with_fp_nex_whole.csv')
                print_(f'\nsaving fps_to_gt csv to: {csv_out_path}\n')

                # out_csv_rows.sort(key=lambda x: x['filename'])
                csv_columns = ['filename', 'class', 'xmin', 'xmax', 'ymin', 'ymax', 'width', 'height']
                if enable_mask:
                    csv_columns += ['mask_h', 'mask_w', 'mask_counts']

                df = pd.DataFrame(out_csv_rows, columns=csv_columns)
                df.to_csv(csv_out_path, index=False)

            json_out_path = utils.linux_path(out_root_dir, 'gt_with_fp_nex_whole.json')

            n_json_imgs = len(json_dict['images'])
            n_json_objs = len(json_dict['annotations'])
            print_(f'saving fps_to_gt json with {n_json_imgs} images and {n_json_objs} objects to: {json_out_path}')

            with open(json_out_path, 'w') as f:
                json_dict_str = json.dumps(json_dict, indent=4)
                f.write(json_dict_str)

        # remove the tmp_files directory
        # if delete_tmp_files:
        #     shutil.rmtree(pkl_files_path)

    if return_eval_dict:
        return eval_result_dict
    else:
        return text_table


def dummy_tqdm(iter_, *args, **kwargs):
    return iter_


def dummy_print(*argv):
    pass


def run(params: Params, det_data_dict, sweep_mode: dict, *argv):
    params = copy.deepcopy(params)

    if not params.verbose:
        print_ = dummy_print
        # tqdm = dummy_tqdm
    else:
        print_ = print

    if argv:
        sweep_params = list(vars(params.sweep).keys())
        for i, sweep_param in enumerate(sweep_params):
            if argv[i] is not None:
                setattr(params, sweep_param, argv[i])

            # param_val = getattr(params, sweep_param)

            # if isinstance(param_val, (list, tuple)):
            #     setattr(params, sweep_param, param_val[0])

    # print('gt_paths', params.gt_paths)
    print_('det_paths', params.det_paths)
    print_('img_paths', params.img_paths)
    print_('labels_path', params.labels_path)

    vid_stride = params.vid_stride  # type: int
    if not isinstance(vid_stride, int):
        vid_stride = 0

    labels_path = params.labels_path

    if params.vid_nms_thresh > 0:
        assert params.vid_det, "vid_nms can only be performed with vid_det data"

    load_samples = params.load_samples
    load_samples_root = params.load_samples_root
    load_samples_suffix = params.load_samples_suffix

    if load_samples_suffix:
        load_samples_root = f'{load_samples_root}_{load_samples_suffix}'

    if len(load_samples) == 1:
        if load_samples[0] == 1:
            load_samples = ['seq_to_samples.txt', ]
        elif load_samples[0] == 0:
            load_samples = []

    if params.labels_root:
        labels_path = utils.linux_path(params.labels_root, labels_path)

    assert labels_path, f"labels_path must be provided"
    assert os.path.isfile(labels_path), f"nonexistent labels_path: {labels_path}"

    vid_info = None

    gt_paths = params.gt_paths
    seq_path_list_file = params.img_paths
    img_paths_suffix = params.img_paths_suffix

    if img_paths_suffix:
        img_paths_suffix = '-'.join(img_paths_suffix)
        seq_path_list_file = utils.add_suffix(seq_path_list_file, img_paths_suffix)

    _det_path_list_file = params.det_paths
    if params.det_nms > 0:
        _det_path_list_file = f'{_det_path_list_file}_nms_{params.det_nms:02d}'

    if params.det_root_dir:
        _det_path_list_file = utils.linux_path(params.det_root_dir, _det_path_list_file)

    if vid_stride > 0:
        assert os.path.isdir(_det_path_list_file), "invalid det_path_list_file for vid_stride filtering"
        assert params.vid_det, "vid_stride filtering can only be performed with vid_det data"

        _det_path_list_parent = os.path.dirname(_det_path_list_file)
        _det_path_list_gparent = os.path.dirname(_det_path_list_parent)
        vid_info_path = utils.linux_path(_det_path_list_gparent, f"vid_info.json.gz")
        if not os.path.isfile(vid_info_path):
            vid_info_path = utils.linux_path(_det_path_list_parent, f"vid_info.json.gz")
            assert os.path.isfile(vid_info_path), f"nonexistent vid_info_path: {vid_info_path}"

        print_(f'loading vid_info from {vid_info_path}')

        import compress_json
        vid_info_dict = compress_json.load(vid_info_path)
        stride_to_video_ids = vid_info_dict['stride_to_video_ids']
        stride_to_filenames = vid_info_dict['stride_to_file_names']

        # seq_to_video_ids = list(map(int, stride_to_video_ids[str(vid_stride)].split(',')))
        vid_info = [stride_to_video_ids[str(vid_stride)], stride_to_filenames[str(vid_stride)]]

        # print(f'\nrestricting video ids to:\n{vid_info[0]}\n')

    img_root_dir = params.img_root_dir
    gt_root_dir = params.gt_root_dir

    img_start_id = params.img_start_id
    img_end_id = params.img_end_id

    seq_start_id = params.start_id
    seq_end_id = params.end_id
    # seq = params.seq
    # if seq >= 0:
    #     seq_start_id = seq_end_id = seq

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
    save_suffix = save_suffix.replace(':', '-')

    assert params.det_nms == 0 or params.nms_thresh == 0, "both nms_thresh and det_nms cannot be nonzero"

    sweep_suffixes = []

    if vid_stride > 0:
        sweep_suffixes.append(f'strd_{vid_stride:02d}')

    if params.det_nms > 0 or sweep_mode['det_nms']:
        sweep_suffixes.append(f'nms_{params.det_nms:02d}')

    if params.nms_thresh > 0 or sweep_mode['nms_thresh']:
        sweep_suffixes.append(f'nms_{params.nms_thresh:02d}')

    if params.vid_nms_thresh > 0 or sweep_mode['vid_nms_thresh']:
        sweep_suffixes.append(f'vnms_{params.vid_nms_thresh:02d}')

    sweep_suffix = '-'.join(sweep_suffixes)

    is_sweep = any(sweep_mode.values())

    out_dir_name = None

    if save_suffix:
        if params.iw:
            save_suffix = f'{save_suffix}-iw'
        if params.class_agnostic:
            save_suffix = f'{save_suffix}-agn'

        # print(f"save_suffix: {save_suffix}")

        out_dir_name = f'{save_suffix}'

        batch_name = params.batch_name
        if batch_name:
            out_dir_name = utils.linux_path(out_dir_name, batch_name)

        if sweep_suffix:
            if params.force_sweep or is_sweep:
                out_dir_name = utils.linux_path(out_dir_name, sweep_suffix)
            else:
                out_dir_name = f'{out_dir_name}-{sweep_suffix}'
    else:
        print_('Using automatically generated suffix')
        params.auto_suffix = 1

    seq_to_samples = None

    if load_samples:
        seq_path_list, seq_to_samples = utils.load_samples_from_txt(load_samples, None, load_samples_root,
                                                                    verbose=params.verbose)
    else:
        seq_path_list_file_temp = seq_path_list_file
        if params.img_root_dir:
            seq_path_list_file_temp = utils.linux_path(params.img_root_dir, seq_path_list_file_temp)

        if os.path.isdir(seq_path_list_file_temp):
            if params.all_img_dirs:
                seq_path_list = [utils.linux_path(seq_path_list_file_temp, name) for name in
                                 os.listdir(seq_path_list_file_temp)
                                 if
                                 os.path.isdir(utils.linux_path(seq_path_list_file_temp, name))]
                seq_path_list.sort(key=utils.sortKey)
            else:
                seq_path_list = [seq_path_list_file_temp, ]
            seq_path_list_file_temp = os.path.abspath(seq_path_list_file_temp)

            if params.auto_suffix:
                db_name = os.path.basename(seq_path_list_file_temp)
                db_root_name = os.path.basename(os.path.dirname(seq_path_list_file_temp))
                out_dir_name = f'{out_dir_name}_{db_root_name}_{db_name}'

        elif os.path.isfile(seq_path_list_file):
            seq_path_list = utils.file_lines_to_list(seq_path_list_file)
            if img_root_dir:
                seq_path_list = [utils.linux_path(img_root_dir, name) for name in seq_path_list]
            if params.auto_suffix:
                db_name = os.path.splitext(os.path.basename(seq_path_list_file))[0]
                out_dir_name = f'{out_dir_name}_{db_name}'

        else:
            raise AssertionError('invalid seq_path_list_file: {}'.format(seq_path_list_file))

    # print_(f'seq_path_list:\n{utils.to_str(seq_path_list)}\n')

    if params.gt_csv_suffix:
        params.gt_csv_name = utils.add_suffix(params.gt_csv_name, params.gt_csv_suffix)

    if not gt_paths:
        gt_path_list = [utils.linux_path(img_path, params.gt_csv_name) for img_path in seq_path_list]
    elif gt_paths.endswith('.csv'):
        if not os.path.isfile(gt_paths) and gt_root_dir:
            gt_paths = utils.linux_path(gt_root_dir, gt_paths)

        assert os.path.isfile(gt_paths), f"invalid gt_paths csv: {gt_paths}"

        gt_path_list = [gt_paths, ]
    elif os.path.isfile(gt_paths):
        gt_path_list = utils.file_lines_to_list(gt_paths)
        # gt_path_list = [utils.linux_path(name, 'annotations.csv') for name in gt_path_list]
        if gt_root_dir:
            gt_path_list = [utils.linux_path(gt_root_dir, name) for name in gt_path_list]
    else:
        if not os.path.isdir(gt_paths) and gt_root_dir:
            gt_paths = utils.linux_path(gt_root_dir, gt_paths)

        assert os.path.isdir(gt_paths), f'invalid gt_paths: {gt_paths}'

        gt_path_list = [utils.linux_path(gt_paths, name) for name in os.listdir(gt_paths)
                        if os.path.isdir(utils.linux_path(gt_paths, name))]
        gt_path_list.sort(key=utils.sortKey)

    # print_(f'gt_path_list:\n{utils.to_str(gt_path_list)}\n')

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

    n_seq = len(seq_path_list)
    n_gts = len(gt_path_list)

    assert n_seq == n_gts, f"mismatch between len of seq_path_list: {n_seq} and gt_path_list: {n_gts}"

    if seq_end_id < seq_start_id:
        seq_end_id = n_seq - 1

    if params.auto_suffix:
        if params.enable_mask:
            out_dir_name = f'{out_dir_name}_mask'

        out_dir_name = f'{out_dir_name}_assoc_{params.assoc_method}'
        out_dir_name = f'{out_dir_name}_{seq_start_id}_{seq_end_id}'

        if params.iw:
            out_dir_name = f'{out_dir_name}-iw'

    if params.out_root_suffix:
        out_root_suffix = '_'.join(params.out_root_suffix)
        out_dir_name = utils.linux_path(out_root_suffix, out_dir_name)

    gt_path_list = gt_path_list[seq_start_id:seq_end_id + 1]

    class_info = open(labels_path, 'r').read().splitlines()
    gt_classes, gt_class_cols = zip(*[k.split('\t') for k in class_info if k])

    if params.class_agnostic:
        gt_classes = ['agnostic', ]
        gt_class_cols = [gt_class_cols[0], ]

    class_name_to_col = {
        x.strip(): col
        for x, col in zip(gt_classes, gt_class_cols)
    }

    out_root_dir = out_root_dir_ = utils.linux_path(params.out_root_dir, f'{out_dir_name}')
    # return out_root_dir

    for _detection_names in all_detection_names:

        _seq_path_list = seq_path_list[seq_start_id:seq_end_id + 1]

        det_path_list_file = _det_path_list_file

        if not det_path_list_file:
            det_path_list = [[utils.linux_path(img_path, detection_name) for detection_name in _detection_names]
                             for img_path in seq_path_list]
        elif det_path_list_file.endswith('.csv'):
            det_path_list = [det_path_list_file, ]
        elif os.path.isdir(det_path_list_file):
            det_path_list = [utils.linux_path(det_path_list_file, name) for name in os.listdir(det_path_list_file) if
                             os.path.isfile(utils.linux_path(det_path_list_file, name)) and name.endswith('.csv')]
            det_path_list.sort(key=utils.sortKey)
        elif os.path.isfile(det_path_list_file):
            det_path_list = utils.file_lines_to_list(det_path_list_file)
            # det_path_list = [name + '.csv' for name in det_path_list]
        else:
            raise AssertionError('invalid det_path_list_file: {}'.format(det_path_list_file))

        n_dets = len(det_path_list)
        if n_dets > 0 and params.combine_dets:
            det_path_list = [det_path_list, ]
            n_dets = 1

        if n_seq != n_dets:
            if len(_seq_path_list) == n_dets:
                print_(f'n_dets = curtailed seq_path_list = {n_dets} so assuming it is already curtailed')
            elif 0 < n_dets < n_seq and params.allow_missing_dets and not params.combine_dets:
                det_seq_names = [os.path.splitext(os.path.basename(k))[0]
                                 for k in det_path_list]
                assert len(set(det_seq_names)) == n_dets, "missing dets can only be handled when det file names " \
                                                          "are sequence names"

                img_seq_names = [os.path.basename(k) for k in seq_path_list]
                missing_det_seq_names = list(set(img_seq_names) - set(det_seq_names))
                print_(
                    f"\n\ncreating empty det files for {len(missing_det_seq_names)} missing sequences:\n"
                    f"{missing_det_seq_names}\n\n")
                for missing_det_seq_name in missing_det_seq_names:
                    missing_det_path = det_path_list[0].replace(det_seq_names[0], missing_det_seq_name)
                    assert not os.path.exists(missing_det_path), \
                        f"missing_det_path exists: {missing_det_path}"
                    open(missing_det_path, 'w').close()
                    det_path_list.append(missing_det_path)
                det_path_list.sort(key=utils.sortKey)
                n_dets = len(det_path_list)
            else:
                raise IOError(f"mismatch between n_seq: {n_seq} and n_dets: {n_dets}")
        else:
            det_path_list = det_path_list[seq_start_id:seq_end_id + 1]

        # print_(f'det_path_list:\n{utils.to_str(det_path_list)}\n')

        # time_stamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")

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

        if params.check_seq_name:
            img_seq_names = [os.path.basename(x) for x in _seq_path_list]
            img_temp = sorted(enumerate(img_seq_names), key=lambda x: x[1])
            img_seq_names_idx, img_seq_names = zip(*img_temp)

            gt_seq_names = [os.path.basename(_gt_path) for _gt_path in gt_path_list]
            if len(gt_seq_names) > 0 and gt_seq_names[0].endswith('.csv'):
                gt_seq_names = [os.path.basename(os.path.dirname(_gt_path)) for _gt_path in gt_path_list]

            det_seq_names = [os.path.splitext(os.path.basename(x))[0] for x in det_path_list]

            gt_seq_names_idx, gt_seq_names = zip(*sorted(enumerate(gt_seq_names), key=lambda x: x[1]))
            det_seq_names_idx, det_seq_names = zip(*sorted(enumerate(det_seq_names), key=lambda x: x[1]))

            assert img_seq_names == gt_seq_names, "mismatch between img_seq_names and gt_seq_names"
            assert img_seq_names == det_seq_names, "mismatch between img_seq_names and det_seq_names"

            _seq_path_list = [_seq_path_list[i] for i in img_seq_names_idx]
            gt_path_list = [gt_path_list[i] for i in gt_seq_names_idx]
            det_path_list = [det_path_list[i] for i in det_seq_names_idx]

        out_root_dir = out_root_dir_
        seq = params.seq
        if isinstance(seq, int) and seq >= 0:
            assert seq < len(_seq_path_list), f"invalid seq: {seq}"
            img_seq_names = [os.path.basename(k) for k in _seq_path_list]
            out_root_dir = utils.linux_path(out_root_dir, f'{img_seq_names[seq]}')
            params.gt_pkl_suffix.append(img_seq_names[seq])

        print_(f'\nrunning eval: {out_root_dir}\n')

        os.makedirs(out_root_dir, exist_ok=True)

        if params.iw:
            assert not params.batch_nms, "batch_nms is not supported with iw"

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

                img_out_root_dir = utils.linux_path(out_root_dir, f'img_{img_id}')

                if not params.gt_pkl:
                    params.gt_pkl_dir = out_root_dir

                img_eval_dict = evaluate(
                    params=params,
                    seq_paths=_seq_path_list,
                    gt_path_list=gt_path_list,
                    all_seq_det_paths=det_path_list,
                    gt_classes=gt_classes,
                    out_root_dir=img_out_root_dir,
                    img_start_id=img_id,
                    img_end_id=img_id,
                    seq_to_samples=seq_to_samples,
                    class_name_to_col=class_name_to_col,
                    fps_to_gt=params.fps_to_gt,
                    show_pbar=params.show_pbar,
                    vid_info=vid_info,
                    raw_det_data_dict=det_data_dict,
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

                img_eval_dict_path = utils.linux_path(img_out_root_dir, 'eval_dict.json')
                open(img_eval_dict_path, 'w').write(json.dumps(img_eval_dict, indent=4))

            eval_dicts_path = utils.linux_path(out_root_dir, 'eval_dict.json')
            print_(f'saving eval_dict to {eval_dicts_path}')
            with open(eval_dicts_path, 'w') as f:
                output_json_data = json.dumps(eval_dicts, indent=4)
                f.write(output_json_data)

            for metric in class_auc_metrics + det_metrics:
                csv_columns_metric = ['img_id', metric]
                metric_list = metrics_dict[metric]
                metric_arr = np.stack((img_end_ids, metric_list), axis=1)

                utils.arr_to_csv(metric_arr, csv_columns_metric, out_root_dir, f'{metric}-iw.csv')

            for metric in roc_auc_metrics + max_tp_metrics:
                metric_list = metrics_dict[metric]
                metric_df = pd.concat(metric_list, axis=0)

                out_fname_csv = utils.linux_path(out_root_dir, f'{metric}-iw.csv')
                metric_df.to_csv(out_fname_csv, index=False, sep='\t')
        else:
            ret = evaluate(
                params=params,
                seq_paths=_seq_path_list,
                gt_path_list=gt_path_list,
                all_seq_det_paths=det_path_list,
                gt_classes=gt_classes,
                out_root_dir=out_root_dir,
                class_name_to_col=class_name_to_col,
                img_start_id=img_start_id,
                img_end_id=img_end_id,
                seq_to_samples=seq_to_samples,
                fps_to_gt=params.fps_to_gt,
                show_pbar=params.show_pbar,
                vid_info=vid_info,
                raw_det_data_dict=det_data_dict,
            )
            if params.batch_nms:
                return ret

            eval_dict = ret
            for gt_class in gt_classes:
                eval_ = eval_dict[gt_class]
                for metric in max_tp_metrics:
                    try:
                        del eval_[metric]
                    except KeyError:
                        pass

            for metric in roc_auc_metrics:
                try:
                    del eval_dict[metric]
                except KeyError:
                    pass

            eval_dict_path = utils.linux_path(out_root_dir, 'eval_dict.json')
            print_(f'saving eval_dict to {eval_dict_path}')
            with open(eval_dict_path, 'w') as f:
                output_json_data = json.dumps(eval_dict, indent=4)
                f.write(output_json_data)
    return out_root_dir


def sweep(params: Params):
    sweep_params = list(vars(params.sweep).keys())

    params.sweep.nms_thresh = [int(k) for k in params.sweep.nms_thresh]
    params.sweep.vid_nms_thresh = [int(k) for k in params.sweep.vid_nms_thresh]

    params_ = copy.deepcopy(params)
    det_data_dict = None

    if params_.batch_nms and (params_.sweep.nms_thresh or params_.sweep.vid_nms_thresh):
        if not params.load_det:
            # det_data_dict = {}

            """batch nms"""
            params_.nms_thresh = 0
            params_.vid_nms_thresh = 0
            params_.load_det = False
            # params_.load_gt = False
            params_.save_det_pkl = True
            params_.save_gt_pkl = True
            sweep_mode = {sweep_param: False for sweep_param in sweep_params}
            params_.force_sweep = 0
            params_.det_nms = 0

            class_agnostic = params.class_agnostic

            if class_agnostic != 1:
                print('performing batch NMS in multi-class mode')
                params_.class_agnostic = 0
                run(params_, None, sweep_mode)
                # det_data_dict['mc'] = det_data_dict_

            if class_agnostic:
                print('performing batch NMS in class agnostic mode')
                params_.class_agnostic = 1
                run(params_, None, sweep_mode)
                # det_data_dict['agn'] = det_data_dict_

        params_ = copy.deepcopy(params)
        params_.batch_nms = False
        params_.load_det = True
        # params_.del_det_pkl = True
        params_.load_gt = True
        # params_.nms_thresh_ = params_.sweep.nms_thresh
        # params_.vid_nms_thresh_ = params_.sweep.vid_nms_thresh
        # params_.sweep.nms_thresh = []
        # params_.sweep.vid_nms_thresh = []

    if params_.seq_wise:
        sweep_params.append('seq')
        if not params_.seq:
            assert params.start_id >= 0 and params.end_id >= 0, \
                "both start_id and end_id must be specified for auto seq_wise mode"
            assert params.end_id >= params.start_id and params.end_id >= 0, "end_id must be >= start_id"

            seq_ids = list(range(params.start_id, params.end_id + 1))
            params_.sweep.seq = list(range(len(seq_ids)))

    if params_.vid_stride:
        sweep_params.append('vid_stride')
        setattr(params_.sweep, 'vid_stride', params_.vid_stride)

    if params_.class_agnostic == 2:
        sweep_params.append('class_agnostic')
        setattr(params_.sweep, 'class_agnostic', [0, 1])
        # params_.class_agnostic = [0, 1]

    sweep_vals = []
    sweep_mode = {}
    for i, sweep_param in enumerate(sweep_params):
        param_val = getattr(params_.sweep, sweep_param)

        is_sweep = len(param_val) > 1

        sweep_mode[sweep_param] = is_sweep

        sweep_vals.append(param_val)

        if is_sweep:
            print('testing over {} {}: {}'.format(len(param_val), sweep_param, param_val))

    import itertools

    sweep_val_combos = list(itertools.product(*sweep_vals))

    n_sweep_val_combos = len(sweep_val_combos)
    n_proc = params_.n_proc
    # n_proc = min(n_proc, n_sweep_val_combos)
    print(f'testing over {n_sweep_val_combos} param combos')

    func = functools.partial(run, params_, det_data_dict, sweep_mode)

    if n_proc > 1:
        params_.show_pbar = False

        print(f'running in parallel over {n_proc} processes')
        # pool = multiprocessing.Pool(n_proc)
        pool = ThreadPool(n_proc)

        out_root_dirs = pool.starmap(func, sweep_val_combos)
        print()
    else:
        out_root_dirs = []
        for sweep_val_combo in tqdm(sweep_val_combos, position=0, leave=True):
            out_root_dir = func(*sweep_val_combo)
            out_root_dirs.append(out_root_dir)

    if params.save_vis or params.show_vis:
        return

    out_zip_paths = []
    if params.class_agnostic == 2:
        agn_root_dirs = [out_root_dir for out_root_dir in out_root_dirs if '-agn/' in out_root_dir]
        agn_zip_path = utils.zip_dirs(agn_root_dirs)

        out_zip_paths.append(agn_zip_path)

        out_root_dirs = [out_root_dir for out_root_dir in out_root_dirs if '-agn/' not in out_root_dir]

    out_zip_path = utils.zip_dirs(out_root_dirs)
    out_zip_paths.append(out_zip_path)

    if params.concat and out_zip_paths is not None:
        concat_params = concat_metrics.Params()
        concat_params.list_dir = ''
        concat_params.list_ext = ''
        concat_params.list_from_cb = 0
        concat_params.list_path_id = 0
        concat_params.auc_mode = 3

        for out_zip_path in out_zip_paths:
            out_zip_dir = os.path.dirname(out_zip_path)
            out_zip_name = os.path.splitext(os.path.basename(out_zip_path))[0]
            concat_params.out_dir = out_zip_dir
            concat_params.out_name = f'{out_zip_name}'

            # time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
            # concat_params.out_name = f'{time_stamp}_{concat_params.out_name}'
            concat_params.list_path = out_zip_path

            """CSV metrics"""
            concat_params.class_name = ''
            concat_params.csv_mode = 1
            concat_params.csv_metrics = ['rec_prec', ]
            concat_params.cfg = 'rec_prec'
            concat_metrics.main(concat_params)

            """JSON metrics"""
            concat_params.class_name = 'overall'
            concat_params.csv_mode = 0
            concat_params.to_clipboard = 0

            concat_params.json_metrics = ['AP', ]
            concat_params.json_metric_names = ['ap', ]
            concat_params.cfg = 'ap'
            concat_metrics.main(concat_params)

            concat_params.json_metrics = ['R=P', ]
            concat_params.json_metric_names = ['mrp', ]
            concat_params.cfg = 'mrp'
            concat_metrics.main(concat_params)


def main():
    params = Params()
    paramparse.process(params)

    wc = '__var__'

    if params.verbose != 2:
        params.verbose = 0
        params.show_pbar = 0

    det_paths = params.det_paths

    if wc not in det_paths:
        sweep(params)
        return

    params.n_threads = 1
    params.n_proc = 1

    wc_start_idx = det_paths.find(wc)
    wc_end_idx = wc_start_idx + len(wc)
    pre_wc = det_paths[:wc_start_idx]
    post_wc = det_paths[wc_end_idx:]
    rep1, rep2 = (pre_wc, post_wc) if len(pre_wc) > len(post_wc) else (post_wc, pre_wc)

    if params.det_root_dir:
        det_paths = utils.linux_path(params.det_root_dir, det_paths)

    det_paths_no_wc = det_paths
    """find the nearest ancestor path without wild card"""
    while wc in det_paths_no_wc:
        det_paths_no_wc = os.path.dirname(det_paths_no_wc)
    assert os.path.isdir(det_paths_no_wc), f"non-existent det_paths_no_wc: {det_paths_no_wc}"

    if params.ckpt_iter:
        det_paths = det_paths.replace(wc, params.ckpt_iter)
        multi_ckpt_mode = False
    else:
        det_paths = det_paths.replace(wc, '*')
        multi_ckpt_mode = True

    proc_det_paths = []
    out_zip_paths = None

    eval_flag_id = '__eval'
    if params.save_suffix:
        eval_flag_id = f'{eval_flag_id}-{params.save_suffix}'

    print(f'eval_flag_id: {eval_flag_id}')

    while True:
        all_matching_paths = glob.glob(det_paths)
        # all_matching_dirs = [os.path.dirname(k) for k in all_matching_paths]

        new_matching_paths = all_matching_paths
        if not params.ignore_inference_flag:
            new_matching_paths = [_path for _path in new_matching_paths
                                  if os.path.isfile(utils.linux_path(_path, '__inference'))]
        if multi_ckpt_mode and not params.ignore_eval_flag:
            new_matching_paths = [_path for _path in new_matching_paths
                                  if not os.path.isfile(utils.linux_path(_path, eval_flag_id))]

        if params.det_root_dir:
            new_matching_paths = [utils.linux_path(os.path.relpath(k, params.det_root_dir)) for k in new_matching_paths]

        new_det_paths = [k for k in new_matching_paths if k not in proc_det_paths]
        new_det_paths.sort(reverse=True, key=lambda x: int(x.replace(rep1, '').replace(rep2, '')))

        if not new_det_paths:
            # print('no new_det_paths found')
            # print(f'all_matching_paths:\n{utils.list_to_str(all_matching_paths)}\n')
            # print(f'new_matching_paths:\n{utils.list_to_str(new_matching_paths)}\n')

            if not multi_ckpt_mode:
                raise AssertionError(f'invalid det_paths: {det_paths} for ckpt_iter: {params.ckpt_iter}')

            if not utils.sleep_with_pbar(params.sleep):
                break
            continue

        det_paths_ = new_det_paths.pop()
        params_ = copy.deepcopy(params)

        match_substr = det_paths_.replace(rep1, '').replace(rep2, '')
        print(f'evaluating {det_paths_}')
        params_.det_paths = det_paths_
        params_.batch_name = f'ckpt-{match_substr}'

        if params.debug:
            sweep(params_)
        else:
            p = utils.Process(target=sweep, args=(params_,))
            p.start()
            p.join()

            if p.is_alive():
                p.terminate()

            proc_det_paths.append(det_paths_)

            if p.exception:
                print(f'\n\nckpt failed to run: {match_substr}\n\n')
                if params.ignore_exceptions:
                    continue
                break

        proc_det_paths.append(det_paths_)
        from datetime import datetime
        time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
        print(f'finished eval at {time_stamp}')
        flag_path = utils.linux_path(det_paths_, eval_flag_id)
        if params.det_root_dir:
            flag_path = utils.linux_path(params.det_root_dir, flag_path)
        with open(flag_path, 'w') as f:
            f.write(time_stamp + '\n')

        if not multi_ckpt_mode:
            break


if __name__ == '__main__':
    main()
