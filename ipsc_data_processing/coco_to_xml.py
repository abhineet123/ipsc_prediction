"""
Script for converting CVAT annotations to Yolo bounding boxes. Based on
https://github.com/newfarms/2d-rock-detection/blob/development/src/preprocessing/cvat_to_csv.py

Currently assumes there is only one class, rock

Requirements:

    pip install lxml tqdm
"""

import os
import glob
import sys
import copy
import paramparse
import shutil

import json
import cv2

import numpy as np
import pandas as pd

import itertools
from collections import OrderedDict

from tqdm import tqdm
from datetime import datetime

sys.path.append('..')

from eval_utils import get_mask_rle_iou, get_iou, contour_pts_from_mask, mask_pts_to_img, \
    col_bgr, resize_ar, add_suffix, drawBox, linux_path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask_util

from terminaltables import AsciiTable


class Params:
    def __init__(self):
        self.cfg = ()
        # self.gt_root_dir = ''
        self.root_dir = ''
        self.gt_json = ''
        self.json = ''
        self.ytvis = 0
        self.eval = 0
        self.metrics = ['bbox', 'segm']
        self.classwise = 1
        self.move_images = False
        self.alpha = 0.75
        self.conf_thresh = 0
        self.incremental = 0
        self.load_coco = 0
        self.save_coco = 0
        self.min_area = 0
        self.fix_category_id = 0
        self.allow_missing_images = 0
        self.labels_name = ''
        self.name_from_title = True
        self.img_dir_name = 'images'
        self.save_csv = 1
        self.save = 0
        self.save_flat = 1
        self.show = 0
        self.csv_suffix = ''
        self.img_root_dir = ''
        self.class_names_path = 'lists/classes///predefined_classes_orig.txt'
        self.output = ''
        self.img_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']

        self.nms_thresh = []
        self.n_proc = 1
        self.enable_mask = 1
        """set to 1 to handle cases of prediction json having only image name for image ID while GT 
        has seq name and image name"""
        self.base_id_in_preds = 0

        self._sweep_params = [
            'nms_thresh',
        ]

def ytvis_to_coco(ytvis_gt, ytvis_preds,
                  coco_gt_json_path, coco_json_path,
                  fix_category_id, incremental, load_coco, save_coco):
    coco_gt = coco_preds = None

    if load_coco and os.path.exists(coco_gt_json_path):
        print(f'loading coco format ytvis GT from {coco_gt_json_path}')
        with open(coco_gt_json_path, 'r') as fid:
            coco_gt = json.load(fid)

    if load_coco and os.path.exists(coco_json_path):
        print(f'loading coco format ytvis predictions from {coco_json_path}')
        with open(coco_json_path, 'r') as fid:
            coco_preds = json.load(fid)

    if coco_gt is not None and coco_preds is not None:
        return coco_gt, coco_preds

    images_list = []
    videos = ytvis_gt['videos']
    categories = ytvis_gt['categories']

    category_id_to_label = {
        category['id']: category['name']
        for category in categories
    }
    print(f'getting ytvis video info')
    video_id_to_file_names = {
        video['id']: video['file_names'] for video in videos
    }

    if coco_gt is None:
        print(f'converting ytvis format gt to coco')
        if incremental:
            print(f'using incremental mode')

        for video in tqdm(videos, desc='images'):
            height, width = video['height'], video['width']
            file_names = video['file_names']
            if incremental:
                file_names = file_names[-1:]

            for file_name in file_names:
                image_id = os.path.splitext(file_name)[0]
                image_dict = dict(
                    height=height,
                    width=width,
                    file_name=file_name,
                    id=image_id,
                )
                images_list.append(image_dict)

        gt_vid_to_img = {}
        gt_img_to_vid = {}

        annotations = ytvis_gt['annotations']
        annotations_list = []
        ann_idx = 1
        for annotation in tqdm(annotations, desc='annotations'):
            video_id = annotation['video_id']

            file_names = video_id_to_file_names[video_id]
            segmentations = annotation['segmentations']
            bboxes = annotation['bboxes']
            areas = annotation['areas']

            n_file_names = len(file_names)
            n_segmentations = len(segmentations)
            n_bboxes = len(bboxes)
            n_areas = len(areas)

            assert n_segmentations == n_file_names, "mismatch between n_segmentations and n_file_names"
            assert n_bboxes == n_file_names, "mismatch between n_bboxes and n_file_names"
            assert n_areas == n_file_names, "mismatch between n_areas and n_file_names"

            if incremental:
                file_names = file_names[-1:]
                segmentations = segmentations[-1:]
                bboxes = bboxes[-1:]
                areas = areas[-1:]

                n_file_names = 1
                n_segmentations = 1
                n_bboxes = 1
                n_areas = 1

            # id = annotation['id']
            iscrowd = annotation['iscrowd']
            category_id = annotation['category_id']
            label = category_id_to_label[category_id]

            # h, w = annotation['height'], annotation['width']

            for obj_id in range(n_segmentations):
                segmentation = segmentations[obj_id]
                bbox = bboxes[obj_id]
                area = areas[obj_id]
                file_name = file_names[obj_id]

                image_id = os.path.splitext(file_name)[0]

                if bbox is None:
                    assert segmentation is None, "bbox is None but segmentation is not"
                    continue

                assert segmentation is not None, "segmentation is None but bbox is not"

                if incremental:
                    """each image should be in only one video"""
                    try:
                        video_id_ = gt_img_to_vid[image_id]
                    except KeyError:
                        gt_img_to_vid[image_id] = video_id
                    else:
                        assert video_id_ == video_id, f"mismatch in gt video_ids for file {image_id}"
                    """each video should have only one image"""
                    try:
                        image_id_ = gt_vid_to_img[video_id]
                    except KeyError:
                        gt_vid_to_img[video_id] = image_id
                    else:
                        assert image_id_ == image_id, f"mismatch in gt image_ids for video {video_id}"

                # ann_segm = segmentation
                # if type(ann_segm) == list:
                #     if len(ann_segm) == 1:
                #         ann_segm = ann_segm[0]
                #
                #     n_pts = int(len(ann_segm) / 2)
                #     contour_pts = np.array(ann_segm).reshape((n_pts, 2))
                #     contour_pts2 = np.asarray([contour_pts, ], dtype=np.int32)
                #
                #     ann_mask = np.zeros((h, w), dtype=np.uint8)
                #     ann_mask = cv2.fillPoly(ann_mask, contour_pts2, 255)
                #
                #     ann_mask_binary = ann_mask > 0
                #
                #     rle = mask_util.encode(ann_mask_binary)
                #     # rle[0]['counts'] = rle[0]['counts'].decode('')
                # else:
                #     if type(ann_segm['counts']) == list:
                #         rle = mask_util.frPyObjects([ann_segm], h, w)
                #     else:
                #         rle = [ann_segm, ]
                #
                # counts = rle[0]["counts"]
                # if isinstance(counts, bytes):
                #     rle[0]["counts"] = counts.decode("utf-8")

                annotations_dict = dict(
                    area=area,
                    iscrowd=iscrowd,
                    bbox=bbox,
                    label=label,
                    category_id=category_id,
                    ignore=0,
                    segmentation=segmentation,
                    image_id=image_id,
                    id=ann_idx,
                )
                ann_idx += 1
                annotations_list.append(annotations_dict)
        coco_gt = dict(
            images=images_list,
            type='instances',
            annotations=annotations_list,
            categories=categories,
        )

        if save_coco:
            print(f'saving coco format GT to {coco_gt_json_path}')
            with open(coco_gt_json_path, 'w') as f:
                output_json_data = json.dumps(coco_gt, indent=4)
                f.write(output_json_data)

    if coco_preds is None:
        print(f'converting ytvis format predictions to coco')

        if incremental:
            print(f'using incremental mode')

        n_total_objs = 0
        n_valid_objs = 0
        coco_preds = []
        # proc_file_names = []
        pbar = tqdm(ytvis_preds, ncols=100)

        det_vid_to_img = {}
        det_img_to_vid = {}

        log_dir ='log/coco_to_xml'

        os.makedirs(log_dir, exist_ok=1)

        for pred in pbar:
            video_id = pred['video_id']

            file_names = video_id_to_file_names[video_id]
            segmentations = pred['segmentations']
            n_file_names = len(file_names)
            n_segmentations = len(segmentations)

            assert n_segmentations == n_file_names, "mismatch between n_segmentations and n_file_names"

            try:
                bboxes = pred['bboxes']
            except KeyError:
                bboxes = None
            else:
                n_bboxes = len(bboxes)
                assert n_bboxes == n_file_names, "mismatch between n_bboxes and n_file_names"

            if incremental:
                file_names = file_names[-1:]
                segmentations = segmentations[-1:]
                n_file_names = 1
                n_segmentations = 1

                if bboxes is not None:
                    bboxes = bboxes[-1:]
                    n_bboxes = 1

            # assert not any(file_name in proc_file_names for file_name in file_names), "duplicate file_names found"
            # proc_file_names += file_names

            score = pred['score']
            category_id = pred['category_id']

            if fix_category_id:
                category_id += 1

            assert category_id in category_id_to_label.keys(), f"category_id not found in gt: {category_id}"

            for idx, segmentation in enumerate(segmentations):
                if segmentation is None:
                    continue
                n_total_objs += 1

                if segmentation['counts'] == 'PPYo1':
                    continue

                mask_orig = mask_util.decode(segmentation).squeeze()
                mask_orig = np.ascontiguousarray(mask_orig, dtype=np.uint8)

                mask_h, mask_w = mask_orig.shape[:2]

                # mask_y, mask_x = np.nonzero(mask_orig)

                # pred_mask = mask_util.decode([segmentation, ]).squeeze()
                # pred_mask_pts = contour_pts_from_mask(pred_mask.copy())
                # pred_mask_pts = list(pred_mask_pts)

                invalid_mask = 0
                xmin = None

                mask_pts, mask_bb, is_multi = contour_pts_from_mask(mask_orig)

                if len(mask_pts) < 4:
                    invalid_mask = 1
                    msg = 'annoying invalid mask'

                else:
                    # mask_pts = np.asarray(mask_pts).squeeze()
                    # mask_x, mask_y = mask_pts[:, 0], mask_pts[:, 1]
                    # xmin2, xmax2 = np.amin(mask_x), np.amax(mask_x)
                    # ymin2, ymax2 = np.amin(mask_y), np.amax(mask_y)
                    #
                    # w2, h2 = xmax2 - xmin2 + 1, ymax2 - ymin2 + 1
                    #
                    # mask_bb2 = (xmin2, ymin2, w2, h2)
                    #
                    # assert mask_bb == mask_bb2, "bb mismatch"

                    xmin, ymin, w, h = mask_bb

                    xmax, ymax = xmin + w, ymin + h
                    mask_bb_area = w * h
                    if mask_bb_area < 4:
                        invalid_mask = 1
                        msg = f'annoying invalid mask with tiny area {mask_bb_area}'

                if invalid_mask:
                    time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
                    mask_rgb = np.zeros((mask_h, mask_w, 3), dtype=np.uint8)
                    mask_y, mask_x = np.nonzero(mask_orig)
                    mask_rgb[mask_y, mask_x, :] = 255
                    if xmin is not None:
                        cv2.rectangle(
                            mask_rgb, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 1)
                    cv2.imwrite(f'{log_dir}/{msg} {time_stamp}.png', mask_rgb)
                    continue

                # bbox = mask_util.toBbox([segmentation, ])
                # xmin, ymin, w, h = np.squeeze(bbox)

                # bbox = list(np.squeeze(bbox).astype(np.float32))

                # xmax, ymax = np.amax(pred_mask_pts, axis=0)
                # xmin, ymin = np.amin(pred_mask_pts, axis=0)
                # w, h = float(xmax - xmin), float(ymax - ymin)
                bbox = [float(xmin), float(ymin), float(w), float(h)]

                file_name = file_names[idx]
                image_id = os.path.splitext(file_name)[0]

                if incremental:
                    """each image should be in only one video"""
                    try:
                        video_id_ = det_img_to_vid[image_id]
                    except KeyError:
                        det_img_to_vid[image_id] = video_id
                    else:
                        assert video_id_ == video_id, f"mismatch in det video_ids for image {image_id}"

                    """each video should have only one image"""
                    try:
                        image_id_ = det_vid_to_img[video_id]
                    except KeyError:
                        det_vid_to_img[video_id] = image_id
                    else:
                        assert image_id_ == image_id, f"mismatch in det image_ids for video {video_id}"

                n_valid_objs += 1

                if is_multi:
                    segmentation = mask_pts_to_img(mask_pts, mask_h, mask_w, to_rle=1)

                coco_pred = dict(
                    image_id=image_id,
                    bbox=bbox,
                    score=score,
                    category_id=category_id,
                    segmentation=segmentation,
                )

                coco_preds.append(coco_pred)
                n_valid_objs_pc = n_valid_objs / n_total_objs * 100.
                pbar.set_description(f'n_valid: {n_valid_objs} / {n_total_objs} ({n_valid_objs_pc:.2f}%)')

        if save_coco:
            print(f'saving coco format predictions to {coco_json_path}')
            with open(coco_json_path, 'w') as f:
                output_json_data = json.dumps(coco_preds, indent=4)
                f.write(output_json_data)

    return coco_gt, coco_preds


def get_mask_iou(mask_1, mask_2, bbox_1, bbox_2, giou=False):
    from skimage.morphology import convex_hull_image

    if bbox_1 is not None and bbox_2 is not None:
        x1, y1, w1, h1 = bbox_2
        x2, y2, w2, h2 = bbox_1

        min_x, min_y = int(min(x1, x2)), int(min(y1, y2))
        max_x, max_y = int(max(x1 + w1, x2 + w2)), int(max(y1 + h1, y2 + h2))

        prev_bin_mask_ = mask_1[min_y:max_y + 1, min_x:max_x + 1]
        bin_mask_ = mask_2[min_y:max_y + 1, min_x:max_x + 1]
    else:
        prev_bin_mask_, bin_mask_ = mask_1, mask_2

    mask_inter = np.logical_and(prev_bin_mask_, bin_mask_)
    n_mask_inter = np.count_nonzero(mask_inter)

    mask_union = np.logical_or(prev_bin_mask_, bin_mask_)
    n_mask_union = np.count_nonzero(mask_union)

    try:
        mask_iou = n_mask_inter / n_mask_union
    except ZeroDivisionError:
        mask_iou = 0

    if not giou:
        return mask_iou

    convex_hull = convex_hull_image(mask_union)
    n_convex_hull = np.count_nonzero(convex_hull)
    g_mask_iou = mask_iou - ((n_convex_hull - n_mask_union) / n_convex_hull)

    return g_mask_iou


def coco_eval(gt_json_path, json_path, class_names, metrics, classwise):
    coco_gt = COCO(gt_json_path)

    coco_pred = coco_gt.loadRes(json_path)
    cat_ids = coco_gt.getCatIds(catNms=class_names)
    img_ids = coco_gt.getImgIds()

    iou_thrs = np.linspace(
        .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)

    eval_results = OrderedDict()
    for metric in metrics:

        cocoEval = COCOeval(coco_gt, coco_pred, metric)
        cocoEval.params.catIds = cat_ids
        cocoEval.params.imgIds = img_ids
        cocoEval.params.iouThrs = iou_thrs
        # mapping of cocoEval.stats
        coco_metric_names = {
            'mAP': 0,
            'mAP_50': 1,
            'mAP_75': 2,
            'mAP_s': 3,
            'mAP_m': 4,
            'mAP_l': 5,
            'AR@100': 6,
            'AR@300': 7,
            'AR@1000': 8,
            'AR_s@1000': 9,
            'AR_m@1000': 10,
            'AR_l@1000': 11
        }
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        if classwise:  # Compute per-category AP
            # Compute per-category AP
            # from https://github.com/facebookresearch/detectron2/
            precisions = cocoEval.eval['precision']
            # precision: (iou, recall, cls, area range, max dets)
            assert len(cat_ids) == precisions.shape[2]

            results_per_category = []
            for idx, catId in enumerate(cat_ids):
                # area range index 0: all area ranges
                # max dets index -1: typically 100 per image
                nm = coco_gt.loadCats(catId)[0]
                precision = precisions[:, :, idx, 0, -1]
                precision = precision[precision > -1]
                if precision.size:
                    ap = np.mean(precision)
                else:
                    ap = float('nan')
                results_per_category.append(
                    (f'{nm["name"]}', f'{float(ap):0.3f}'))

            num_columns = min(6, len(results_per_category) * 2)
            results_flatten = list(
                itertools.chain(*results_per_category))
            headers = ['category', 'AP'] * (num_columns // 2)
            results_2d = itertools.zip_longest(*[
                results_flatten[i::num_columns]
                for i in range(num_columns)
            ])
            table_data = [headers]
            table_data += [result for result in results_2d]
            table = AsciiTable(table_data)
            print('\n' + table.table)

            metric_items = [
                'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
            ]

        for metric_item in metric_items:
            key = f'{metric}_{metric_item}'
            val = float(
                f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
            )
            eval_results[key] = val
        ap = cocoEval.stats[:6]
        eval_results[f'{metric}_mAP_copypaste'] = (
            f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
            f'{ap[4]:.3f} {ap[5]:.3f}')

    return eval_results


def run(params, *argv):
    """

    :param Params params:
    :param argv:
    :return:
    """
    params = copy.deepcopy(params)

    # params.pred_path = os.path.join(params.db_root_dir, params.pred_path)

    for i, sweep_param in enumerate(params._sweep_params):

        if argv[i] is not None:
            setattr(params, sweep_param, argv[i])

        param_val = getattr(params, sweep_param)

        if isinstance(param_val, (list, tuple)):
            setattr(params, sweep_param, param_val[0])

    out_root_path = params.output
    class_names_path = params.class_names_path
    img_root_dir = params.img_root_dir

    nms_thresh = params.nms_thresh  # type: float

    class_names = [k.strip() for k in open(class_names_path, 'r').readlines() if k.strip()]
    class_names, class_cols = zip(*[k.split('\t') for k in class_names])

    class_name_to_col = {
        x.strip(): col
        for x, col in zip(class_names, class_cols)
    }
    # class_dict = {x.strip(): i for (i, x) in enumerate(class_names)}

    if not out_root_path:
        out_root_path = params.root_dir

    labels_name = params.labels_name

    if not labels_name:
        labels_name = 'annotations'

    assert params.json, "json must be provided"
    assert params.gt_json, "gt_json must be provided"

    pred_json_path = params.json
    gt_json_path = params.gt_json

    if params.root_dir:
        pred_json_path = linux_path(params.root_dir, pred_json_path)
        gt_json_path = linux_path(params.root_dir, gt_json_path)

    assert os.path.exists(pred_json_path), f"pred_json_path does not exist: {pred_json_path}"
    assert os.path.exists(gt_json_path), f"gt_json_path does not exist: {gt_json_path}"

    print(f'loading GT from {gt_json_path}')
    with open(gt_json_path, 'r') as fid:
        gt_json_data = json.load(fid)

    if os.path.isdir(pred_json_path):
        print(f'looking for predictions json files in {pred_json_path}')
        json_file_paths = glob.glob(linux_path(pred_json_path, '*.json'))
        n_json_file_paths = len(json_file_paths)
        assert n_json_file_paths > 0, "no json files found"
        print(f'found {n_json_file_paths} json files')
        pred_json_data = []
        for json_file_path in tqdm(json_file_paths):
            with open(json_file_path, 'r') as fid:
                json_datum = json.load(fid)
                pred_json_data.extend(json_datum)
    else:
        print(f'loading predictions from {pred_json_path}')
        with open(pred_json_path, 'r') as fid:
            pred_json_data = json.load(fid)

    if params.ytvis:
        coco_suffix = 'coco'
        if params.incremental:
            coco_suffix += '_incremental'
        coco_gt_json_path = add_suffix(gt_json_path, coco_suffix, dst_ext='.json')
        coco_pred_json_path = add_suffix(pred_json_path, coco_suffix, dst_ext='.json')

        gt_json_path = coco_gt_json_path
        pred_json_path = coco_pred_json_path

        gt_json_data, pred_json_data, = ytvis_to_coco(gt_json_data, pred_json_data,
                                                      coco_gt_json_path, coco_pred_json_path,
                                                      fix_category_id=params.fix_category_id,
                                                      incremental=params.incremental,
                                                      load_coco=params.load_coco,
                                                      save_coco=params.save_coco,
                                                      )
    else:
        coco_gt_json_path = gt_json_path
        coco_pred_json_path = pred_json_path

    categories = gt_json_data['categories']
    category_id_to_label = {
        category['id']: category['name']
        for category in categories
    }

    if params.eval == 2:
        eval_results = coco_eval(gt_json_path, pred_json_path, class_names,
                                 params.metrics, params.classwise)

        print(eval_results)

        # exit()

    if not img_root_dir:
        img_root_dir = os.path.dirname(gt_json_path)

        if params.ytvis:
            img_root_dir = os.path.dirname(img_root_dir)

    image_data = gt_json_data['images']

    image_id_to_name = {
        image['id']: image['file_name']
        for image in image_data
    }

    image_id_base_to_name = {
        os.path.basename(image['id']): image['file_name']
        for image in image_data
    }

    # coco_gt = COCO(gt_json_path)
    # cat_ids = coco_gt.getCatIds(catNms=class_names)

    # image_id_to_name = [image['file_name'] for image in image_data]
    # image_id_to_name = [linux_path(img_root_dir, image_name) for image_name in image_names]

    for _pred_id, _pred in enumerate(pred_json_data):
        _pred['global_id'] = _pred_id

    gt_images_df = pd.DataFrame.from_dict(gt_json_data['images'])
    ann_df = pd.DataFrame.from_dict(gt_json_data['annotations'])

    pred_df = pd.DataFrame.from_dict(pred_json_data)

    gt_image_ids = list(gt_images_df['id'].unique())

    seq_names = [os.path.dirname(image_id) for image_id in gt_image_ids]
    seq_names = list(set(seq_names))

    n_gt_image_ids = len(gt_image_ids)
    n_seq_names = len(seq_names)

    print(f'n_gt_image_ids:  {n_gt_image_ids}')
    print(f'n_seq_names:  {n_seq_names}')

    json_dir = os.path.dirname(pred_json_path)

    seq_to_csv_rows = {}

    vis = params.save or params.show

    if params.save:
        out_vis_dir = linux_path(json_dir, 'coco_to_xml')
        if params.incremental:
            out_vis_dir = f'{out_vis_dir}_inc'
        if params.nms_thresh > 0:
            out_vis_dir = f'{out_vis_dir}_nms_{int(nms_thresh * 100):02d}'
        os.makedirs(out_vis_dir, exist_ok=1)

        print(f'saving vis images to {out_vis_dir}')

    n_valid_objs = 0
    n_all_objs = 0

    # print('generating mask visualizations and csv')

    # images_df_id = gt_images_df.groupby("id").agg(list)
    pred_df_id = pred_df.groupby("image_id").agg(list)

    # assert images_df_id.shape[0] == n_gt_image_ids, "n_gt_image_ids mismatch"

    # pbar = tqdm(images_df_id.iterrows(), ncols=100, total=n_gt_image_ids)
    pbar = tqdm(gt_images_df.iterrows(), ncols=100, total=n_gt_image_ids)
    # pbar = tqdm(gt_image_ids, ncols=100)
    global_objs_to_delete = []

    print('post processing predictions...')
    if nms_thresh > 0:
        print(f'\tNMS with threshold {nms_thresh:.2f}')
    if params.conf_thresh > 0:
        print(f'confidence thresholding at {params.conf_thresh:.2f}')

    # for gt_image_id in pbar:
    for gt_image_id, gt_img_info in pbar:

        # det_img_info = gt_images_df.loc[gt_images_df['id'] == gt_image_id]

        # gt_images_df = gt_images_df.drop(det_img_info.index)
        # n_df = gt_images_df.shape[0]

        gt_image_id = str(gt_img_info['id'])

        h, w = int(gt_img_info['height']), int(gt_img_info['width'])

        if params.base_id_in_preds:
            det_image_id = os.path.basename(gt_image_id)
        else:
            det_image_id = gt_image_id

        # preds = pred_df.loc[pred_df['image_id'] == det_image_id]

        try:
            preds = pred_df_id.loc[det_image_id]
        except KeyError:
            # preds = []
            print(f'no preds for {det_image_id}')
            continue

        # pred_df = pred_df.drop(preds.index)
        # n_preds = pred_df.shape[0]

        img_name = image_id_to_name[gt_image_id]
        if params.save:
            out_img_name = img_name
            if params.save_flat:
                out_img_name = out_img_name.replace(os.sep, '_').replace('/', '_')
            out_vis_path = linux_path(out_vis_dir, out_img_name)
            out_vis_parent_path = os.path.dirname(out_vis_path)
            os.makedirs(out_vis_parent_path, exist_ok=True)

        img = ann_img = pred_img = None
        img_path = linux_path(img_root_dir, img_name)

        pred_global_ids = preds.loc['global_id']
        pred_scores = preds.loc['score']
        pred_bboxes = preds.loc['bbox']
        category_ids = preds.loc['category_id']

        if params.enable_mask:
            pred_segms = preds.loc['segmentation']
        else:
            pred_segms = [None for _ in pred_bboxes]

        if vis:
            assert os.path.exists(img_path), f'img does not exist: {img_path}'
            img = cv2.imread(img_path)
            ann_img = np.copy(img)

            if params.enable_mask:
                ann_img = ann_img.astype(np.float32)

            anns = ann_df.loc[ann_df['image_id'] == gt_image_id]

            for ann_id, ann in anns.iterrows():
                label = ann['label']
                col = class_name_to_col[label]
                ann_bbox = ann['bbox']
                if params.enable_mask:
                    ann_segm = ann['segmentation']
                    if type(ann_segm) == list:
                        if len(ann_segm) == 1:
                            ann_segm = ann_segm[0]

                        n_pts = int(len(ann_segm) / 2)
                        contour_pts = np.array(ann_segm).reshape((n_pts, 2))
                        contour_pts2 = np.asarray([contour_pts, ], dtype=np.int32)

                        ann_mask = np.zeros((h, w), dtype=np.uint8)
                        ann_mask = cv2.fillPoly(ann_mask, contour_pts2, 255)
                        ann_mask_binary = ann_mask > 0

                        mask_cv = contour_pts.reshape((-1, 1, 2)).astype(np.int32)
                        cv2.drawContours(ann_img, mask_cv, -1, col_bgr[col], thickness=2)

                        # rle = mask_util.frPyObjects([ann_segm], h, w)
                    else:
                        if type(ann_segm['counts']) == list:
                            rle = mask_util.frPyObjects([ann_segm], h, w)
                        else:
                            rle = [ann_segm, ]

                        ann_mask_binary = mask_util.decode(rle).squeeze().astype(bool)
                        # ann_mask = ann_mask_binary.astype(np.uint8)

                    ann_mask_img = np.zeros_like(ann_img)

                    ann_mask_img[ann_mask_binary] = col_bgr[col]

                    # ann_mask_vis = resize_ar(ann_mask_img, width=1280, height=720)
                    # cv2.imshow('ann_mask', ann_mask_vis)
                    # cv2.waitKey(0)

                    ann_img[ann_mask_binary] = (params.alpha * ann_img[ann_mask_binary] +
                                                (1 - params.alpha) * ann_mask_img[ann_mask_binary])
                else:
                    drawBox(ann_img, np.asarray(ann_bbox), color=col, xywh=True)

            ann_img = ann_img.astype(np.uint8)

            # ann_img_vis = resize_ar(ann_img, width=1280, height=720)
            # cv2.imshow('ann_img', ann_img_vis)
            # cv2.waitKey(0)

        pred_objs = []
        objs_to_delete = []

        # for pred_id, pred in preds.iterrows():
        for pred_id, (pred_score, pred_bbox, pred_segm, category_id, global_id) in enumerate(
                zip(pred_scores, pred_bboxes, pred_segms, category_ids, pred_global_ids)):

            n_all_objs += 1
            # print(f'pred_id: {pred_id}')

            # pred_score = pred['score']
            # pred_bbox = pred['bbox']
            # pred_segm = pred['segmentation']
            # category_id = pred['category_id']

            label = category_id_to_label[category_id]
            col = class_name_to_col[label]
            if params.enable_mask:
                if pred_segm['counts'] == 'Xjil6':
                    continue

                pred_mask_binary = mask_util.decode([pred_segm, ]).squeeze().astype(bool)
            else:
                pred_mask_binary = None

            if pred_score < params.conf_thresh:
                global_objs_to_delete.append(global_id)
            else:
                pred_objs.append((pred_id, pred_bbox, pred_segm, pred_mask_binary, pred_score, col, label, global_id))

        if nms_thresh > 0:
            # print("running non-maximum suppression")

            pred_obj_pairs = list(itertools.combinations(pred_objs, 2))

            # n_pred_objs = len(pred_objs)
            # n_pred_obj_pairs = len(pred_obj_pairs)

            n_deleted = 0

            for pred_obj_pair in pred_obj_pairs:
                obj1, obj2 = pred_obj_pair

                idx1, bbox1, rle1, _, score1, col1, _, global_id1 = obj1
                idx2, bbox2, rle2, _, score2, col2, _, global_id2 = obj2

                assert idx1 != idx2, "invalid object pair with identical IDs"

                if idx1 in objs_to_delete or idx2 in objs_to_delete:
                    continue

                if params.enable_mask:
                    pred_iou = get_mask_rle_iou(rle1, rle2)
                else:
                    pred_iou = get_iou(bbox1, bbox2, xywh=True)

                # mask_iou2 = get_mask_iou(mask1, mask2, bbox1, bbox2, giou=False)
                # assert pred_iou == mask_iou2, "mask_iou2 mismatch found"

                if pred_iou >= nms_thresh:
                    n_deleted += 1
                    # print(f'found matching object pair with iou {pred_iou:.3f}')
                    if score1 > score2:
                        objs_to_delete.append(idx2)
                        global_objs_to_delete.append(global_id2)
                        # print(f'removing obj {idx2} with score {score2} < {score1}')
                    else:
                        objs_to_delete.append(idx1)
                        global_objs_to_delete.append(global_id1)
                        # print(f'removing obj {idx1} with score {score1} < {score2}')
            # print(f'\nn_preds: {n_pred_objs}\tn_pairs: {n_pred_obj_pairs}\tn_deleted: {n_deleted}\n')
        if vis:
            pred_img = np.copy(img)

        for pred_obj in pred_objs:
            # pred_mask = pred_mask_binary.astype(np.uint8) * 255

            idx, bbox, rle, mask, score, col, label, global_id = pred_obj

            if idx in objs_to_delete:
                continue

            n_valid_objs += 1

            if vis:
                if params.enable_mask:
                    pred_mask_img = np.zeros_like(pred_img)
                    pred_mask_img[mask] = col_bgr[col]
                    pred_img[mask] = (params.alpha * pred_img[mask] +
                                      (1 - params.alpha) * pred_mask_img[mask])
                else:
                    drawBox(pred_img, np.asarray(bbox), color=col, xywh=True)

            xmin, ymin, w, h = bbox

            xmax, ymax = xmin + w, ymin + h

            csv_seq_name = os.path.dirname(gt_image_id)
            assert csv_seq_name, f"no csv_seq_name found in gt_image_id {gt_image_id}"

            csv_img_id = os.path.basename(img_path)

            if params.enable_mask:
                mask_h, mask_w = rle['size']
                mask_counts = rle['counts']
                if isinstance(mask_counts, bytes):
                    mask_counts = mask_counts.decode('utf-8')

            if params.save_csv:
                row = {
                    "ImageID": csv_img_id,
                    "LabelName": label,
                    "XMin": xmin,
                    "XMax": xmax,
                    "YMin": ymin,
                    "YMax": ymax,
                    "Confidence": score,
                }
                if params.enable_mask:
                    row.update(
                        {
                            "mask_w": mask_w,
                            "mask_h": mask_h,
                            "mask_counts": mask_counts,
                        }
                    )

                if csv_seq_name not in seq_to_csv_rows:
                    seq_to_csv_rows[csv_seq_name] = []

                seq_to_csv_rows[csv_seq_name].append(row)

        n_filtered_objs = n_all_objs - n_valid_objs
        filtered_pc = n_filtered_objs / n_all_objs * 100.
        pbar.set_description(f'n_filtered_objs {n_filtered_objs} / {n_all_objs} ({filtered_pc:.2f}%)'
                             f'')

        # print('here we are')
        if vis:
            cat_img = np.concatenate((ann_img, pred_img), axis=1)
            if params.save:
                cv2.imwrite(out_vis_path, cat_img)

            if params.show:
                cat_img_vis = resize_ar(cat_img, width=1280)
                cv2.imshow('cat_img_vis', cat_img_vis)
                cv2.waitKey(1)

    if params.save_csv:
        csv_columns = [
            "ImageID", "LabelName",
            "XMin", "XMax", "YMin", "YMax", "Confidence",
        ]
        if params.enable_mask:
            csv_columns += ['mask_w', 'mask_h', 'mask_counts']

        out_csv_dir_name = "csv"
        if params.csv_suffix:
            out_csv_dir_name = f'{out_csv_dir_name}_{params.csv_suffix}'

        if params.incremental:
            out_csv_dir_name = f'{out_csv_dir_name}_incremental'

        if nms_thresh > 0:
            out_csv_dir_name = f'{out_csv_dir_name}_nms_{int(nms_thresh * 100):02d}'

        out_csv_dir = os.path.join(json_dir, out_csv_dir_name)
        print(f'writing csv files to: {out_csv_dir}')
        os.makedirs(out_csv_dir, exist_ok=1)
        for csv_seq_name, csv_rows in seq_to_csv_rows.items():
            if not csv_rows:
                print(f'{csv_seq_name}: no csv data found')
                continue
            out_csv_name = f"{csv_seq_name}.csv"
            out_csv_path = os.path.join(out_csv_dir, out_csv_name)
            # print(f'{csv_seq_name} :: saving csv to {out_csv_path}')
            df = pd.DataFrame(csv_rows, columns=csv_columns)
            df.to_csv(out_csv_path, index=False)

    if global_objs_to_delete:
        n_preds = len(pred_json_data)
        n_invalid = len(global_objs_to_delete)
        invalid_pc = n_valid_objs / n_all_objs * 100.
        print(f'removing {n_invalid} / {n_preds} ({invalid_pc}%) predictions')
        global_objs_to_keep = set(range(n_preds)) - set(global_objs_to_delete)
        pred_json_data = [pred_json_data[i] for i in global_objs_to_keep]

        coco_filtered_suffix = 'filtered'
        if nms_thresh > 0:
            coco_filtered_suffix = f'{coco_filtered_suffix}_nms_{int(nms_thresh * 100):02d}'

        coco_filtered_pred_json_path = add_suffix(coco_pred_json_path, coco_filtered_suffix, dst_ext='.json')

        print(f'saving coco format filtered predictions to {coco_filtered_pred_json_path}')
        with open(coco_filtered_pred_json_path, 'w') as f:
            output_json_data = json.dumps(pred_json_data, indent=4)
            f.write(output_json_data)

        if params.eval:
            eval_results = coco_eval(gt_json_path, coco_filtered_pred_json_path,
                                     class_names,
                                     params.metrics, params.classwise)

            print(eval_results)


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
        import multiprocessing
        import functools

        print(f'running in parallel over {n_proc} processes')
        pool = multiprocessing.Pool(n_proc)
        func = functools.partial(run, params)

        pool.starmap(func, sweep_val_combos)
    else:
        for sweep_val_combo in sweep_val_combos:
            run(params, *sweep_val_combo)


if __name__ == "__main__":
    main()
