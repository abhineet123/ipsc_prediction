import os
import sys
import glob
import random
import ast

import pandas as pd
import numpy as np
import cv2
import json

from pprint import pformat

from datetime import datetime
from tqdm import tqdm

import imagesize
import paramparse

from itertools import groupby
import pycocotools.mask as mask_util

import multiprocessing

from eval_utils import (col_bgr, sortKey, linux_path, draw_box,
                        annotate, load_samples_from_txt, get_iou, add_suffix)

sys.path.append(linux_path(os.path.expanduser('~'), 'pix2seq'))

from tasks import task_utils


# from multiprocessing.pool import ThreadPool


class Params(paramparse.CFG):
    def __init__(self):
        paramparse.CFG.__init__(self, cfg_prefix='xml_to_ytvis')

        self.batch_size = 1
        self.description = ''
        self.excluded_images_list = ''

        self.class_names_path = ''
        self.map_classes = 0
        self.auto_class_cols = 0

        self.codec = 'H264'
        self.csv_file_name = ''
        self.fps = 20
        self.img_ext = 'jpg'
        self.load_path = ''
        self.load_samples = []
        self.load_samples_root = ''
        self.load_samples_suffix = []

        self.map_folder = ''
        self.min_val = 0
        self.recursive = 0
        self.n_classes = 4
        self.n_frames = 0
        self.img_dir_name = 'images'
        self.xml_dir_name = 'annotations'
        self.dir_suffix = ''
        self.out_dir_name = ''

        self.incremental = 0

        self.out_json_name = ''
        self.root_dir = ''
        self.xml_root_dir = ''
        self.seq_paths = ''
        self.seq_paths_suffix = ''
        self.show_img = 0
        self.shuffle = 0
        self.subseq = 0
        self.subseq_split_ids = []
        self.sources_to_include = []
        self.val_ratio = 0
        self.allow_missing_images = 0
        self.remove_mj_dir_suffix = 0
        self.infer_target_id = 0
        self.get_img_stats = 0
        self.ignore_invalid_label = 0
        self.allow_ignored_class = 0
        self.ignore_missing_target = 0
        self.zero_based_target_ids = 0

        self.target_id_field = 'id_number'
        self.add_ext_to_image = 0

        self.start_frame_id = 0
        self.end_frame_id = -1
        self.frame_stride = -1

        self.n_seq = 0
        self.start_seq_id = 0
        self.end_seq_id = -1

        self.frame_gap = 1
        self.length = 0
        self.stride = 1
        self.sample = -1

        self.max_length = 0
        self.min_length = 0
        self.coco_rle = 0
        self.n_proc = 0

        self.json_gz = 1
        self.xml_zip = 1

        """in order to not exclude empty images without any xml files"""
        self.xml_from_img = 1

        self.enable_masks = 0
        self.save_masks = 0
        self.check_img_size = 0
        self.quant_bins = [24, 32, 40, 48, 64, 80, 128, 160, 200, 256, 320]

        self.vis = 0

        self.strides = []


def offset_target_ids(vid_to_target_ids, annotations, set_type):
    print(f'adding offsets to {set_type} targets IDs')
    vid_ids = sorted(list(vid_to_target_ids.keys()))
    target_id_offsets = {}
    curr_offset = 0
    for vid_id in vid_ids:
        target_id_offsets[vid_id] = curr_offset
        curr_offset += len(vid_to_target_ids[vid_id])

    for annotation in tqdm(annotations, position=0, leave=True):
        vid_id = annotation['video_id']
        annotation['id'] += target_id_offsets[vid_id]


def binary_mask_to_rle_coco(binary_mask):
    # binary_mask_fortran = np.array(binary_mask[:, :, None], order="F", dtype="uint8")
    binary_mask_fortran = np.asfortranarray(binary_mask, dtype="uint8")
    rle = mask_util.encode(binary_mask_fortran)
    # rle_encoded = rle['counts']
    # rle_decoded = rle_encoded.decode('ascii')
    # rle_decoded_mask = mask_util.decode(rle)

    # import pycocotools._mask as _mask_util
    # rle_decoded_string = _mask_util.decompress(rle)

    rle["counts"] = rle["counts"].decode("utf-8")

    return rle


def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle


def read_xml_file(params: Params,
                  db_root_dir,
                  excluded_images,
                  img_path_to_stats,
                  seq_name_to_xml_paths,
                  seq_name_to_info,
                  quant_bin_to_ious,
                  class_dict,
                  class_map_dict,
                  xml_data):
    xml_path, xml_path_id, seq_path, seq_name = xml_data

    all_pix_vals_mean = []
    all_pix_vals_std = []

    import xml.etree.ElementTree as ET

    if params.xml_zip:
        xml_name = os.path.basename(xml_path)
        xml_dir_path = os.path.dirname(xml_path)
        xml_zip_path = xml_dir_path + ".zip"

        from zipfile import ZipFile
        with ZipFile(xml_zip_path, 'r') as xml_zip_file:
            with xml_zip_file.open(xml_name, "r") as xml_file:
                ann_tree = ET.parse(xml_file)
    else:
        ann_tree = ET.parse(xml_path)
    ann_root = ann_tree.getroot()

    filename = ann_tree.findtext('filename')

    if params.add_ext_to_image:
        assert params.img_ext not in filename, f"filename already has img_ext: {filename}"
        filename = f'{filename}.{params.img_ext}'

    """img_path is relative to db_root_dir"""
    # img_rel_path = ann_tree.findtext('path')
    # if img_rel_path is None:
    #     img_rel_path = linux_path(seq_name, filename)
    # else:
    #     _filename = os.path.basename(img_rel_path)
    #     assert _filename == filename, f"mismatch between filename: {filename} and path: {img_rel_path}"
    img_path = linux_path(seq_path, filename)
    img_name = filename
    img_rel_path = linux_path(os.path.relpath(img_path, db_root_dir))

    if params.remove_mj_dir_suffix:
        img_file_rel_path_list = img_rel_path.split('/')
        img_rel_path_ = linux_path(img_file_rel_path_list[0], img_file_rel_path_list[1], img_name)
        img_path_ = linux_path(db_root_dir, img_rel_path_)

        if os.path.exists(img_path) and not os.path.exists(img_path_):
            os.rename(img_path, img_path_)

        img_rel_path = img_rel_path_
        img_path = img_path_

    # img_file_path = linux_path(seq_path, img_file_name)

    if excluded_images is not None and img_name in excluded_images[seq_path]:
        print(f'\n{seq_name} :: skipping excluded image {img_name}')
        return None

    if not os.path.exists(img_path):
        msg = f"img_file_path does not exist: {img_path}"
        if params.allow_missing_images:
            print('\n' + msg + '\n')
            return None
        else:
            raise AssertionError(msg)

    size_from_xml = ann_root.find('size')
    img_w = int(size_from_xml.findtext('width'))
    img_h = int(size_from_xml.findtext('height'))

    if params.check_img_size:
        img_w_, img_h_ = imagesize.get(img_path)
        assert img_h_ == img_h and img_w_ == img_w, (f"mismatch between image dimensions in XML: {(img_h, img_w)} and "
                                                     f"actual: ({img_h_, img_w_})")

    if params.get_img_stats:
        try:
            img_stat = img_path_to_stats[img_path]
        except KeyError:
            img = cv2.imread(img_path)
            img_h, img_w = img.shape[:2]

            img_reshaped = np.reshape(img, (img_h * img_w, 3))

            pix_vals_mean = list(np.mean(img_reshaped, axis=0))
            pix_vals_std = list(np.std(img_reshaped, axis=0))

            img_path_to_stats[img_path] = (pix_vals_mean, pix_vals_std)
        else:
            pix_vals_mean, pix_vals_std = img_stat

        all_pix_vals_mean.append(pix_vals_mean)
        all_pix_vals_std.append(pix_vals_std)

    objs = ann_root.findall('object')
    src_shape = (img_h, img_w)

    xml_dict = dict(
        img_path=img_path,
        img_id=xml_path_id,
        img_rel_path=img_rel_path,
        filename=filename,
        src_shape=src_shape,
    )

    xml_data_list = []
    target_ids = []

    for obj_id, obj in enumerate(objs):

        label = obj.findtext('name')

        if params.allow_ignored_class and label == 'ignored':
            """special class to mark ignored regions in the image that have not been annotated"""
            continue

        try:
            label = class_map_dict[label]
        except KeyError as e:
            if params.ignore_invalid_label:
                print(f'{xml_path}: ignoring obj with invalid label: {label}')
                continue
            raise AssertionError(e)

        try:
            label_id = class_dict[label]
        except KeyError as e:
            if params.ignore_invalid_label:
                print(f'{xml_path}: ignoring obj with invalid label: {label}')
                continue
            raise AssertionError(e)

        try:
            target_id = int(obj.findtext(params.target_id_field))
        except ValueError as e:
            if params.ignore_missing_target:
                print(f'{xml_path}: ignoring obj with missing or invalid target_id')
                continue
            raise ValueError(e)

        if params.zero_based_target_ids:
            target_id += 1

        target_ids.append(target_id)

        bbox = obj.find('bndbox')
        # score = float(obj.find('score').text)
        # difficult = int(obj.find('difficult').text)
        # bbox_source = obj.find('bbox_source').text
        xmin = bbox.find('xmin')
        ymin = bbox.find('ymin')
        xmax = bbox.find('xmax')
        ymax = bbox.find('ymax')

        xmin, ymin, xmax, ymax = float(xmin.text), float(ymin.text), float(xmax.text), float(ymax.text)
        bbox_w, bbox_h = xmax - xmin, ymax - ymin

        bbox = [xmin, ymin, bbox_w, bbox_h]

        xmin_norm, ymin_norm, xmax_norm, ymax_norm = xmin / img_w, ymin / img_h, xmax / img_w, ymax / img_h
        bbox_norm = [xmin_norm, ymin_norm, xmax_norm, ymax_norm]

        for quant_bin in params.quant_bins:
            bbox_quant = [int(k * quant_bin) for k in bbox_norm]
            xmin_rec, ymin_rec, xmax_rec, ymax_rec = [float(k) / quant_bin for k in bbox_quant]

            xmin_rec, ymin_rec, xmax_rec, ymax_rec = (xmin_rec * img_w, ymin_rec * img_h, xmax_rec * img_w,
                                                      ymax_rec * img_h)
            iou = get_iou(
                [xmin, ymin, xmax, ymax],
                [xmin_rec, ymin_rec, xmax_rec, ymax_rec],
                xywh=False
            )
            quant_bin_to_ious[quant_bin].append(iou)

        xml_data = dict(
            label=label,
            target_id=target_id,
            bbox=bbox,
        )

        if params.enable_masks:
            mask_obj = obj.find('mask')
            if mask_obj is None:
                msg = 'no mask found for object:\n{}'.format(img_name)
                raise AssertionError(msg)

            mask = mask_obj.text

            mask = [k.strip().split(',') for k in mask.strip().split(';') if k]
            # pprint(mask)
            mask_pts = [(float(_pt[0]), float(_pt[1])) for _pt in mask]

            bin_mask = np.zeros(src_shape, dtype=np.uint8)
            bin_mask = cv2.fillPoly(bin_mask, np.array([mask_pts, ], dtype=np.int32), 255)
            if params.coco_rle:
                bin_mask_rle = binary_mask_to_rle_coco(bin_mask)
            else:
                bin_mask_rle = binary_mask_to_rle(bin_mask)

            seg_area = np.count_nonzero(bin_mask)

            xml_data.update(dict(
                bin_mask_rle=bin_mask_rle,
                seg_area=seg_area,
            ))

        xml_data_list.append(xml_data)

    target_ids = list(set(target_ids))

    n_objs = len(xml_data_list)
    try:
        seq_info = seq_name_to_info[seq_name]
    except KeyError:
        all_xml_files = seq_name_to_xml_paths[seq_name]
        seq_name_to_info[seq_name] = dict(
            name=seq_name,
            height=img_h,
            width=img_w,
            aspect_ratio=float(img_w) / float(img_h),
            length=len(all_xml_files),
            target_ids=target_ids,
            n_objs=[n_objs, ],
        )
    else:
        seq_info['n_objs'].append(n_objs)
        seq_info['target_ids'] += target_ids

    xml_dict['objs'] = xml_data_list

    return xml_path, xml_dict, all_pix_vals_mean, all_pix_vals_std


def save_annotations_ytvis(
        xml_data_dict,
        class_to_id,
        ignore_invalid_label,
        infer_target_id,
        use_tqdm,
        enable_masks,
        vis,
        xml_data,
):
    xml_files, vid_id = xml_data

    video_dict = {}
    all_annotations = []
    target_ids = []

    if not xml_files:
        return vid_id, video_dict, all_annotations, target_ids

    n_valid_images = 0
    n_images = 0
    n_objs = 0
    label_to_n_objs = {
        label: 0 for label in class_to_id
    }
    vid_size = None
    file_names = []
    file_ids = []
    ann_objs = {}
    sec_ann_objs = {}
    n_files = len(xml_files)

    vid_seq_path = vid_seq_name = None
    next_target_id = 0
    target_ids = []
    pause = 1

    pbar = tqdm(xml_files, ncols=100, position=0, leave=True) if use_tqdm else xml_files
    for frame_id, (xml_path, xml_id, seq_path, seq_name) in enumerate(pbar):

        xml_data = xml_data_dict[xml_path]

        if infer_target_id:
            if frame_id > 0 and ann_objs:
                max_target_id = max(list(ann_objs.keys()))
                prev_objs = [
                    [
                        _id,
                        ann_obj['category_id'],
                        ann_obj['bboxes'][frame_id - 1],
                        ann_obj['bin_mask_rle'][frame_id - 1],
                        ann_obj['bin_mask'][frame_id - 1],
                        True,
                    ]
                    for _id, ann_obj in sec_ann_objs.items() if ann_obj['bboxes'][frame_id - 1] is not None]
            else:
                prev_objs = []
                max_target_id = 0
            next_target_id = max(max_target_id + 1, next_target_id)

        if vid_seq_path is None:
            vid_seq_path = seq_path
        else:
            assert vid_seq_path == seq_path, f"mismatch in seq_path: {seq_path}"

        if vid_seq_name is None:
            vid_seq_name = seq_name
        else:
            assert vid_seq_name == seq_name, f"mismatch in seq_name: {seq_name}"

        n_images += 1

        img_rel_path = xml_data['img_rel_path']

        img_id = xml_data['img_id']

        assert img_id == xml_id, f"mismatch between img_id {img_id} and xml_id: {xml_id}"

        src_shape = xml_data['src_shape']

        h, w = src_shape

        if vid_size is None:
            vid_size = (w, h)
        else:
            assert vid_size == (w, h), f"mismatch between size of image: {(w, h)} and video: {vid_size}"

        file_names.append(img_rel_path)
        file_ids.append(img_id)

        objs = xml_data['objs']

        vis_img = None
        if vis:
            img_path = xml_data['img_path']
            vis_img = cv2.imread(img_path)

        for obj_id, obj in enumerate(objs):

            label = obj['label']
            target_id = obj['target_id']
            bbox = obj['bbox']

            bbox_cx, bbox_cy, bbox_w, bbox_h = bbox
            if vis:
                draw_box(vis_img, np.asarray(bbox), f'{label}-{target_id}', 'green', 1)

            if enable_masks:
                bin_mask_rle = obj['bin_mask_rle']
                obj_area = obj['seg_area']
            else:
                obj_area = bbox_w * bbox_h

            try:
                category_id = class_to_id[label]
            except KeyError:
                msg = f"label {label} is not in class_to_id_and_col"
                if ignore_invalid_label:
                    print(msg)
                    continue
                else:
                    raise AssertionError(msg)

            if infer_target_id:
                assert enable_masks, "infer_target_id is currently only supported with masks"

                assert target_id <= 0, f"existing target_id: {target_id} found in {xml_path}"

                valid_prev_objs = [prev_obj for prev_obj in prev_objs
                                   if prev_obj[1] == category_id and prev_obj[-1]]

                bin_mask = mask_util.decode(bin_mask_rle).astype(bool)

                if not valid_prev_objs:
                    target_id = next_target_id
                    next_target_id += 1
                else:
                    # prev_bboxes = [k[1] for k in prev_objs]
                    # prev_bin_masks = [k[2] for k in prev_objs]
                    # prev_masks_pts = [k[3] for k in prev_objs]

                    # prev_bboxes_np = np.asarray(prev_bboxes)
                    # bbox_np = np.asarray(bbox)

                    # max_iou, max_iou_idx = get_max_overlap_obj(prev_bboxes_np, bbox_np)

                    from skimage.morphology import convex_hull_image

                    mask_ious = []
                    for prev_obj in valid_prev_objs:
                        prev_bbox = prev_obj[2]
                        prev_bin_mask_rle = prev_obj[3]
                        prev_bin_mask = prev_obj[4]

                        x1, y1, w1, h1 = bbox
                        x2, y2, w2, h2 = prev_bbox

                        min_x, min_y = int(min(x1, x2)), int(min(y1, y2))
                        max_x, max_y = int(max(x1 + w1, x2 + w2)), int(max(y1 + h1, y2 + h2))

                        mask_iou = mask_util.iou([prev_bin_mask_rle, ], [bin_mask_rle, ], 0)

                        prev_bin_mask_ = prev_bin_mask[min_y:max_y + 1, min_x:max_x + 1]
                        bin_mask_ = bin_mask[min_y:max_y + 1, min_x:max_x + 1]

                        mask_inter = np.logical_and(prev_bin_mask_, bin_mask_)
                        n_mask_inter = np.count_nonzero(mask_inter)
                        mask_iou_ = n_mask_inter / n_mask_union

                        assert mask_iou_ == mask_iou, "mask_iou mismatch"

                        mask_union = np.logical_or(prev_bin_mask_, bin_mask_)
                        n_mask_union = np.count_nonzero(mask_union)

                        convex_hull = convex_hull_image(mask_union)
                        n_convex_hull = np.count_nonzero(convex_hull)
                        g_mask_iou = mask_iou - ((n_convex_hull - n_mask_union) / n_convex_hull)

                        mask_ious.append(g_mask_iou)

                    max_mask_iou_idx = np.argmax(mask_ious)

                    max_mask_iou_obj = valid_prev_objs[max_mask_iou_idx]
                    matched_target_id = max_mask_iou_obj[0]
                    target_id = matched_target_id
                    max_mask_iou_obj[-1] = False

            assert target_id > 0, "target_id must be > 0"

            if target_id not in target_ids:
                target_ids.append(target_id)

            if target_id not in ann_objs:
                ann_objs[target_id] = {
                    'iscrowd': 0,
                    'category_id': category_id,
                    'bboxes': [None, ] * n_files,
                    'areas': [None, ] * n_files,
                }
                if enable_masks:
                    ann_objs[target_id].update({
                        'segmentations': [None, ] * n_files,
                        'bin_mask_rle': [None, ] * n_files,
                        'mask_pts': [None, ] * n_files,
                    })

                if infer_target_id:
                    sec_ann_objs[target_id] = {
                        'category_id': category_id,
                        'bboxes': [None, ] * n_files,
                        'bin_mask_rle': [None, ] * n_files,
                        'bin_mask': [None, ] * n_files,
                    }

            curr_ann_obj = ann_objs[target_id]

            curr_ann_obj['bboxes'][frame_id] = bbox
            curr_ann_obj['areas'][frame_id] = obj_area

            if enable_masks:
                curr_ann_obj['segmentations'][frame_id] = bin_mask_rle

            if infer_target_id:
                sec_curr_ann_obj = sec_ann_objs[target_id]
                sec_curr_ann_obj['bboxes'][frame_id] = bbox
                sec_curr_ann_obj['bin_mask_rle'][frame_id] = bin_mask_rle
                sec_curr_ann_obj['bin_mask'][frame_id] = bin_mask

            n_objs += 1

            label_to_n_objs[label] += 1

        if vis:
            vis_img = annotate(vis_img, img_rel_path)
            cv2.imshow('vis_img', vis_img)
            k = cv2.waitKey(1 - pause)
            if k == 32:
                pause = 1 - pause
        n_valid_images += 1

    assert len(file_names) == n_valid_images, f"mismatch in n_valid_images: {n_valid_images}"

    vid_w, vid_h = vid_size

    all_annotations = []

    for target_id in ann_objs:
        ann_obj = ann_objs[target_id]
        annotations_dict = {
            "width": vid_w,
            "height": vid_h,
            "length": 1,
            "id": target_id,
            "video_id": vid_id,
            "category_id": ann_obj['category_id'],
            "bboxes": ann_obj['bboxes'],
            "iscrowd": ann_obj['iscrowd'],
            "areas": ann_obj['areas'],
        }

        if enable_masks:
            annotations_dict.update({
                "segmentations": ann_obj['segmentations'],
            })
        all_annotations.append(annotations_dict)

    assert all(i < j for i, j in zip(file_ids, file_ids[1:])), \
        "file_ids should be strictly increasing"

    video_dict = {
        "width": vid_w,
        "height": vid_h,
        "length": n_valid_images,
        "date_captured": "",
        "license": 1,
        "flickr_url": "",
        "file_names": file_names,
        "file_ids": file_ids,
        "id": vid_id,
        "coco_url": "",
    }

    return vid_id, video_dict, all_annotations, target_ids


def get_xml_files(
        params: Params,
        excluded_images_list, all_excluded_images,
        all_val_files, all_train_files,
        seq_name_to_xml_paths,
        seq_to_samples,
        xml_data
):
    xml_dir_path, all_xml_files, seq_name, seq_path, subseq_info, vid_start_id = xml_data

    if all_xml_files is None:
        if seq_to_samples is not None:
            all_xml_files = seq_to_samples[seq_path]
        else:
            if params.xml_from_img:
                all_img_files = glob.glob(linux_path(seq_path, f'*.{params.img_ext}'), recursive=False)
                assert len(all_img_files) > 0, 'No image files found in {}'.format(seq_path)

                all_img_names = [os.path.splitext(os.path.basename(img_file))[0] for img_file in all_img_files]
                all_xml_files = [linux_path(xml_dir_path, f'{img_name}.xml') for img_name in all_img_names]
                # print()
            else:
                if params.xml_zip:
                    from zipfile import ZipFile

                    xml_zip_path = xml_dir_path + ".zip"
                    print(f'loading xml files from  zip file {xml_zip_path}')
                    with ZipFile(xml_zip_path, 'r') as xml_zip_file:
                        all_xml_files = xml_zip_file.namelist()

                    all_xml_files = [linux_path(xml_dir_path, xml_file) for xml_file in all_xml_files]
                else:
                    # xml_file_gen = [[os.path.join(dirpath, f) for f in filenames if
                    #                  os.path.splitext(f.lower())[1] == '.xml']
                    #                 for (dirpath, dirnames, filenames) in os.walk(xml_dir_path, followlinks=True)]
                    # all_xml_files = [item for sublist in xml_file_gen for item in sublist]

                    all_xml_files = glob.glob(linux_path(xml_dir_path, '*.xml'), recursive=params.recursive)
                    # print()

        if params.shuffle:
            random.shuffle(all_xml_files)
        else:
            """seq_to_samples might have overlapping sequences so sorting will mess it up"""
            if seq_to_samples is None:
                all_xml_files.sort(key=lambda fname: os.path.basename(fname))
    else:
        assert seq_to_samples is None, "seq_to_samples cannot be provided alongside all_xml_files"

    start_frame_id = params.start_frame_id
    end_frame_id = params.end_frame_id
    frame_stride = params.frame_stride

    if frame_stride <= 0:
        frame_stride = 1

    if end_frame_id < start_frame_id:
        end_frame_id = len(all_xml_files) - 1

    all_xml_files = all_xml_files[start_frame_id:end_frame_id + 1:frame_stride]

    if seq_name not in seq_name_to_xml_paths:
        seq_name_to_xml_paths[seq_name] = all_xml_files

    n_all_files = len(all_xml_files)

    excluded_images = []
    if excluded_images_list:
        excluded_images_list_path = linux_path(xml_dir_path, excluded_images_list)
        if os.path.exists(excluded_images_list_path):
            print(f'reading excluded_images_list from {excluded_images_list_path}')

            excluded_images = open(excluded_images_list_path, 'r').readlines()
            excluded_images = [k.strip() for k in set(excluded_images)]
            print(f'found {len(excluded_images)} excluded_images')
        else:
            print(f'excluded_images_list does not exist {excluded_images_list_path}')

    all_excluded_images[seq_path] = excluded_images

    assert n_all_files > 0, 'No xml files found in {}'.format(xml_dir_path)

    all_subseq_xml_files = []

    if params.subseq:
        assert not excluded_images, "subseq cannot be reconstructed with non-empty excluded_images"

        if params.subseq_split_ids:
            subseq_start_end_ids = []
            start_id = 0
            for end_id in params.subseq_split_ids:
                subseq_start_end_ids.append((start_id, end_id))
                start_id = end_id + 1
            end_id = len(all_xml_files) - 1
            subseq_start_end_ids.append((start_id, end_id))
        else:
            subseq_start_end_ids = [(row["start_id"], row["end_id"]) for _, row in subseq_info.iterrows()]

        pbar = tqdm(subseq_start_end_ids, position=0, leave=True)

        global_subseq_id = 0

        for subseq_id, (start_id, end_id) in enumerate(pbar):
            assert end_id < n_all_files, f"invalid end_id: {end_id} for n_files: {n_all_files}" \
                                         f" in subseq_id {subseq_id} of {xml_dir_path}"

            subseq_xml_files = all_xml_files[start_id:end_id + 1]
            n_subseq_xml_files = len(subseq_xml_files)

            if n_subseq_xml_files < params.min_length:
                print(f'out of range subseq {subseq_id + 1} ({start_id} -> {end_id})'
                      f' with length {n_subseq_xml_files} < {params.min_length} ')
                gap = params.min_length - n_subseq_xml_files
                assert gap <= start_id, "subseq shifting is not possible"
                start_id -= gap
                print(f'shifting to ({start_id} -> {end_id})')
                subseq_xml_files = all_xml_files[start_id:end_id + 1]
                n_subseq_xml_files = len(subseq_xml_files)
            elif n_subseq_xml_files > params.max_length > 0:
                # n_subsubseq = int(round(float(n_subseq_xml_files) / params.max_length))
                # subsubseq_start_id = 0

                subsubseq_start_ids = list(range(0, n_subseq_xml_files, params.stride))
                n_subsubseq = len(subsubseq_start_ids)

                for subsubseq_id, subsubseq_start_id in enumerate(subsubseq_start_ids):

                    subsubseq_end_id = min(subsubseq_start_id + (params.max_length - 1) * params.frame_gap,
                                           n_subseq_xml_files - 1)

                    if subsubseq_start_id > subsubseq_end_id:
                        break

                    subsubseq_xml_files = subseq_xml_files[subsubseq_start_id:subsubseq_end_id + 1:params.frame_gap]

                    n_subsubseq_xml_files = len(subsubseq_xml_files)

                    if n_subsubseq_xml_files < params.min_length:
                        print(
                            f'skipping subseq {subseq_id + 1} - {subsubseq_id + 1} with length {n_subsubseq_xml_files}')
                        continue

                    print(f'\t{seq_name} :: subseq {global_subseq_id + 1} '
                          f'({subseq_id + 1} - {subsubseq_id + 1} / {n_subsubseq})  length {n_subsubseq_xml_files} '
                          f'({start_id + subsubseq_start_id} -> {start_id + subsubseq_end_id})')

                    assert len(subsubseq_xml_files) > 0, "no subsubseq_xml_files found"

                    all_subseq_xml_files.append(subsubseq_xml_files)

                    # subsubseq_start_id = subsubseq_end_id + 1

                    global_subseq_id += 1
            else:
                print(f'{seq_name}: subseq {global_subseq_id + 1} length {n_subseq_xml_files} ({start_id} -> {end_id})')
                assert len(subseq_xml_files) > 0, "no subseq_xml_files found"

                all_subseq_xml_files.append(subseq_xml_files)
                global_subseq_id += 1
    else:
        if n_all_files < params.min_length:
            print(f'skipping seq {seq_name} with length {n_all_files} < {params.min_length}')
            return
        elif n_all_files > params.max_length > 0:
            # n_subseq = int(round(float(n_all_files) / params.max_length))
            subseq_start_ids = list(range(0, n_all_files, params.stride))
            # n_subseq = len(subseq_start_ids)

            # subseq_start_id = 0
            for subseq_id, subseq_start_id in enumerate(subseq_start_ids):

                subseq_end_id = min(subseq_start_id + (params.max_length - 1) * params.frame_gap, n_all_files - 1)

                if subseq_start_id > subseq_end_id:
                    break

                subseq_xml_files = all_xml_files[subseq_start_id:subseq_end_id + 1:params.frame_gap]

                n_subseq_xml_files = len(subseq_xml_files)

                if n_subseq_xml_files < params.min_length:
                    print(f'\nout of range subseq {subseq_id + 1} ({subseq_start_id} -> {subseq_end_id})'
                          f' with length {n_subseq_xml_files} < {params.min_length} ')
                    gap = params.min_length - n_subseq_xml_files
                    assert gap <= subseq_start_id, "subseq shifting is not possible"
                    subseq_start_id -= gap
                    print(f'\tshifting to ({subseq_start_id} -> {subseq_end_id})\n')
                    subseq_xml_files = all_xml_files[subseq_start_id:subseq_end_id + 1:params.frame_gap]
                    n_subseq_xml_files = len(subseq_xml_files)

                    # print(f'skipping subseq {subseq_id + 1} with length {n_subseq_xml_files}')
                    # continue

                print(f'{seq_name} :: subseq {subseq_id + 1} length {n_subseq_xml_files} '
                      f'({subseq_start_id} -> {subseq_end_id})')

                assert n_subseq_xml_files > 0, "no subseq_xml_files found"

                all_subseq_xml_files.append(subseq_xml_files)

                # subseq_start_id = subseq_end_id + 1

        else:
            assert n_all_files > 0, "no xml_files found for seq_name"

            print(f'{seq_name}: length {n_all_files}')
            all_subseq_xml_files.append(all_xml_files)

    for subseq_id, subseq_xml_files in enumerate(all_subseq_xml_files):
        # vid_id = vid_start_id + subseq_id + 1
        n_files = len(subseq_xml_files)

        assert n_files > 0, f'No xml files found for subseq {subseq_id} of {xml_dir_path}'

        n_val_files = max(int(n_files * params.val_ratio), params.min_val)

        # n_train_files = n_files - n_val_files

        # print(f'{vid_id} / {n_vids} :: {seq_name} (subseq: {subseq_id + 1}) : '
        #       f'n_train, n_val: {[n_train_files, n_val_files]} ')
        subseq_xml_file_ids = [all_xml_files.index(file) for file in subseq_xml_files]

        subseq_xml_files = tuple(
            zip(subseq_xml_files, subseq_xml_file_ids, [seq_path, ] * n_files, [seq_name, ] * n_files))

        val_xml_files = subseq_xml_files[:n_val_files]
        train_xml_files = subseq_xml_files[n_val_files:]
        n_train_xml_files = len(train_xml_files)
        n_val_xml_files = len(val_xml_files)

        if n_val_xml_files and n_val_xml_files >= params.min_length:
            if params.incremental:
                for _end_id in range(n_val_xml_files - 1):
                    _val_xml_files = val_xml_files[:_end_id + 1]
                    all_val_files.append(_val_xml_files)
            all_val_files.append(val_xml_files)

        if n_train_xml_files and n_train_xml_files >= params.min_length:
            if params.incremental:
                for _end_id in range(n_train_xml_files - 1):
                    _train_xml_files = train_xml_files[:_end_id + 1]
                    all_train_files.append(_train_xml_files)
            all_train_files.append(train_xml_files)


def get_iou_stats(ious, bins=100):
    stats = dict(
        mean=np.mean(ious),
        median=np.median(ious),
        min=np.amin(ious),
        max=np.amax(ious),
    )
    hist, bin_edges = np.histogram(ious, bins=100, range=(0, 1))
    bin_edges = bin_edges[:-1]
    stats.update({
        f'hist-{bin_edge:.2f}': hist_val for bin_edge, hist_val in zip(bin_edges, hist, strict=True)
    })

    return stats


def get_n_objs_stats(seq_info: dict, params: Params):
    n_objs_list = np.asarray(seq_info['n_objs'])

    n_seq_frames = len(n_objs_list)
    assert n_seq_frames >= 1, "n_objs_list must have non-zero length"
    n_frames = seq_info['length']
    if params.stride == 1 and params.frame_gap == 1:
        assert n_seq_frames == n_frames, "n_seq_frames mismatch"

    seq_info['n_objects'] = np.sum(n_objs_list)
    seq_info['mean'] = np.mean(n_objs_list)
    seq_info['median'] = np.median(n_objs_list)
    seq_info['min'] = np.amin(n_objs_list)
    seq_info['max'] = np.amax(n_objs_list)

    # seq_len_threshs = [256 * i for i in range(2, 33)]
    # bbox_threshs = [int(seq_len // n_tokens_per_obj) for seq_len in seq_len_threshs]

    bbox_threshs = list(range(10, 51))

    n_exceed_list = [int(np.count_nonzero(n_objs_list > bbox_thresh)) for bbox_thresh in bbox_threshs]
    exceed_percent_list = [n_exceed / n_seq_frames * 100 for n_exceed in n_exceed_list]
    seq_info.update(
        {
            f'{k}': exceed_percent for k, exceed_percent in zip(
            # seq_len_threshs,
            bbox_threshs,
            exceed_percent_list)
        }
    )
    del (seq_info['n_objs'])


def run(params: Params):
    seq_paths = params.seq_paths
    root_dir = params.root_dir
    xml_root_dir = params.xml_root_dir

    load_samples = params.load_samples
    load_samples_root = params.load_samples_root
    load_samples_suffix = params.load_samples_suffix

    if len(load_samples) == 1:
        if load_samples[0] == 1:
            load_samples = ['seq_to_samples.txt', ]
        elif load_samples[0] == 0:
            load_samples = []

    class_names_path = params.class_names_path
    auto_class_cols = params.auto_class_cols
    map_classes = params.map_classes
    excluded_images_list = params.excluded_images_list
    description = params.description
    out_dir_name = params.out_dir_name
    out_json_name = params.out_json_name
    n_seq = params.n_seq
    start_seq_id = params.start_seq_id
    end_seq_id = params.end_seq_id

    assert description, "dataset description must be provided"

    if params.vis:
        assert params.n_proc <= 1, "visualization is not supported in multiprocessing mode"

    class_names = [k.strip() for k in open(class_names_path, 'r').readlines() if k.strip()]

    if map_classes:
        if auto_class_cols:
            class_names, mapped_class_names = zip(*[k.split('\t') for k in class_names])
        else:
            class_names, mapped_class_names, class_cols = zip(*[k.split('\t') for k in class_names])

        class_map_dict = {class_name: mapped_class_name for (class_name, mapped_class_name) in
                          zip(class_names, mapped_class_names, strict=True)}
    elif not auto_class_cols:
        class_names, class_cols = zip(*[k.split('\t') for k in class_names])

    n_classes = len(class_names)

    if auto_class_cols:
        class_cols = []
        class_cols_rgb = task_utils.get_cols_rgb(n_classes)

        for class_col_rgb in class_cols_rgb:
            r, g, b = class_col_rgb
            col_name = f'{b}_{g}_{r}'
            col_bgr[col_name] = (b, g, r)
            class_cols.append(col_name)

    if not map_classes:
        class_map_dict = {class_name: class_name for class_name in class_names}

    """class id 0 is for background"""
    class_dict = {x.strip(): i + 1 for (i, x) in enumerate(class_names)}
    if map_classes:
        """assign same class IDs to the mapped class names as the corresponding original class names
        and remove the original class names"""
        for (class_name, mapped_class_name) in class_map_dict.items():
            class_dict[mapped_class_name] = class_dict.pop(class_name)

    """background is class 0 with color black"""
    palette = [[0, 0, 0], ]
    for class_id in range(n_classes):
        col = class_cols[class_id]

        col_rgb = col_bgr[col][::-1]

        palette.append(col_rgb)

    # palette_flat = [value for color in palette for value in color]

    xml_dir_name = params.xml_dir_name
    if params.dir_suffix:
        xml_dir_name = f'{xml_dir_name}_{params.dir_suffix}'

    print(f'xml_dir_name: {xml_dir_name}')

    if params.start_frame_id > 0 or params.end_frame_id >= 0 or params.frame_stride > 1:
        frame_suffix = f'{params.start_frame_id}_{params.end_frame_id}'
        if params.frame_stride > 1:
            frame_suffix = f'{frame_suffix}_{params.frame_stride}'
        description = f'{description}-{frame_suffix}'

    if params.dir_suffix:
        print(f'dir_suffix: {params.dir_suffix}')
        description = f'{description}-{params.dir_suffix}'

    if params.length:
        print(f'setting max_length and min_length to {params.length}')
        params.max_length = params.min_length = params.length
        description = f'{description}-length-{params.length}'
        if not params.stride:
            print('setting stride to be equal to length')
            params.stride = params.length
    else:
        if params.max_length:
            print(f'max_length: {params.max_length}')
            description = f'{description}-max_length-{params.max_length}'
            if not params.stride:
                print('setting stride to be equal to max_length')
                params.stride = params.max_length

        if params.min_length:
            print(f'min_length: {params.min_length}')
            description = f'{description}-min_length-{params.max_length}'
            if not params.stride:
                print('setting stride to be equal to min_length')
                params.stride = params.min_length

    if params.stride:
        print(f'stride: {params.stride}')
        description = f'{description}-stride-{params.stride}'

    if params.frame_gap > 1:
        print(f'stride: {params.stride}')
        description = f'{description}-frame_gap-{params.frame_gap}'

    if params.incremental:
        print(f'saving incremental clips')
        description = f'{description}-incremental'

    if params.seq_paths_suffix:
        name_, ext_ = os.path.splitext(seq_paths)
        seq_paths = f'{name_}_{params.seq_paths_suffix}{ext_}'
        description = f'{description}-{params.seq_paths_suffix}'

    if load_samples_suffix:
        load_samples_suffix = '_'.join(load_samples_suffix)
        load_samples_root = f'{load_samples_root}_{load_samples_suffix}'
        description = f'{description}-{load_samples_suffix}'

    if load_samples:
        seq_paths, seq_to_samples = load_samples_from_txt(
            load_samples, xml_dir_name, load_samples_root,
            xml_root_dir=xml_root_dir, root_dir=root_dir,
        )
    else:
        seq_to_samples = None

        if seq_paths:
            if seq_paths.endswith('.txt'):
                assert os.path.isfile(seq_paths), f"nonexistent seq_paths file: {seq_paths}"

                seq_paths = [x.strip() for x in open(seq_paths).readlines() if x.strip()]
            else:
                seq_paths = seq_paths.split(',')
            if root_dir:
                seq_paths = [linux_path(root_dir, name) for name in seq_paths]

        elif root_dir:
            seq_paths = [linux_path(root_dir, name) for name in os.listdir(root_dir) if
                         os.path.isdir(linux_path(root_dir, name))]
            seq_paths.sort(key=sortKey)
        else:
            raise IOError('Either seq_paths or root_dir must be provided')

    n_seq_paths = len(seq_paths)
    if 0 < n_seq < n_seq_paths:
        assert end_seq_id < 0, "n_seq cannot be specified along with end_seq_id"
        end_seq_id = start_seq_id + n_seq - 1

    if start_seq_id > 0 or end_seq_id >= 0:
        if end_seq_id < 0:
            end_seq_id = n_seq_paths - 1
        assert end_seq_id < len(seq_paths), f"invalid end_seq_id: {end_seq_id} for n_seq_paths: {n_seq_paths}"
        seq_paths = seq_paths[start_seq_id:end_seq_id + 1]
        seq_suffix = f'seq-{start_seq_id}_{end_seq_id}'
        description = f'{description}-{seq_suffix}'
        print(f'start_seq_id: {start_seq_id}')
        print(f'end_seq_id: {end_seq_id}')

    if params.sample > 0:
        sample_suffix = f'sample-{params.sample}'
        description = f'{description}-{sample_suffix}'

    print(f'description: {description}')

    n_seq = len(seq_paths)
    assert n_seq > 0, "no sequences found"

    if xml_root_dir:
        assert root_dir, "root_dir must be provided with xml_root_dir"
        xml_dir_paths = [seq_path.replace(root_dir, xml_root_dir) for seq_path in seq_paths]
    else:
        xml_dir_paths = [linux_path(seq_path, xml_dir_name) for seq_path in seq_paths]

    # img_dir_paths = [linux_path(seq_path, params.img_dir_name) for seq_path in seq_paths]
    seq_names = [os.path.basename(seq_path) for seq_path in seq_paths]

    if params.subseq and not params.subseq_split_ids:
        assert not params.shuffle, "subseq cannot be reconstructed from shuffled xml files"

        subseq_info_suffix = 'subseq_info'
        sorted_files_suffix = 'sorted_src_files'
        if params.dir_suffix:
            subseq_info_suffix = f'{params.dir_suffix}_{subseq_info_suffix}'
            sorted_files_suffix = f'{params.dir_suffix}_{sorted_files_suffix}'

        sorted_files_paths = [f'{seq_path}_{sorted_files_suffix}.txt' for seq_path in seq_paths]
        subseq_info_paths = [f'{seq_path}_{subseq_info_suffix}.csv' for seq_path in seq_paths]
        subseq_info_list = [pd.read_csv(subseq_info_path) for subseq_info_path in subseq_info_paths]
        sorted_files_list = [open(sorted_files_path, 'r').read().splitlines() for sorted_files_path in
                             sorted_files_paths]
        n_subseq_list = [len(subseq_info) for subseq_info in subseq_info_list]

        vid_start_ids = [0, ] * n_seq

        for i in range(1, len(n_subseq_list)):
            vid_start_ids[i] = vid_start_ids[i - 1] + n_subseq_list[i - 1]

        n_vids = vid_start_ids[-1] + n_subseq_list[-1]
    else:
        vid_start_ids = list(range(n_seq))
        n_vids = n_seq
        subseq_info_list = [None, ] * n_vids

        sorted_files_list = [None, ] * n_vids

    _all_train_files = []
    all_val_files = []

    all_excluded_images = {}

    if not out_dir_name:
        out_dir_name = 'ytvis19'

    """folder containing all sequence folders"""
    if root_dir:
        db_root_dir = root_dir
    else:
        db_root_dir = os.path.dirname(seq_paths[0])

    ann_dir_path = linux_path(db_root_dir, out_dir_name, "Annotations")

    if params.save_masks:
        print(f'saving mask images to {ann_dir_path}')
        os.makedirs(ann_dir_path, exist_ok=True)
    else:
        print(f'not saving mask images')

    xml_data_list = list(zip(xml_dir_paths, sorted_files_list, seq_names, seq_paths,
                             subseq_info_list, vid_start_ids, strict=True))
    seq_name_to_xml_paths = {}

    import functools
    print(f'generating video clips from {n_vids} source videos...')
    for xml_data in tqdm(xml_data_list, position=0, leave=True):
        get_xml_files(
            params,
            excluded_images_list, all_excluded_images,
            all_val_files, _all_train_files,
            seq_name_to_xml_paths,
            seq_to_samples,
            xml_data)

    if params.sample > 0:
        print(f'sampling every {params.sample} video')
        _all_train_files = _all_train_files[::params.sample]
        all_val_files = all_val_files[::params.sample]

    n_train_vids = len(_all_train_files)
    n_val_vids = len(all_val_files)

    print(f'got {n_train_vids} training video clips')
    print(f'got {n_val_vids} validation video clips')

    if not out_json_name:
        out_json_name = description.lower().replace(' ', '_')

    time_stamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")

    info = {
        "version": "1.0",
        "year": datetime.now().strftime("%y"),
        "contributor": "asingh1",
        "date_created": time_stamp
    }

    licenses = [
        {
            "url": "https://creativecommons.org/licenses/by/4.0/",
            "id": 1,
            "name": "Creative Commons Attribution 4.0 License"
        }
    ]

    categories = []
    for label, label_id in class_dict.items():
        category_info = {
            'supercategory': 'object',
            'id': label_id,
            'name': label
        }
        categories.append(category_info)

    img_path_to_stats = {}

    if params.get_img_stats:
        stats_file_path = linux_path(db_root_dir, 'img_stats.txt')
        if os.path.exists(stats_file_path):
            print(f'reading img_stats from {stats_file_path}')
            img_stats = open(stats_file_path, 'r').read().splitlines()
            img_stats = [k.split('\t') for k in img_stats if k]
            img_path_to_stats = {
                img_stat[0]: (
                    list(paramparse.str_to_tuple(img_stat[1])),
                    list(paramparse.str_to_tuple(img_stat[2])),
                )
                for img_stat in img_stats
            }
            print(f'found stats for {len(img_path_to_stats)} images')

    all_files = {
        'train': _all_train_files,
        'val': all_val_files,
    }

    for split_type, all_split_files in all_files.items():

        n_split_vids = len(all_split_files)
        if not n_split_vids:
            print(f'no {split_type} videos found')
            continue

        all_data_xml_paths = list(set([item for sublist in all_split_files for item in sublist]))
        from collections import defaultdict, OrderedDict
        seq_name_to_info = OrderedDict()
        quant_bin_to_ious = defaultdict(list)

        read_xml_func = functools.partial(
            read_xml_file,
            params,
            db_root_dir,
            all_excluded_images,
            img_path_to_stats,
            seq_name_to_xml_paths,
            seq_name_to_info,
            quant_bin_to_ious,
            class_dict,
            class_map_dict,
        )
        print(f'reading {len(all_data_xml_paths)} {split_type} xml files')
        if params.n_proc > 1:
            print('running in parallel over {} processes'.format(params.n_proc))
            with multiprocessing.Pool(params.n_proc) as p:
                xml_out_data = list(tqdm(p.imap(read_xml_func, all_data_xml_paths),
                                         total=n_split_vids, ncols=100, position=0, leave=True))
        else:
            xml_out_data = []
            for xml_info in tqdm(all_data_xml_paths, ncols=100, position=0, leave=True):
                xml_out_data.append(read_xml_func(xml_info))
                # xml_out_data.append(None)

        n_objs_list_all = []
        n_tokens_per_obj = 4 * params.length + 1

        n_target_ids_all = 0

        for __id, (seq_name, seq_info) in enumerate(seq_name_to_info.items()):
            n_objs_list_all += seq_info['n_objs']

            target_ids = seq_info['target_ids']
            target_ids = list(set(target_ids))
            n_target_ids = len(target_ids)
            n_target_ids_all += n_target_ids
            seq_info['n_target_ids'] = n_target_ids
            del (seq_info['target_ids'])
            get_n_objs_stats(seq_info, params)

        seq_info_all = dict(
            name='__all__',
            height=0,
            width=0,
            aspect_ratio=0,
            length=len(n_objs_list_all),
            n_objs=n_objs_list_all,
            n_target_ids=n_target_ids_all,
        )

        get_n_objs_stats(seq_info_all, params)
        seq_name_to_info['__all__'] = seq_info_all
        seq_name_to_info.move_to_end('__all__', last=False)

        seq_name_to_info_df = pd.DataFrame.from_dict(seq_name_to_info, orient='index')
        seq_name_to_info_dir = os.path.join(db_root_dir, out_dir_name)
        os.makedirs(seq_name_to_info_dir, exist_ok=True)
        seq_name_to_info_csv = os.path.join(seq_name_to_info_dir, f"{out_json_name}-seq_name_to_info.csv")
        print(f'\nseq_name_to_info_csv: {seq_name_to_info_csv}\n')
        seq_name_to_info_df.to_csv(seq_name_to_info_csv, index=False)

        for quant_bin, ious in quant_bin_to_ious.items():
            quant_bin_to_ious[quant_bin] = get_iou_stats(np.asarray(ious))

        quant_bin_to_ious_df = pd.DataFrame.from_dict(quant_bin_to_ious, orient='columns')
        quant_bin_to_ious_csv = os.path.join(seq_name_to_info_dir, f"{out_json_name}-quant_bin_to_ious.csv")
        print(f'\nquant_bin_to_ious_csv: {quant_bin_to_ious_csv}\n')
        quant_bin_to_ious_df.to_csv(quant_bin_to_ious_csv, index=True)

        xml_data_dict = {}

        all_pix_vals_mean = []
        all_pix_vals_std = []

        for xml_out_datum in xml_out_data:
            if xml_out_datum is None:
                continue
            xml_path, xml_dict, pix_vals_mean, pix_vals_std = xml_out_datum
            xml_data_dict[xml_path] = xml_dict
            all_pix_vals_mean += pix_vals_mean
            all_pix_vals_std += pix_vals_std

        # use_tqdm = params.n_proc <= 1
        use_tqdm = 0
        json_func = functools.partial(
            save_annotations_ytvis,
            xml_data_dict,
            class_dict,
            params.ignore_invalid_label,
            params.infer_target_id,
            use_tqdm,
            params.enable_masks,
            params.vis,
        )

        print(f'generating json data for {n_split_vids} {split_type} videos')
        vid_ids = list(range(1, n_split_vids + 1))
        vid_info_list = list(zip(all_split_files, vid_ids))

        if params.n_proc > 1:
            print('running in parallel over {} processes'.format(params.n_proc))
            with multiprocessing.Pool(params.n_proc) as p:
                json_out_data = list(tqdm(p.imap(json_func, vid_info_list),
                                          total=n_split_vids, ncols=100, position=0, leave=True))
        else:
            json_out_data = []
            for vid_info in tqdm(vid_info_list, ncols=100, position=0, leave=True):
                json_out_data.append(json_func(vid_info))

        videos = []
        annotations = []
        vid_to_target_ids = {}

        for vid_id, k1, k2, k3 in json_out_data:
            videos.append(k1)
            annotations += k2
            vid_to_target_ids[vid_id] = k3

        offset_target_ids(vid_to_target_ids, annotations, f'{split_type}')
        if params.val_ratio > 0:
            info["description"] = description + f'-{split_type}'
        else:
            info["description"] = description

        info["counts"] = dict(
            videos=len(videos),
            annotations=len(annotations),
        ),
        json_dict = {
            "info": info,
            "licenses": licenses,
            "videos": videos,
            "categories": categories,
            "annotations": annotations,
        }
        n_xml = len(all_split_files)

        if n_val_vids > 0:
            json_name = f'{out_json_name}-{split_type}.json'
        else:
            json_name = f'{out_json_name}.json'

        if params.json_gz:
            json_name += '.gz'

        json_path = linux_path(db_root_dir, out_dir_name, json_name)

        json_dir = os.path.dirname(json_path)
        os.makedirs(json_dir, exist_ok=True)

        json_kwargs = dict(
            indent=4
        )
        print(f'saving json for {n_xml} {split_type} images to: {json_path}')

        if params.json_gz:
            import compress_json
            compress_json.dump(json_dict, json_path, json_kwargs=json_kwargs)
        else:
            output_json = json.dumps(json_dict, **json_kwargs)
            with open(json_path, 'w') as f:
                f.write(output_json)

        if params.get_img_stats:
            pix_vals_mean = list(np.mean(all_pix_vals_mean, axis=0))
            pix_vals_std = list(np.mean(all_pix_vals_std, axis=0))  #
            print(f'pix_vals_mean: {pix_vals_mean}')
            print(f'pix_vals_std: {pix_vals_std}')

            if not os.path.exists(stats_file_path) and img_path_to_stats:
                print(f'writing stats for {len(img_path_to_stats)} images to {stats_file_path}')
                with open(stats_file_path, 'w') as fid:
                    for img_path, img_stat in tqdm(img_path_to_stats.items(), position=0, leave=True):
                        pix_vals_mean, pix_vals_std = img_stat
                        fid.write(f'{img_path}\t{pix_vals_mean}\t{pix_vals_std}\n')


def main():
    params: Params = paramparse.process(Params)
    if params.strides:
        if len(params.strides) == 1 and params.strides[0] == 0:
            params.strides = list(range(1, params.length + 1))
        for stride in params.strides:
            params.stride = stride
            run(params)
    else:
        run(params)


if __name__ == '__main__':
    main()
