import os
import glob
import random
import ast

import pandas as pd
import numpy as np
import cv2

from pprint import pformat

import json
import xml.etree.ElementTree as ET
from datetime import datetime
from tqdm import tqdm

import imagesize
import paramparse

from eval_utils import col_bgr, sortKey, linux_path

from itertools import groupby
import pycocotools.mask as mask_util

import multiprocessing


# from multiprocessing.pool import ThreadPool


class Params:
    def __init__(self):
        self.cfg = ()
        self.batch_size = 1
        self.description = ''
        self.excluded_images_list = ''
        self.class_names_path = 'lists/classes///predefined_classes_orig.txt'
        self.codec = 'H264'
        self.csv_file_name = ''
        self.fps = 20
        self.img_ext = 'png'
        self.load_path = ''
        self.load_samples = []
        self.load_samples_root = ''
        self.map_folder = ''
        self.min_val = 0
        self.recursive = 0
        self.n_classes = 4
        self.n_frames = 0
        self.img_dir_name = 'images'
        self.xml_dir_name = 'annotations'
        self.dir_suffix = ''
        self.out_dir_name = ''
        self.n_seq = 0
        self.incremental = 0

        self.out_json_name = ''
        self.root_dir = ''
        self.save_file_name = ''
        self.save_video = 1
        self.seq_paths = ''
        self.show_img = 0
        self.shuffle = 0
        self.subseq = 0
        self.subseq_split_ids = []
        self.sources_to_include = []
        self.val_ratio = 0
        self.save_masks = 0
        self.allow_missing_images = 0
        self.remove_mj_dir_suffix = 0
        self.infer_target_id = 0
        self.get_img_stats = 1
        self.ignore_invalid_label = 0
        self.start_frame_id = 0
        self.end_frame_id = -1
        self.max_length = 0
        self.min_length = 0
        self.coco_rle = 0
        self.n_proc = 1


def offset_target_ids(vid_to_target_ids, annotations, set_type):
    print(f'adding offsets to {set_type} targets IDs')
    vid_ids = sorted(list(vid_to_target_ids.keys()))
    target_id_offsets = {}
    curr_offset = 0
    for vid_id in vid_ids:
        target_id_offsets[vid_id] = curr_offset
        curr_offset += len(vid_to_target_ids[vid_id])

    for annotation in tqdm(annotations):
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


def read_xml_file(db_root_dir, excluded_images, allow_missing_images, coco_rle,
                  get_img_stats, img_path_to_stats, remove_mj_dir_suffix, xml_data):
    xml_path, seq_path, seq_name = xml_data

    all_pix_vals_mean = []
    all_pix_vals_std = []

    ann_tree = ET.parse(xml_path)
    ann_root = ann_tree.getroot()

    """img_path is relative to db_root_dir"""
    img_rel_path = ann_tree.findtext('path')
    filename = ann_tree.findtext('filename')
    if img_rel_path is None:
        img_rel_path = linux_path(seq_name, filename)
    else:
        _filename = os.path.basename(img_rel_path)
        assert _filename == filename, f"mismatch between filename: {filename} and path: {img_rel_path}"

    img_name = filename
    img_path = linux_path(db_root_dir, img_rel_path)

    if remove_mj_dir_suffix:
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
        if allow_missing_images:
            print('\n' + msg + '\n')
            return None
        else:
            raise AssertionError(msg)

    w, h = imagesize.get(img_path)

    size_from_xml = ann_root.find('size')
    w_from_xml = int(size_from_xml.findtext('width'))
    h_from_xml = int(size_from_xml.findtext('height'))

    assert h_from_xml == h and w_from_xml == w, f"incorrect image dimensions in XML: {(h_from_xml, w_from_xml)}"

    if get_img_stats:
        try:
            img_stat = img_path_to_stats[img_path]
        except KeyError:
            img = cv2.imread(img_path)
            h, w = img.shape[:2]

            img_reshaped = np.reshape(img, (h * w, 3))

            pix_vals_mean = list(np.mean(img_reshaped, axis=0))
            pix_vals_std = list(np.std(img_reshaped, axis=0))

            img_path_to_stats[img_path] = (pix_vals_mean, pix_vals_std)
        else:
            pix_vals_mean, pix_vals_std = img_stat

        all_pix_vals_mean.append(pix_vals_mean)
        all_pix_vals_std.append(pix_vals_std)

    objs = ann_root.findall('object')

    src_shape = (h, w)

    xml_dict = dict(
        img_path=img_path,
        img_rel_path=img_rel_path,
        filename=filename,
        src_shape=src_shape,
    )

    xml_data_list = []

    for obj_id, obj in enumerate(objs):

        label = obj.findtext('name')
        target_id = int(obj.findtext('id_number'))

        bbox = obj.find('bndbox')
        # score = float(obj.find('score').text)
        # difficult = int(obj.find('difficult').text)
        # bbox_source = obj.find('bbox_source').text
        xmin = bbox.find('xmin')
        ymin = bbox.find('ymin')
        xmax = bbox.find('xmax')
        ymax = bbox.find('ymax')

        xmin, ymin, xmax, ymax = float(xmin.text), float(ymin.text), float(xmax.text), float(ymax.text)
        w, h = xmax - xmin, ymax - ymin

        bbox = [xmin, ymin, w, h]

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
        if coco_rle:
            bin_mask_rle = binary_mask_to_rle_coco(bin_mask)
        else:
            bin_mask_rle = binary_mask_to_rle(bin_mask)

        seg_area = np.count_nonzero(bin_mask)

        xml_data = dict(
            label=label,
            target_id=target_id,
            bbox=bbox,
            bin_mask_rle=bin_mask_rle,
            seg_area=seg_area,
        )

        xml_data_list.append(xml_data)

    xml_dict['objs'] = xml_data_list

    return xml_path, xml_dict, all_pix_vals_mean, all_pix_vals_std


def save_annotations_ytvis(
        xml_data_dict,
        class_to_id,
        ignore_invalid_label,
        infer_target_id,
        use_tqdm,
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

    ann_objs = {}
    sec_ann_objs = {}

    n_files = len(xml_files)

    if use_tqdm:
        pbar = tqdm(xml_files, ncols=100)
    else:
        pbar = xml_files

    vid_seq_path = vid_seq_name = None
    next_target_id = 0

    target_ids = []

    for frame_id, (xml_path, seq_path, seq_name) in enumerate(pbar):

        xml_data = xml_data_dict[xml_path]

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
        # img_path = xml_data['img_path']
        src_shape = xml_data['src_shape']

        h, w = src_shape

        if vid_size is None:
            vid_size = (w, h)
        else:
            assert vid_size == (w, h), f"mismatch between size of image: {(w, h)} and video: {vid_size}"

        file_names.append(img_rel_path)

        objs = xml_data['objs']

        for obj_id, obj in enumerate(objs):

            label = obj['label']
            target_id = obj['target_id']
            bbox = obj['bbox']
            bin_mask_rle = obj['bin_mask_rle']
            seg_area = obj['seg_area']

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
                    'segmentations': [None, ] * n_files,
                    'bboxes': [None, ] * n_files,
                    'areas': [None, ] * n_files,

                    'bin_mask_rle': [None, ] * n_files,
                    'mask_pts': [None, ] * n_files,
                }
                if infer_target_id:
                    sec_ann_objs[target_id] = {
                        'category_id': category_id,
                        'bboxes': [None, ] * n_files,
                        'bin_mask_rle': [None, ] * n_files,
                        'bin_mask': [None, ] * n_files,
                    }

            curr_ann_obj = ann_objs[target_id]

            curr_ann_obj['bboxes'][frame_id] = bbox
            curr_ann_obj['segmentations'][frame_id] = bin_mask_rle
            curr_ann_obj['areas'][frame_id] = seg_area

            if infer_target_id:
                sec_curr_ann_obj = sec_ann_objs[target_id]
                sec_curr_ann_obj['bboxes'][frame_id] = bbox
                sec_curr_ann_obj['bin_mask_rle'][frame_id] = bin_mask_rle
                sec_curr_ann_obj['bin_mask'][frame_id] = bin_mask

            n_objs += 1

            label_to_n_objs[label] += 1

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
            "segmentations": ann_obj['segmentations'],
            "bboxes": ann_obj['bboxes'],
            "iscrowd": ann_obj['iscrowd'],
            "areas": ann_obj['areas'],
        }

        all_annotations.append(annotations_dict)

    video_dict = {
        "width": vid_w,
        "height": vid_h,
        "length": n_valid_images,
        "date_captured": "",
        "license": 1,
        "flickr_url": "",
        "file_names": file_names,
        "id": vid_id,
        "coco_url": "",
    }

    return vid_id, video_dict, all_annotations, target_ids


def get_xml_files(
        params: Params,
        excluded_images_list, all_excluded_images,
        all_val_files, all_train_files,
        xml_data
):
    xml_dir_path, all_xml_files, seq_name, seq_path, subseq_info, vid_start_id = xml_data

    if all_xml_files is None:
        # xml_file_gen = [[os.path.join(dirpath, f) for f in filenames if
        #                  os.path.splitext(f.lower())[1] == '.xml']
        #                 for (dirpath, dirnames, filenames) in os.walk(xml_dir_path, followlinks=True)]
        # all_xml_files = [item for sublist in xml_file_gen for item in sublist]

        all_xml_files = glob.glob(linux_path(xml_dir_path, '*.xml'), recursive=params.recursive)

        if params.shuffle:
            random.shuffle(all_xml_files)
        else:
            all_xml_files.sort(key=lambda fname: os.path.basename(fname))

    start_frame_id = params.start_frame_id
    end_frame_id = params.end_frame_id

    if end_frame_id < start_frame_id:
        end_frame_id = len(all_xml_files) - 1

    all_xml_files = all_xml_files[start_frame_id:end_frame_id + 1]

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

        pbar = tqdm(subseq_start_end_ids)

        global_subseq_id = 0

        for subseq_id, (start_id, end_id) in enumerate(pbar):
            assert end_id < n_all_files, f"invalid end_id: {end_id} for n_files: {n_all_files}" \
                f" in subseq_id {subseq_id} of {xml_dir_path}"

            subseq_xml_files = all_xml_files[start_id:end_id + 1]
            n_subseq_xml_files = len(subseq_xml_files)
            if n_subseq_xml_files < params.min_length:
                print(f'skipping {seq_name} subseq {subseq_id + 1} ({start_id} -> {end_id})'
                      f' with length {n_subseq_xml_files} < {params.min_length}')
                continue
            elif n_subseq_xml_files > params.max_length > 0:
                n_subsubseq = int(round(float(n_subseq_xml_files) / params.max_length))
                subsubseq_start_id = 0
                for subsubseq_id in range(n_subsubseq):

                    subsubseq_end_id = min(subsubseq_start_id + params.max_length - 1, n_subseq_xml_files - 1)

                    if subsubseq_start_id > subsubseq_end_id:
                        break

                    subsubseq_xml_files = subseq_xml_files[subsubseq_start_id:subsubseq_end_id + 1]

                    n_subsubseq_xml_files = len(subsubseq_xml_files)

                    print(f'\t{seq_name} :: subseq {global_subseq_id + 1} '
                          f'({subseq_id + 1} - {subsubseq_id + 1} / {n_subsubseq})  length {n_subsubseq_xml_files} '
                          f'({start_id + subsubseq_start_id} -> {start_id + subsubseq_end_id})')

                    assert len(subsubseq_xml_files) > 0, "no subsubseq_xml_files found"

                    all_subseq_xml_files.append(subsubseq_xml_files)

                    subsubseq_start_id = subsubseq_end_id + 1

                    global_subseq_id += 1
            else:
                print(f'{seq_name}: subseq {global_subseq_id + 1} length {n_subseq_xml_files} ({start_id} -> {end_id})')
                assert len(subseq_xml_files) > 0, "no subseq_xml_files found"

                all_subseq_xml_files.append(subseq_xml_files)
                global_subseq_id += 1
    else:
        if n_all_files < params.min_length:
            print(f'skipping {seq_name} with length {n_all_files} < {params.min_length}')
            return
        elif n_all_files > params.max_length > 0:
            n_subseq = int(round(float(n_all_files) / params.max_length))
            subseq_start_id = 0
            for subseq_id in range(n_subseq):

                subseq_end_id = min(subseq_start_id + params.max_length - 1, n_all_files - 1)

                if subseq_start_id > subseq_end_id:
                    break

                subseq_xml_files = all_xml_files[subseq_start_id:subseq_end_id + 1]

                n_subseq_xml_files = len(subseq_xml_files)

                print(f'{seq_name} :: subseq {subseq_id + 1} length {n_subseq_xml_files} '
                      f'({subseq_start_id} -> {subseq_end_id})')

                assert n_subseq_xml_files > 0, "no subseq_xml_files found"

                all_subseq_xml_files.append(subseq_xml_files)

                subseq_start_id = subseq_end_id + 1

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

        subseq_xml_files = tuple(zip(subseq_xml_files, [seq_path, ] * n_files, [seq_name, ] * n_files))

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


def main():
    params = Params()

    paramparse.process(params)

    seq_paths = params.seq_paths
    root_dir = params.root_dir
    load_samples = params.load_samples
    load_samples_root = params.load_samples_root
    class_names_path = params.class_names_path
    excluded_images_list = params.excluded_images_list
    description = params.description
    out_dir_name = params.out_dir_name
    out_json_name = params.out_json_name
    n_seq = params.n_seq

    assert description, "dataset description must be provided"

    if params.dir_suffix:
        print(f'dir_suffix: {params.dir_suffix}')
        description = f'{description}-{params.dir_suffix}'

    if params.max_length:
        print(f'max_length: {params.max_length}')
        description = f'{description}-max_length-{params.max_length}'

    if params.incremental:
        print(f'saving incremental clips')
        description = f'{description}-incremental'

    if seq_paths:
        if os.path.isfile(seq_paths):
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

    if 0 < n_seq < len(seq_paths):
        seq_paths = seq_paths[:n_seq]

    n_seq = len(seq_paths)
    assert n_seq > 0, "no sequences found"

    seq_to_samples = {}

    if len(load_samples) == 1:
        if load_samples[0] == 1:
            load_samples = ['seq_to_samples.txt', ]
        elif load_samples[0] == 0:
            load_samples = []

    if load_samples:
        # if load_samples == '1':
        #     load_samples = 'seq_to_samples.txt'
        print('load_samples: {}'.format(pformat(load_samples)))
        if load_samples_root:
            load_samples = [linux_path(load_samples_root, k) for k in load_samples]
        print('Loading samples from : {}'.format(load_samples))
        for _f in load_samples:
            if os.path.isdir(_f):
                _f = linux_path(_f, 'seq_to_samples.txt')
            with open(_f, 'r') as fid:
                curr_seq_to_samples = ast.literal_eval(fid.read())
                for _seq in curr_seq_to_samples:
                    if _seq in seq_to_samples:
                        seq_to_samples[_seq] += curr_seq_to_samples[_seq]
                    else:
                        seq_to_samples[_seq] = curr_seq_to_samples[_seq]

    class_names = [k.strip() for k in open(class_names_path, 'r').readlines() if k.strip()]
    class_names, class_cols = zip(*[k.split('\t') for k in class_names])

    """class id 0 is for background"""
    class_dict = {x.strip(): i + 1 for (i, x) in enumerate(class_names)}

    n_classes = len(class_cols)
    """background is class 0 with color black"""
    palette = [[0, 0, 0], ]
    for class_id in range(n_classes):
        col = class_cols[class_id]

        col_rgb = col_bgr[col][::-1]

        palette.append(col_rgb)

    palette_flat = [value for color in palette for value in color]

    xml_dir_name = params.xml_dir_name
    if params.dir_suffix:
        xml_dir_name = f'{xml_dir_name}_{params.dir_suffix}'

    print(f'xml_dir_name: {xml_dir_name}')

    n_seq = len(seq_paths)

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
        os.makedirs(ann_dir_path, exist_ok=1)
    else:
        print(f'not saving mask images')

    xml_data_list = list(zip(xml_dir_paths, sorted_files_list, seq_names, seq_paths, subseq_info_list, vid_start_ids))

    import functools
    print(f'generating video clips from {n_vids} source videos...')
    for xml_data in tqdm(xml_data_list):
        get_xml_files(
            params,
            excluded_images_list, all_excluded_images,
            all_val_files, _all_train_files,
            xml_data)

    n_train_vids = len(_all_train_files)
    n_val_vids = len(all_val_files)

    print(f'got {n_train_vids} training video clips')
    print(f'got {n_val_vids} validation video clips')

    if not out_json_name:
        out_json_name = description.lower().replace(' ', '_')

    time_stamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")

    info = {
        "version": "1.0",
        "year": 2022,
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

    for div_type, all_div_files in all_files.items():

        n_div_vids = len(all_div_files)
        if not n_div_vids:
            print(f'no {div_type} videos found')
            continue

        all_data_xml_paths = list(set([item for sublist in all_div_files for item in sublist]))

        read_xml_func = functools.partial(
            read_xml_file,
            db_root_dir,
            all_excluded_images,
            params.allow_missing_images,
            params.coco_rle,
            params.get_img_stats,
            img_path_to_stats,
            params.remove_mj_dir_suffix
        )
        print(f'reading {len(all_data_xml_paths)} {div_type} xml files')
        if params.n_proc > 1:
            print('running in parallel over {} processes'.format(params.n_proc))
            with multiprocessing.Pool(params.n_proc) as p:
                xml_out_data = list(tqdm(p.imap(read_xml_func, all_data_xml_paths), total=n_div_vids, ncols=100))
        else:
            xml_out_data = []
            for xml_info in tqdm(all_data_xml_paths, ncols=100):
                xml_out_data.append(read_xml_func(xml_info))

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

        json_func = functools.partial(
            save_annotations_ytvis,
            xml_data_dict,
            class_dict,
            params.ignore_invalid_label,
            params.infer_target_id,
            False,
        )

        print(f'generating json data for {n_div_vids} {div_type} videos')
        vid_ids = list(range(1, n_div_vids + 1))
        vid_info_list = list(zip(all_div_files, vid_ids))

        if params.n_proc > 1:
            print('running in parallel over {} processes'.format(params.n_proc))
            with multiprocessing.Pool(params.n_proc) as p:
                json_out_data = list(tqdm(p.imap(json_func, vid_info_list), total=n_div_vids, ncols=100))
        else:
            json_out_data = []
            for vid_info in tqdm(vid_info_list, ncols=100):
                json_out_data.append(json_func(vid_info))

        videos = []
        annotations = []
        vid_to_target_ids = {}

        for vid_id, k1, k2, k3 in json_out_data:
            videos.append(k1)
            annotations += k2
            vid_to_target_ids[vid_id] = k3

        offset_target_ids(vid_to_target_ids, annotations, f'{div_type}')
        info["description"] = description + f'-{div_type}'
        json_dict = {
            "info": info,
            "licenses": licenses,
            "videos": videos,
            "categories": categories,
            "annotations": annotations,
        }
        n_xml = len(all_div_files)

        if n_val_vids > 0:
            json_name = f'{out_json_name}-{div_type}.json'
        else:
            json_name = f'{out_json_name}.json'

        json_path = linux_path(db_root_dir, out_dir_name, json_name)

        json_dir = os.path.dirname(json_path)
        os.makedirs(json_dir, exist_ok=1)

        print(f'saving json for {n_xml} {div_type} images to: {json_path}')
        with open(json_path, 'w') as f:
            output_json = json.dumps(json_dict, indent=4)
            f.write(output_json)

        if params.get_img_stats:
            pix_vals_mean = list(np.mean(all_pix_vals_mean, axis=0))
            pix_vals_std = list(np.mean(all_pix_vals_std, axis=0))  #
            print(f'pix_vals_mean: {pix_vals_mean}')
            print(f'pix_vals_std: {pix_vals_std}')

            if not os.path.exists(stats_file_path) and img_path_to_stats:
                print(f'writing stats for {len(img_path_to_stats)} images to {stats_file_path}')
                with open(stats_file_path, 'w') as fid:
                    for img_path, img_stat in tqdm(img_path_to_stats.items()):
                        pix_vals_mean, pix_vals_std = img_stat
                        fid.write(f'{img_path}\t{pix_vals_mean}\t{pix_vals_std}\n')


if __name__ == '__main__':
    main()
