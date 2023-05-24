import os
import glob
import random
import ast
import re

import numpy as np
import cv2
from PIL import Image

from pprint import pformat

import json
import xml.etree.ElementTree as ET
from typing import Dict, List
from tqdm import tqdm
import imagesize

import paramparse

from eval_utils import sortKey, col_bgr, linux_path


class Params:
    def __init__(self):
        self.cfg = ()
        self.batch_size = 1
        self.excluded_images_list = ''
        self.class_names_path = 'lists/classes///predefined_classes_orig.txt'
        self.codec = 'H264'
        self.csv_file_name = ''
        self.enable_mask = 2
        self.extract_num_from_imgid = 0
        self.fps = 20
        self.img_ext = 'png'
        self.load_path = ''
        self.load_samples = []
        self.load_samples_root = ''
        self.map_folder = ''
        self.min_val = 0
        self.n_classes = 4
        self.n_frames = 0
        self.dir_name = 'annotations'
        self.dir_suffix = ''
        self.output_json = 'coco.json'
        self.root_dir = ''
        self.seq_paths = ''
        self.show_img = 0
        self.shuffle = 0
        self.sources_to_include = []
        self.val_ratio = 0
        self.no_annotations = 0
        self.allow_missing_images = 0
        self.remove_mj_dir_suffix = 0
        self.get_img_stats = 1
        self.write_masks = 0
        self.mask_dir_name = 'masks'
        self.ignore_invalid_label = 0

        self.start_frame_id = 0
        self.end_frame_id = -1

        self.n_seq = 0
        self.only_list = 0


def get_image_info(seq_name, annotation_root, extract_num_from_imgid=0):
    rel_path = annotation_root.findtext('path')
    if rel_path is None:
        filename = annotation_root.findtext('filename')
        rel_path = linux_path(seq_name, filename)
    else:
        filename = os.path.basename(rel_path)

    # folder = annotation_root.findtext('folder')
    # path = linux_path(folder, rel_path)

    img_name = os.path.basename(filename)

    img_id = os.path.splitext(img_name)[0]
    if extract_num_from_imgid and isinstance(img_id, str):
        img_id = int(re.findall(r'\d+', img_id)[0])

    size = annotation_root.find('size')
    width = int(size.findtext('width'))
    height = int(size.findtext('height'))

    image_info = {
        'file_name': rel_path,
        'height': height,
        'width': width,
        'id': seq_name + '/' + img_id
    }
    return image_info


def get_coco_annotation_from_obj(obj, label2id, enable_mask, ignore_invalid_label):
    label = obj.findtext('name')

    if label not in label2id:
        msg = f"label {label} is not in label2id"
        if ignore_invalid_label:
            print(msg)
            return None
        else:
            raise AssertionError(msg)

    category_id = label2id[label]
    bndbox = obj.find('bndbox')
    xmin = int(float(bndbox.findtext('xmin'))) - 1
    ymin = int(float(bndbox.findtext('ymin'))) - 1
    xmax = int(float(bndbox.findtext('xmax')))
    ymax = int(float(bndbox.findtext('ymax')))
    assert xmax > xmin and ymax > ymin, f"Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
    o_width = xmax - xmin
    o_height = ymax - ymin
    ann = {
        'area': o_width * o_height,
        'iscrowd': 0,
        'bbox': [xmin, ymin, o_width, o_height],
        'label': label,
        'category_id': category_id,
        'ignore': 0,
    }
    if enable_mask:
        mask_obj = obj.find('mask')
        if mask_obj is None:
            msg = 'no mask found for object:\n{}'.format(ann)
            if enable_mask == 2:
                # print(msg)
                """ignore mask-less objects"""
                return None
            else:
                raise AssertionError(msg)
        mask = mask_obj.text

        mask = [k.strip().split(',') for k in mask.strip().split(';') if k]
        # pprint(mask)
        mask_pts = []
        mask_pts_flat = []
        for _pt in mask:
            _pt_float = [float(_pt[0]), float(_pt[1])]
            mask_pts_flat.append(_pt_float[0])
            mask_pts_flat.append(_pt_float[1])

            mask_pts.append(_pt_float)

        ann.update({
            'segmentation': [mask_pts_flat, ],
            'mask_pts': mask_pts
        }
        )
    return ann


def save_boxes_coco(annotation_paths: List[str],
                    label2id: Dict[str, int],
                    output_json: str,
                    extract_num_from_imgid: int,
                    enable_mask: int,
                    allow_missing_images: int,
                    ignore_invalid_label: int,
                    remove_mj_dir_suffix: int,
                    get_img_stats: int,
                    write_masks: int,
                    list_path: str,
                    mask_dir_name: str,
                    palette_flat: list,
                    only_list: int,
                    excluded_images=None
                    ):
    output_json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": []
    }
    bnd_id = 1  # START_BOUNDING_BOX_ID, TODO input as args ?
    pbar = tqdm(annotation_paths)

    out_root_dir = os.path.dirname(output_json)

    img_path_to_stats = {}
    if get_img_stats:
        stats_file_path = linux_path(out_root_dir, 'img_stats.txt')
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

        all_pix_vals_mean = []
        all_pix_vals_std = []

    n_valid_images = 0
    n_images = 0

    n_objs = 0

    label_to_n_objs = {
        label: 0 for label in label2id
    }

    img_paths = []

    for xml_path, seq_path in pbar:
        seq_name = os.path.basename(seq_path)

        n_images += 1

        # Read annotation xml
        ann_tree = ET.parse(xml_path)
        ann_root = ann_tree.getroot()

        img_info = get_image_info(
            seq_name=seq_name,
            annotation_root=ann_root,
            extract_num_from_imgid=extract_num_from_imgid)

        img_file_rel_path = img_info['file_name']
        img_file_name = os.path.basename(img_file_rel_path)

        if remove_mj_dir_suffix:
            img_file_rel_path_list = img_file_rel_path.split('/')
            img_file_rel_path = linux_path(img_file_rel_path_list[0], img_file_rel_path_list[1], img_file_name)

        # img_file_path = os.path.join(seq_path, img_file_name)

        if excluded_images is not None and img_file_name in excluded_images[seq_path]:
            print(f'\n{seq_name} :: skipping excluded image {img_file_name}')
            continue

        img_file_path = linux_path(out_root_dir, img_file_rel_path)

        if not os.path.exists(img_file_path):
            print(f'img_file_rel_path: {img_file_rel_path}')
            print(f'out_root_dir: {out_root_dir}')

            msg = f"img_file_path does not exist: {img_file_path}"
            if allow_missing_images:
                print('\n' + msg + '\n')
                continue
            else:
                raise AssertionError(msg)

        img_paths.append(img_file_path)

        if only_list:
            continue

        img_fname_noext = os.path.splitext(img_file_name)[0]
        img_dir_path = os.path.dirname(img_file_path)

        if write_masks:
            mask_dir_path = linux_path(img_dir_path, mask_dir_name)
            os.makedirs(mask_dir_path, exist_ok=True)

            width, height = imagesize.get(str(img_file_path))
            mask_img = np.zeros((height, width), dtype=np.uint8)

        if get_img_stats:
            try:
                img_stat = img_path_to_stats[img_file_path]
            except KeyError:
                img = cv2.imread(img_file_path)
                h, w = img.shape[:2]

                assert img_info['height'] == h and img_info['width'] == w, "incorrect image dimensions in XML"

                img_reshaped = np.reshape(img, (h * w, 3))

                pix_vals_mean = list(np.mean(img_reshaped, axis=0))
                pix_vals_std = list(np.std(img_reshaped, axis=0))
                with open(stats_file_path, 'a') as fid:
                    fid.write(f'{img_file_path}\t{pix_vals_mean}\t{pix_vals_std}\n')
            else:
                pix_vals_mean, pix_vals_std = img_stat

                # print(f'{img_file_path} : {pix_vals_mean}, {pix_vals_std}')

            all_pix_vals_mean.append(pix_vals_mean)
            all_pix_vals_std.append(pix_vals_std)

        img_id = img_info['id']
        output_json_dict['images'].append(img_info)

        objs = ann_root.findall('object')

        # print()
        for obj_id, obj in enumerate(objs):
            ann = get_coco_annotation_from_obj(obj=obj, label2id=label2id, enable_mask=enable_mask,
                                               ignore_invalid_label=ignore_invalid_label)
            if ann is None:
                print(f'\nskipping object {obj_id + 1} in {xml_path}')
                continue

            ann.update(
                {
                    'image_id': img_id,
                    'id': bnd_id
                }
            )
            output_json_dict['annotations'].append(ann)
            n_objs += 1

            label_to_n_objs[ann['label']] += 1

            bnd_id += 1

            if write_masks:
                category_id = ann['category_id']
                """0 is background"""
                class_id = category_id + 1
                mask_pts = ann['mask_pts']
                mask_pts_arr = np.array([mask_pts, ], dtype=np.int32)
                mask_img = cv2.fillPoly(mask_img, mask_pts_arr, class_id)

            if enable_mask:
                del ann['mask_pts']

        n_valid_images += 1
        desc = f'{n_valid_images} / {n_images} valid images :: {n_objs} objects '
        for label in label2id:
            desc += f' {label}: {label_to_n_objs[label]}'

        if write_masks:
            mask_img_pil = Image.fromarray(mask_img)
            mask_img_pil = mask_img_pil.convert('P')
            mask_img_pil.putpalette(palette_flat)

            mask_fname = f'{img_fname_noext}.png'
            mask_path = linux_path(mask_dir_path, mask_fname)
            mask_parent_path = os.path.dirname(mask_path)
            os.makedirs(mask_parent_path, exist_ok=1)
            mask_img_pil.save(mask_path)

        pbar.set_description(desc)

    print(f'saving img list to {list_path}')
    with open(list_path, 'w') as fid:
        fid.write('\n'.join(img_paths))

    if only_list:
        return

    for label, label_id in label2id.items():
        category_info = {'supercategory': 'none', 'id': label_id, 'name': label}
        output_json_dict['categories'].append(category_info)

    if get_img_stats:
        pix_vals_mean = list(np.mean(all_pix_vals_mean, axis=0))
        pix_vals_std = list(np.mean(all_pix_vals_std, axis=0))  #
        print(f'pix_vals_mean: {pix_vals_mean}')
        print(f'pix_vals_std: {pix_vals_std}')

    with open(output_json, 'w') as f:
        output_json = json.dumps(output_json_dict, indent=4)
        f.write(output_json)


def main():
    params = Params()

    paramparse.process(params)

    seq_paths = params.seq_paths
    root_dir = params.root_dir
    # sources_to_include = params.sources_to_include
    enable_mask = params.enable_mask
    val_ratio = params.val_ratio
    min_val = params.min_val
    shuffle = params.shuffle
    load_samples = params.load_samples
    load_samples_root = params.load_samples_root
    class_names_path = params.class_names_path
    output_json = params.output_json
    extract_num_from_imgid = params.extract_num_from_imgid
    no_annotations = params.no_annotations
    excluded_images_list = params.excluded_images_list
    n_seq = params.n_seq

    if seq_paths:
        if os.path.isfile(seq_paths):
            seq_paths = [x.strip() for x in open(seq_paths).readlines() if x.strip()]
        else:
            seq_paths = seq_paths.split(',')
        if root_dir:
            seq_paths = [os.path.join(root_dir, name) for name in seq_paths]

    elif root_dir:
        seq_paths = [os.path.join(root_dir, name) for name in os.listdir(root_dir) if
                     os.path.isdir(os.path.join(root_dir, name))]
        seq_paths.sort(key=sortKey)
    else:
        raise IOError('Either seq_paths or root_dir must be provided')

    if 0 < n_seq < len(seq_paths):
        seq_paths = seq_paths[:n_seq]

    n_seq = len(seq_paths)
    assert n_seq > 0, "no sequences found"

    output_json_dir, output_json_fname = os.path.dirname(output_json), os.path.basename(output_json)
    output_json_fname_noext, output_json_fname_ext = os.path.splitext(output_json_fname)

    if not output_json_dir:
        if root_dir:
            output_json_dir = root_dir
        else:
            output_json_dir = os.path.dirname(seq_paths[0])

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
            load_samples = [os.path.join(load_samples_root, k) for k in load_samples]
        print('Loading samples from : {}'.format(load_samples))
        for _f in load_samples:
            if os.path.isdir(_f):
                _f = os.path.join(_f, 'seq_to_samples.txt')
            with open(_f, 'r') as fid:
                curr_seq_to_samples = ast.literal_eval(fid.read())
                for _seq in curr_seq_to_samples:
                    if _seq in seq_to_samples:
                        seq_to_samples[_seq] += curr_seq_to_samples[_seq]
                    else:
                        seq_to_samples[_seq] = curr_seq_to_samples[_seq]

    class_info = [k.strip() for k in open(class_names_path, 'r').readlines() if k.strip()]
    class_names, class_cols = zip(*[k.split('\t') for k in class_info])

    n_classes = len(class_cols)
    """background is class ID 0 with color black"""
    palette = [[0, 0, 0], ]
    for class_id in range(n_classes):
        col = class_cols[class_id]

        col_rgb = col_bgr[col][::-1]

        palette.append(col_rgb)

    palette_flat = [value for color in palette for value in color]

    class_dict = {x.strip(): i for (i, x) in enumerate(class_names)}

    if no_annotations:

        print('creating json without annotations')
        output_json_dict = {
            "images": [],
            "type": "instances",
            "annotations": [],
            "categories": []
        }
        for seq_id, seq_path in enumerate(seq_paths):
            img_files = glob.glob(os.path.join(seq_path, '**/*.jpg'), recursive=True)

            img_files.sort(key=lambda fname: os.path.basename(fname))

            start_frame_id = params.start_frame_id
            end_frame_id = params.end_frame_id

            if end_frame_id < start_frame_id:
                end_frame_id = len(img_files) - 1

            # print(f'params.end_frame_id: {params.end_frame_id}')
            # print(f'end_frame_id: {end_frame_id}')

            img_files = img_files[start_frame_id:end_frame_id + 1]

            if shuffle:
                random.shuffle(img_files)

            seq_name = os.path.basename(seq_path)

            print(f'\n sequence {seq_id + 1} / {n_seq}: {seq_name}\n')

            for img_file in tqdm(img_files):
                # img = cv2.imread(img_file)
                width, height = imagesize.get(img_file)
                filename = os.path.basename(img_file)
                img_id = os.path.splitext(filename)[0]
                image_info = {
                    'file_name': seq_name + '/' + filename,
                    'height': height,
                    'width': width,
                    'id': seq_name + '/' + img_id
                }
                output_json_dict['images'].append(image_info)

        for label, label_id in class_dict.items():
            category_info = {'supercategory': 'none', 'id': label_id, 'name': label}
            output_json_dict['categories'].append(category_info)

        json_path = os.path.join(output_json_dir, output_json_fname)

        print('saving output json to: {}'.format(json_path))
        with open(json_path, 'w') as f:
            output_json_data = json.dumps(output_json_dict, indent=4)
            f.write(output_json_data)
        return

    dir_name = params.dir_name
    if params.dir_suffix:
        dir_name = f'{dir_name}_{params.dir_suffix}'

    print(f'dir_name: {dir_name}')

    xml_paths = [linux_path(seq_path, dir_name) for seq_path in seq_paths]
    seq_names = [os.path.basename(seq_path) for seq_path in seq_paths]
    # samples = [seq_to_samples[seq_path] for seq_path in seq_paths]

    train_xml = []
    val_xml = []
    all_excluded_images = {}

    inv_val = 0
    if val_ratio < 0:
        inv_val = 1
        val_ratio = - val_ratio

    for xml_path, seq_name in zip(xml_paths, seq_names):

        seq_path = os.path.dirname(xml_path)

        xml_files = glob.glob(os.path.join(xml_path, '**/*.xml'), recursive=True)

        xml_files.sort(key=lambda fname: os.path.basename(fname))

        start_frame_id = params.start_frame_id
        end_frame_id = params.end_frame_id

        if end_frame_id < start_frame_id:
            end_frame_id = len(xml_files) - 1

        xml_files = xml_files[start_frame_id:end_frame_id + 1]

        if shuffle:
            random.shuffle(xml_files)

        n_files = len(xml_files)

        excluded_images = []
        if excluded_images_list:
            excluded_images_list_path = linux_path(xml_path, excluded_images_list)
            if os.path.exists(excluded_images_list_path):
                print(f'reading excluded_images_list from {excluded_images_list_path}')

                excluded_images = open(excluded_images_list_path, 'r').readlines()
                excluded_images = [k.strip() for k in set(excluded_images)]
                print(f'found {len(excluded_images)} excluded_images')
            else:
                print(f'excluded_images_list does not exist {excluded_images_list_path}')

        all_excluded_images[seq_path] = excluded_images

        assert n_files > 0, 'No xml files found in {}'.format(xml_path)


        n_val_files = max(int(n_files * val_ratio), min_val)

        n_train_files = n_files - n_val_files

        print(f'{seq_name} :: n_train, n_val: {[n_train_files, n_val_files]} ')

        if inv_val:
            val_files = tuple(zip(xml_files[:n_val_files], [seq_path, ] * n_val_files))
            train_files = tuple(zip(xml_files[n_val_files:], [seq_path, ] * n_train_files))
        else:
            train_files = tuple(zip(xml_files[:n_train_files], [seq_path, ] * n_train_files))
            val_files = tuple(zip(xml_files[n_train_files:], [seq_path, ] * n_val_files))

        val_xml += val_files
        train_xml += train_files

    train_json_fname = output_json_fname_noext

    n_val_xml = len(val_xml)

    if n_val_xml > 0:
        train_json_fname += '-train'
        val_json_fname = output_json_fname_noext + '-val' + output_json_fname_ext
        val_json_path = os.path.join(output_json_dir, val_json_fname)

        val_list_path = val_json_path.replace('.json', '.txt')
        if params.write_masks:
            print(f'\nsaving img list for {n_val_xml} validation files to: {val_list_path}\n')
        else:
            print(f'\nsaving JSON annotations for {n_val_xml} validation files to: {val_json_path}\n')

        save_boxes_coco(val_xml, class_dict, val_json_path,
                        extract_num_from_imgid, enable_mask,
                        excluded_images=all_excluded_images,
                        ignore_invalid_label=params.ignore_invalid_label,
                        allow_missing_images=params.allow_missing_images,
                        remove_mj_dir_suffix=params.remove_mj_dir_suffix,
                        get_img_stats=params.get_img_stats,
                        write_masks=params.write_masks,
                        list_path=val_list_path,
                        mask_dir_name=params.mask_dir_name,
                        only_list=params.only_list,
                        palette_flat=palette_flat,
                        )

    n_train_xml = len(train_xml)
    if n_train_xml > 0:
        train_json_fname += output_json_fname_ext

        train_json_path = os.path.join(output_json_dir, train_json_fname)
        train_list_path = train_json_path.replace('.json', '.txt')
        if params.write_masks:
            print(f'\nsaving imag list for {n_train_xml} train files to: {train_list_path}\n')
        else:
            print(f'\nsaving JSON annotations for {n_train_xml} train files to: {train_json_path}\n')

        save_boxes_coco(train_xml, class_dict, train_json_path,
                        extract_num_from_imgid, enable_mask,
                        excluded_images=all_excluded_images,
                        allow_missing_images=params.allow_missing_images,
                        ignore_invalid_label=params.ignore_invalid_label,
                        remove_mj_dir_suffix=params.remove_mj_dir_suffix,
                        get_img_stats=params.get_img_stats,
                        list_path=train_list_path,
                        write_masks=params.write_masks,
                        mask_dir_name=params.mask_dir_name,
                        only_list=params.only_list,
                        palette_flat=palette_flat,
                        )


if __name__ == '__main__':
    main()
