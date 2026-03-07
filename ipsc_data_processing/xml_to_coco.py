import os
import glob
import random
import sys
import re

import numpy as np
import cv2
from PIL import Image

from pprint import pformat

import json
from typing import Dict, List
from tqdm import tqdm
import imagesize

import paramparse

from eval_utils import sortKey, col_bgr, linux_path, add_suffix, load_samples_from_txt

sys.path.append(linux_path(os.path.expanduser('~'), 'pix2seq'))

from tasks import task_utils


class Params(paramparse.CFG):
    def __init__(self):
        paramparse.CFG.__init__(self, cfg_prefix='xml_to_coco')

        self.batch_size = 1
        self.excluded_images_list = ''

        self.class_names_path = ''
        self.map_classes = 0
        self.auto_class_cols = 0

        self.codec = 'H264'
        self.csv_file_name = ''
        self.extract_num_from_imgid = 0
        self.fps = 20

        self.img_ext = 'png'
        self.add_ext_to_image = 'png'

        self.load_path = ''

        self.samples_per_seq = 0.0

        self.load_samples = []
        self.load_samples_root = ''
        self.load_samples_suffix = []

        self.map_folder = ''
        self.min_val = 0
        self.n_classes = 4
        self.n_frames = 0
        self.dir_name = 'annotations'
        self.dir_suffix = ''
        self.output_json = []

        self.enable_masks = 0
        self.save_masks = 0
        self.mask_dir_name = 'masks'

        self.root_dir = ''
        self.xml_root_dir = ''

        self.seq_paths = ''
        self.seq_paths_suffix = ''
        self.show_img = 0
        self.shuffle = 0
        self.sources_to_include = []
        self.val_ratio = 0
        self.no_annotations = 0
        self.write_empty = 0
        self.allow_missing_images = 0
        self.remove_mj_dir_suffix = 0
        self.get_img_stats = 0
        self.ignore_invalid_label = 0
        self.allow_ignored_class = 0
        self.skip_invalid = 0

        self.add_empty_images = 0
        self.ignore_empty_images = 0
        self.delete_empty_images = 0
        self.delete_empty_xml = 0

        self.start_frame_id = 0
        self.end_frame_id = -1
        self.frame_stride = 1

        self.n_seq = 0
        self.start_id = 0
        self.end_id = -1
        self.seq_stride = 1

        self.only_list = 0

        self.json_gz = 1
        self.xml_zip = 0


def save_json(json_dict, json_path, json_gz):
    n_json_imgs = len(json_dict['images'])
    n_json_objs = len(json_dict['annotations'])

    out_csv_path = json_path.replace('.json', '.csv')
    if json_gz:
        json_path += '.gz'

    print(f'saving output json with {n_json_imgs} images and {n_json_objs} objects to: {json_path}')
    json_kwargs = dict(
        indent=4
    )

    import pandas as pd
    print(f'out_csv_path: {out_csv_path}')

    img_info_df = pd.DataFrame.from_records(json_dict['images'])
    img_info_df.to_csv(out_csv_path, index=False)

    if json_gz:
        import compress_json
        compress_json.dump(json_dict, json_path, json_kwargs=json_kwargs)
    else:
        with open(json_path, 'w') as f:
            output_json_data = json.dumps(json_dict, **json_kwargs)
            f.write(output_json_data)


def save_ytvis_json(json_dict, json_path, json_gz=True):
    n_vids = len(json_dict['videos'])
    n_json_objs = len(json_dict['annotations'])

    if json_gz:
        json_path += '.gz'

    print(f'saving ytvis json with {n_vids} videos and {n_json_objs} objects to: {json_path}')

    json_kwargs = dict(
        indent=4
    )
    if json_gz:
        import compress_json
        compress_json.dump(json_dict, json_path, json_kwargs=json_kwargs)
    else:
        with open(json_path, 'w') as f:
            output_json_data = json.dumps(json_dict, indent=4)
            f.write(output_json_data)


def save_boxes_coco(
        params: Params,
        db_root_dir,
        xml_paths,
        output_json_dict,
        class_dict,
        output_json,
        extract_num_from_imgid,
        enable_mask,
        list_path,
        palette_flat,
        excluded_images,
        class_map_dict,
):
    bnd_id = 1  # START_BOUNDING_BOX_ID, TODO input as args ?

    out_root_dir = os.path.dirname(output_json)

    img_id_to_info = {img_info['id']: img_info for img_info in output_json_dict['images']}

    img_path_to_stats = {}
    if params.get_img_stats:
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
        label: 0 for label in class_dict
    }

    img_paths = []

    pbar = tqdm(xml_paths, position=0, leave=True)

    for seq_id, (xml_path, seq_path) in enumerate(pbar):
        seq_name = os.path.basename(seq_path)

        n_images += 1

        # Read annotation xml
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

        filename = ann_root.findtext('filename')
        if params.add_ext_to_image:
            assert params.img_ext not in filename, f"filename already has img_ext: {filename}"
            filename = f'{filename}.{params.img_ext}'

        # img_file_rel_path = linux_path(seq_name, filename)

        img_path = linux_path(seq_path, filename)
        img_file_rel_path = linux_path(os.path.relpath(img_path, db_root_dir))
        img_file_name = filename

        # rel_path = annotation_root.findtext('path')
        # if rel_path is None:
        #     filename = annotation_root.findtext('filename')
        #     rel_path = linux_path(seq_name, filename)
        # else:
        #     filename = os.path.basename(rel_path)

        # folder = annotation_root.findtext('folder')
        # path = linux_path(folder, rel_path)

        img_name = os.path.basename(filename)

        img_id = os.path.splitext(img_name)[0]
        if extract_num_from_imgid and isinstance(img_id, str):
            img_id = int(re.findall(r'\d+', img_id)[0])

        size = ann_root.find('size')
        xml_width = int(size.findtext('width'))
        xml_height = int(size.findtext('height'))

        img_info = {
            'file_name': img_file_rel_path,
            'height': xml_height,
            'width': xml_width,
            'id': seq_name + '/' + img_id
        }

        if params.remove_mj_dir_suffix:
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
            if params.allow_missing_images:
                print('\n' + msg + '\n')
                continue
            else:
                raise AssertionError(msg)

        img_paths.append(img_file_path)

        if params.only_list:
            continue

        img_fname_noext = os.path.splitext(img_file_name)[0]
        img_dir_path = os.path.dirname(img_file_path)

        if params.save_masks:
            mask_dir_path = linux_path(img_dir_path, params.mask_dir_name)
            os.makedirs(mask_dir_path, exist_ok=True)

            # print(f'{seq_name}: mask_dir_path: {mask_dir_path}')

            width, height = imagesize.get(str(img_file_path))
            mask_img = np.zeros((height, width), dtype=np.uint8)

        if params.get_img_stats:
            try:
                img_stat = img_path_to_stats[img_file_path]
            except KeyError:
                img = cv2.imread(img_file_path)
                h, w = img.shape[:2]

                assert xml_height == h and xml_width == w, "incorrect image dimensions in XML"

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

        objs = ann_root.findall('object')

        # print()
        n_valid_objs = 0
        for obj_id, obj in enumerate(objs):
            label = obj.findtext('name')

            if params.allow_ignored_class and label == 'ignored':
                """special class to mark ignored regions in the image that have not been annotated"""
                continue

            try:
                label = class_map_dict[label]
            except KeyError as e:
                if params.ignore_invalid_label:
                    # print(f'{xml_path}: ignoring obj with invalid label: {label}')
                    continue
                raise AssertionError(e)

            if label not in class_dict:
                msg = f"label {label} is not in class_dict"
                if params.ignore_invalid_label:
                    print(msg)
                    continue
                else:
                    raise AssertionError(msg)

            category_id = class_dict[label]
            bndbox = obj.find('bndbox')
            xmin = int(float(bndbox.findtext('xmin'))) - 1
            ymin = int(float(bndbox.findtext('ymin'))) - 1
            xmax = int(float(bndbox.findtext('xmax')))
            ymax = int(float(bndbox.findtext('ymax')))

            assert xmax > xmin and ymax > ymin, f"Box size error: ({xmin, ymin, xmax, ymax})"

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
                    if enable_mask == 2 and params.skip_invalid:
                        # print(msg)
                        """ignore mask-less objects"""
                        continue
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
                })

            # if ann is None:
            #     if params.skip_invalid:
            #         print(f'\nskipping object {obj_id + 1} in {xml_path}')
            #         continue
            #     raise AssertionError(f'\ninvalid object {obj_id + 1} in {xml_path}')

            ann.update(
                {
                    'image_id': img_info['id'],
                    'id': bnd_id
                }
            )
            output_json_dict['annotations'].append(ann)

            n_objs += 1
            n_valid_objs += 1

            label_to_n_objs[ann['label']] += 1

            bnd_id += 1

            if params.save_masks:
                # category_id = ann['category_id']
                # class_id = category_id + 1
                category_name = ann['label']
                class_id = class_dict[category_name]
                mask_pts = ann['mask_pts']
                mask_pts_arr = np.array([mask_pts, ], dtype=np.int32)
                mask_img = cv2.fillPoly(mask_img, mask_pts_arr, class_id)

            if enable_mask:
                del ann['mask_pts']

        if n_valid_objs == 0:
            if params.delete_empty_images:
                assert not params.add_empty_images, "empty images can either be added or deleted"

                print(f"deleting empty image with no valid objects: {img_file_path}")
                os.remove(img_file_path)
                if params.delete_empty_xml:
                    print(f"also deleting corresponding xml: {xml_path}")
                    os.remove(xml_path)
            elif params.ignore_empty_images:
                print(f"ignoring empty image with no valid objects: {img_file_path}")
            else:
                raise AssertionError(f"empty image found with no valid objects: {img_file_path}")

        if n_valid_objs > 0 or params.add_empty_images:
            try:
                img_info_existing = img_id_to_info[img_info['id']]
            except KeyError:
                output_json_dict['images'].append(img_info)
                n_valid_images += 1
            else:
                assert img_info_existing == img_info, "img_info mismatch"

        desc = f'{seq_name} {n_valid_images} / {n_images} valid images :: {n_objs} objects '

        if len(class_dict) <= 2:
            for label in class_dict:
                desc += f' {label}: {label_to_n_objs[label]}'

        if params.save_masks:
            mask_img_pil = Image.fromarray(mask_img)
            mask_img_pil = mask_img_pil.convert('P')
            mask_img_pil.putpalette(palette_flat)

            mask_fname = f'{img_fname_noext}.png'
            mask_path = linux_path(mask_dir_path, mask_fname)
            mask_parent_path = os.path.dirname(mask_path)
            os.makedirs(mask_parent_path, exist_ok=True)
            mask_img_pil.save(mask_path)

        pbar.set_description(desc)

    print(f'saving img list to {list_path}')
    with open(list_path, 'w') as fid:
        fid.write('\n'.join(img_paths))

    if params.only_list:
        return

    for label, label_id in class_dict.items():
        category_info = {'supercategory': 'none', 'id': label_id, 'name': label}
        output_json_dict['categories'].append(category_info)

    if params.get_img_stats:
        pix_vals_mean = list(np.mean(all_pix_vals_mean, axis=0))
        pix_vals_std = list(np.mean(all_pix_vals_std, axis=0))  #
        print(f'pix_vals_mean: {pix_vals_mean}')
        print(f'pix_vals_std: {pix_vals_std}')

    save_json(output_json_dict, output_json, params.json_gz)


def main():
    params: Params = paramparse.process(Params)

    if params.save_masks:
        params.enable_masks = 1

    seq_paths = params.seq_paths
    root_dir = params.root_dir
    xml_root_dir = params.xml_root_dir

    enable_mask = params.enable_masks
    val_ratio = params.val_ratio
    min_val = params.min_val
    shuffle = params.shuffle

    load_samples = params.load_samples
    load_samples_root = params.load_samples_root
    load_samples_suffix = params.load_samples_suffix

    if len(load_samples) == 1:
        if load_samples[0] == 1:
            load_samples = ['seq_to_samples.txt', ]
        elif load_samples[0] == 0:
            load_samples = []

    class_names_path = params.class_names_path
    map_classes = params.map_classes
    auto_class_cols = params.auto_class_cols

    assert class_names_path, "class_names_path must be provided"

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

    """background is class ID 0 with color black"""
    palette = [[0, 0, 0], ]
    for class_id in range(n_classes):
        col = class_cols[class_id]

        col_rgb = col_bgr[col][::-1]

        palette.append(col_rgb)

    palette_flat = [value for color in palette for value in color]

    output_json = params.output_json
    output_json = '_'.join(output_json)

    extract_num_from_imgid = params.extract_num_from_imgid
    no_annotations = params.no_annotations
    write_empty = params.write_empty
    excluded_images_list = params.excluded_images_list

    start_id = params.start_id
    end_id = params.end_id
    n_seq = params.n_seq
    seq_stride = params.seq_stride

    xml_dir_name = params.dir_name
    if params.dir_suffix:
        xml_dir_name = f'{xml_dir_name}_{params.dir_suffix}'

    print(f'xml_dir_name: {xml_dir_name}')

    if n_seq > 0:
        end_id = start_id + n_seq - 1

    if params.seq_paths_suffix:
        output_json = add_suffix(output_json, params.seq_paths_suffix, sep='-')

    if load_samples_suffix:
        load_samples_suffix = '_'.join(load_samples_suffix)
        load_samples_root = f'{load_samples_root}_{load_samples_suffix}'
        output_json = add_suffix(output_json, load_samples_suffix, sep='-')

    if load_samples:
        seq_paths, seq_to_samples = load_samples_from_txt(
            load_samples, xml_dir_name, load_samples_root,
            xml_root_dir=xml_root_dir, root_dir=root_dir)
    else:
        seq_to_samples = None

        if seq_paths:
            if seq_paths.endswith('.txt'):
                if params.seq_paths_suffix:
                    seq_paths = add_suffix(seq_paths, params.seq_paths_suffix)

                assert os.path.isfile(seq_paths), f"nonexistent seq_paths file: {seq_paths}"

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

    if start_id > 0 or end_id >= 0 or seq_stride > 1:
        if end_id < 0:
            end_id = len(seq_paths) - 1

        seq_suffix = f'{start_id}_{end_id}'
        if seq_stride > 1:
            seq_suffix = f'{seq_suffix}_{seq_stride}'
        output_json = add_suffix(output_json, f'seq-{seq_suffix}', sep='-')

    if end_id < 0:
        end_id = len(seq_paths) - 1

    if params.start_frame_id > 0 or params.end_frame_id >= 0 or params.frame_stride > 1:
        frame_suffix = f'{params.start_frame_id}_{params.end_frame_id}'
        if params.frame_stride > 1:
            frame_suffix = f'{frame_suffix}_{params.frame_stride}'

        output_json = add_suffix(output_json, frame_suffix, sep='-')

    seq_paths = seq_paths[start_id:end_id + 1:seq_stride]

    n_seq = len(seq_paths)
    assert n_seq > 0, "no sequences found"

    output_json_dir, output_json_fname = os.path.dirname(output_json), os.path.basename(output_json)
    output_json_fname_noext, output_json_fname_ext = os.path.splitext(output_json_fname)
    if not output_json_fname_ext:
        output_json_fname_ext = '.json'

    if not output_json_dir:
        if root_dir:
            output_json_dir = root_dir
        else:
            output_json_dir = os.path.dirname(seq_paths[0])

    """folder containing all sequence folders"""
    if root_dir:
        db_root_dir = root_dir
    else:
        db_root_dir = os.path.dirname(seq_paths[0])

    description = output_json_fname_noext

    output_json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": [],
        "description": description,
    }

    if no_annotations or write_empty:
        if no_annotations:
            print('writing json for all images without annotations')

        if write_empty:
            print('including images without annotations in the json')
            assert val_ratio == 0, "validation set with empty images is meaningless"

        from datetime import datetime
        time_stamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")
        info = {
            "version": "1.0",
            "year": 2024,
            "contributor": "asingh1",
            "date_created": time_stamp,
            "description": description,
        }

        licenses = [
            {
                "url": "https://creativecommons.org/licenses/by/4.0/",
                "id": 1,
                "name": "Creative Commons Attribution 4.0 License"
            }
        ]

        ytvis_json_dict = {
            "info": info,
            "licenses": licenses,
            "videos": [],
            "categories": [],
            "annotations": [],
        }

        for seq_id, seq_path in enumerate(seq_paths):
            img_files = glob.glob(os.path.join(seq_path, '**/*.jpg'), recursive=True)

            assert len(img_files) > 0, f"no img_files found in {seq_path}"

            img_files.sort(key=lambda fname: os.path.basename(fname))

            start_frame_id = params.start_frame_id
            end_frame_id = params.end_frame_id
            frame_stride = params.frame_stride

            if frame_stride <= 0:
                frame_stride = 1

            if end_frame_id < start_frame_id:
                end_frame_id = len(img_files) - 1

            # print(f'params.end_frame_id: {params.end_frame_id}')
            # print(f'end_frame_id: {end_frame_id}')

            img_files = img_files[start_frame_id:end_frame_id + 1:frame_stride]

            if shuffle:
                random.shuffle(img_files)

            seq_name = os.path.basename(seq_path)
            seq_root_dir = os.path.dirname(seq_path)

            print(f'\n sequence {seq_id + 1} / {n_seq}: {seq_name}\n')

            for img_file_id, img_file in enumerate(tqdm(img_files, position=0, leave=True)):
                # img = cv2.imread(img_file)
                width, height = imagesize.get(img_file)
                filename = os.path.basename(img_file)
                img_id = os.path.splitext(filename)[0]

                rel_path = os.path.relpath(img_file, seq_root_dir).rstrip('.' + os.sep).replace(os.sep, '/')
                # rel_path = seq_name + '/' + filename
                image_info = {
                    'file_name': rel_path,
                    'height': height,
                    'width': width,
                    'id': seq_name + '/' + img_id
                }
                video_dict = {
                    "width": width,
                    "height": height,
                    "length": 1,
                    "date_captured": "",
                    "license": 1,
                    "flickr_url": "",
                    "file_names": [rel_path, ],
                    "id": img_file_id + 1,
                    "coco_url": "",
                }
                output_json_dict['images'].append(image_info)
                ytvis_json_dict['videos'].append(video_dict)

        if no_annotations:
            for label, label_id in class_dict.items():
                category_info = {'supercategory': 'none', 'id': label_id, 'name': label}
                output_json_dict['categories'].append(category_info)

                ytvis_category_info = {'supercategory': 'none', 'id': label_id + 1, 'name': label}
                ytvis_json_dict['categories'].append(ytvis_category_info)

            json_path = os.path.join(output_json_dir, output_json_fname)
            save_json(output_json_dict, json_path)

            ytvis_json_dir = linux_path(output_json_dir, 'ytvis19')
            ytvis_json_name = f'ytvis_{output_json_fname}'
            os.makedirs(ytvis_json_dir, exist_ok=1)
            ytvis_json_path = os.path.join(ytvis_json_dir, ytvis_json_name)
            save_ytvis_json(ytvis_json_dict, ytvis_json_path)
            return

    if xml_root_dir:
        assert root_dir, "root_dir must be provided with xml_root_dir"
        xml_paths = [seq_path.replace(root_dir, xml_root_dir) for seq_path in seq_paths]
    else:
        xml_paths = [linux_path(seq_path, xml_dir_name) for seq_path in seq_paths]

    seq_names = [os.path.basename(seq_path) for seq_path in seq_paths]
    # samples = [seq_to_samples[seq_path] for seq_path in seq_paths]

    train_xml = []
    val_xml = []
    all_excluded_images = {}

    inv_val = 0
    if val_ratio < 0:
        inv_val = 1
        val_ratio = - val_ratio

    pbar = tqdm(zip(xml_paths, seq_paths, seq_names, strict=True), position=0, leave=True, total=len(xml_paths))
    for xml_path, seq_path, seq_name in pbar:

        if seq_to_samples is not None:
            xml_files = seq_to_samples[seq_path]
        else:
            if params.xml_zip:
                from zipfile import ZipFile

                xml_zip_path = xml_path + ".zip"
                print(f'loading xml files from  zip file {xml_zip_path}')
                with ZipFile(xml_zip_path, 'r') as xml_zip_file:
                    all_xml_files = xml_zip_file.namelist()

                xml_files = [linux_path(xml_path, xml_file) for xml_file in all_xml_files]
            else:
                xml_files = glob.glob(os.path.join(xml_path, '**/*.xml'), recursive=True)

            xml_files.sort(key=lambda fname: os.path.basename(fname))

        start_frame_id = params.start_frame_id
        end_frame_id = params.end_frame_id
        frame_stride = params.frame_stride

        if frame_stride <= 0:
            frame_stride = 1

        if end_frame_id < start_frame_id:
            end_frame_id = len(xml_files) - 1

        xml_files = xml_files[start_frame_id:end_frame_id + 1:frame_stride]

        if shuffle:
            random.shuffle(xml_files)

        n_files = len(xml_files)

        if params.samples_per_seq > 0:
            if params.samples_per_seq < 1.0:
                """fractional"""
                n_samples = int(n_files * params.samples_per_seq)
            else:
                n_samples = int(params.samples_per_seq)
                n_samples = min(n_samples, n_files)

            print(f'{seq_name}: n_samples: {n_samples} / {n_files}')
            xml_files = xml_files[:n_samples]
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

        pbar.set_description(f'{seq_name} :: n_train, n_val: {[n_train_files, n_val_files]}')

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
        if params.save_masks:
            print(f'\nsaving img list for {n_val_xml} validation files to: {val_list_path}\n')
        else:
            print(f'\nsaving JSON annotations for {n_val_xml} validation files to: {val_json_path}\n')

        save_boxes_coco(
            params=params,
            db_root_dir=db_root_dir,
            xml_paths=val_xml,
            output_json_dict=output_json_dict,
            class_dict=class_dict,
            output_json=val_json_path,
            extract_num_from_imgid=extract_num_from_imgid,
            enable_mask=enable_mask,
            excluded_images=all_excluded_images,
            list_path=val_list_path,
            palette_flat=palette_flat,
            class_map_dict=class_map_dict,
        )

    n_train_xml = len(train_xml)
    if n_train_xml > 0:
        train_json_fname += output_json_fname_ext

        train_json_path = os.path.join(output_json_dir, train_json_fname)
        train_list_path = train_json_path.replace('.json', '.txt')
        if params.save_masks:
            print(f'\nsaving imag list for {n_train_xml} train files to: {train_list_path}\n')
        else:
            print(f'\nsaving JSON annotations for {n_train_xml} train files to: {train_json_path}\n')

        save_boxes_coco(
            params=params,
            db_root_dir=db_root_dir,
            xml_paths=train_xml,
            output_json_dict=output_json_dict,
            class_dict=class_dict,
            output_json=train_json_path,
            extract_num_from_imgid=extract_num_from_imgid,
            enable_mask=enable_mask,
            excluded_images=all_excluded_images,
            list_path=train_list_path,
            palette_flat=palette_flat,
            class_map_dict=class_map_dict,
        )


if __name__ == '__main__':
    main()
