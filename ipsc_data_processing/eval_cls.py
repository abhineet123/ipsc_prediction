import os
import glob
import random
import ast
import math

import numpy as np
import pandas as pd

import cv2

from pprint import pformat

import json
import xml.etree.ElementTree as ET
from tqdm import tqdm

import imagesize
import paramparse

from eval_utils import compute_binary_cls_metrics, arr_to_csv, sortKey, linux_path


class Params:
    class XGB:
        def __init__(self):
            self.max_depth = 6
            self.eta = 0.3
            self.objective = 'multi:softprob'
            self.nthread = 12
            self.eval_metric = 'auc'
            self.num_round = 1000
            self.verbose = 1

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
        self.min_val = 1
        self.n_classes = 4
        self.n_frames = 0
        self.input_dir_name = 'annotations'
        self.input_dir_suffix = ''
        self.out_dir = ''
        self.test_dir = ''
        self.db_out_dir = ''
        self.read_colors = 1
        self.root_dir = ''
        self.save_file_name = ''
        self.save_video = 1
        self.seq_paths = ''
        self.show_img = 0
        self.shuffle = 0
        self.sources_to_include = []
        self.n_val = 0
        self.val_ratio = 0
        self.save_masks = 1
        self.allow_missing_images = 0
        self.remove_mj_dir_suffix = 0
        self.ignore_invalid_label = 0
        self.save_patches = 0
        self.get_img_stats = 1
        self.patch_border = 5
        self.enable_mask = 0
        self.conf_thresholds = [0.5, ]

        self.start_frame_id = 0
        self.end_frame_id = -1

        self.start_id = 0
        self.end_id = -1

        self.xgb = Params.XGB()

        self.iw = 0
        self.load = 0
        self.load_model = ''
        self.balanced_train = 0


# Data conventions: A point is a pair of floats (x, y). A circle is a triple of floats (center x, center y, radius).

# Returns the smallest circle that encloses all the given points. Runs in expected O(n) time, randomized.
# Input: A sequence of pairs of floats or ints, e.g. [(0,5), (3.1,-2.7)].
# Output: A triple of floats representing a circle.
# Note: If 0 points are given, None is returned. If 1 point is given, a circle of radius 0 is returned.
#
# Initially: No boundary points known
def make_circle(points):
    # Convert to float and randomize order
    shuffled = [(float(x), float(y)) for (x, y) in points]
    random.shuffle(shuffled)

    # Progressively add points to circle or recompute circle
    c = None
    for (i, p) in enumerate(shuffled):
        if c is None or not is_in_circle(c, p):
            c = _make_circle_one_point(shuffled[: i + 1], p)
    return c


# One boundary point known
def _make_circle_one_point(points, p):
    c = (p[0], p[1], 0.0)
    for (i, q) in enumerate(points):
        if not is_in_circle(c, q):
            if c[2] == 0.0:
                c = make_diameter(p, q)
            else:
                c = _make_circle_two_points(points[: i + 1], p, q)
    return c


# Two boundary points known
def _make_circle_two_points(points, p, q):
    circ = make_diameter(p, q)
    left = None
    right = None
    px, py = p
    qx, qy = q

    # For each point not in the two-point circle
    for r in points:
        if is_in_circle(circ, r):
            continue

        # Form a circumcircle and classify it on left or right side
        cross = _cross_product(px, py, qx, qy, r[0], r[1])
        c = make_circumcircle(p, q, r)
        if c is None:
            continue
        elif cross > 0.0 and (
                left is None or _cross_product(px, py, qx, qy, c[0], c[1]) > _cross_product(px, py, qx, qy, left[0],
                                                                                            left[1])):
            left = c
        elif cross < 0.0 and (
                right is None or _cross_product(px, py, qx, qy, c[0], c[1]) < _cross_product(px, py, qx, qy, right[0],
                                                                                             right[1])):
            right = c

    # Select which circle to return
    if left is None and right is None:
        return circ
    elif left is None:
        return right
    elif right is None:
        return left
    else:
        return left if (left[2] <= right[2]) else right


def make_diameter(a, b):
    cx = (a[0] + b[0]) / 2
    cy = (a[1] + b[1]) / 2
    r0 = math.hypot(cx - a[0], cy - a[1])
    r1 = math.hypot(cx - b[0], cy - b[1])
    return (cx, cy, max(r0, r1))


def make_circumcircle(a, b, c):
    # Mathematical algorithm from Wikipedia: Circumscribed circle
    ox = (min(a[0], b[0], c[0]) + max(a[0], b[0], c[0])) / 2
    oy = (min(a[1], b[1], c[1]) + max(a[1], b[1], c[1])) / 2
    ax = a[0] - ox
    ay = a[1] - oy
    bx = b[0] - ox
    by = b[1] - oy
    cx = c[0] - ox
    cy = c[1] - oy
    d = (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) * 2.0
    if d == 0.0:
        return None
    x = ox + ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
    y = oy + ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
    ra = math.hypot(x - a[0], y - a[1])
    rb = math.hypot(x - b[0], y - b[1])
    rc = math.hypot(x - c[0], y - c[1])
    return (x, y, max(ra, rb, rc))


_MULTIPLICATIVE_EPSILON = 1 + 1e-14


def is_in_circle(c, p):
    return c is not None and math.hypot(p[0] - c[0], p[1] - c[1]) <= c[2] * _MULTIPLICATIVE_EPSILON


# Returns twice the signed area of the triangle defined by (x0, y0), (x1, y1), (x2, y2).
def _cross_product(x0, y0, x1, y1, x2, y2):
    return (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)


def save_cell_features(
        xml_files,
        class_names,
        db_root_dir,
        allow_missing_images,
        ignore_invalid_label,
        excluded_images,
        out_root_dir,
        save_patches,
        get_img_stats,
        patch_border,
        enable_mask,
        all_pix_vals_mean,
        all_pix_vals_std,
):
    n_non_empty_images = 0
    n_images = 0

    n_objs = 0

    label_to_n_objs = {
        label: 0 for label in class_names
    }

    vid_size = None

    file_names = []

    pbar = tqdm(xml_files)
    target_id_to_objs = {}

    if get_img_stats:
        print(f'collecting image stats')

    if save_patches:
        print(f'saving patches to {out_root_dir}')

    patch_labels = []
    patch_name_to_src_id = {}

    for src_id, (xml_path, seq_path, seq_name) in enumerate(pbar):

        n_images += 1

        # Read annotation xml
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

        img_dir = os.path.dirname(img_path)
        seq_name = os.path.basename(img_dir)
        img_name_noext, img_ext = os.path.splitext(img_name)

        # if save_patches:
        #     patch_out_dir = linux_path(out_root_dir, seq_name)
        #     os.makedirs(patch_out_dir, exist_ok=1)

        file_names.append(img_rel_path)

        # img_file_path = linux_path(seq_path, img_file_name)

        if excluded_images is not None and img_name in excluded_images[seq_path]:
            print(f'\n{seq_name} :: skipping excluded image {img_name}')
            continue

        if not os.path.exists(img_path):
            msg = f"img_file_path does not exist: {img_path}"
            if allow_missing_images:
                print('\n' + msg + '\n')
                continue
            else:
                raise AssertionError(msg)

        img_w, img_h = imagesize.get(img_path)

        if vid_size is None:
            vid_size = (img_w, img_h)
        else:
            assert vid_size == (img_w, img_h), f"mismatch between size of image: {(img_w, img_h)} and video: {vid_size}"

        size_from_xml = ann_root.find('size')
        w_from_xml = int(size_from_xml.findtext('width'))
        h_from_xml = int(size_from_xml.findtext('height'))

        assert h_from_xml == img_h and w_from_xml == img_w, \
            f"incorrect image dimensions in XML: {(h_from_xml, w_from_xml)}"

        objs = ann_root.findall('object')

        src_shape = (img_h, img_w)
        img = cv2.imread(img_path)

        for obj_id, obj in enumerate(objs):

            id_number = int(obj.findtext('id_number'))

            if id_number <= 0:
                continue

            label = obj.findtext('name')

            try:
                category_id = class_names.index(label)
            except ValueError:
                msg = f"label {label} is not in class_to_id_and_col"
                if ignore_invalid_label:
                    print(msg)
                    continue
                else:
                    raise AssertionError(msg)

            bndbox = obj.find('bndbox')

            xmin = float(bndbox.findtext('xmin')) - 1
            ymin = float(bndbox.findtext('ymin')) - 1
            xmax = float(bndbox.findtext('xmax'))
            ymax = float(bndbox.findtext('ymax'))

            assert xmax > xmin and ymax > ymin, f"invalid box: {xmin, ymin, xmax, ymax}"

            mask_obj = obj.find('mask')
            if mask_obj is None:
                msg = 'no mask found for object:\n{}'.format(img_name)
                raise AssertionError(msg)

            mask = mask_obj.text
            mask = [k.strip().split(',') for k in mask.strip().split(';') if k]
            # pprint(mask)
            mask_pts = [(float(_pt[0]), float(_pt[1])) for _pt in mask]

            mask_img = np.zeros(src_shape, dtype=np.uint8)
            mask_img = cv2.fillPoly(mask_img, np.array([mask_pts, ], dtype=np.int32), 1)

            start_row = int(max(0, ymin - patch_border))
            start_col = int(max(0, xmin - patch_border))
            end_row = int(min(img_h - 1, ymax + patch_border))
            end_col = int(min(img_w - 1, xmax + patch_border))

            patch_out_dir = linux_path(out_root_dir, label)
            patch_h, patch_w = end_row - start_row + 1, end_col - start_col + 1
            patch_img_name = f'{seq_name}_{img_name_noext}_{obj_id:03d}_{patch_w}x{patch_h}{img_ext}'
            patch_out_path = linux_path(patch_out_dir, patch_img_name)

            if save_patches:
                if enable_mask:
                    masked_img = img.copy()
                    masked_img[mask_img == 0] = 0
                    img_patch = masked_img[start_row:end_row + 1, start_col:end_col + 1, ...]
                    # cv2.imshow('masked_img', masked_img)
                    # cv2.imshow('img_patch', img_patch)
                    # cv2.waitKey(0)
                else:
                    img_patch = img[start_row:end_row + 1, start_col:end_col + 1, ...]

                os.makedirs(patch_out_dir, exist_ok=1)

                if get_img_stats:
                    h, w = img_patch.shape[:2]
                    img_reshaped = np.reshape(img_patch, (h * w, 3))

                    pix_vals_mean = list(np.mean(img_reshaped, axis=0))
                    pix_vals_std = list(np.std(img_reshaped, axis=0))

                    all_pix_vals_mean.append(pix_vals_mean)
                    all_pix_vals_std.append(pix_vals_std)

                cv2.imwrite(patch_out_path, img_patch)

                patch_labels.append(dict(path=patch_out_path, name=patch_img_name, label=label))

                if save_patches == 2:
                    continue

            cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2

            pix_vals = img[mask_img == 1].flatten()

            area = np.count_nonzero(mask_img)

            pix_min = np.amin(pix_vals)
            pix_max = np.amax(pix_vals)
            pix_std = np.std(pix_vals)

            center_x, center_y, radius = make_circle(mask_pts)

            circum_circle_area = math.pi * radius * radius

            assert circum_circle_area > 0, "zero circum_circle_area"
            assert area > 0, "zero area"

            circlicity = circum_circle_area / area

            curr_obj = {
                'bbox': [xmin, ymin, xmax, ymax],
                'mask': mask_pts,
                'cx': cx,
                'cy': cy,
                'area': area,
                'pix_min': pix_min,
                'pix_max': pix_max,
                'pix_std': pix_std,
                'circlicity': circlicity,
                'label': category_id,
                'src_id': src_id,
            }

            if id_number not in target_id_to_objs:
                target_id_to_objs[id_number] = []

            target_id_to_objs[id_number].append(curr_obj)

            patch_name_to_src_id[patch_img_name] = src_id

            n_objs += 1

            label_to_n_objs[label] += 1

        n_non_empty_images += 1
        desc = f'{n_images - n_non_empty_images} / {n_images} empty images :: {n_objs} objects '
        for label in class_names:
            desc += f' {label}: {label_to_n_objs[label]}'

        pbar.set_description(desc)

    features = []
    labels = []
    src_ids = []

    if save_patches:
        csv_columns = ['path', 'name', 'label']
        out_csv_path = linux_path(out_root_dir, "annotations.csv")
        df = pd.DataFrame(patch_labels, columns=csv_columns)
        """incremental CSV writing"""
        if not os.path.exists(out_csv_path):
            """create a new CSV file"""
            df.to_csv(out_csv_path, index=False)
        else:
            """write to an existing CSV file"""
            df.to_csv(out_csv_path, index=False, mode='a', header=False)

        if save_patches == 2:
            return features, labels, src_ids

    for target_id in target_id_to_objs:
        objs = target_id_to_objs[target_id]

        for obj_id, obj in enumerate(objs):

            src_id = obj['src_id']

            if obj_id > 0:
                prev_obj = objs[obj_id - 1]
                disp_x = obj['cx'] - prev_obj['cx']
                disp_y = obj['cy'] - prev_obj['cy']
                speed = math.sqrt(disp_x ** 2 + disp_y ** 2)

                init__obj = objs[0]
                init_disp_x = obj['cx'] - init__obj['cx']
                init_disp_y = obj['cy'] - init__obj['cy']
                init_disp = math.sqrt(init_disp_x ** 2 + init_disp_y ** 2)
            else:
                speed = 0
                init_disp = 0

            feature = [
                obj['area'],
                obj['circlicity'],
                speed,
                init_disp,
                obj['pix_std'],
                obj['pix_min'],
                obj['pix_max'],
            ]
            label = obj['label']

            src_ids.append(src_id)
            features.append(feature)
            labels.append(label)

    return features, labels, src_ids, patch_name_to_src_id


def add_image_id(arr_list):
    cmb_arrs = []
    for img_id, arr in enumerate(arr_list):
        img_id_arr = np.full((arr.shape[0], 1), img_id)
        cmb_arr = np.concatenate((img_id_arr, arr), axis=1)
        cmb_arrs.append(cmb_arr)

    cmb_arrs = np.concatenate(cmb_arrs, axis=0)

    return cmb_arrs


def save_metrics(sample_out_dir, class_tp, class_fp, conf_to_acc, roc_aucs, tp_fp, class_names, iw=0):
    os.makedirs(sample_out_dir, exist_ok=1)
    csv_columns_tp_fp = ['confidence_threshold', 'TP', 'FP']
    csv_columns_acc = ["confidence_threshold", "overall"] + list(class_names)
    csv_columns_roc_auc = ['FP_threshold', 'AUC']

    acc_names = [
        'acc'
    ]
    tp_fp_names = [
        f'tp_fp_uex',
        f'tp_fp_uex_fn',
        f'tp_fp_ex',
        f'tp_fp_ex_fn',
    ]
    roc_aucs_names = [
        f'roc_auc_uex',
        f'roc_auc_ex',
        f'roc_auc_uex_fn',
        f'roc_auc_ex_fn',
    ]
    if iw:
        auc_names = [
            f'auc_uex',
            f'auc_ex',
            f'auc_uex_fn',
            f'auc_ex_fn',
        ]
        csv_columns_auc = ['Image_ID', 'AUC']

        roc_aucs_names = [f'{roc_aucs_name}-iw' for roc_aucs_name in roc_aucs_names]
        tp_fp_names = [f'{tp_fp_name}-iw' for tp_fp_name in tp_fp_names]
        acc_names = [f'{acc_name}-iw' for acc_name in acc_names]
        auc_names = [f'{auc_name}-iw' for auc_name in auc_names]

        csv_columns_roc_auc = ['Image_ID', ] + csv_columns_roc_auc
        csv_columns_acc = ['Image_ID', ] + csv_columns_acc
        csv_columns_tp_fp = ['Image_ID', ] + csv_columns_tp_fp

        aucs = np.asarray([roc_auc[-1][1] for roc_auc in roc_aucs]).reshape((-1, 1))
        img_id_arr = np.asarray(range(len(aucs))).reshape((-1, 1))
        cmb_auc_arr = np.concatenate((img_id_arr, aucs), axis=1)
        for auc_name in auc_names:
            arr_to_csv(cmb_auc_arr, csv_columns_auc, sample_out_dir, f'{auc_name}.csv')

        conf_to_acc = add_image_id(conf_to_acc)
        roc_aucs = add_image_id(roc_aucs)
        class_tp = add_image_id(class_tp)
        class_fp = add_image_id(class_fp)
        if tp_fp is not None:
            tp_fp = add_image_id(tp_fp)

    for acc_name in acc_names:
        arr_to_csv(conf_to_acc, csv_columns_acc, sample_out_dir, f'{acc_name}.csv')

    for class_id, class_name in enumerate(class_names):
        if iw:
            class_tp_fp = np.concatenate((
                conf_to_acc[:, :2].reshape((-1, 2)),
                class_tp[:, class_id + 1].reshape((-1, 1)),
                class_fp[:, class_id + 1].reshape((-1, 1)),
            ), axis=1)
        else:
            class_tp_fp = np.concatenate((
                conf_to_acc[:, 0].reshape((-1, 1)),
                class_tp[:, class_id].reshape((-1, 1)),
                class_fp[:, class_id].reshape((-1, 1)),
            ), axis=1)

        for tp_fp_name in tp_fp_names:
            arr_to_csv(class_tp_fp, csv_columns_tp_fp, sample_out_dir, f'{class_name}-{tp_fp_name}.csv')



    class_dict = {}
    for roc_aucs_name in roc_aucs_names:
        arr_to_csv(roc_aucs, csv_columns_roc_auc, sample_out_dir, f'{roc_aucs_name}.csv')
        for _id, fp_threshold in enumerate(roc_aucs[:, 0]):
            class_dict[f'roc_auc_uex-{fp_threshold:.1f}'] = float(roc_aucs[_id, 1])

    eval_dict = {
        class_names[0]: class_dict
    }

    eval_dict_path = linux_path(sample_out_dir, 'eval_dict.json')
    print(f'saving eval_dict to {eval_dict_path}')
    with open(eval_dict_path, 'w') as f:
        output_json_data = json.dumps(eval_dict, indent=4)
        f.write(output_json_data)

    if tp_fp is not None:
        for tp_fp_name in tp_fp_names:
            arr_to_csv(tp_fp, csv_columns_tp_fp, sample_out_dir, f'{tp_fp_name}.csv')


def main():
    params = Params()

    paramparse.process(params)

    seq_paths = params.seq_paths
    root_dir = params.root_dir
    n_val_files = params.n_val
    val_ratio = params.val_ratio
    min_val = params.min_val
    shuffle = params.shuffle
    load_samples = params.load_samples
    load_samples_root = params.load_samples_root
    class_names_path = params.class_names_path
    read_colors = params.read_colors
    excluded_images_list = params.excluded_images_list
    description = params.description
    out_dir = params.out_dir
    test_dir = params.test_dir
    db_out_dir = params.db_out_dir

    assert description, "dataset description must be provided"

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

    n_seq = len(seq_paths)
    assert n_seq > 0, "no sequences found"

    start_id = params.start_id
    end_id = params.end_id

    if end_id < start_id:
        end_id = n_seq - 1

    seq_paths = seq_paths[start_id:end_id + 1]
    n_seq = len(seq_paths)

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
    if read_colors:
        class_names, class_cols = zip(*[k.split('\t') for k in class_names])

    n_classes = len(class_names)
    # class_ids = list(range(n_classes))
    # class_name_to_id = {x: i for (i, x) in enumerate(class_names)}

    input_dir_name = params.input_dir_name
    if params.input_dir_suffix:
        input_dir_name = f'{input_dir_name}_{params.input_dir_suffix}'

    print(f'input_dir_name: {input_dir_name}')

    xml_dir_paths = [linux_path(seq_path, input_dir_name) for seq_path in seq_paths]
    seq_names = [os.path.basename(seq_path) for seq_path in seq_paths]

    all_train_files = []
    all_train_features = []
    all_train_labels = []
    all_train_src_ids = []

    all_train_src_name_to_id = {}

    all_test_files = []
    all_test_features = []
    all_test_labels = []
    all_test_src_ids = []

    all_test_src_name_to_id = {}

    all_excluded_images = {}

    """folder containing all sequence folders"""
    if root_dir:
        db_root_dir = root_dir
    else:
        db_root_dir = os.path.dirname(seq_paths[0])

    reverse_split = False
    if n_val_files < 0:
        n_val_files = -n_val_files
        reverse_split = True
    elif val_ratio < 0:
        val_ratio = -val_ratio
        reverse_split = True

    if reverse_split:
        print('using reverse split')

    out_template = description
    if params.balanced_train:
        out_template = f'{out_template}_balanced'

    if params.enable_mask:
        out_template = f'{out_template}_masked'

    if not db_out_dir:
        db_out_dir = linux_path(root_dir, 'swc', out_template)

    if not out_dir:
        out_dir = linux_path('log', 'xgb', out_template)

    print(f'saving metrics to {out_dir}')

    os.makedirs(out_dir, exist_ok=1)
    os.makedirs(db_out_dir, exist_ok=1)

    model_out = linux_path(out_dir, 'xgb_trained.model')

    if params.load_model:
        try:
            load_model = int(params.load_model)
        except ValueError:
            model_out = params.load_model

    train_features_out_csv = linux_path(out_dir, 'train-features.csv')
    train_labels_out_csv = linux_path(out_dir, 'train-labels.csv')
    train_src_ids_out_csv = linux_path(out_dir, 'train-src_ids.csv')
    train_src_name_to_id_out_json = linux_path(out_dir, 'train-src_name_to_id.json')

    if not test_dir:
        test_dir = out_dir

    test_features_out_csv = linux_path(test_dir, 'test-features.csv')
    test_labels_out_csv = linux_path(test_dir, 'test-labels.csv')
    test_src_ids_out_csv = linux_path(test_dir, 'test-src_ids.csv')
    test_src_name_to_id_out_json = linux_path(test_dir, 'test-src_name_to_id.json')
    test_probs_out_csv = linux_path(test_dir, 'test-probs.csv')
    # test_probs_out_csv = linux_path(out_dir, 'test-probs2.csv')

    if params.load:
        print(f'Loading test labels from {test_labels_out_csv}')
        all_test_labels = np.loadtxt(test_labels_out_csv)

        if params.iw:
            if not os.path.exists(test_src_ids_out_csv):
                test_src_names_csv = linux_path(test_dir, 'test-all_src_names.csv')
                print(f'Loading test src_names from {test_src_names_csv}')
                all_test_src_names = open(test_src_names_csv, 'r').read().splitlines()

                print(f'Loading test src_name_to_id from {test_src_name_to_id_out_json}')
                test_src_name_to_id = json.load(open(test_src_name_to_id_out_json, 'r'))

                all_test_src_ids = [test_src_name_to_id[test_src_name] for test_src_name in all_test_src_names]
            else:
                print(f'Loading test src_ids from {test_src_ids_out_csv}')
                all_test_src_ids = np.loadtxt(test_src_ids_out_csv)

        if params.load == 2:
            print(f'Loading test probs from {test_probs_out_csv}')
            try:
                all_test_probs = np.loadtxt(test_probs_out_csv, delimiter=',')
            except ValueError:
                all_test_probs = np.loadtxt(test_probs_out_csv, delimiter='\t')

        else:
            print(f'Loading training features from {train_features_out_csv}')
            all_train_features = np.loadtxt(train_features_out_csv, delimiter=',')

            print(f'Loading training src_ids from {train_labels_out_csv}')
            all_train_src_ids = np.loadtxt(train_src_ids_out_csv)

            print(f'Loading training features from {train_labels_out_csv}')
            all_train_labels = np.loadtxt(train_labels_out_csv)

            print(f'Loading test features from {test_features_out_csv}')
            all_test_features = np.loadtxt(test_features_out_csv, delimiter=',')
    else:
        all_pix_vals_mean = []
        all_pix_vals_std = []

        for seq_id, (xml_dir_path, seq_name, seq_path) in enumerate(zip(xml_dir_paths, seq_names, seq_paths)):

            xml_files = glob.glob(linux_path(xml_dir_path, '*.xml'), recursive=False)

            if shuffle:
                assert not reverse_split, "reverse_split makes no sense with shuffle on"
                random.shuffle(xml_files)
            else:
                xml_files.sort(key=lambda fname: os.path.basename(fname))

            start_frame_id = params.start_frame_id
            end_frame_id = params.end_frame_id

            if end_frame_id < start_frame_id:
                end_frame_id = len(xml_files) - 1

            xml_files = xml_files[start_frame_id:end_frame_id + 1]

            n_files = len(xml_files)

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

            assert n_files > 0, 'No xml xml_files found in {}'.format(xml_dir_path)

            if n_val_files == 0 and val_ratio > 0:
                n_val_files = max(int(n_files * val_ratio), min_val)
            else:
                val_ratio = float(n_val_files) / n_files

            n_train_files = n_files - n_val_files

            print(f'\n\n{seq_id + 1} / {n_seq} {seq_name} :: n_train, n_test: {[n_train_files, n_val_files]} ')

            xml_files = tuple(zip(xml_files, [seq_path, ] * n_files, [seq_name, ] * n_files))

            if reverse_split:
                val_xml_files = xml_files[:n_val_files]
                train_xml_files = xml_files[n_val_files:]
            else:
                train_xml_files = xml_files[:n_train_files]
                val_xml_files = xml_files[n_train_files:]

            all_test_files += val_xml_files
            all_train_files += train_xml_files

            if n_val_files > 0:
                val_out_dir = linux_path(db_out_dir, 'val')

                # if params.save_patches:
                #     print(f'writing validation images to {val_out_dir}')

                test_features, test_labels, test_src_ids, test_src_name_to_id = save_cell_features(
                    xml_files=val_xml_files,
                    class_names=class_names,
                    db_root_dir=db_root_dir,
                    allow_missing_images=params.allow_missing_images,
                    ignore_invalid_label=params.ignore_invalid_label,
                    excluded_images=all_excluded_images,
                    out_root_dir=val_out_dir,
                    save_patches=params.save_patches,
                    get_img_stats=params.get_img_stats,
                    patch_border=params.patch_border,
                    enable_mask=params.enable_mask,
                    all_pix_vals_mean=all_pix_vals_mean,
                    all_pix_vals_std=all_pix_vals_std,
                )

                all_test_src_ids += test_src_ids
                all_test_features += test_features
                all_test_labels += test_labels

                all_test_src_name_to_id.update(test_src_name_to_id)

            if n_train_files > 0:
                train_out_dir = linux_path(db_out_dir, 'train')

                # if params.save_patches:
                #     print(f'writing training images to {train_out_dir}')

                train_features, train_labels, train_src_ids, train_src_name_to_id = save_cell_features(
                    xml_files=train_xml_files,
                    class_names=class_names,
                    db_root_dir=db_root_dir,
                    allow_missing_images=params.allow_missing_images,
                    ignore_invalid_label=params.ignore_invalid_label,
                    excluded_images=all_excluded_images,
                    out_root_dir=train_out_dir,
                    get_img_stats=params.get_img_stats,
                    save_patches=params.save_patches,
                    patch_border=params.patch_border,
                    enable_mask=params.enable_mask,
                    all_pix_vals_mean=all_pix_vals_mean,
                    all_pix_vals_std=all_pix_vals_std,
                )

                all_train_features += train_features
                all_train_labels += train_labels
                all_train_src_ids += train_src_ids
                all_train_src_name_to_id.update(train_src_name_to_id)

        if params.get_img_stats:
            pix_vals_mean = list(np.mean(all_pix_vals_mean, axis=0))
            pix_vals_std = list(np.mean(all_pix_vals_std, axis=0))  #
            print(f'pix_vals_mean: {pix_vals_mean}')
            print(f'pix_vals_std: {pix_vals_std}')

        # time_stamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")

    if params.save_patches == 2:
        return

    n_train_features = len(all_train_features)
    n_val_features = len(all_test_features)

    print(f'n_train_features: {n_train_features}')
    print(f'n_val_features: {n_val_features}')

    if params.load != 2:
        n_train = len(all_train_labels)

        all_train_features = np.asarray(all_train_features)
        all_train_labels = np.asarray(all_train_labels)
        all_train_src_ids = np.asarray(all_train_src_ids)

        if not params.load:
            print(f'generated {n_train} training samples...')

            np.savetxt(train_features_out_csv, all_train_features, fmt='%f', delimiter=',', newline='\n')
            np.savetxt(train_labels_out_csv, all_train_labels, fmt='%f', newline='\n')
            np.savetxt(train_src_ids_out_csv, all_train_src_ids, fmt='%f', newline='\n')

            open(train_src_name_to_id_out_json, 'w').write(json.dumps(all_train_src_name_to_id, indent=4))

        class_sample_info = {}
        class_n_samples = []

        for class_id, class_name in enumerate(class_names):
            class_idx = np.nonzero(all_train_labels == class_id)

            assert len(class_idx) == 1, "non-unit class_idx"

            class_idx = class_idx[0]

            class_train_labels = all_train_labels[class_idx]
            n_class_train_labels = len(class_train_labels)
            class_pc = float(n_class_train_labels) / n_train * 100

            class_sample_info[class_name] = (class_idx, n_class_train_labels, class_pc)
            print(f'\tclass {class_name}: {n_class_train_labels} / {n_train} ({class_pc:.4f}%)')

            class_n_samples.append(n_class_train_labels)

        if params.balanced_train:
            n_min_samples = min(class_n_samples)
            print(f'balancing training set by using {n_min_samples} samples for each class')
            all_idx = []
            for class_id, class_name in enumerate(class_names):
                class_idx, n_class_train_labels, class_pc = class_sample_info[class_name]
                if n_class_train_labels > n_min_samples:
                    class_idx = np.random.permutation(class_idx)
                    class_idx = class_idx[:n_min_samples]
                all_idx += list(class_idx)

            all_idx = np.asarray(all_idx)
            all_train_features = all_train_features[all_idx, :]
            all_train_labels = all_train_labels[all_idx]

            n_train = len(all_train_labels)

        all_train_labels = all_train_labels.reshape((n_train, 1))

        import xgboost as xgb

        xgb_params = {
            'max_depth': params.xgb.max_depth,
            'eta': params.xgb.eta,
            'verbosity': params.xgb.verbose,
            'objective': params.xgb.objective,
            'nthread': params.xgb.nthread,
            'eval_metric': params.xgb.eval_metric,
            'num_class': n_classes,
        }
        if load_model:
            print(f'loading model from: {model_out}')

            bst = xgb.Booster({'nthread': 4})  # init model
            bst.load_model(model_out)  # load data
        else:
            dtrain = xgb.DMatrix(all_train_features, label=all_train_labels)

            print(f'training xgboost on {n_train} samples...')
            for class_id, class_name in enumerate(class_names):
                class_train_labels = all_train_labels[all_train_labels == class_id]
                n_class_train_labels = len(class_train_labels)
                class_pc = float(n_class_train_labels) / n_train * 100
                print(f'\tclass {class_name}: {n_class_train_labels} / {n_train} ({class_pc:.4f}%)')
            print()

            bst = xgb.train(xgb_params, dtrain, params.xgb.num_round)

            bst.save_model(model_out)

    n_all_test = len(all_test_labels)

    all_test_labels = np.asarray(all_test_labels).reshape((n_all_test, 1))
    all_test_features = np.asarray(all_test_features)

    if params.iw:
        all_test_src_ids = np.asarray(all_test_src_ids).reshape((n_all_test,))
        all_sample_ids = []
        for src_id in range(n_val_files):
            samples_ids = np.nonzero(all_test_src_ids == src_id)[0]
            all_sample_ids.append(samples_ids)
    else:
        all_sample_ids = [list(range(n_all_test)), ]

    if not params.load:
        np.savetxt(test_features_out_csv, all_test_features, delimiter=',', newline='\n', fmt='%f')
        np.savetxt(test_labels_out_csv, all_test_labels, newline='\n', fmt='%d')
        np.savetxt(test_src_ids_out_csv, all_test_src_ids, newline='\n', fmt='%d')
        open(test_src_name_to_id_out_json, 'w').write(json.dumps(all_test_src_name_to_id, indent=4))

    if params.load != 2:
        all_test_probs = np.zeros((n_all_test, 2), dtype=np.float32)

    all_metrics = []
    aucs = []
    for test_id, samples_ids in enumerate(all_sample_ids):
        test_labels = all_test_labels[samples_ids, ...]
        n_test = len(test_labels)

        if params.load != 2:
            test_features = all_test_features[samples_ids, ...]

            dtest = xgb.DMatrix(test_features)
            if params.iw:
                print(f'testing on image {test_id}')

            print(f'testing xgboost on {n_test} samples...')
            for class_id, class_name in enumerate(class_names):
                class_val_labels = test_labels[test_labels == class_id]
                n_class_val = len(class_val_labels)
                class_pc = float(n_class_val) / n_test * 100
                print(f'\tclass {class_name}: {n_class_val} / {n_test} ({class_pc:.4f}%)')
            print()
            test_probs = bst.predict(dtest)
            all_test_probs[samples_ids, ...] = test_probs
        else:
            test_probs = all_test_probs[samples_ids, ...]

        test_probs = test_probs.reshape((n_test, n_classes))

        n_conf_thresholds = len(params.conf_thresholds)
        conf_thresholds = np.asarray(params.conf_thresholds).reshape((n_conf_thresholds, 1))

        class_tp, class_fp, conf_to_acc, roc_aucs, tp_fp = compute_binary_cls_metrics(
            conf_thresholds, test_probs, test_labels, class_names)

        class_tp *= 100
        class_fp *= 100
        conf_to_acc[:, 1:] *= 100
        roc_aucs *= 100
        tp_fp[:, 1:] *= 100

        auc = roc_aucs[-1][1]
        aucs.append(auc)

        all_metrics.append((class_tp, class_fp, conf_to_acc, roc_aucs, tp_fp))

        if params.iw:
            sample_out_dir = linux_path(f'{test_dir}-iw', f'img_{test_id}')
        else:
            sample_out_dir = test_dir

        print(f'saving metrics to {sample_out_dir}')

        save_metrics(sample_out_dir, class_tp, class_fp, conf_to_acc, roc_aucs, tp_fp, class_names)
        # predictions = predictions.reshape((n_test, 1))

        if params.load != 2:
            sample_probs_out_csv = linux_path(sample_out_dir, 'test-probs.csv')
            print(f'saving test probs to {sample_probs_out_csv}')
            np.savetxt(sample_probs_out_csv, test_probs, delimiter='\t', newline='\n', fmt='%f')

    if params.iw:
        class_tp, class_fp, conf_to_acc, roc_aucs, tp_fp = zip(*all_metrics)

        # class_tp = np.concatenate(class_tp, axis=1)
        # class_fp = np.concatenate(class_fp, axis=1)
        # conf_to_acc = np.concatenate(conf_to_acc, axis=1)
        # roc_aucs = np.concatenate(roc_aucs, axis=1)
        # tp_fp = np.concatenate(tp_fp, axis=1)

        iw_out_dir = linux_path(f'{test_dir}-iw')
        save_metrics(iw_out_dir, class_tp, class_fp, conf_to_acc, roc_aucs, None, class_names, iw=1)

        class_tp, class_fp, conf_to_acc, roc_aucs, tp_fp = compute_binary_cls_metrics(
            conf_thresholds, all_test_probs, all_test_labels, class_names)
        save_metrics(test_dir, class_tp, class_fp, conf_to_acc, roc_aucs, tp_fp, class_names)


if __name__ == '__main__':
    main()
