import copy

import numpy as np
import os
import sys
import cv2

import paramparse

from tqdm import tqdm

from xml_to_coco import save_json

from eval_utils import ImageSequenceWriter as ImageWriter
from pascal_voc_io import PascalVocWriter
from eval_utils import contour_pts_to_mask, mask_img_to_pts, sortKey, resize_ar, \
    drawBox, show_labels, clamp, linux_path, add_suffix, mask_pts_to_img, col_bgr


class Params(paramparse.CFG):
    def __init__(self):
        self.cfg = ()
        paramparse.CFG.__init__(self, cfg_prefix='mot_csv_to_xml_coco')

        self.bbox_source = ''
        self.clamp_scores = 0
        self.codec = 'mp4v'
        self.csv_file_name = ''
        self.data_type = 'annotations'
        self.ext = 'mp4'
        self.extrapolate_seg = 0
        self.allow_missing_seg = 1
        self.file_name = ''
        self.vid_ext = ''

        self.allow_ignored = 0
        self.ignore_invalid_class = 0
        self.ignore_invalid = 0
        self.ignore_missing = 0
        self.ignore_occl = 1
        self.img_dir = ''
        self.seg_dir = ''
        self.img_ext = 'jpg'
        self.seg_ext = 'png'
        self.list_file_name = ''
        self.list_file_suffix = ''
        self.map_folder = ''
        self.min_vis = 0.5
        self.class_names_path = ''

        """
        if mode == 0:
            ann_path = linux_path(img_path, 'gt', 'gt.txt')
        elif mode == 1:
            ann_path = linux_path(img_path, f'../../{data_type.capitalize()}/{seq_name}.txt')
        elif mode == 2:
            is_csv = 1
            ann_path = linux_path(img_path, f'{seq_name}.csv')
        """
        self.mode = 1
        self.n_classes = 4
        self.n_frames = 0
        self.out_suffix = ''
        self.percent_scores = 0
        self.resize_factor = 1.0
        self.root_dir = ''

        self.out_root_dir = ''
        self.out_root_suffix = ''

        self.save_file_name = ''
        self.save_path = ''
        self.save_raw = 0
        self.save_video = 0
        self.fps = 30
        self.save_img_seq = 0

        self.show_class = 0
        self.show_img = 0

        self.start_id = 0
        self.end_id = -1

        self.allow_empty = 0
        self.stats_only = 0
        self.raw_ctc_seg = 0
        self.vis_size = ''
        self.vis_root_dir = ''

        self.sample = 0
        self.start_frame_id = 0
        self.end_frame_id = -1
        self.frame_stride = 1

        self.json_gz = 1
        self.json_dir = 0
        self.json_fname = 0
        self.json_desc = ''
        """save all the XML files for a sequence into as zip file rather than separate XML files"""
        self.zip = 1


def parse_mot(ann_path, valid_frame_ids, label, ignore_invalid, percent_scores, clamp_scores, allow_ignored):
    ann_lines = open(ann_path).readlines()
    ann_data = [[float(x) for x in _line.strip().split(',')] for _line in ann_lines if _line.strip()]

    """sort by frame IDs"""
    ann_data.sort(key=lambda x: x[0])
    # ann_data = np.asarray(ann_data)

    obj_ids = []
    obj_dict = {valid_frame_id: [] for valid_frame_id in valid_frame_ids}
    if allow_ignored:
        obj_dict[-1] = []

    for __id, _datum in enumerate(ann_data):

        # if _datum[7] != 1 or _datum[8] < min_vis:
        #     continue

        is_ignored = 0
        obj_id = int(_datum[1])

        if allow_ignored and (_datum[0] == -1 and _datum[1] == -1):
            label_ = 'ignored'
            is_ignored = 1
            frame_id = -1
        else:
            frame_id = int(_datum[0]) - 1
            label_ = label
            if frame_id not in valid_frame_ids:
                continue
            obj_ids.append(obj_id)

        # Bounding box sanity check
        bbox = [float(x) for x in _datum[2:6]]
        l, t, w, h = bbox
        xmin = l
        ymin = t
        xmax = xmin + w
        ymax = ymin + h

        bbox = [xmin, ymin, xmax, ymax]

        if xmin >= xmax or ymin >= ymax:
            msg = f'Invalid box {[xmin, ymin, xmax, ymax]}\n in line {__id} : {_datum}\n'
            if ignore_invalid:
                print(msg)
            else:
                raise AssertionError(msg)

        confidence = float(_datum[6])

        if w <= 0 or h <= 0 or confidence == 0:
            """annoying meaningless unexplained crappy boxes that exist for no apparent reason at all"""
            continue

        # xmin, ymin, w, h = _datum[2:]

        if percent_scores:
            confidence /= 100.0

        if clamp_scores:
            confidence = max(min(confidence, 1), 0)

        if 0 <= confidence <= 1:
            pass
        elif is_ignored:
            confidence = 1
        else:
            msg = "Invalid confidence: {} in line {} : {}".format(
                confidence, __id, _datum)

            if ignore_invalid == 2:
                confidence = 1
            elif ignore_invalid == 1:
                print(msg)
            else:
                raise AssertionError(msg)

        obj_entry = {
            'id': obj_id,
            'label': label_,
            'bbox': bbox,
            'confidence': confidence
        }
        obj_dict[frame_id].append(obj_entry)

    if allow_ignored and -1 in obj_dict:
        ignored_areas = copy.deepcopy(obj_dict[-1])
        del obj_dict[-1]
        for frame_id, obj in obj_dict.items():
            obj += ignored_areas

    # ann_frame_ids = list(obj_dict.keys())
    # empty_frame_ids = list(set(valid_frame_ids) - set(ann_frame_ids))
    # for empty_frame_id in empty_frame_ids:
    #     obj_dict[empty_frame_id] = []

    return obj_ids, obj_dict


def parse_csv(ann_path, valid_frame_ids, ignore_invalid, percent_scores, clamp_scores):
    obj_ids = []

    obj_dict = {valid_frame_id: [] for valid_frame_id in valid_frame_ids}

    import pandas as pd
    df = pd.read_csv(ann_path)
    n_predictions = len(df)

    df['filename'] = df['filename'].astype(str)

    grouped_predictions = df.groupby("filename")
    filenames = list(grouped_predictions.groups.keys())

    filenames.sort()

    n_filenames = len(filenames)
    print(f'{ann_path} --> {n_predictions} labels for {n_filenames} images')

    pbar = tqdm(filenames, total=n_filenames)

    for frame_id, filename in enumerate(pbar):

        if frame_id not in valid_frame_ids:
            continue

        row_ids = grouped_predictions.groups[filename]

        img_df = df.loc[row_ids]
        for _, row in img_df.iterrows():

            try:
                confidence = row['confidence']
            except KeyError:
                confidence = 1.0

            xmin = float(row['xmin'])
            ymin = float(row['ymin'])
            xmax = float(row['xmax'])
            ymax = float(row['ymax'])

            if xmin >= xmax or ymin >= ymax:
                msg = f'Invalid box {[xmin, ymin, xmax, ymax]}\n for file {filename}\n'
                if ignore_invalid:
                    print(msg)
                else:
                    raise AssertionError(msg)

            if confidence == 0:
                """annoying meaningless unexplained crappy boxes that exist for no apparent reason at all"""
                continue

            # xmin, ymin, w, h = _datum[2:]

            if percent_scores:
                confidence /= 100.0

            if clamp_scores:
                confidence = max(min(confidence, 1), 0)

            if 0 <= confidence <= 1:
                pass
            else:
                msg = f"Invalid confidence: {confidence} for file {filename}"

                if ignore_invalid == 2:
                    confidence = 1
                elif ignore_invalid == 1:
                    print(msg)
                else:
                    raise AssertionError(msg)

            # width = float(row['width'])
            # height = float(row['height'])
            label = str(row['class'])

            try:
                target_id = int(row['target_id'])
            except KeyError:
                target_id = -1

            bbox = [xmin, ymin, xmax, ymax]
            obj_entry = {
                'id': target_id,
                'label': label,
                'bbox': bbox,
                'confidence': confidence
            }
            try:
                mask_str = str(row['mask']).strip('"')
            except KeyError:
                pass
            else:
                x_coordinates = []
                y_coordinates = []

                mask = [[float(k) for k in point_string.split(",")]
                        for point_string in mask_str.split(";") if point_string]

                # for point_string in mask_str.split(";"):
                #     if not point_string:
                #         continue
                #     x_coordinate, y_coordinate = point_string.split(",")
                #     x_coordinate = float(x_coordinate)
                #     y_coordinate = float(y_coordinate)
                #     x_coordinates.append(x_coordinate)
                #     y_coordinates.append(y_coordinate)
                # mask = [(x, y) for x, y in zip(x_coordinates, y_coordinates)]

                obj_entry['mask'] = mask

            if frame_id not in obj_dict:
                obj_dict[frame_id] = []
            obj_dict[frame_id].append(obj_entry)
            obj_ids.append(target_id)
    return obj_ids, obj_dict


# def process_seq(
#         seq_idx, n_seq, params, img_root_dir, mode, vid_ext, img_ext, root_dir, out_root_dir,
#         data_type, ignore_missing, ignore_invalid, percent_scores,
#         clamp_scores, class_names, total_unique_obj_ids, total_n_frames, total_sampled_n_frames,
#         stats_only, seq_to_n_unique_obj_ids, vis_size, seg_dir_path, vis_root_dir, ext, save_video,
#         image_exts, codec, fps, out_suffix, raw_ctc_seg, json_fname, json_dict, save_raw, seg_ext,
#         allow_missing_seg,extrapolate_seg, label2id, bnd_id, bbox_source, show_img, show_class,
# ):


def main():
    params = Params()
    paramparse.process(params)

    out_suffix = params.out_suffix
    data_type = params.data_type
    file_name = params.file_name
    root_dir = params.root_dir
    out_root_dir = params.out_root_dir
    out_root_suffix = params.out_root_suffix
    list_file_name = params.list_file_name
    list_file_suffix = params.list_file_suffix
    img_ext = params.img_ext
    # ignore_occl = params.ignore_occl
    show_img = params.show_img
    show_class = params.show_class
    # resize_factor = params.resize_factor
    ext = params.ext
    codec = params.codec
    fps = params.fps
    vis_size = params.vis_size
    vis_root_dir = params.vis_root_dir
    save_path = params.save_path
    # min_vis = params.min_vis
    save_raw = params.save_raw
    save_video = params.save_video
    mode = params.mode
    start_id = params.start_id
    end_id = params.end_id
    ignore_missing = params.ignore_missing
    # label = params.label
    percent_scores = params.percent_scores
    clamp_scores = params.clamp_scores
    ignore_invalid = params.ignore_invalid
    stats_only = params.stats_only
    bbox_source = params.bbox_source
    img_dir = params.img_dir

    seg_dir = params.seg_dir
    seg_ext = params.seg_ext

    raw_ctc_seg = params.raw_ctc_seg
    allow_missing_seg = params.allow_missing_seg
    extrapolate_seg = params.extrapolate_seg
    class_names_path = params.class_names_path

    assert class_names_path, "class_names_path must be provided"

    vid_ext = params.vid_ext

    image_exts = ['jpg', 'bmp', 'png', 'tif']

    if img_dir:
        img_root_dir = linux_path(root_dir, img_dir)
    else:
        img_root_dir = root_dir

    if seg_dir:
        seg_dir_path = linux_path(root_dir, seg_dir)
        assert os.path.exists(seg_dir_path), f"seg_dir: {seg_dir_path} does not exist"
        print(f'reading segmentation images from {seg_dir_path}')
    else:
        seg_dir_path = None

    if list_file_name:
        if list_file_suffix:
            list_file_name = add_suffix(list_file_name, list_file_suffix)

        assert os.path.isfile(list_file_name), f'List file: {list_file_name} does not exist'
        seq_paths = [x.strip() for x in open(list_file_name).readlines() if x.strip()]
        if img_root_dir:
            seq_paths = [linux_path(img_root_dir, x) for x in seq_paths]

    elif img_root_dir:
        if vid_ext:
            seq_paths = [linux_path(img_root_dir, name) for name in os.listdir(img_root_dir) if
                         os.path.isfile(linux_path(img_root_dir, name)) and name.endswith(vid_ext)]

            """remove ext to be compatible with image sequence paths"""
            seq_paths = [os.path.splitext(seq_path)[0] for seq_path in seq_paths]
            seq_paths.sort(key=sortKey)
        else:
            seq_paths = [linux_path(img_root_dir, name) for name in os.listdir(img_root_dir) if
                         os.path.isdir(linux_path(img_root_dir, name))]
            seq_paths.sort(key=sortKey)
    else:
        if not file_name:
            raise IOError('Either list file or a single sequence file must be provided')
        seq_paths = [file_name]

    if not vis_root_dir:
        vis_root_dir = linux_path(os.path.dirname(seq_paths[0]), 'vis')

    class_info = [k.strip() for k in open(class_names_path, 'r').readlines() if k.strip()]
    class_names, class_cols = zip(*[[m.strip() for m in k.split('\t')] for k in class_info])
    class_to_id = {x: i for (i, x) in enumerate(class_names)}
    class_to_col = {x: c for (x, c) in zip(class_names, class_cols)}

    if not out_root_dir and out_root_suffix:
        out_root_dir = f'{root_dir}_{out_root_suffix}'

    pause_after_frame = 1
    total_n_frames = 0
    total_sampled_n_frames = 0
    total_unique_obj_ids = 0

    if end_id < 0:
        end_id = len(seq_paths) - 1

    seq_paths = seq_paths[start_id:end_id + 1]
    seq_to_n_unique_obj_ids = {}

    n_seq = len(seq_paths)
    print('Running over {} sequences'.format(n_seq))

    if not bbox_source:
        if data_type == 'annotation':
            bbox_source = 'ground_truth'
        elif data_type == 'detection':
            bbox_source = 'object_detector'
        else:
            bbox_source = data_type

    json_fname = params.json_fname

    json_path = None
    xml_writer = None

    if json_fname:
        json_dict = {
            "images": [],
            "type": "instances",
            "annotations": [],
            "categories": []
        }
        if params.json_desc:
            json_dict['description'] = params.json_desc

        for label, label_id in class_to_id.items():
            category_info = {'supercategory': 'none', 'id': label_id, 'name': label}
            json_dict['categories'].append(category_info)

        bnd_id = 1

        json_dir = params.json_dir

        if not json_dir:
            json_dir = os.path.dirname(seq_paths[0])

        if out_root_dir:
            json_dir = json_dir.replace(root_dir, out_root_dir, 1)

        os.makedirs(json_dir, exist_ok=True)
        json_path = os.path.join(json_dir, json_fname)

        print(f'saving json to {json_path}')

    n_classes = len(class_names)

    for seq_idx, img_root_dir in enumerate(seq_paths):
        seq_name = os.path.basename(img_root_dir)

        if mode == 0:
            src_path = linux_path(img_root_dir, 'img1')
        else:
            src_path = img_root_dir

        vid_cap = None
        src_files = None

        if vid_ext:
            is_vid = 1
            vid_path = f'{src_path}.{vid_ext}'
            assert os.path.isfile(vid_path), f"invalid vid_path: {vid_path}"
            vid_cap = cv2.VideoCapture()
            if not vid_cap.open(vid_path):
                raise AssertionError(f'Video file {vid_path} could not be opened')
            n_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            img_seq_out_dir = src_path
            if out_root_dir:
                img_seq_out_dir = img_seq_out_dir.replace(root_dir, out_root_dir, 1)

            if params.save_img_seq:
                print(f'saving image sequence to {img_seq_out_dir}')
                os.makedirs(img_seq_out_dir, exist_ok=True)
        else:
            assert os.path.isdir(src_path), f'invalid source path: {src_path}'
            src_files = [f for f in os.listdir(src_path) if
                         os.path.isfile(linux_path(src_path, f)) and f.endswith(img_ext)]
            src_files.sort(key=sortKey)
            n_frames = len(src_files)
            is_vid = 0

        print(f'n_frames: {n_frames}')

        sampled_frame_ids = list(range(n_frames))

        start_frame_id = params.start_frame_id
        end_frame_id = params.end_frame_id
        frame_stride = params.frame_stride

        if frame_stride <= 0:
            frame_stride = 1

        if end_frame_id < start_frame_id:
            end_frame_id = len(sampled_frame_ids) - 1

        # print(f'params.end_frame_id: {params.end_frame_id}')
        # print(f'end_frame_id: {end_frame_id}')

        sampled_frame_ids = sampled_frame_ids[start_frame_id:end_frame_id + 1:frame_stride]

        if params.sample:
            print(f'sampling 1 in {params.sample} frames')
            sampled_frame_ids = sampled_frame_ids[::params.sample]

            n_sampled_frames = len(sampled_frame_ids)
            print(f'n_sampled_frames: {n_sampled_frames}')

        total_n_frames += n_frames
        if params.sample:
            total_sampled_n_frames += n_sampled_frames

        # if is_vid:
        #     seq_name = os.path.splitext(seq_name)[0]

        print('seq_name: ', seq_name)

        print('sequence {}/{}: {}: '.format(seq_idx + 1, n_seq, seq_name))

        is_csv = 0

        if mode == 0:
            ann_path = linux_path(img_root_dir, 'gt', 'gt.txt')
        elif mode == 1:
            ann_path = linux_path(img_root_dir, f'../../{data_type.capitalize()}/{seq_name}.txt')
        elif mode == 2:
            is_csv = 1
            if is_vid:
                # img_path_noext = os.path.splitext(img_root_dir)[0]
                ann_path = f'{img_root_dir}.csv'
            else:
                ann_path = linux_path(img_root_dir, f'{data_type}.csv')
        else:
            raise AssertionError(f'Invalid mode: {mode}')

        ann_path = os.path.abspath(ann_path)

        if not os.path.exists(ann_path):
            msg = f"Annotation file for sequence {seq_name} not found: {ann_path}"
            if ignore_missing:
                print(msg)
                return
            else:
                raise AssertionError(msg)

        print(f'Reading {data_type} from {ann_path}')

        if is_csv:
            obj_ids, obj_dict = parse_csv(ann_path, sampled_frame_ids,
                                          ignore_invalid, percent_scores, clamp_scores)
        else:
            assert len(class_names) == 1 or params.allow_ignored, \
                "multiple class names not supported in MOT mode unless allow_ignored is on"
            obj_ids, obj_dict = parse_mot(
                ann_path, sampled_frame_ids, class_names[0], ignore_invalid, percent_scores, clamp_scores,
                params.allow_ignored)

        print(f'Done reading {data_type}')

        enable_resize = 0

        unique_obj_ids = list(set(obj_ids))
        max_obj_id = max(obj_ids)
        n_unique_obj_ids = len(unique_obj_ids)

        total_unique_obj_ids += n_unique_obj_ids

        print(f'{seq_name}: {n_unique_obj_ids}')

        n_col_levels = int(max_obj_id ** (1. / 3) + 1)

        col_levels = [int(x) for x in np.linspace(
            # exclude too light and too dark colours to avoid confusion with white and black
            50, 200,
            n_col_levels, dtype=int)]
        import itertools
        import random
        rgb_cols = list(itertools.product(col_levels, repeat=3))
        random.shuffle(rgb_cols)

        assert len(rgb_cols) >= max_obj_id, "insufficient number of colours created"

        seq_to_n_unique_obj_ids[seq_name] = n_unique_obj_ids

        if stats_only:
            continue

        if not vis_size:
            if is_vid:
                vis_height = vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                vis_width = vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            else:
                temp_img = cv2.imread(linux_path(src_path, src_files[0]))
                vis_height, vis_width, _ = temp_img.shape

            if seg_dir_path is not None:
                """concatenate segmentation image to the right of the source image"""
                vis_width *= 2
        else:
            vis_width, vis_height = [int(x) for x in vis_size.split('x')]
            enable_resize = 1

        if not save_path:
            save_path = linux_path(vis_root_dir, os.path.basename(img_root_dir) + '.' + ext)

        save_dir = os.path.dirname(save_path)
        save_seq_name = os.path.splitext(os.path.basename(save_path))[0]

        if save_video:
            if ext in image_exts:
                video_out = ImageWriter(save_path)
                print('Saving {}x{} visualization sequence to {}'.format(
                    vis_width, vis_height, save_path))
            else:
                enable_resize = 1
                save_dir = os.path.dirname(save_path)
                if save_dir and not os.path.isdir(save_dir):
                    os.makedirs(save_dir)

                fourcc = cv2.VideoWriter_fourcc(*codec)
                video_out = cv2.VideoWriter(save_path, fourcc, fps, (vis_width, vis_height))
                if video_out is None:
                    raise IOError('Output video file could not be opened: {}'.format(save_path))
                print('Saving {}x{} visualization video to {}'.format(
                    vis_width, vis_height, save_path))

        if out_suffix:
            out_dir_name = f'{data_type}_{out_suffix}'
        else:
            out_dir_name = f'{data_type}'

        if seg_dir_path is not None and raw_ctc_seg:
            gold_seg_dir_path = linux_path(seg_dir_path, f'{seq_name}_GT', 'SEG')
            silver_seg_dir_path = linux_path(seg_dir_path, f'{seq_name}_ST', 'SEG')
            print(f'looking for gold standard CTC segmentations in {gold_seg_dir_path}')
            print(f'looking for silver standard CTC segmentations in {silver_seg_dir_path}')

        xml_zip_file = xml_dir_path = None
        if save_video != 2 and not json_fname:
            if mode == 0:
                xml_dir_path = linux_path(save_dir, save_seq_name, out_dir_name)
            else:
                if is_vid:
                    # img_path_noext = os.path.splitext(img_root_dir)[0]
                    xml_dir_path = linux_path(img_root_dir, out_dir_name)
                else:
                    xml_dir_path = linux_path(img_root_dir, out_dir_name)

            if params.zip:
                from zipfile import ZipFile

                xml_zip_path = xml_dir_path + '.zip'
                print(f'saving xml files to zip file {xml_zip_path}')
                xml_zip_file = ZipFile(xml_zip_path, 'w')
            else:
                os.makedirs(xml_dir_path, exist_ok=True)
                print(f'saving xml files to {xml_dir_path}')

        obj_ids_to_bboxes = {}
        obj_ids_to_seg_pts = {}

        missing_seg_images = []

        vid_frame_id = -1

        for frame_id in tqdm(sampled_frame_ids):
            if is_vid:
                filename = f'image{frame_id + 1:06d}.jpg'

                assert vid_frame_id <= frame_id, "vid_frame_id exceeds frame_id"

                image = None

                while vid_frame_id < frame_id:
                    ret, image = vid_cap.read()
                    vid_frame_id += 1
                    if not ret:
                        raise AssertionError('Frame {:d} could not be read'.format(vid_frame_id + 1))

                img_file_path = linux_path(img_seq_out_dir, filename)

                if params.save_img_seq and not os.path.isfile(img_file_path):
                    cv2.imwrite(img_file_path, image)
            else:
                filename = src_files[frame_id]
                img_file_path = linux_path(src_path, filename)
                if not os.path.exists(img_file_path):
                    raise SystemError('Image file {} does not exist'.format(img_file_path))

                image = cv2.imread(img_file_path)

            filename_no_ext = os.path.splitext(filename)[0]
            height, width = image.shape[:2]

            if save_video != 2:
                if json_fname:
                    rel_path = linux_path(f'{seq_name}/{filename}')
                    img_id = linux_path(f'{seq_name}/{filename_no_ext}')

                    json_img_info = {
                        'file_name': rel_path,
                        'height': height,
                        'width': width,
                        'id': seq_name + '/' + img_id
                    }
                    json_dict['images'].append(json_img_info)

                else:
                    xml_fname = filename_no_ext + '.xml'
                    if params.zip:
                        out_xml_path = xml_fname
                    else:
                        out_xml_path = linux_path(xml_dir_path, xml_fname)

            vis_img = image.copy()
            seg_img_vis = None

            curr_obj_ids = []
            n_objs = 0
            try:
                objects = obj_dict[frame_id]
            except KeyError:
                msg = f'No {data_type} found for frame {frame_id}: {img_file_path}'
                if params.allow_empty:
                    print(msg)
                    objects = []
                else:
                    raise AssertionError(msg)

            if save_video and save_raw:
                video_out.write(image)

            # out_frame_id += 1
            # out_filename = 'image{:06d}.jpg'.format(out_frame_id)

            image_shape = [height, width, 3]
            if save_video != 2 and not json_fname:
                xml_writer = PascalVocWriter(src_path, filename, image_shape)

            n_objs = len(objects)

            # obj_ids = [obj['id'] for obj in objects]

            if seg_dir_path is not None:

                seg_img = None

                if raw_ctc_seg:
                    gold_seg_file_path = linux_path(gold_seg_dir_path, f'{filename_no_ext}.{seg_ext}')
                    silver_seg_file_path = linux_path(silver_seg_dir_path, f'{filename_no_ext}.{seg_ext}')

                    unique_ids = []
                    unique_ids_dict = dict(
                        silver=[],
                        gold=[],
                    )
                    gold_seg_img = silver_seg_img = None
                    if os.path.exists(gold_seg_file_path):
                        gold_seg_img = cv2.imread(gold_seg_file_path, cv2.IMREAD_UNCHANGED)
                        gold_unique_ids = np.unique(gold_seg_img)

                        assert len(gold_unique_ids) > 1, \
                            f"invalid gold_seg_img with single unique_id: {gold_unique_ids}"

                        unique_ids_dict['gold'] = list(gold_unique_ids)
                        unique_ids += unique_ids_dict['gold']

                        seg_img = gold_seg_img

                    if os.path.exists(silver_seg_file_path):
                        silver_seg_img = cv2.imread(silver_seg_file_path, cv2.IMREAD_UNCHANGED)
                        silver_unique_ids = np.unique(silver_seg_img)

                        assert len(silver_unique_ids) > 1, \
                            f"invalid silver_seg_img with single unique_id: {silver_unique_ids}"

                        unique_ids_dict['silver'] = list(silver_unique_ids)
                        unique_ids += unique_ids_dict['silver']

                        seg_img = silver_seg_img

                    if gold_seg_img is None and silver_seg_img is None:
                        msg = f"neither gold not silver segmentation file found for {filename_no_ext}"
                        if allow_missing_seg:
                            print(msg)
                        else:
                            raise AssertionError(msg)

                else:
                    seg_file_path = linux_path(seg_dir_path, seq_name, f'{filename_no_ext}.{seg_ext}')
                    assert os.path.exists(seg_file_path), f"seg_file: {seg_file_path} does not exist"

                    seg_img = cv2.imread(seg_file_path, cv2.IMREAD_GRAYSCALE)

                    unique_ids = np.unique(seg_img)
                    # seg_pts = contour_pts_from_mask(seg_img)

            for obj in objects:
                label = obj['label']
                try:
                    class_col = col_bgr[class_to_col[label]]
                except KeyError:
                    msg = f'Invalid class {label}\n'
                    if params.ignore_invalid_class:
                        print(msg)
                        continue
                    else:
                        raise AssertionError(msg)

                obj_id = obj['id']
                curr_obj_ids.append(obj_id)

                bbox = obj['bbox']
                confidence = obj['confidence']

                xmin, ymin, xmax, ymax = bbox

                if extrapolate_seg:
                    if obj_id in obj_ids_to_bboxes:
                        prev_bbox = obj_ids_to_bboxes[obj_id][-1]
                    else:
                        obj_ids_to_bboxes[obj_id] = []
                        prev_bbox = None

                    obj_ids_to_bboxes[obj_id].append(bbox)

                xmin, xmax = clamp([xmin, xmax], 0, width - 1)
                ymin, ymax = clamp([ymin, ymax], 0, height - 1)

                if json_fname:
                    o_width = xmax - xmin
                    o_height = ymax - ymin
                    category_id = class_to_id[label]

                    json_ann = {
                        'image_id': json_img_info['id'],
                        'area': o_width * o_height,
                        'iscrowd': 0,
                        'bbox': [xmin, ymin, o_width, o_height],
                        'label': label,
                        'category_id': category_id,
                        'ignore': 0,
                        'id': bnd_id
                    }
                    bnd_id += 1
                else:
                    xml_dict = dict(
                        xmin=int(xmin),
                        ymin=int(ymin),
                        xmax=int(xmax),
                        ymax=int(ymax),
                        name=label,
                        difficult=False,
                        bbox_source=bbox_source,
                        id_number=obj_id,
                        score=confidence,
                        mask=None,
                        mask_img=None
                    )

                seg_mask = None
                if seg_dir_path is None:
                    try:
                        mask_pts = obj['mask']
                    except KeyError:
                        pass
                    else:
                        seg_mask = mask_pts_to_img(mask_pts, height, width, to_rle=False)
                else:
                    if raw_ctc_seg:
                        obj_seg_img = None
                        """first try silver segmentations and then the gold once since the former 
                        are more consistent and numerous"""
                        if obj_id in unique_ids_dict['silver']:
                            obj_seg_img = silver_seg_img
                        elif obj_id in unique_ids_dict['gold']:
                            obj_seg_img = gold_seg_img
                        else:
                            if show_img:
                                if silver_seg_img is not None:
                                    silver_seg_img_vis = (silver_seg_img.astype(np.float32) *
                                                          (255 / silver_unique_ids[-1])).astype(np.uint8)
                                    cv2.imshow(f'{silver_seg_file_path}', silver_seg_img_vis)

                                if gold_seg_img is not None:
                                    gold_seg_img_vis = (gold_seg_img.astype(np.float32) *
                                                        (255 / gold_unique_ids[-1])).astype(np.uint8)
                                    cv2.imshow(f'{gold_seg_file_path}', gold_seg_img_vis)

                            print(
                                f'neither gold nor silver segmentation found in frame {frame_id}: {filename} '
                                f'for object {obj_id} :: {unique_ids}'
                            )

                            if extrapolate_seg:
                                """try to extrapolate segmentation points from the most recent points and box"""
                                if prev_bbox is not None:
                                    prev_seg_pts = obj_ids_to_seg_pts[obj_id][-1]

                                    tx = bbox[0] - prev_bbox[0]
                                    ty = bbox[1] - prev_bbox[1]

                                    seg_pts = [[x + tx, y + ty] for x, y, f in prev_seg_pts]

                                    obj_seg_img, blended_seg_img = contour_pts_to_mask(seg_pts, vis_img)

                                    if show_img:
                                        cv2.imshow(f'extrapolated obj_seg_img', obj_seg_img)
                                        cv2.imshow(f'extrapolated blended_seg_img', blended_seg_img)
                                else:
                                    print('no previous data exists for this object to extrapolate from')

                            if show_img:
                                cv2.waitKey(0)

                            # print()
                    else:
                        if obj_id not in unique_ids:
                            print(
                                # raise AssertionError(
                                f'No segmentation found for object {obj_id} :: {unique_ids}'
                            )

                    if obj_seg_img is None:
                        msg = f'no segmentation found in frame {frame_id}: {filename} for object {obj_id}'
                        if allow_missing_seg:
                            """skip this object since it has missing segmentation"""
                            missing_seg_images.append(filename)
                            print(msg)
                            # continue
                            seg_mask = None
                        else:
                            raise AssertionError(msg)
                    else:
                        seg_mask = np.zeros_like(obj_seg_img, dtype=np.uint8)
                        seg_mask[obj_seg_img == obj_id] = 255

                        # cv2.imshow('seg_mask', seg_mask)
                        # cv2.waitKey(0)

                        seg_pts, _, _ = mask_img_to_pts(seg_mask)

                        if extrapolate_seg:
                            if obj_id not in obj_ids_to_seg_pts:
                                obj_ids_to_seg_pts[obj_id] = []

                            obj_ids_to_seg_pts[obj_id].append(seg_pts)

                        if json_fname:
                            mask_pts_flat = []
                            for _pt in seg_pts:
                                mask_pts_flat.append(float(_pt[0]))
                                mask_pts_flat.append(float(_pt[1]))
                            json_ann.update({
                                'segmentation': [mask_pts_flat, ],
                            })
                        else:
                            xml_dict['mask'] = seg_pts

                    # print()

                if json_fname:
                    json_dict['annotations'].append(json_ann)
                elif xml_writer is not None:
                    xml_writer.addBndBox(**xml_dict)

                if show_img or (save_video and not save_raw):
                    obj_col = rgb_cols[obj_id]
                    if show_class:
                        _label = '{} {}'.format(label, obj_id)
                    else:
                        _label = '{}'.format(obj_id)

                    if seg_mask is not None:
                        if seg_img_vis is None:
                            seg_img_vis = np.zeros_like(vis_img)

                        seg_mask_binary = seg_mask.astype(bool)
                        vis_img[seg_mask_binary] = vis_img[seg_mask_binary] * 0.5 + np.asarray(obj_col) * 0.5
                        seg_col = class_col if n_classes > 1 else obj_col
                        if n_classes > 1:
                            seg_img_vis[seg_mask_binary] = np.asarray(seg_col)

                    drawBox(vis_img, xmin, ymin, xmax, ymax, label=_label,
                            font_size=1.0, box_color=obj_col, thickness=6)

            if xml_writer is not None:
                xml_writer.save(targetFile=out_xml_path, verbose=False, zip_file=xml_zip_file)

            if save_video or show_img:
                curr_obj_cols = [rgb_cols[k] for k in curr_obj_ids]

                if seg_img_vis is not None:
                    vis_img = np.concatenate((vis_img, seg_img_vis), axis=1)
                if enable_resize:
                    vis_img = resize_ar(vis_img, vis_width, vis_height)

                obj_txt = "object" if n_objs == 1 else "objects"
                cv2.putText(vis_img, f'frame {frame_id}: {filename} :: {n_objs} {obj_txt}', (5, 15),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255))

                show_labels(vis_img, curr_obj_ids, curr_obj_cols)

                if save_video and not save_raw:
                    video_out.write(vis_img)

                if show_img:
                    cv2.imshow(seq_name, vis_img)

                    k = cv2.waitKey(1 - pause_after_frame) & 0xFF
                    if k == ord('q'):
                        sys.exit(0)
                    elif k == 27:
                        break
                    elif k == 32:
                        pause_after_frame = 1 - pause_after_frame

        if xml_zip_file is not None:
            xml_zip_file.close()

        if missing_seg_images:
            missing_seg_images = list(set(missing_seg_images))
            missing_seg_images_file = linux_path(xml_dir_path, 'missing_seg_images.txt')
            print(f'writing {len(missing_seg_images)} missing_seg_images to {missing_seg_images_file}')
            with open(missing_seg_images_file, 'w') as fid:
                fid.write('\n'.join(missing_seg_images))

        if save_video:
            video_out.release()

        save_path = ''

        if show_img:
            cv2.destroyWindow(seq_name)
        # total_n_frames += out_frame_id
        # print('out_n_frames: ', out_frame_id)

    if json_fname:
        save_json(json_dict, json_path, params.json_gz)

    print('total_n_frames: ', total_n_frames)
    print('total_sampled_n_frames: ', total_sampled_n_frames)
    print('total_unique_obj_ids: ', total_unique_obj_ids)


if __name__ == '__main__':
    main()
