import numpy as np
import os
import sys
import cv2

import paramparse

from tqdm import tqdm

from eval_utils import ImageSequenceWriter as ImageWriter
from pascal_voc_io import PascalVocWriter
from eval_utils import contour_pts_to_mask, contour_pts_from_mask, sortKey, resize_ar, \
    drawBox, show_labels, clamp, linux_path


class Params:
    def __init__(self):
        self.cfg = ()
        self.bbox_source = ''
        self.clamp_scores = 0
        self.codec = 'H264'
        self.csv_file_name = ''
        self.data_type = 'annotations'
        self.ext = 'mkv'
        self.extrapolate_seg = 0
        self.allow_missing_seg = 1
        self.file_name = ''
        self.fps = 30
        self.ignore_invalid = 0
        self.ignore_missing = 0
        self.ignore_occl = 1
        self.img_dir = ''
        self.seg_dir = ''
        self.img_ext = 'jpg'
        self.seg_ext = 'png'
        self.label = 'person'
        self.list_file_name = ''
        self.map_folder = ''
        self.min_vis = 0.5
        self.mode = 1
        self.n_classes = 4
        self.n_frames = 0
        self.out_root_dir = ''
        self.out_suffix = ''
        self.percent_scores = 0
        self.resize_factor = 1.0
        self.root_dir = ''
        self.save_file_name = ''
        self.save_path = ''
        self.save_raw = 0
        self.save_video = 0
        self.show_class = 0
        self.show_img = 1
        self.start_id = 0
        self.stats_only = 0
        self.raw_ctc_seg = 0
        self.vis_size = ''



def main():
    params = Params()
    paramparse.process(params)

    out_suffix = params.out_suffix
    data_type = params.data_type
    file_name = params.file_name
    root_dir = params.root_dir
    list_file_name = params.list_file_name
    img_ext = params.img_ext
    # ignore_occl = params.ignore_occl
    show_img = params.show_img
    show_class = params.show_class
    # resize_factor = params.resize_factor
    ext = params.ext
    codec = params.codec
    fps = params.fps
    vis_size = params.vis_size
    out_root_dir = params.out_root_dir
    save_path = params.save_path
    # min_vis = params.min_vis
    save_raw = params.save_raw
    save_video = params.save_video
    mode = params.mode
    start_id = params.start_id
    ignore_missing = params.ignore_missing
    label = params.label
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

    image_exts = ['jpg', 'bmp', 'png', 'tif']

    if img_dir:
        img_path = linux_path(root_dir, img_dir)
    else:
        img_path = root_dir

    if seg_dir:
        seg_dir_path = linux_path(root_dir, seg_dir)
        assert os.path.exists(seg_dir_path), f"seg_dir: {seg_dir_path} does not exist"
        print(f'reading segmentation images from {seg_dir_path}')
    else:
        seg_dir_path = None

    if list_file_name:
        if not os.path.exists(list_file_name):
            raise IOError('List file: {} does not exist'.format(list_file_name))
        file_list = [x.strip() for x in open(list_file_name).readlines() if x.strip()]
        if img_path:
            file_list = [linux_path(img_path, x) for x in file_list]
    elif img_path:
        if img_path.startswith('camera'):
            file_list = [img_path]
        else:
            file_list = [linux_path(img_path, name) for name in os.listdir(img_path) if
                         os.path.isdir(linux_path(img_path, name))]
            file_list.sort(key=sortKey)
    else:
        if not file_name:
            raise IOError('Either list file or a single sequence file must be provided')
        file_list = [file_name]

    if not out_root_dir:
        out_root_dir = linux_path(os.path.dirname(file_list[0]), 'vis')

    pause_after_frame = 1
    total_n_frames = 0
    total_unique_obj_ids = 0
    file_list = file_list[start_id:]
    seq_to_n_unique_obj_ids = {}

    n_seq = len(file_list)
    print('Running over {} sequences'.format(n_seq))

    if not bbox_source:
        if data_type == 'annotation':
            bbox_source = 'ground_truth'
        elif data_type == 'detection':
            bbox_source = 'object_detector'
        else:
            bbox_source = data_type

    for seq_idx, img_path in enumerate(file_list):
        seq_name = os.path.basename(img_path)
        print('seq_name: ', seq_name)

        print('sequence {}/{}: {}: '.format(seq_idx + 1, n_seq, seq_name))

        if mode == 0:
            ann_path = linux_path(img_path, 'gt', 'gt.txt')
        else:
            ann_path = linux_path(img_path, f'../../{data_type.capitalize()}/{seq_name}.txt')

        ann_path = os.path.abspath(ann_path)

        if not os.path.exists(ann_path):
            msg = "Annotation file for sequence {} not found: {}".format(seq_name, ann_path)
            if ignore_missing:
                print(msg)
                continue
            else:
                raise IOError(msg)

        print('Reading {} from {}'.format(data_type, ann_path))

        ann_lines = open(ann_path).readlines()

        ann_data = [[float(x) for x in _line.strip().split(',')] for _line in ann_lines if _line.strip()]
        # ann_data.sort(key=lambda x: x[0])
        # ann_data = np.asarray(ann_data)

        if mode == 0:
            src_path = linux_path(img_path, 'img1')
        else:
            src_path = img_path

        src_files = [f for f in os.listdir(src_path) if
                     os.path.isfile(linux_path(src_path, f)) and f.endswith(img_ext)]
        src_files.sort(key=sortKey)
        n_frames = len(src_files)

        print('n_frames: ', n_frames)

        total_n_frames += n_frames

        obj_ids = []

        obj_dict = {}
        for __id, _data in enumerate(ann_data):

            # if _data[7] != 1 or _data[8] < min_vis:
            #     continue
            obj_id = int(_data[1])

            obj_ids.append(obj_id)

            # Bounding box sanity check
            bbox = [float(x) for x in _data[2:6]]
            l, t, w, h = bbox
            xmin = int(l)
            ymin = int(t)
            xmax = int(xmin + w)
            ymax = int(ymin + h)
            if xmin >= xmax or ymin >= ymax:
                msg = f'Invalid box {[xmin, ymin, xmax, ymax]}\n in line {__id} : {_data}\n'
                if ignore_invalid:
                    print(msg)
                else:
                    raise AssertionError(msg)

            confidence = float(_data[6])

            if w <= 0 or h <= 0 or confidence == 0:
                """annoying meaningless unexplained crappy boxes that exist for no apparent reason at all"""
                continue

            frame_id = int(_data[0]) - 1
            # xmin, ymin, w, h = _data[2:]

            if percent_scores:
                confidence /= 100.0

            if clamp_scores:
                confidence = max(min(confidence, 1), 0)

            if 0 <= confidence <= 1:
                pass
            else:
                msg = "Invalid confidence: {} in line {} : {}".format(
                    confidence, __id, _data)

                if ignore_invalid == 2:
                    confidence = 1
                elif ignore_invalid == 1:
                    print(msg)
                else:
                    raise AssertionError(msg)

            obj_entry = {
                'id': obj_id,
                'label': label,
                'bbox': bbox,
                'confidence': confidence
            }
            if frame_id not in obj_dict:
                obj_dict[frame_id] = []
            obj_dict[frame_id].append(obj_entry)

        print('Done reading {}'.format(data_type))

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
            temp_img = cv2.imread(linux_path(src_path, src_files[0]))
            vis_height, vis_width, _ = temp_img.shape

            if seg_dir_path is not None:
                """concatenate segmentation image to the right of the source image"""
                vis_width *= 2
        else:
            vis_width, vis_height = [int(x) for x in vis_size.split('x')]
            enable_resize = 1

        if not save_path:
            save_path = linux_path(out_root_dir, os.path.basename(img_path) + '.' + ext)

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
            out_dir_name = '{}_{}'.format(data_type, out_suffix)
        else:
            out_dir_name = '{}'.format(data_type)
        if mode == 0:
            xml_dir_path = linux_path(save_dir, save_seq_name, out_dir_name)
        else:
            xml_dir_path = linux_path(img_path, out_dir_name)

        os.makedirs(xml_dir_path, exist_ok=1)

        if seg_dir_path is not None and raw_ctc_seg:
            gold_seg_dir_path = linux_path(seg_dir_path, f'{seq_name}_GT', 'SEG')
            silver_seg_dir_path = linux_path(seg_dir_path, f'{seq_name}_ST', 'SEG')
            print(f'looking for gold standard CTC segmentations in {gold_seg_dir_path}')
            print(f'looking for silver standard CTC segmentations in {silver_seg_dir_path}')

        print('saving xml files to {}'.format(xml_dir_path))

        obj_ids_to_bboxes = {}
        obj_ids_to_seg_pts = {}

        missing_seg_images = []

        for frame_id in tqdm(range(n_frames)):
            filename = src_files[frame_id]

            filename_no_ext = os.path.splitext(filename)[0]

            xml_fname = filename_no_ext + '.xml'

            out_xml_path = linux_path(xml_dir_path, xml_fname)

            img_file_path = linux_path(src_path, filename)
            if not os.path.exists(img_file_path):
                raise SystemError('Image file {} does not exist'.format(img_file_path))

            image = cv2.imread(img_file_path)

            vis_img = image.copy()
            seg_img_vis = np.zeros_like(vis_img)

            height, width = image.shape[:2]

            if frame_id in obj_dict:
                if save_video and save_raw:
                    video_out.write(image)

                # out_frame_id += 1
                # out_filename = 'image{:06d}.jpg'.format(out_frame_id)

                image_shape = [height, width, 3]
                xml_writer = PascalVocWriter(src_path, filename, image_shape)

                objects = obj_dict[frame_id]

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

                curr_obj_ids = []
                for obj in objects:
                    obj_id = obj['id']

                    curr_obj_ids.append(obj_id)

                    label = obj['label']
                    bbox = obj['bbox']
                    confidence = obj['confidence']

                    l, t, w, h = bbox
                    xmin = int(l)
                    ymin = int(t)

                    xmax = int(xmin + w)
                    ymax = int(ymin + h)

                    curr_bbox = [xmin, ymin, xmax, ymax]

                    if extrapolate_seg:
                        if obj_id in obj_ids_to_bboxes:
                            prev_bbox = obj_ids_to_bboxes[obj_id][-1]
                        else:
                            obj_ids_to_bboxes[obj_id] = []
                            prev_bbox = None

                        obj_ids_to_bboxes[obj_id].append(curr_bbox)

                    xmin, xmax = clamp([xmin, xmax], 0, width - 1)
                    ymin, ymax = clamp([ymin, ymax], 0, height - 1)

                    xml_dict = dict(
                        xmin=xmin,
                        ymin=ymin,
                        xmax=xmax,
                        ymax=ymax,
                        name=label,
                        difficult=False,
                        bbox_source=bbox_source,
                        id_number=obj_id,
                        score=confidence,
                        mask=None,
                        mask_img=None
                    )

                    if seg_dir_path is not None:

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

                                        tx = curr_bbox[0] - prev_bbox[0]
                                        ty = curr_bbox[1] - prev_bbox[1]

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

                            seg_pts, _, _ = contour_pts_from_mask(seg_mask)

                            if extrapolate_seg:
                                if obj_id not in obj_ids_to_seg_pts:
                                    obj_ids_to_seg_pts[obj_id] = []

                                obj_ids_to_seg_pts[obj_id].append(seg_pts)

                            xml_dict['mask'] = seg_pts

                        # print()

                    xml_writer.addBndBox(**xml_dict)

                    if show_img or (save_video and not save_raw):
                        obj_col = rgb_cols[obj_id]
                        if show_class:
                            _label = '{} {}'.format(label, obj_id)
                        else:
                            _label = '{}'.format(obj_id)

                        if seg_dir_path is not None and seg_mask is not None:
                            seg_mask_binary = seg_mask.astype(bool)
                            color_mask = np.asarray(obj_col)
                            vis_img[seg_mask_binary] = vis_img[seg_mask_binary] * 0.5 + color_mask * 0.5
                            seg_img_vis[seg_mask_binary] = color_mask
                        else:
                            drawBox(vis_img, xmin, ymin, xmax, ymax, label=_label, font_size=0.5, box_color=obj_col)

                xml_writer.save(targetFile=out_xml_path)
            else:
                print('No {} found for frame {}: {}'.format(data_type, frame_id, img_file_path))

            if save_video or show_img:
                curr_obj_cols = [rgb_cols[k] for k in curr_obj_ids]

                if seg_dir_path is not None:
                    # if seg_img is not None:
                    #     seg_img_vis = (seg_img.astype(np.float32) * (255 / unique_ids[-1])).astype(np.uint8)
                    # else:
                    #     seg_img_vis = np.zeros_like(vis_img)

                    vis_img = np.concatenate((vis_img, seg_img_vis), axis=1)

                    # cv2.imshow('seg_img_vis', seg_img_vis)

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

    print('total_n_frames: ', total_n_frames)
    print('total_unique_obj_ids: ', total_unique_obj_ids)


if __name__ == '__main__':
    main()
