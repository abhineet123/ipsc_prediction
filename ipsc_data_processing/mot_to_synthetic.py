import numpy as np
import os
import cv2

import paramparse

from eval_utils import sortKey, get_shifted_boxes, clamp, linux_path

class Params:

    def __init__(self):
        self.cfg = ('',)
        self.clamp_scores = 0
        self.codec = 'H264'
        self.csv_file_name = ''
        self.data_type = 'annotations'
        self.ext = 'mkv'
        self.file_name = ''
        self.fps = 30
        self.ignore_invalid = 0
        self.ignore_missing = 0
        self.ignore_occl = 1
        self.img_ext = 'jpg'
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
        self.vis_size = ''

        self.synthetic_samples = 3
        self.synthetic_neg_iou = 0.3
        self.min_shift_ratio = 0
        self.replace_existing = 1


def main():
    params = Params()

    paramparse.process(params)

    out_suffix = params.out_suffix
    data_type = params.data_type
    file_name = params.file_name
    root_dir = params.root_dir
    list_file_name = params.list_file_name
    img_ext = params.img_ext
    ignore_occl = params.ignore_occl
    show_img = params.show_img
    show_class = params.show_class
    resize_factor = params.resize_factor
    ext = params.ext
    codec = params.codec
    fps = params.fps
    vis_size = params.vis_size
    out_root_dir = params.out_root_dir
    save_path = params.save_path
    min_vis = params.min_vis
    save_raw = params.save_raw
    save_video = params.save_video
    mode = params.mode
    start_id = params.start_id
    ignore_missing = params.ignore_missing
    label = params.label
    percent_scores = params.percent_scores
    clamp_scores = params.clamp_scores
    ignore_invalid = params.ignore_invalid

    synthetic_samples = params.synthetic_samples
    synthetic_neg_iou = params.synthetic_neg_iou
    min_shift_ratio = params.min_shift_ratio
    replace_existing = params.replace_existing

    image_exts = ['jpg', 'bmp', 'png', 'tif']

    if list_file_name:
        if not os.path.exists(list_file_name):
            raise IOError('List file: {} does not exist'.format(list_file_name))
        file_list = [x.strip() for x in open(list_file_name).readlines() if x.strip()]
        if root_dir:
            file_list = [os.path.join(root_dir, x) for x in file_list]
    elif root_dir:
        if root_dir.startswith('camera'):
            file_list = [root_dir]
        else:
            file_list = [os.path.join(root_dir, name) for name in os.listdir(root_dir) if
                         os.path.isdir(os.path.join(root_dir, name))]
            file_list.sort(key=sortKey)
    else:
        if not file_name:
            raise IOError('Either list file or a single sequence file must be provided')
        file_list = [file_name]
    n_seq = len(file_list)

    if not out_root_dir:
        out_root_dir = os.path.join(os.path.dirname(file_list[0]), 'vis')

    print('Running over {} sequences'.format(n_seq))
    pause_after_frame = 1
    total_n_frames = 0
    file_list = file_list[start_id:]
    for seq_idx, img_path in enumerate(file_list):
        seq_name = os.path.basename(img_path)
        print('seq_name: ', seq_name)

        print('sequence {}/{}: {}: '.format(seq_idx + 1, n_seq, seq_name))

        if mode == 0:
            ann_path = os.path.join(img_path, 'gt', 'gt.txt')
        else:
            ann_path = os.path.join(img_path, '../../{}/{}.txt'.format(
                data_type.capitalize(), seq_name))

        ann_path = os.path.abspath(ann_path)

        if not os.path.exists(ann_path):
            msg = "Annotation file for sequence {} not found: {}".format(seq_name, ann_path)
            if ignore_missing:
                print(msg)
                continue
            else:
                raise IOError(msg)

        ann_dir = os.path.dirname(ann_path)
        out_root_dir = os.path.dirname(ann_dir)
        ann_dir_name = os.path.basename(ann_dir)

        out_dir_path = linux_path(out_root_dir, '{}_syn_{}'.format(ann_dir_name, synthetic_samples))
        os.makedirs(out_dir_path, exist_ok=True)
        mot_path = linux_path(out_dir_path, '{}.txt'.format(seq_name))

        if os.path.exists(mot_path) and not replace_existing:
            print('\nskipping existing syn file: {} for mot: {}'.format(mot_path, ann_path))
            continue

        print('Reading {} from {}'.format(data_type, ann_path))

        ann_lines = open(ann_path).readlines()

        ann_data = [[float(x) for x in _line.strip().split(',')] for _line in ann_lines if _line.strip()]
        # ann_data.sort(key=lambda x: x[0])
        # ann_data = np.asarray(ann_data)

        print('Writing to {}'.format(mot_path))
        out_fid = open(mot_path, 'w')

        src_path = img_path

        src_files = [f for f in os.listdir(src_path) if
                     os.path.isfile(os.path.join(src_path, f)) and f.endswith(img_ext)]
        src_files.sort(key=sortKey)
        n_frames = len(src_files)

        print('n_frames: ', n_frames)

        total_n_frames += n_frames

        obj_dict = {}
        for __id, _data in enumerate(ann_data):

            # if _data[7] != 1 or _data[8] < min_vis:
            #     continue
            obj_id = int(_data[1])
            # Bounding box sanity check
            bbox = [float(x) for x in _data[2:6]]
            l, t, w, h = bbox
            xmin = int(l)
            ymin = int(t)
            xmax = int(xmin + w)
            ymax = int(ymin + h)
            if xmin >= xmax or ymin >= ymax:
                msg = 'Invalid box {}\n in line {} : {}\n'.format(
                    [xmin, ymin, xmax, ymax], __id, _data
                )
                if ignore_invalid:
                    print(msg)
                else:
                    raise IOError(msg)

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

                if ignore_invalid:
                    print(msg)
                    continue
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

        # csv_raw = []

        total_boxes = 0
        total_gt_boxes = 0
        total_syn_boxes = 0
        for frame_id in range(n_frames):
            if frame_id not in obj_dict:
                print('No {} found for frame {}: {}'.format(data_type, frame_id, file_path))
                continue

            filename = src_files[frame_id]
            file_path = os.path.join(src_path, filename)
            if not os.path.exists(file_path):
                raise SystemError('Image file {} does not exist'.format(file_path))

            image = cv2.imread(file_path)

            height, width = image.shape[:2]

            objects = obj_dict[frame_id]
            gt_boxes = []

            for obj in objects:
                bbox = obj['bbox']
                l, t, w, h = bbox

                xmin = int(l)
                ymin = int(t)

                xmax = int(xmin + w)
                ymax = int(ymin + h)

                xmin, xmax = clamp([xmin, xmax], 0, width - 1)
                ymin, ymax = clamp([ymin, ymax], 0, height - 1)

                if xmax <= xmin or ymax <= ymin:
                    print('Annoying invalid GT bounding box found for frame {}: {}'.format(frame_id, bbox))

                raw_data = {
                    'filename': filename,
                    'width': int(width),
                    'height': int(height),
                    'class': label,
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax,
                    'confidence': confidence,
                    'target_id': obj_id,
                }
                # csv_raw.append(raw_data)

                out_fid.write('{:d},{:d},{:f},{:f},{:f},{:f},{:f},-1,-1,-1\n'.format(
                    frame_id + 1, obj_id, xmin, ymin, w, h, confidence))
                total_boxes += 1

                gt_boxes.append((raw_data, (xmin, ymin, xmax - xmin, ymax - ymin)))

            gt_boxes_list = [k[1] for k in gt_boxes]
            gt_boxes_arr = np.asarray(gt_boxes_list)
            all_sampled_boxes = []

            total_gt_boxes += len(gt_boxes)

            for _raw_data, _bbox in gt_boxes:
                sampled_boxes = get_shifted_boxes(np.asarray(_bbox), image, synthetic_samples,
                                                  min_anchor_iou=0,
                                                  max_anchor_iou=synthetic_neg_iou,
                                                  min_shift_ratio=min_shift_ratio,
                                                  max_shift_ratio=1.0,
                                                  gt_boxes=gt_boxes_arr,
                                                  sampled_boxes=all_sampled_boxes,
                                                  vis=0,
                                                  # vis=params.show_img,
                                                  )
                for _sampled_box in sampled_boxes:
                    l, t, w, h = _sampled_box
                    xmin = l
                    ymin = t

                    # raw_data = {
                    #     'filename': _raw_data['filename'],
                    #     'width': _raw_data['width'],
                    #     'height': _raw_data['height'],
                    #     'class': _raw_data['class'],
                    #     'xmin': xmin,
                    #     'ymin': ymin,
                    #     'xmax': xmax,
                    #     'ymax': ymax,
                    #     'confidence': confidence,
                    #     'target_id': -1,
                    # }
                    # csv_raw.append(raw_data)

                    out_fid.write('{:d},{:d},{:f},{:f},{:f},{:f},{:f},-1,-1,-1\n'.format(
                        frame_id + 1, -1, xmin, ymin, w, h, _raw_data['confidence']))
                    total_boxes += 1

                all_sampled_boxes += sampled_boxes

            total_syn_boxes += len(all_sampled_boxes)

        print('generated {} synthetic boxes out of {} GT boxes for a total of {} boxes'.format(
            total_syn_boxes, total_gt_boxes, total_boxes))

        out_fid.close()

    print('total_n_frames: ', total_n_frames)


if __name__ == '__main__':
    main()
