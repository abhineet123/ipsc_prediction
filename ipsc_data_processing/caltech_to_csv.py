import pandas as pd
import numpy as np
import os, cv2

import paramparse

from eval_utils import ImageSequenceWriter
from eval_utils import sortKey, resize_ar, drawBox

params = {
    'file_name': '',
    'save_path': '',
    'save_file_name': '',
    'csv_file_name': '',
    'map_folder': '',
    'root_dir': '',
    'list_file_name': '',
    'n_classes': 4,
    'img_ext': 'jpg',
    'ignore_occl': 0,
    'only_person': 0,
    'show_img': 1,
    'resize_factor': 1.0,
    'n_frames': 0,
    'save_raw': 0,
    'vis_width': 0,
    'vis_height': 0,
    'vis_root': '',
    'ext': 'mkv',
    'codec': 'H264',
    'fps': 30,
}

paramparse.process_dict(params)

file_name = params['file_name']
root_dir = params['root_dir']
list_file_name = params['list_file_name']
img_ext = params['img_ext']
ignore_occl = params['ignore_occl']
only_person = params['only_person']
show_img = params['show_img']
resize_factor = params['resize_factor']
ext = params['ext']
codec = params['codec']
fps = params['fps']
_vis_width = params['vis_width']
_vis_height = params['vis_height']
vis_root = params['vis_root']
save_path = params['save_path']
save_raw = params['save_raw']

image_exts = ['jpg', 'bmp', 'png', 'tif']

if list_file_name:
    if not os.path.exists(list_file_name):
        raise IOError('List file: {} does not exist'.format(list_file_name))
    file_list = [x.strip() for x in open(list_file_name).readlines()]
    if root_dir:
        file_list = [os.path.join(root_dir, x) for x in file_list]
elif root_dir:
    if root_dir.startswith('camera'):
        file_list = [root_dir]
    else:
        file_list = [os.path.join(root_dir, name) for name in os.listdir(root_dir) if
                     os.path.isdir(os.path.join(root_dir, name)) and
                     not name.endswith('_vis') and not name.startswith('vis_')]
        file_list.sort(key=sortKey)
else:
    if not file_name:
        raise IOError('Either list file or a single sequence file must be provided')
    file_list = [file_name]
n_seq = len(file_list)

print('Running over {} sequences'.format(n_seq))
pause_after_frame = 1

if not vis_root:
    vis_root = 'vis' if not save_raw else 'raw'

total_n_frames = 0

for seq_idx, img_path in enumerate(file_list):
    seq_name = os.path.basename(img_path)

    print('sequence {}/{}: {}: '.format(seq_idx + 1, n_seq, img_path))

    ann_path = os.path.join(os.path.dirname(img_path), os.path.basename(img_path) + '.txt')

    ann_lines = open(ann_path).readlines()
    # print('ann_lines: ', ann_lines)

    _ann_header = {k: int(v) for k, v in [x.strip().split('=') for x in ann_lines[1].split(' ')]}
    print('seq_name: ', seq_name)
    print('_ann_header: ', _ann_header)
    n_frames = _ann_header['nFrame']
    src_files = [f for f in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, f)) and f.endswith(img_ext)]
    src_files.sort(key=sortKey)
    if len(src_files) != n_frames:
        print('n_frames: ', n_frames)
        print('len(src_files): ', len(src_files))
        raise IOError('Mismatch detected')

    n_objs = _ann_header['n']
    n_lines = len(ann_lines)
    label_lines_idx = [i for i in range(n_lines) if ann_lines[i].startswith('lbl=')]

    if len(label_lines_idx) != n_objs:
        print('n_objs: ', n_objs)
        print('len(label_lines_idx): ', len(label_lines_idx))
        raise IOError('Mismatch detected')

    obj_dict = {}
    for _idx in label_lines_idx:
        obj_header = {k: v for k, v in [x.strip().split('=') for x in ann_lines[_idx].split(' ')]}
        obj_label = obj_header['lbl']
        # print('obj_header: ', obj_header)

        start_frame_id = int(obj_header['str'])
        end_frame_id = int(obj_header['end'])

        ann_line = ann_lines[_idx + 1].strip()

        pos_line = ann_line.replace('pos =[', '').replace('; ]', '')
        obj_poa = [bbox for bbox in [x.strip().split(' ') for x in pos_line.strip().split(';')]]

        n_obj_frames = end_frame_id - start_frame_id + 1

        if len(obj_poa) != n_obj_frames:
            print('n_obj_frames: ', n_obj_frames)
            print('len(obj_poa): ', len(obj_poa))
            print('obj_poa: ', obj_poa)
            raise IOError('Mismatch detected')

        occl_line = ann_lines[_idx + 3].strip().replace('occl=[', '').replace(' ]', '')
        obj_occl = [int(x) for x in occl_line.strip().split(' ')]
        if len(obj_occl) != n_obj_frames:
            print('n_obj_frames: ', n_obj_frames)
            print('len(obj_occl): ', len(obj_occl))
            print('obj_occl: ', obj_occl)
            raise IOError('Mismatch detected')

        for j in range(n_obj_frames):
            frame_id = start_frame_id + j - 1

            # Bounding box sanity check
            bbox = [float(x) for x in obj_poa[j]]
            l, t, w, h = bbox
            xmin = int(l)
            ymin = int(t)
            xmax = int(xmin + w)
            ymax = int(ymin + h)
            if xmin >= xmax or ymin >= ymax:
                raise IOError('Invalid box {}\n in line {}\n with pos_line: {}\n and obj_poa:{}\n'.format(
                    [xmin, ymin, xmax, ymax], ann_line, pos_line, obj_poa
                ))

            obj_entry = {'label': obj_label.replace("'", ""),
                         'bbox': bbox,
                         'occl': obj_occl[j]}
            if frame_id in obj_dict:
                obj_dict[frame_id].append(obj_entry)
            else:
                obj_dict[frame_id] = [obj_entry]

    enable_resize = 0
    if _vis_height <= 0 or _vis_width <= 0:
        temp_img = cv2.imread(os.path.join(img_path, src_files[0]))
        vis_height, vis_width, _ = temp_img.shape
    else:
        vis_height, vis_width = _vis_height, _vis_width
        enable_resize = 1

    if ext in image_exts:
        if not save_path:
            save_path = os.path.join(os.path.dirname(img_path), vis_root, os.path.basename(img_path) + '.' + ext)
        video_out = ImageSequenceWriter(save_path)
        print('Saving {}x{} visualization sequence to {}'.format(
            vis_width, vis_height, save_path))
    else:
        enable_resize = 1
        if not save_path:
            save_path = os.path.join(os.path.dirname(img_path), os.path.basename(img_path) + '.' + ext)

        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        fourcc = cv2.VideoWriter_fourcc(*codec)
        video_out = cv2.VideoWriter(save_path, fourcc, fps, (vis_width, vis_height))
        if video_out is None:
            raise IOError('Output video file could not be opened: {}'.format(save_path))
        print('Saving {}x{} visualization video to {}'.format(
            vis_width, vis_height, save_path))

    csv_raw = []

    if cv2.__version__.startswith('2'):
        font_line_type = cv2.CV_AA
    else:
        font_line_type = cv2.LINE_AA

    out_frame_id = 0
    for frame_id in range(n_frames):
        filename = src_files[frame_id]
        file_path = os.path.join(img_path, filename)
        if not os.path.exists(file_path):
            raise SystemError('Image file {} does not exist'.format(file_path))

        if only_person:
            if frame_id not in obj_dict or np.any([obj['label'] != 'person' for obj in obj_dict[frame_id]]):
                continue

            if np.count_nonzero([obj['label'] == 'person' for obj in obj_dict[frame_id]]) < only_person:
                continue

            if ignore_occl and np.any([obj['occl'] for obj in obj_dict[frame_id]]):
                continue

        image = cv2.imread(file_path)
        height, width = image.shape[:2]

        out_frame_id += 1
        out_filename = 'image{:06d}.jpg'.format(out_frame_id)
        if save_raw:
            video_out.write(image)

        if frame_id in obj_dict:
            objects = obj_dict[frame_id]
            for obj in objects:
                occl = obj['occl']

                if ignore_occl and occl:
                    continue

                label = obj['label']

                bbox = obj['bbox']

                l, t, w, h = bbox
                xmin = int(l)
                ymin = int(t)

                xmax = int(xmin + w)
                ymax = int(ymin + h)

                raw_data = {
                    'filename': out_filename,
                    'width': int(width),
                    'height': int(height),
                    'class': label,
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax,
                }
                csv_raw.append(raw_data)

                box_color = (0, 255, 0) if not occl else (0, 0, 255)

                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), box_color)

                _bb = [xmin, ymin, xmax, ymax]
                if _bb[1] > 10:
                    y_loc = int(_bb[1] - 5)
                else:
                    y_loc = int(_bb[3] + 5)
                cv2.putText(image, label, (int(_bb[0] - 1), y_loc), cv2.FONT_HERSHEY_SIMPLEX,
                            0.3, box_color, 1, font_line_type)

        if enable_resize:
            image = resize_ar(image, vis_width, vis_height)
        if not save_raw:
            video_out.write(image)

        # if resize_factor != 1:
        #     image = cv2.resize(image, (0, 0), fx=resize_factor, fy=resize_factor)

        if show_img:
            cv2.imshow(seq_name, image)
            k = cv2.waitKey(1 - pause_after_frame) & 0xFF
            if k == ord('q') or k == 27:
                break
            elif k == 32:
                pause_after_frame = 1 - pause_after_frame

    if save_raw:
        csv_file = os.path.join(os.path.dirname(save_path), seq_name, 'annotations.csv')
    else:
        csv_file = os.path.join(img_path, 'annotations.csv')
    print('saving csv file to {}'.format(csv_file))
    print('out_n_frames: ', out_frame_id)
    total_n_frames += out_frame_id
    df = pd.DataFrame(csv_raw)
    df.to_csv(csv_file)
    video_out.release()
    save_path = ''
    cv2.destroyWindow(seq_name)

print('total_n_frames: ', total_n_frames)
