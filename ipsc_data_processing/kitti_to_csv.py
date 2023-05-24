import pandas as pd
import numpy as np
import os, cv2

import paramparse

from eval_utils import ImageSequenceWriter as ImageWriter
from eval_utils import sortKey, resize_ar

params = {
    'file_name': '',
    'save_path': '',
    'save_file_name': '',
    'csv_file_name': '',
    'map_folder': '',
    'root_dir': '',
    'list_file_name': '',
    'n_classes': 4,
    'img_ext': 'png',
    'ignore_dc': 1,
    'show_img': 1,
    'min_vis': 0.5,
    'resize_factor': 1.0,
    'n_frames': 0,
    'vis_width': 0,
    'vis_height': 0,
    'ext': 'mkv',
    'codec': 'H264',
    'fps': 30,
    'only_person': 0,
    'save_raw': 0,
    'vis_root': '',
}

paramparse.process_dict(params)

file_name = params['file_name']
root_dir = params['root_dir']
list_file_name = params['list_file_name']
img_ext = params['img_ext']
ignore_dc = params['ignore_dc']
show_img = params['show_img']
resize_factor = params['resize_factor']
ext = params['ext']
codec = params['codec']
fps = params['fps']
_vis_width = params['vis_width']
_vis_height = params['vis_height']
save_path = params['save_path']
min_vis = params['min_vis']
only_person = params['only_person']
save_raw = params['save_raw']
vis_root = params['vis_root']

image_exts = ['jpg', 'bmp', 'png', 'tif']

if list_file_name:
    if not os.path.exists(list_file_name):
        raise IOError('List file: {} does not exist'.format(list_file_name))
    file_list = [x.strip() for x in open(list_file_name).readlines()]
    if root_dir:
        file_list = [os.path.join(root_dir, x) for x in file_list]
elif root_dir:
    img_root_dir = os.path.join(root_dir, 'image_02')
    file_list = [os.path.join(img_root_dir, name) for name in os.listdir(img_root_dir) if
                 os.path.isdir(os.path.join(img_root_dir, name))]
    file_list.sort(key=sortKey)
else:
    if not file_name:
        raise IOError('Either list file or a single sequence file must be provided')
    file_list = [file_name]

if not vis_root:
    vis_root = 'vis'

n_seq = len(file_list)

print('Running over {} sequences'.format(n_seq))
pause_after_frame = 1
total_n_frames = 0

for seq_idx, img_path in enumerate(file_list):
    seq_name = os.path.basename(img_path)
    print('seq_name: ', seq_name)

    print('sequence {}/{}: {}: '.format(seq_idx + 1, n_seq, seq_name))

    ann_path = os.path.join(root_dir, 'label_02', seq_name + '.txt')
    ann_lines = open(ann_path).readlines()

    ann_data = [[x for x in _line.strip().split(' ')] for _line in ann_lines]
    # ann_data.sort(key=lambda x: x[0])
    # ann_data = np.asarray(ann_data)

    src_path = img_path
    src_files = [f for f in os.listdir(src_path) if
                 os.path.isfile(os.path.join(src_path, f)) and f.endswith(img_ext)]
    src_files.sort(key=sortKey)
    n_frames = len(src_files)

    print('n_frames: ', n_frames)

    obj_dict = {}
    for _data in ann_data:

        class_label = _data[2]

        if class_label != 'Pedestrian' and (ignore_dc or class_label != 'DontCare'):
            continue

        frame_id = int(_data[0]) - 1
        # xmin, ymin, w, h = _data[2:]

        obj_entry = {'label': 'person',
                     'bbox': [float(x) for x in _data[6:10]]}
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
            save_path = os.path.join(root_dir, vis_root, os.path.basename(img_path) + '.' + ext)
        video_out = ImageWriter(save_path)
        print('Saving {}x{} visualization sequence to {}'.format(
            vis_width, vis_height, save_path))
    else:
        enable_resize = 1
        if not save_path:
            save_path = os.path.join(root_dir, os.path.basename(img_path) + '.' + ext)

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

        if only_person:
            if frame_id not in obj_dict or np.any([obj['label'] != 'person' for obj in obj_dict[frame_id]]):
                continue

            if np.count_nonzero([obj['label'] == 'person' for obj in obj_dict[frame_id]])< only_person:
                continue

        filename = src_files[frame_id]
        file_path = os.path.join(src_path, filename)
        if not os.path.exists(file_path):
            raise SystemError('Image file {} does not exist'.format(file_path))

        image = cv2.imread(file_path)
        height, width = image.shape[:2]
        if frame_id in obj_dict:
            if save_raw:
                video_out.write(image)
            out_frame_id += 1
            out_filename = 'image{:06d}.jpg'.format(out_frame_id)

            objects = obj_dict[frame_id]
            for obj in objects:
                label = obj['label']
                bbox = obj['bbox']

                # xmin, ymin, xmax, ymax = bbox

                l, t, w, h = bbox
                xmin = int(l)
                ymin = int(t)

                xmax = int(w)
                ymax = int(h)

                # xmax = int(xmin + w)
                # ymax = int(ymin + h)

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

                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0))

                _bb = [xmin, ymin, xmax, ymax]
                if _bb[1] > 10:
                    y_loc = int(_bb[1] - 5)
                else:
                    y_loc = int(_bb[3] + 5)
                cv2.putText(image, label, (int(_bb[0] - 1), y_loc), cv2.FONT_HERSHEY_SIMPLEX,
                            0.3, (0, 255, 0), 1, font_line_type)

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
