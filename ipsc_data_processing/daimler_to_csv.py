import pandas as pd
import numpy as np
import os, sys, cv2

import paramparse

from eval_utils import ImageSequenceWriter as ImageWriter
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
    'img_ext': 'pgm',
    'ignore_occl': 1,
    'only_person': 0,
    'show_img': 1,
    'min_vis': 0.5,
    'resize_factor': 1.0,
    'save_raw': 0,
    'n_frames': 0,
    'vis_width': 0,
    'vis_height': 0,
    'ext': 'mkv',
    'codec': 'H264',
    'fps': 30,
    'vis_root': '',
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
save_path = params['save_path']
min_vis = params['min_vis']
save_raw = params['save_raw']
vis_root = params['vis_root']

image_exts = ['jpg', 'bmp', 'png', 'tif']

if not vis_root:
    vis_root = 'vis'

pause_after_frame = 1

img_path = os.path.join(root_dir, 'Data', 'TestData')

ann_path = os.path.join(root_dir, 'GroundTruth', 'GroundTruth2D.db')
ann_lines = open(ann_path).readlines()
n_ann_lines = len(ann_lines)

frame_ann_start_ids = [i + 1 for i in range(n_ann_lines) if ann_lines[i].strip() == ';']
n_ann_frames = len(frame_ann_start_ids)

src_path = img_path
src_files = [f for f in os.listdir(src_path) if
             os.path.isfile(os.path.join(src_path, f)) and f.endswith(img_ext)]
src_files.sort(key=sortKey)
n_frames = len(src_files)

print('n_frames: ', n_frames)

if n_frames != n_ann_frames:
    print('n_frames: ', n_frames)
    print('n_ann_frames: ', n_ann_frames)
    raise IOError('Mismatch detected')

if _vis_height <= 0 or _vis_width <= 0:
    temp_img = cv2.imread(os.path.join(img_path, src_files[0]))
    vis_height, vis_width, _ = temp_img.shape
else:
    vis_height, vis_width = _vis_height, _vis_width

if ext in image_exts:
    if not save_path:
        save_path = os.path.join(os.path.dirname(img_path), vis_root, os.path.basename(img_path) + '.' + ext)
    video_out = ImageWriter(save_path)
    print('Saving {}x{} visualization sequence to {}'.format(
        vis_width, vis_height, save_path))
else:
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

class_id_to_label = {
    0: 'person',
    1: 'bicyclist',
    2: 'motorcyclist',
    10: 'people',
    255: 'occluded',
}

csv_raw = []
out_frame_id = 0
for frame_id in range(n_frames):

    frame_ann_id = frame_ann_start_ids[frame_id]

    filename = ann_lines[frame_ann_id].strip()
    width, height = [int(x) for x in ann_lines[frame_ann_id + 1].strip().split(' ')]
    _, n_objs = [int(x) for x in ann_lines[frame_ann_id + 2].strip().split(' ')]

    if filename != src_files[frame_id]:
        print('filename: ', filename)
        print('src_files[frame_id]: ', src_files[frame_id])
        raise IOError('Mismatch detected')

    curr_ann_id = frame_ann_id + 3
    curr_csv_raw = []
    for obj_id in range(n_objs):
        try:
            class_id = int(ann_lines[curr_ann_id].strip().split(' ')[1])
            curr_ann_id += 1
        except IndexError:
            raise IOError('Invalid formatting on line {}: {}'.format(curr_ann_id + 1, ann_lines[curr_ann_id]))
        try:
            unique_id, unique_id_2 = [int(x) for x in ann_lines[curr_ann_id].strip().split(' ')]
            curr_ann_id += 1
        except IndexError:
            raise IOError('Invalid formatting on line {}: {}'.format(curr_ann_id + 1, ann_lines[curr_ann_id]))
        try:
            confidence = float(ann_lines[curr_ann_id].strip())
            curr_ann_id += 1
        except IndexError:
            raise IOError('Invalid formatting on line {}: {}'.format(curr_ann_id + 1, ann_lines[curr_ann_id]))

        try:
            ann_line = ann_lines[curr_ann_id].strip()
            _xmin, _ymin, _xmax, _ymax = [int(x) for x in ann_line.split(' ')]
            curr_ann_id += 1
        except IndexError:
            raise IOError('Invalid formatting on line {}: {}'.format(curr_ann_id + 1, ann_lines[curr_ann_id]))

        # Bounding box sanity check
        if _xmin > _xmax or _ymin > _ymax:
            print('Invalid box {}\n in line {}\n'.format(
                [xmin, ymin, xmax, ymax], ann_line,
            ))

        # if ymin < ymax:
        #     raise IOError('Invalid box {}\n in line {}\n'.format(
        #         [xmin, ymin, xmax, ymax], ann_line
        #     ))
        xmin = min(_xmin,_xmax)
        xmax = max(_xmin,_xmax)

        ymin = min(_ymin,_ymax)
        ymax = max(_ymin,_ymax)

        label = class_id_to_label[class_id]
        out_filename = 'image{:06d}.jpg'.format(out_frame_id + 1)

        raw_data = {
            'filename': out_filename,
            'width': int(width),
            'height': int(height),
            'class': label,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
            'confidence': confidence,
        }
        curr_csv_raw.append(raw_data)
        curr_ann_id += 1

    if only_person:
        if np.any([obj['class'] != 'person' for obj in curr_csv_raw]):
            continue

        if np.count_nonzero([obj['class'] == 'person' for obj in curr_csv_raw]) < only_person:
            continue

    file_path = os.path.join(src_path, filename)
    if not os.path.exists(file_path):
        raise SystemError('Image file {} does not exist'.format(file_path))

    image = cv2.imread(file_path)
    if save_raw:
        video_out.write(image)

    _height, _width = image.shape[:2]

    if width != _width:
        print('width: ', width)
        print('_width: ', _width)
        raise IOError('Mismatch detected')

    if height != _height:
        print('height: ', height)
        print('_height: ', _height)
        raise IOError('Mismatch detected')

    for _data in curr_csv_raw:
        confidence = _data['confidence']
        xmin = _data['xmin']
        ymin = _data['ymin']
        xmax = _data['xmax']
        ymax = _data['ymax']
        label = _data['class']

        box_color = (0, 255, 0) if confidence == 1.0 else (0, 0, 255)
        drawBox(image, xmin, ymin, xmax, ymax, box_color, label)

    image = resize_ar(image, vis_width, vis_height)
    csv_raw += curr_csv_raw

    if not save_raw:
        video_out.write(image)

    out_frame_id += 1

    if show_img:
        cv2.imshow('Daimler', image)
        k = cv2.waitKey(1 - pause_after_frame) & 0xFF
        if k == ord('q') or k == 27:
            break
        elif k == 32:
            pause_after_frame = 1 - pause_after_frame

csv_file = os.path.join(img_path, 'annotations.csv')
print('saving csv file to {}'.format(csv_file))

df = pd.DataFrame(csv_raw)
df.to_csv(csv_file)
video_out.release()
cv2.destroyWindow('Daimler')

print('out_n_frames: ', out_frame_id)
