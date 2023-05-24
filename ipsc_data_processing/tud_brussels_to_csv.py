import pandas as pd
import os, cv2

import paramparse

from eval_utils import ImageSequenceWriter as ImageWriter
from eval_utils import sortKey, resize_ar, drawBox

params = {
    'file_name': '',
    'save_path': '',
    'save_file_name': '',
    'csv_file_name': '',
    'map_folder': '',
    'img_path': '',
    'ann_path': '',
    'list_file_name': '',
    'n_classes': 4,
    'img_ext': 'png',
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
    'out_root_dir': '',
    'mode': 0,
    'db_type': 0,
}

paramparse.process_dict(params)

file_name = params['file_name']
img_path = params['img_path']
ann_path = params['ann_path']
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
out_root_dir = params['out_root_dir']
mode = params['mode']
db_type = params['db_type']

image_exts = ['jpg', 'bmp', 'png', 'tif']

pause_after_frame = 1

ann_lines = open(ann_path).readlines()
n_ann_lines = len(ann_lines)

if db_type == 1:
    src_path = os.path.join(img_path, 'left')
else:
    src_path = img_path

src_files = [f for f in os.listdir(src_path) if
             os.path.isfile(os.path.join(src_path, f)) and f.endswith(img_ext)]
src_files.sort(key=sortKey)
n_frames = len(src_files)

if not out_root_dir:
    out_root_dir = os.path.join(os.path.dirname(img_path), 'vis')

print('n_frames: ', n_frames)

if _vis_height <= 0 or _vis_width <= 0:
    temp_img = cv2.imread(os.path.join(src_path, src_files[0]))
    vis_height, vis_width, _ = temp_img.shape
else:
    vis_height, vis_width = _vis_height, _vis_width

if not save_path:
    save_path = os.path.join(out_root_dir, os.path.basename(img_path) + '.' + ext)

save_dir = os.path.dirname(save_path)
save_seq_name = os.path.splitext(os.path.basename(save_path))[0]

if ext in image_exts:
    video_out = ImageWriter(save_path)
    print('Saving {}x{} visualization sequence to {}'.format(
        vis_width, vis_height, save_path))
else:
    if save_dir and not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    video_out = cv2.VideoWriter(save_path, fourcc, fps, (vis_width, vis_height))
    if video_out is None:
        raise IOError('Output video file could not be opened: {}'.format(save_path))
    print('Saving {}x{} visualization video to {}'.format(
        vis_width, vis_height, save_path))

if cv2.__version__.startswith('2'):
	font_line_type = cv2.CV_AA
else:
	font_line_type = cv2.LINE_AA

csv_raw = []
out_frame_id = 0
for ann_line_id in range(n_ann_lines):

    ann_line = ann_lines[ann_line_id].strip().replace(';', '')
    ann_line_list = ann_line.split(': ')
    filename = ann_line_list[0].replace('"', '')

    file_path = os.path.join(img_path, filename)
    if not os.path.exists(file_path):
        raise SystemError('Image file {} does not exist'.format(file_path))

    image = cv2.imread(file_path)
    _height, _width = image.shape[:2]

    try:
        if ann_line_list[1].endswith('.'):
            ann_line_list[1] = ann_line_list[1].replace('.', '')
        ann_nums = ann_line_list[1].replace('(', '').replace(')', '').split(',')
    except:
        continue

    n_ann_nums = len(ann_nums)

    if n_ann_nums == 0:
        continue

    if n_ann_nums % 4 != 0:
        raise IOError('Invalid num data found in GT: {}'.format(ann_line_list[1]))

    n_objs = int(n_ann_nums / 4)

    if save_raw:
        video_out.write(image)

    curr_id = 0
    for obj_id in range(n_objs):
        try:
            _xmin = int(ann_nums[curr_id])
            curr_id += 1

            _ymin = int(ann_nums[curr_id])
            curr_id += 1

            _xmax = int(ann_nums[curr_id])
            curr_id += 1

            if mode == 0:
                _ymax = int(ann_nums[curr_id])
            else:
                _ymax = int(ann_nums[curr_id].split(':')[0])
            curr_id += 1

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

        except ValueError as e:
            print('ann_line: ', ann_line)
            print('ann_nums: ', ann_nums)
            print('n_ann_nums: ', n_ann_nums)
            print('n_objs: ', n_objs)
            raise IOError('Error in line {}: {}\n{}'.format(ann_line_id + 1, ann_line, e))

        label = 'person'
        out_filename = 'image{:06d}.jpg'.format(out_frame_id + 1)

        raw_data = {
            'filename': out_filename,
            'width': int(_width),
            'height': int(_height),
            'class': label,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
        }
        csv_raw.append(raw_data)

        if show_img or not save_raw:
            drawBox(image, xmin, ymin, xmax, ymax, label=label)

    image = resize_ar(image, vis_width, vis_height)

    if not save_raw:
        video_out.write(image)

    out_frame_id += 1

    if show_img:
        cv2.imshow('TUD-Brussels', image)
        k = cv2.waitKey(1 - pause_after_frame) & 0xFF
        if k == ord('q') or k == 27:
            break
        elif k == 32:
            pause_after_frame = 1 - pause_after_frame

csv_file = os.path.join(save_dir, save_seq_name, 'annotations.csv')
print('saving csv file to {}'.format(csv_file))

df = pd.DataFrame(csv_raw)
df.to_csv(csv_file)
video_out.release()
cv2.destroyWindow('Daimler')

print('out_n_frames: ', out_frame_id)
