import pandas as pd
import os
import cv2

from tqdm import tqdm

import paramparse

from eval_utils import ImageSequenceWriter as ImageWriter
from eval_utils import sortKey, resize_ar, drawBox, clamp

params = {
    'out_suffix': '',
    'data_type': 'annotations',
    'file_name': '',
    'save_path': '',
    'save_file_name': '',
    'csv_file_name': '',
    'map_folder': '',
    'root_dir': '',
    'list_file_name': '',
    'n_classes': 4,
    'img_ext': 'jpg',
    'ignore_occl': 1,
    'show_img': 1,
    'show_class': 0,
    'min_vis': 0.5,
    'resize_factor': 1.0,
    'n_frames': 0,
    'vis_size': '',
    'out_root_dir': '',
    'ext': 'mkv',
    'codec': 'H264',
    'fps': 30,
    'save_raw': 0,
    'mode': 1,
    'save_video': 0,
    'start_id': 0,
    'ignore_missing': 0,
    'percent_scores': 0,
    'clamp_scores': 0,
    'ignore_invalid': 0,
    'stats_only': 0,
    'label': 'person',
}

paramparse.process_dict(params)

out_suffix = params['out_suffix']
data_type = params['data_type']
file_name = params['file_name']
root_dir = params['root_dir']
list_file_name = params['list_file_name']
img_ext = params['img_ext']
ignore_occl = params['ignore_occl']
show_img = params['show_img']
show_class = params['show_class']
resize_factor = params['resize_factor']
ext = params['ext']
codec = params['codec']
fps = params['fps']
vis_size = params['vis_size']
out_root_dir = params['out_root_dir']
save_path = params['save_path']
min_vis = params['min_vis']
save_raw = params['save_raw']
save_video = params['save_video']
mode = params['mode']
start_id = params['start_id']
ignore_missing = params['ignore_missing']
label = params['label']
percent_scores = params['percent_scores']
clamp_scores = params['clamp_scores']
ignore_invalid = params['ignore_invalid']
stats_only = params['stats_only']

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
total_unique_obj_ids = 0
file_list = file_list[start_id:]
seq_to_n_unique_obj_ids = {}
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

    print('Reading {} from {}'.format(data_type, ann_path))

    ann_lines = open(ann_path).readlines()

    ann_data = [[float(x) for x in _line.strip().split(',')] for _line in ann_lines if _line.strip()]
    # ann_data.sort(key=lambda x: x[0])
    # ann_data = np.asarray(ann_data)

    if mode == 0:
        src_path = os.path.join(img_path, 'img1')
    else:
        src_path = img_path

    src_files = [f for f in os.listdir(src_path) if
                 os.path.isfile(os.path.join(src_path, f)) and f.endswith(img_ext)]
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
            msg = 'Invalid box {}\n in line {} : {}\n'.format(
                [xmin, ymin, xmax, ymax], __id, _data
            )
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
    n_unique_obj_ids = len(unique_obj_ids)

    total_unique_obj_ids += n_unique_obj_ids

    print(f'{seq_name}: {n_unique_obj_ids}')

    seq_to_n_unique_obj_ids[seq_name] = n_unique_obj_ids

    if stats_only:
        continue

    if not vis_size:
        temp_img = cv2.imread(os.path.join(src_path, src_files[0]))
        vis_height, vis_width, _ = temp_img.shape
    else:
        vis_width, vis_height = [int(x) for x in vis_size.split('x')]
        enable_resize = 1

    if not save_path:
        save_path = os.path.join(out_root_dir, os.path.basename(img_path) + '.' + ext)

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

    csv_raw = []

    if cv2.__version__.startswith('2'):
        font_line_type = cv2.CV_AA
    else:
        font_line_type = cv2.LINE_AA

    # out_frame_id = 0
    for frame_id in tqdm(range(n_frames)):
        filename = src_files[frame_id]
        file_path = os.path.join(src_path, filename)
        if not os.path.exists(file_path):
            raise SystemError('Image file {} does not exist'.format(file_path))

        image = cv2.imread(file_path)

        height, width = image.shape[:2]

        if frame_id in obj_dict:
            if save_video and save_raw:
                video_out.write(image)

            # out_frame_id += 1
            # out_filename = 'image{:06d}.jpg'.format(out_frame_id)

            objects = obj_dict[frame_id]
            for obj in objects:
                obj_id = obj['id']
                label = obj['label']
                bbox = obj['bbox']
                confidence = obj['confidence']

                l, t, w, h = bbox
                xmin = int(l)
                ymin = int(t)

                xmax = int(xmin + w)
                ymax = int(ymin + h)

                xmin, xmax = clamp([xmin, xmax], 0, width - 1)
                ymin, ymax = clamp([ymin, ymax], 0, height - 1)

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
                csv_raw.append(raw_data)

                if show_img or not save_raw:
                    if show_class:
                        _label = '{} {}'.format(label, obj_id)
                    else:
                        _label = '{}'.format(obj_id)

                    drawBox(image, xmin, ymin, xmax, ymax, label=_label, font_size=0.5)
        else:
            print('No {} found for frame {}: {}'.format(data_type, frame_id, file_path))

        if save_video or show_img:
            if enable_resize:
                image = resize_ar(image, vis_width, vis_height)

            if save_video and not save_raw:
                video_out.write(image)

            if show_img:
                cv2.imshow(seq_name, image)
                k = cv2.waitKey(1 - pause_after_frame) & 0xFF
                if k == ord('q') or k == 27:
                    break
                elif k == 32:
                    pause_after_frame = 1 - pause_after_frame

    if out_suffix:
        out_fname = '{}_{}.csv'.format(data_type, out_suffix)
    else:
        out_fname = '{}.csv'.format(data_type)
    if mode == 0:
        csv_file = os.path.join(save_dir, save_seq_name, out_fname)
    else:
        csv_file = os.path.join(img_path, out_fname)

    print('saving csv file to {}'.format(csv_file))
    df = pd.DataFrame(csv_raw)
    df.to_csv(csv_file)
    if save_video:
        video_out.release()
    save_path = ''

    if show_img:
        cv2.destroyWindow(seq_name)
    # total_n_frames += out_frame_id
    # print('out_n_frames: ', out_frame_id)

print('total_n_frames: ', total_n_frames)
print('total_unique_obj_ids: ', total_unique_obj_ids)
