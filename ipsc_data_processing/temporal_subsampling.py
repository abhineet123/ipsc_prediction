import pandas as pd
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
    'img_ext': 'jpg',
    'ignore_occl': 1,
    'show_img': 1,
    'start_id': 0,
    'frame_gap': 1,
    'resize_factor': 1.0,
    'n_frames': 0,
    'vis_width': 0,
    'vis_height': 0,
    'fixed_ar': 0,
    'out_root_dir': '',
    'out_postfix': '',
    'ext': 'jpg',
    'codec': 'H264',
    'fps': 30,
    'save_raw': 1,
    'invert_selection': 0,
}
paramparse.process_dict(params)

file_name = params['file_name']
root_dir = params['root_dir']
list_file_name = params['list_file_name']
img_ext = params['img_ext']
ignore_occl = params['ignore_occl']
show_img = params['show_img']
resize_factor = params['resize_factor']
ext = params['ext']
codec = params['codec']
fps = params['fps']
_vis_width = params['vis_width']
_vis_height = params['vis_height']
out_root_dir = params['out_root_dir']
out_postfix = params['out_postfix']
save_path = params['save_path']
start_id = params['start_id']
frame_gap = params['frame_gap']
save_raw = params['save_raw']
invert_selection = params['invert_selection']
fixed_ar = params['fixed_ar']

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
                     os.path.isdir(os.path.join(root_dir, name))]
        file_list.sort(key=sortKey)
else:
    if not file_name:
        raise IOError('Either list file or a single sequence file must be provided')
    file_list = [file_name]
n_seq = len(file_list)

if not out_root_dir:
    out_root_dir = os.path.dirname(file_list[0])
    if not out_postfix:
        out_postfix = '_ss_{}_{}'.format(start_id, frame_gap)

print('Running over {} sequences'.format(n_seq))
pause_after_frame = 1
total_n_frames = 0
for seq_idx, img_path in enumerate(file_list):
    seq_name = os.path.basename(img_path)
    print('seq_name: ', seq_name)

    print('sequence {}/{}: {}: '.format(seq_idx + 1, n_seq, seq_name))

    csv_path = os.path.join(img_path, 'annotations.csv')
    df = pd.read_csv(csv_path)

    src_path = img_path
    src_files = [f for f in os.listdir(src_path) if
                 os.path.isfile(os.path.join(src_path, f)) and f.endswith(img_ext)]
    src_files.sort(key=sortKey)
    n_frames = len(src_files)

    print('n_frames: ', n_frames)

    enable_resize = 0
    if _vis_height <= 0 or _vis_width <= 0:
        temp_img = cv2.imread(os.path.join(src_path, src_files[0]))
        vis_height, vis_width, _ = temp_img.shape
    else:
        vis_height, vis_width = _vis_height, _vis_width
        enable_resize = 1

    if ext in image_exts:
        out_seq_name = os.path.basename(img_path)
        if out_postfix:
            out_seq_name = '{}_{}'.format(out_seq_name, out_postfix)
        out_seq_name = '{}.{}'.format(out_seq_name, ext)
        save_path = os.path.join(out_root_dir, out_seq_name)
        video_out = ImageWriter(save_path)
        print('Saving {}x{} output sequence to {}'.format(vis_width, vis_height, save_path))
    else:
        raise IOError('Output must be an image sequence')

    save_dir = os.path.dirname(save_path)
    save_seq_name = os.path.splitext(os.path.basename(save_path))[0]

    frame_ids = range(start_id, n_frames, frame_gap)
    if invert_selection:
        frame_ids = [x for x in range(n_frames) if x not in frame_ids]

    csv_raw = []

    out_frame_id = 0
    n_out_frames = len(frame_ids)
    for frame_id in frame_ids:
        filename = src_files[frame_id]
        file_path = os.path.join(src_path, filename)
        if not os.path.exists(file_path):
            raise SystemError('Image file {} does not exist'.format(file_path))

        image = cv2.imread(file_path)

        if fixed_ar:
            image, resize_factor, start_row, start_col = resize_ar(image, vis_width, vis_height, return_factors=1)

        if save_raw:
            video_out.write(image)

        height, width = image.shape[:2]

        multiple_instance = df.loc[df['filename'] == filename]
        no_instances = len(multiple_instance.index)
        df = df.drop(multiple_instance.index[:no_instances])

        out_frame_id += 1
        out_filename = 'image{:06d}.jpg'.format(out_frame_id)

        for instance in range(0, len(multiple_instance.index)):
            xmin = multiple_instance.iloc[instance].loc['xmin']
            ymin = multiple_instance.iloc[instance].loc['ymin']
            xmax = multiple_instance.iloc[instance].loc['xmax']
            ymax = multiple_instance.iloc[instance].loc['ymax']
            label = multiple_instance.iloc[instance].loc['class']

            if fixed_ar:
                xmin = int((xmin + start_col) * resize_factor)
                xmax = int((xmax + start_col) * resize_factor)
                ymin = int((ymin + start_row) * resize_factor)
                ymax = int((ymax + start_row) * resize_factor)

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

            if show_img or not save_raw:
                drawBox(image, xmin, ymin, xmax, ymax, label=label)

        if enable_resize and not fixed_ar:
            image = resize_ar(image, vis_width, vis_height)

        if not save_raw:
            video_out.write(image)

        if show_img:
            cv2.imshow(seq_name, image)
            k = cv2.waitKey(1 - pause_after_frame) & 0xFF
            if k == ord('q') or k == 27:
                break
            elif k == 32:
                pause_after_frame = 1 - pause_after_frame
        else:
            sys.stdout.write('\rDone {:d}/{:d} frames'.format(out_frame_id, n_out_frames))
            sys.stdout.flush()

    if show_img:
        cv2.destroyWindow(seq_name)
    else:
        sys.stdout.write('\n')
        sys.stdout.flush()

    csv_file = os.path.join(save_dir, save_seq_name, 'annotations.csv')
    print('saving csv file to {}'.format(csv_file))
    pd.DataFrame(csv_raw).to_csv(csv_file)
    video_out.release()

    total_n_frames += out_frame_id
    print('out_n_frames: ', out_frame_id)

print('total_n_frames: ', total_n_frames)
