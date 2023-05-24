import pandas as pd
import numpy as np
import os
import cv2

import paramparse

from eval_utils import ImageSequenceWriter, sortKey, drawBox

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
    'n_frames': 0,
    'vis_root': '',
    'ext': 'jpg',
    'save_raw': 0,
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
                     os.path.isdir(os.path.join(root_dir, name))]
        file_list.sort(key=sortKey)
else:
    if not file_name:
        raise IOError('Either list file or a single sequence file must be provided')
    file_list = [file_name]
n_seq = len(file_list)

if n_seq == 0:
    raise IOError('No input sequences found')

if not vis_root:
    vis_root = 'vis'

enable_resize = 0

if ext in image_exts:
    if not save_path:
        save_path = os.path.join(os.path.dirname(file_list[0]), 'combined.{}'.format(ext))
    video_out = ImageSequenceWriter(save_path)
    print('Saving combined sequence to {}'.format(save_path))
else:
    raise IOError('Output must be an image sequence')

save_dir = os.path.dirname(save_path)
save_seq_name = os.path.splitext(os.path.basename(save_path))[0]

print('Running over {} sequences'.format(n_seq))
pause_after_frame = 1
total_n_frames = 0
csv_raw = []
out_frame_id = 0

for seq_idx, img_path in enumerate(file_list):
    seq_name = os.path.basename(img_path)
    print('seq_name: ', seq_name)

    print('sequence {}/{}: {}: '.format(seq_idx + 1, n_seq, seq_name))

    csv_path = os.path.join(img_path, 'annotations.csv')
    df_in = pd.read_csv(csv_path)

    src_path = img_path
    src_files = [f for f in os.listdir(src_path) if
                 os.path.isfile(os.path.join(src_path, f)) and f.endswith(img_ext)]
    src_files.sort(key=sortKey)
    n_frames = len(src_files)

    print('n_frames: ', n_frames)

    for frame_id in range(n_frames):
        filename = src_files[frame_id]
        file_path = os.path.join(src_path, filename)
        if not os.path.exists(file_path):
            raise SystemError('Image file {} does not exist'.format(file_path))

        image = cv2.imread(file_path)

        height, width = image.shape[:2]

        multiple_instance = df_in.loc[df_in['filename'] == filename]
        no_instances = len(multiple_instance.index)
        df_in = df_in.drop(multiple_instance.index[:no_instances])

        out_frame_id += 1
        out_filename = 'image{:06d}.jpg'.format(out_frame_id)

        for instance in range(0, len(multiple_instance.index)):
            xmin = multiple_instance.iloc[instance].loc['xmin']
            ymin = multiple_instance.iloc[instance].loc['ymin']
            xmax = multiple_instance.iloc[instance].loc['xmax']
            ymax = multiple_instance.iloc[instance].loc['ymax']
            label = multiple_instance.iloc[instance].loc['class']

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

            drawBox(image, xmin, ymin, xmax, ymax, label=label)

        if not save_raw:
            video_out.write(image)

        if show_img:
            cv2.imshow(seq_name, image)
            k = cv2.waitKey(1 - pause_after_frame) & 0xFF
            if k == ord('q') or k == 27:
                break
            elif k == 32:
                pause_after_frame = 1 - pause_after_frame

    cv2.destroyWindow(seq_name)
    total_n_frames += out_frame_id
    print('out_n_frames: ', out_frame_id)

video_out.release()
csv_file = os.path.join(save_dir, save_seq_name, 'annotations.csv')
print('saving csv file to {}'.format(csv_file))
pd.DataFrame(csv_raw).to_csv(csv_file)
print('total_n_frames: ', total_n_frames)
