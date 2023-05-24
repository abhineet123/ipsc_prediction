import os
from tqdm import tqdm
import pandas as pd

import paramparse

from eval_utils import sortKey, linux_path


_params = {
    'root_dir': '.',
    'img_root_dir': '',
    'start_id': 0,
    'end_id': -1,
    'ignored_region_only': 0,
    'speed': 0.5,
    'show_img': 0,
    'quality': 3,
    'resize': 0,
    'mode': 0,
    'auto_progress': 0,
    'recursive': 0,
    'replace_existing': 0,
    'data_type': '',
}

paramparse.process_dict(_params)

root_dir = _params['root_dir']
_img_root_dir = _params['img_root_dir']
start_id = _params['start_id']
ignored_region_only = _params['ignored_region_only']
end_id = _params['end_id']
recursive = _params['recursive']
replace_existing = _params['replace_existing']
data_type = _params['data_type']

csv_exts = ('.csv',)
img_exts = ('.jpg', '.bmp', '.jpeg', '.png', '.tif', '.tiff', '.webp')

if recursive:
    __csv_files_gen = [[linux_path(dirpath, f) for f in filenames if
                        os.path.splitext(f.lower())[1] in csv_exts]
                       for (dirpath, dirnames, filenames) in os.walk(root_dir, followlinks=True)]
    __csv_files_list = [item for sublist in __csv_files_gen for item in sublist]
else:
    __csv_files_list = [linux_path(root_dir, k) for k in os.listdir(root_dir) if
                        os.path.splitext(k.lower())[1] in csv_exts]

if end_id <= start_id:
    end_id = len(__csv_files_list) - 1

print('root_dir: {}'.format(root_dir))
print('start_id: {}'.format(start_id))
print('end_id: {}'.format(end_id))

# print('__csv_files_list: {}'.format(__csv_files_list))
__csv_files_list.sort(key=sortKey)

n_csv_files_list = len(__csv_files_list)

n_frames_list = []

for seq_id in tqdm(range(start_id, end_id + 1), ncols=10):
    csv_path = __csv_files_list[seq_id]
    if recursive:
        csv_dir = os.path.dirname(csv_path)
        seq_name = os.path.basename(csv_dir)
        csv_root_dir = os.path.dirname(csv_dir)
        if not _img_root_dir:
            img_root_dir = csv_root_dir
        else:
            img_root_dir = _img_root_dir

        out_root_dir = os.path.dirname(csv_root_dir)
        out_dir_name = os.path.splitext(os.path.basename(csv_path))[0]

        if data_type and not out_dir_name.startswith(data_type):
            print('\nskipping file with invalid data type: {}'.format(csv_path))
            continue

        out_dir_path = linux_path(out_root_dir, out_dir_name.capitalize())
        os.makedirs(out_dir_path, exist_ok=True)
        mot_path = linux_path(out_dir_path, '{}.txt'.format(seq_name))

        if os.path.exists(mot_path) and not replace_existing:
            print('\nskipping existing mot file: {} for csv: {}'.format(mot_path, csv_path))
            continue
    else:
        seq_name = os.path.splitext(os.path.basename(csv_path))[0]
        mot_path = csv_path.replace('.csv', '.txt')

    img_path = linux_path(img_root_dir, seq_name)
    _src_files = [k for k in os.listdir(img_path) if
                  os.path.splitext(k.lower())[1] in img_exts]
    _src_files.sort(key=sortKey)

    bounding_boxes = []

    print('\nProcessing file {:d} / {:d} :: {:s}'.format(seq_id + 1, n_csv_files_list, csv_path))
    # print('Reading from {}'.format(csv_path))
    print('Writing to {}'.format(mot_path))

    # continue

    out_fid = open(mot_path, 'w')

    df_det = pd.read_csv(csv_path)

    for _, row in df_det.iterrows():
        filename = row['filename']

        try:
            confidence = row['confidence']
        except KeyError:
            confidence = 1.0

        xmin = float(row['xmin'])
        ymin = float(row['ymin'])
        xmax = float(row['xmax'])
        ymax = float(row['ymax'])

        # width = float(row['width'])
        # height = float(row['height'])
        class_name = row['class']

        try:
            target_id = row['target_id']
        except KeyError:
            target_id = -1

        w, h = xmax - xmin, ymax - ymin

        try:
            frame_id = _src_files.index(filename)
        except:
            raise IOError('Invalid filename found: {}'.format(filename))

        out_fid.write('{:d},{:d},{:f},{:f},{:f},{:f},{:f},-1,-1,-1\n'.format(
            frame_id + 1, target_id, xmin, ymin, w, h, confidence))

        # bounding_boxes.append(
        #     {"class": class_name,
        #      "confidence": confidence,
        #      "filename": filename,
        #      # "width": width,
        #      # "height": height,
        #      "bbox": [xmin, ymin, xmax, ymax]}
        # )

    # for bndbox in bounding_boxes:
    #
    #     xmin, ymin, xmax, ymax = bndbox['bbox']

    out_fid.close()
