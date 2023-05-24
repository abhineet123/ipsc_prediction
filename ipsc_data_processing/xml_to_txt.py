import os, sys, re

import pandas as pd

import paramparse

from eval_utils import sortKey


def save_boxes_txt(_type, csv_path, class_dict, out_dir=''):
    if _type == 0:
        _type_str = 'mAP'
    else:
        _type_str = 'yolo'

    if not csv_path or not os.path.exists(csv_path):
        print('CSV annotations file does not exist: {}'.format(csv_path))
        return None

    def convert_to_yolo(size, box):
        dw = 1. / size[0]
        dh = 1. / size[1]
        x = (box[0] + box[1]) / 2.0
        y = (box[2] + box[3]) / 2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return x, y, w, h

    def getint(fn):
        basename = os.path.basename(fn)
        num = re.sub("\D", "", basename)
        try:
            return int(num)
        except:
            return 0

    if not out_dir:
        out_dir = os.path.join(os.path.dirname(csv_path), _type_str)

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    list_file = None
    if _type == 1:
        list_path = os.path.join(out_dir, 'list.txt')
        list_file = open(list_path, 'w')

    print('Loading CSV annotations from {:s}...'.format(csv_path))
    print('Writing {} annotations to {:s}...'.format(_type_str, out_dir))

    # Read .csv and store as dataframe
    df = pd.read_csv(csv_path)
    file_id = 0
    n_boxes = 0

    # Collect instances of objects and remove from df
    while not df.empty:
        filename = df.iloc[0].loc['filename']
        # Look for objects with similar filenames, group them, send them to csv_to_record function and remove from df
        multiple_instance = df.loc[df['filename'] == filename]
        # Total # of object instances in a file
        no_instances = len(multiple_instance.index)
        # Remove from df (avoids duplication)
        df = df.drop(multiple_instance.index[:no_instances])

        img_width = int(multiple_instance.iloc[0].loc['width'])
        img_height = int(multiple_instance.iloc[0].loc['height'])

        file_no_ext = os.path.splitext(filename)[0]
        out_file_path = os.path.join(out_dir, '{}.txt'.format(file_no_ext))
        out_file = open(out_file_path, 'w')

        for instance in range(0, len(multiple_instance.index)):
            xmin = multiple_instance.iloc[instance].loc['xmin']
            ymin = multiple_instance.iloc[instance].loc['ymin']
            xmax = multiple_instance.iloc[instance].loc['xmax']
            ymax = multiple_instance.iloc[instance].loc['ymax']
            label = multiple_instance.iloc[instance].loc['class']

            def clamp(x, min_value=0.0, max_value=1.0):
                return max(min(x, max_value), min_value)

            xmin = int(clamp(xmin, 0, img_width - 1))
            xmax = int(clamp(xmax, 0, img_width - 1))

            ymin = int(clamp(ymin, 0, img_height - 1))
            ymax = int(clamp(ymax, 0, img_height - 1))

            if _type == 0:
                out_file.write('{:s} {:d} {:d} {:d} {:d}\n'.format(label, xmin, ymin, xmax, ymax))
            else:
                class_id = class_dict[label] + 1
                bb = convert_to_yolo((img_width, img_height), [xmin, xmax, ymin, ymax])
                out_file.write('{:d} {:f} {:f} {:f} {:f}\n'.format(class_id, bb[0], bb[1], bb[2], bb[3]))
            if _type == 1:
                list_file.write('{:s}\n'.format(filename))
            n_boxes += 1
        file_id += 1
        sys.stdout.write('\rDone {:d} files with {:d} boxes ({:d}x{:d})'.format(
            file_id, n_boxes, img_width, img_height))
        sys.stdout.flush()

        out_file.close(),

    if _type == 1:
        list_file.close()

    sys.stdout.write('\n')
    sys.stdout.flush()

    return out_dir


if __name__ == '__main__':
    params = {
        'list_file': '',
        'class_names_path': '',
        'type': 0,
        'file_name': '',
        'out_dir': '',
        'root_dir': '',
        'save_file_name': '',
        'csv_file_name': '',
        'map_folder': '',
        'load_path': '',
        'n_classes': 4,
        'img_ext': 'png',
        'batch_size': 1,
        'show_img': 0,
        'save_video': 1,
        'n_frames': 0,
        'codec': 'H264',
        'fps': 20,
    }

    paramparse.process_dict(params)

    list_file = params['list_file']
    root_dir = params['root_dir']
    class_names_path = params['class_names_path']
    _type = params['type']
    file_name = params['file_name']
    out_dir = params['out_dir']

    class_names = open(class_names_path, 'r').readlines()
    class_dict = {x.strip(): i for (i, x) in enumerate(class_names)}

    if list_file:
        if os.path.isdir(list_file):
            img_paths = [os.path.join(list_file, name) for name in os.listdir(list_file) if
                         os.path.isdir(os.path.join(list_file, name))]
            img_paths.sort(key=sortKey)
        else:
            img_paths = [x.strip() for x in open(list_file).readlines() if x.strip()]
            if root_dir:
                img_paths = [os.path.join(root_dir, name) for name in img_paths]
    else:
        img_paths = [file_name]

    for img_path in img_paths:
        csv_path = os.path.join(img_path, 'annotations.csv')
        seq_out_dir = out_dir
        if seq_out_dir:
            seq_name = os.path.basename(img_path)
            seq_out_dir = os.path.join(seq_out_dir, seq_name)
        save_boxes_txt(_type, csv_path, class_dict, seq_out_dir)
