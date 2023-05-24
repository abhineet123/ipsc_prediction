import os, sys, re

import pandas as pd

from eval_utils import sortKey


def save_boxes_yolov5(csv_path, class_dict, enable_mask, ignore_invalid_class, consolidate_db, out_dir=''):
    """

    :param csv_path:
    :param class_dict:
    :param enable_mask:
    :param ignore_invalid_class:

    :param consolidate_db: put labels for all sequences in the data set in the same folder in the same folder
    as the dataset folder itself
    if disabled, the labels for each sequence are put under that sequence's folder

    :param out_dir:
    :return:
    """
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

    seq_path = os.path.dirname(csv_path)
    seq_name = os.path.basename(seq_path)
    db_path = os.path.dirname(seq_path)
    db_root_path = os.path.dirname(db_path)

    if not out_dir:
        if consolidate_db:
            out_dir = os.path.join(db_root_path, 'labels', seq_name)
        else:
            out_dir = os.path.join(seq_path, 'labels')

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    list_path = os.path.join(out_dir, 'list.txt')
    list_file = open(list_path, 'w')

    print('Loading CSV annotations from {:s}...'.format(csv_path))
    print('Writing YOLOv5 annotations to {:s}...'.format(out_dir))

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

            try:
                class_id = class_dict[label]
            except KeyError:
                if ignore_invalid_class:
                    print(f'ignoring invalid_class: {label}')
                    continue
                raise AssertionError(f'invalid_class: {label}')

            if enable_mask:
                mask_str = multiple_instance.iloc[instance].loc['mask']
                mask_pts_1 = mask_str.split(';')
                mask_pts_2 = [k.split(',') for k in mask_pts_1 if k]
                mask_pts = [(float(x) / img_width, float(y) / img_height) for x, y in mask_pts_2]

                mask_pts = [(clamp(x), clamp(y)) for x, y in mask_pts]
                # xs, ys = zip(*mask_pts)
                # print()

                yolo_mask_str = ' '.join(f'{x} {y}' for x, y in mask_pts)
                out_file.write('{:d} {:s}\n'.format(class_id, yolo_mask_str))
            else:
                bb = convert_to_yolo((img_width, img_height), [xmin, xmax, ymin, ymax])
                out_file.write('{:d} {:f} {:f} {:f} {:f}\n'.format(class_id, bb[0], bb[1], bb[2], bb[3]))

            list_file.write('{:s}\n'.format(filename))

            n_boxes += 1
        file_id += 1
        sys.stdout.write('\rDone {:d} files with {:d} boxes ({:d}x{:d})'.format(
            file_id, n_boxes, img_width, img_height))
        sys.stdout.flush()

        out_file.close(),

    list_file.close()

    sys.stdout.write('\n')
    sys.stdout.flush()

    return out_dir


import paramparse


class Params:
    """
    :ivar batch_size:
    :type batch_size: int

    :ivar class_names_path:
    :type class_names_path: str

    :ivar codec:
    :type codec: str

    :ivar csv_file_name:
    :type csv_file_name: str

    :ivar file_name:
    :type file_name: str

    :ivar fps:
    :type fps: int

    :ivar img_ext:
    :type img_ext: str

    :ivar list_file:
    :type list_file: str

    :ivar load_path:
    :type load_path: str

    :ivar map_folder:
    :type map_folder: str

    :ivar n_classes:
    :type n_classes: int

    :ivar n_frames:
    :type n_frames: int

    :ivar out_dir:
    :type out_dir: str

    :ivar read_colors:
    :type read_colors: int

    :ivar root_dir:
    :type root_dir: str

    :ivar save_file_name:
    :type save_file_name: str

    :ivar save_video:
    :type save_video: int

    :ivar show_img:
    :type show_img: int

    """

    def __init__(self):
        self.cfg = ()
        self.batch_size = 1
        self.class_names_path = ''
        self.codec = 'H264'
        self.csv_file_name = ''
        self.file_name = ''
        self.fps = 20
        self.img_ext = 'png'
        self.list_file = ''
        self.load_path = ''
        self.map_folder = ''
        self.n_classes = 4
        self.n_frames = 0
        self.out_dir = ''
        self.read_colors = 1
        self.root_dir = ''
        self.save_file_name = ''
        self.save_video = 1
        self.show_img = 0
        self.enable_mask = 0
        self.ignore_invalid_class = 0
        self.consolidate_db = 0


if __name__ == '__main__':

    params = Params()
    paramparse.process(params)

    seq_paths = params.list_file
    root_dir = params.root_dir
    class_names_path = params.class_names_path
    read_colors = params.read_colors
    file_name = params.file_name
    out_dir = params.out_dir
    enable_mask = params.enable_mask
    ignore_invalid_class = params.ignore_invalid_class

    if enable_mask:
        print("running in mask mode")

    assert class_names_path, "class_names_path must be provided"

    class_names = [k.strip() for k in open(class_names_path, 'r').readlines() if k.strip()]
    if read_colors:
        class_names, class_cols = zip(*[k.split('\t') for k in class_names])

    class_dict = {x.strip(): i for (i, x) in enumerate(class_names)}

    if seq_paths:
        if os.path.isfile(seq_paths):
            seq_paths = [x.strip() for x in open(seq_paths).readlines() if x.strip()]
        else:
            seq_paths = seq_paths.split(',')

        if root_dir:
            seq_paths = [os.path.join(root_dir, name) for name in seq_paths]

    elif root_dir:
        seq_paths = [os.path.join(root_dir, name) for name in os.listdir(root_dir) if
                     os.path.isdir(os.path.join(root_dir, name))]
        seq_paths.sort(key=sortKey)
    else:
        raise IOError('Either seq_paths or root_dir must be provided')

    for seq_path in seq_paths:
        csv_path = os.path.join(seq_path, 'annotations.csv')
        save_boxes_yolov5(csv_path, class_dict, enable_mask, ignore_invalid_class, params.consolidate_db)
