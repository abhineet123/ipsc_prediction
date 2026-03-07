import pandas as pd
import numpy as np
import os, cv2

import paramparse

from tqdm import tqdm

from eval_utils import ImageSequenceWriter as ImageWriter
from eval_utils import sortKey, resize_ar


class Params:
    """
    :ivar codec:
    :type codec: str

    :ivar csv_file_name:
    :type csv_file_name: str

    :ivar ext:
    :type ext: str

    :ivar file_name:
    :type file_name: str

    :ivar fps:
    :type fps: int

    :ivar img_ext:
    :type img_ext: str

    :ivar list_file_name:
    :type list_file_name: str

    :ivar map_folder:
    :type map_folder: str

    :ivar min_vis:
    :type min_vis: float

    :ivar n_classes:
    :type n_classes: int

    :ivar n_frames:
    :type n_frames: int

    :ivar only_person:
    :type only_person: int

    :ivar resize_factor:
    :type resize_factor: float

    :ivar root_dir:
    :type root_dir: str

    :ivar save_file_name:
    :type save_file_name: str

    :ivar save_path:
    :type save_path: str

    :ivar save_raw:
    :type save_raw: int

    :ivar show_img:
    :type show_img: int

    :ivar vis_height:
    :type vis_height: int

    :ivar vis_root:
    :type vis_root: str

    :ivar vis_width:
    :type vis_width: int

    """

    def __init__(self):
        self.cfg = ()
        self.codec = 'mp4v'
        self.csv_file_name = ''
        self.ext = 'mp4'
        self.file_name = ''
        self.fps = 30
        self.img_ext = 'jpg'
        self.list_file_name = 'lists/idot.txt'
        self.map_folder = ''
        self.n_classes = 4
        self.n_frames = 0
        self.resize_factor = 1.0
        self.root_dir = '/data/GRAM'
        self.save_file_name = ''
        self.save_path = ''
        self.save_raw = 0
        self.show_img = 0
        self.vis_height = 0
        self.save_vis = 0
        self.vis_root = ''
        self.vis_width = 0


def main():
    params = Params()
    paramparse.process(params)

    file_name = params.file_name
    root_dir = params.root_dir
    list_file_name = params.list_file_name
    img_ext = params.img_ext
    show_img = params.show_img
    resize_factor = params.resize_factor
    ext = params.ext
    codec = params.codec
    fps = params.fps
    _vis_width = params.vis_width
    _vis_height = params.vis_height
    save_path = params.save_path
    save_raw = params.save_raw
    save_vis = params.save_vis
    vis_root = params.vis_root

    image_exts = ['jpg', 'bmp', 'png', 'tif']

    if list_file_name:
        if not os.path.exists(list_file_name):
            raise IOError('List file: {} does not exist'.format(list_file_name))
        file_list = [x.strip() for x in open(list_file_name).readlines()]
        if root_dir:
            file_list = [os.path.join(root_dir, 'Images', x) for x in file_list]
    elif root_dir:
        img_root_dir = os.path.join(root_dir, 'Images')
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

        ann_path = os.path.join(root_dir, 'Annotations', seq_name + '.gt')
        ann_lines = open(ann_path).readlines()

        ann_data = [[x for x in _line.strip().split(' ')] for _line in ann_lines]
        # ann_data.sort(key=lambda x: x[0])
        # ann_data = np.asarray(ann_data)

        src_path = img_path
        print(f'getting source images from {src_path}')
        src_files = [f for f in os.listdir(src_path) if
                     os.path.isfile(os.path.join(src_path, f)) and f.endswith(img_ext)]
        src_files.sort(key=sortKey)
        n_frames = len(src_files)

        print('n_frames: ', n_frames)

        obj_dict = {}
        for _data in ann_data:

            object_id, x, y, width, height, frame_id, if_lost, class_label = _data

            class_label = class_label.strip('"')
            object_id = int(object_id) + 1
            frame_id = int(frame_id)

            obj_entry = {
                'object_id': object_id,
                'label': class_label,
                'bbox': [int(x) for x in [x, y, width, height]]
            }
            if frame_id in obj_dict:
                obj_dict[frame_id].append(obj_entry)
            else:
                obj_dict[frame_id] = [obj_entry]

        if save_vis:
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

        font_line_type = cv2.LINE_AA
        out_frame_id = 0
        for frame_id in tqdm(range(n_frames), desc=seq_name):
            filename = src_files[frame_id]
            file_path = os.path.join(src_path, filename)
            if not os.path.exists(file_path):
                raise SystemError('Image file {} does not exist'.format(file_path))

            image = cv2.imread(file_path)
            height, width = image.shape[:2]
            if frame_id in obj_dict:
                if save_vis and save_raw:
                    video_out.write(image)
                out_frame_id += 1
                out_filename = 'image{:06d}.jpg'.format(out_frame_id)

                objects = obj_dict[frame_id]
                for obj in objects:
                    object_id = obj['object_id']
                    label = obj['label']
                    bbox = obj['bbox']

                    # xmin, ymin, xmax, ymax = bbox

                    l, t, w, h = bbox
                    xmin = int(l)
                    ymin = int(t)

                    xmax = int(xmin + w)
                    ymax = int(ymin + h)

                    raw_data = {
                        'filename': out_filename,
                        'width': int(width),
                        'height': int(height),
                        'target_id': object_id,
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

            if save_vis:
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

        print('out_n_frames: ', out_frame_id)
        total_n_frames += out_frame_id

        csv_file = os.path.join(img_path, 'annotations.csv')
        print('saving csv file to {}'.format(csv_file))
        df = pd.DataFrame(csv_raw)
        df.to_csv(csv_file, index=False)

        save_path = ''
        if save_vis:
            video_out.release()
        if show_img:
            cv2.destroyWindow(seq_name)

    print('total_n_frames: ', total_n_frames)


if __name__ == '__main__':
    main()
