import pandas as pd
import os
import cv2

from tqdm import tqdm

import paramparse

from mot_csv_to_xml_coco import parse_csv, parse_mot
from eval_utils import ImageSequenceWriter as ImageWriter
from eval_utils import sortKey, resize_ar, drawBox, clamp, linux_path


#
# def parse_mot_old():
#     ann_data = [[float(x) for x in _line.strip().split(',')] for _line in ann_lines if _line.strip()]
#     # ann_data.sort(key=lambda x: x[0])
#     # ann_data = np.asarray(ann_data)
#
#     obj_ids = []
#     obj_dict = {}
#     for __id, _data in enumerate(ann_data):
#
#         # if _data[7] != 1 or _data[8] < min_vis:
#         #     continue
#         obj_id = int(_data[1])
#
#         obj_ids.append(obj_id)
#
#         # Bounding box sanity check
#         bbox = [float(x) for x in _data[2:6]]
#         l, t, w, h = bbox
#         xmin = int(l)
#         ymin = int(t)
#         xmax = int(xmin + w)
#         ymax = int(ymin + h)
#         if xmin >= xmax or ymin >= ymax:
#             msg = 'Invalid box {}\n in line {} : {}\n'.format(
#                 [xmin, ymin, xmax, ymax], __id, _data
#             )
#             if ignore_invalid:
#                 print(msg)
#             else:
#                 raise AssertionError(msg)
#
#         confidence = float(_data[6])
#
#         if w <= 0 or h <= 0 or confidence == 0:
#             """annoying meaningless unexplained crappy boxes that exist for no apparent reason at all"""
#             continue
#
#         frame_id = int(_data[0]) - 1
#         # xmin, ymin, w, h = _data[2:]
#
#         if percent_scores:
#             confidence /= 100.0
#
#         if clamp_scores:
#             confidence = max(min(confidence, 1), 0)
#
#         if 0 <= confidence <= 1:
#             pass
#         else:
#             msg = "Invalid confidence: {} in line {} : {}".format(
#                 confidence, __id, _data)
#
#             if ignore_invalid == 2:
#                 confidence = 1
#             elif ignore_invalid == 1:
#                 print(msg)
#             else:
#                 raise AssertionError(msg)
#
#         obj_entry = {
#             'id': obj_id,
#             'label': label,
#             'bbox': bbox,
#             'confidence': confidence
#         }
#         if frame_id not in obj_dict:
#             obj_dict[frame_id] = []
#         obj_dict[frame_id].append(obj_entry)
#
#     print('Done reading {}'.format(data_type))


class Params:
    def __init__(self):
        self.cfg = ()
        self.clamp_scores = 0
        self.codec = 'mp4v'
        self.csv_file_name = ''
        self.data_type = 'annotations'
        self.ext = 'mp4'
        self.file_name = ''
        self.fps = 30

        self.allow_ignored = 0
        self.ignore_invalid = 0
        self.ignore_missing = 0
        self.img_ext = 'jpg'
        self.list_file_name = ''
        self.map_folder = ''

        """
        if mode == 0:
            ann_path = linux_path(img_path, 'gt', 'gt.txt')
        elif mode == 1:
            ann_path = linux_path(img_path, f'../../{data_type.capitalize()}/{seq_name}.txt')
        elif mode == 2:
            is_csv = 1
            ann_path = linux_path(img_path, f'{seq_name}.csv')
        """
        self.mode = 1

        self.n_classes = 4
        self.n_frames = 0
        self.out_root_dir = ''
        self.out_root_suffix = ''
        self.out_suffix = ''
        self.percent_scores = 0
        self.root_dir = ''
        self.save_file_name = ''
        self.save_path = ''
        self.save_raw = 0
        self.save_video = 0
        self.show_class = 2
        self.show_img = 1
        self.start_id = 0
        self.end_id = -1
        self.stats_only = 0
        self.vis_size = ''

        self.allow_empty = 1
        self.sample = 0
        self.vid_ext = ''
        self.class_names_path = ''
        self.img_dir = ''


def main():
    params = Params()

    paramparse.process(params)

    out_suffix = params.out_suffix
    data_type = params.data_type
    file_name = params.file_name
    root_dir = params.root_dir
    list_file_name = params.list_file_name
    img_ext = params.img_ext
    show_img = params.show_img
    show_class = params.show_class
    ext = params.ext
    codec = params.codec
    fps = params.fps
    vis_size = params.vis_size
    save_path = params.save_path
    save_raw = params.save_raw
    save_video = params.save_video
    mode = params.mode
    start_id = params.start_id
    end_id = params.end_id
    ignore_missing = params.ignore_missing
    percent_scores = params.percent_scores
    clamp_scores = params.clamp_scores
    ignore_invalid = params.ignore_invalid
    stats_only = params.stats_only
    vid_ext = params.vid_ext
    out_root_dir = params.out_root_dir
    out_root_suffix = params.out_root_suffix
    class_names_path = params.class_names_path
    img_dir = params.img_dir

    if img_dir:
        img_root_dir = linux_path(root_dir, img_dir)
    else:
        img_root_dir = root_dir

    image_exts = ['jpg', 'bmp', 'png', 'tif']

    class_info = [k.strip() for k in open(class_names_path, 'r').readlines() if k.strip()]
    class_names, class_cols = zip(*[k.split('\t') for k in class_info])
    # label2id = {x.strip(): i for (i, x) in enumerate(class_names)}
    class_to_id = {x: i for (i, x) in enumerate(class_names)}
    class_to_col = {x: c for (x, c) in zip(class_names, class_cols)}

    if list_file_name:
        if not os.path.exists(list_file_name):
            raise IOError('List file: {} does not exist'.format(list_file_name))
        seq_paths = [x.strip() for x in open(list_file_name).readlines() if x.strip()]
        if img_root_dir:
            seq_paths = [linux_path(img_root_dir, x) for x in seq_paths]
    elif img_root_dir:
        if vid_ext:
            seq_paths = [linux_path(img_root_dir, name) for name in os.listdir(img_root_dir) if
                         os.path.isfile(linux_path(img_root_dir, name)) and name.endswith(vid_ext)]

            """remove ext to be compatible with image sequence paths"""
            seq_paths = [os.path.splitext(seq_path)[0] for seq_path in seq_paths]
            seq_paths.sort(key=sortKey)
        else:
            seq_paths = [linux_path(img_root_dir, name) for name in os.listdir(img_root_dir) if
                         os.path.isdir(linux_path(img_root_dir, name))]
            seq_paths.sort(key=sortKey)
    else:
        if not file_name:
            raise IOError('Either list file or a single sequence file must be provided')
        seq_paths = [file_name]
    n_seq = len(seq_paths)
    print('Found {} sequences'.format(n_seq))

    if not out_root_dir:
        if out_root_suffix and root_dir:
            out_root_dir = f'{root_dir}_{out_root_suffix}'
        else:
            out_root_dir = os.path.dirname(seq_paths[0])

    pause_after_frame = 1
    total_n_frames = 0
    total_unique_obj_ids = 0

    if end_id < 0:
        end_id = len(seq_paths) - 1
    seq_paths = seq_paths[start_id:end_id + 1]
    n_seq = len(seq_paths)
    print('Running over {} sequences'.format(n_seq))

    seq_to_n_unique_obj_ids = {}
    for seq_idx, seq_path in enumerate(seq_paths):
        seq_name = os.path.basename(seq_path)
        print('seq_name: ', seq_name)

        print('sequence {}/{}: {}: '.format(seq_idx + 1, n_seq, seq_name))

        vid_cap = None
        src_path = None

        if vid_ext:
            is_vid = 1
            vid_path = f'{seq_path}.{vid_ext}'
            assert os.path.isfile(vid_path), f"invalid vid_path: {vid_path}"
            vid_cap = cv2.VideoCapture()
            if not vid_cap.open(vid_path):
                raise AssertionError(f'Video file {vid_path} could not be opened')
            n_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))

            src_files = [linux_path(seq_path, f'image{frame_id + 1:06d}.jpg')
                         for frame_id in range(n_frames)]
        else:
            is_vid = 0
            if mode == 0:
                src_path = linux_path(seq_path, 'img1')
            else:
                src_path = seq_path

            src_files = [f for f in os.listdir(src_path) if
                         os.path.isfile(linux_path(src_path, f)) and f.endswith(img_ext)]
            src_files.sort(key=sortKey)

            n_frames = len(src_files)

        sampled_frame_ids = list(range(n_frames))

        is_csv = 0

        if mode == 0:
            ann_path = linux_path(seq_path, 'gt', 'gt.txt')
        elif mode == 1:
            ann_path = linux_path(seq_path, f'../../{data_type.capitalize()}/{seq_name}.txt')
        elif mode == 2:
            is_csv = 1
            if is_vid:
                ann_path = f'{seq_path}.csv'
            else:
                ann_path = linux_path(seq_path, f'{data_type}.csv')

        ann_path = os.path.abspath(ann_path)

        if not os.path.exists(ann_path):
            msg = f"{data_type} file for sequence {seq_name} not found: {ann_path}"
            if ignore_missing:
                print(msg)
                continue
            else:
                raise IOError(msg)

        print(f'Reading {data_type} from {ann_path}')

        # ann_lines = open(ann_path).readlines()

        if params.sample:
            print(f'sampling 1 in {params.sample} frames')
            sampled_frame_ids = sampled_frame_ids[::params.sample]

        n_sampled_frames = len(sampled_frame_ids)

        print(f'n_frames: {n_frames}')
        print(f'n_sampled_frames: {n_sampled_frames}', )

        total_n_frames += n_frames

        if is_csv:
            obj_ids, obj_dict = parse_csv(ann_path, sampled_frame_ids, ignore_invalid, percent_scores, clamp_scores)
        else:
            assert len(class_names) == 1 or params.allow_ignored, \
                "multiple class names not supported in MOT mode unless allow_ignored is on"
            obj_ids, obj_dict = parse_mot(
                ann_path, sampled_frame_ids, class_names[0], ignore_invalid, percent_scores,
                clamp_scores, params.allow_ignored)

        # print(f'{list(obj_dict.keys())}')
        enable_resize = 0

        unique_obj_ids = list(set(obj_ids))
        n_unique_obj_ids = len(unique_obj_ids)

        total_unique_obj_ids += n_unique_obj_ids

        print(f'{seq_name}: {n_unique_obj_ids}')

        seq_to_n_unique_obj_ids[seq_name] = n_unique_obj_ids

        if stats_only:
            continue

        if not vis_size:
            if is_vid:
                vis_height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                vis_width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))

            else:
                temp_img = cv2.imread(linux_path(src_path, src_files[0]))
                vis_height, vis_width, _ = temp_img.shape
        else:
            vis_width, vis_height = [int(x) for x in vis_size.split('x')]
            enable_resize = 1

        if not save_path:
            save_path = linux_path(out_root_dir, os.path.basename(seq_path) + '.' + ext)

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

        vid_frame_id = -1

        n_empty = 0

        pbar = tqdm(sampled_frame_ids)
        for frame_id in pbar:
            filename = src_files[frame_id]

            if is_vid:
                image = None

                while vid_frame_id < frame_id:
                    ret, image = vid_cap.read()
                    vid_frame_id += 1
                    if not ret:
                        raise AssertionError('Frame {:d} could not be read'.format(vid_frame_id + 1))
            else:
                file_path = linux_path(src_path, filename)
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

                    # l, t, w, h = bbox
                    # xmin = int(l)
                    # ymin = int(t)
                    #
                    # xmax = int(xmin + w)
                    # ymax = int(ymin + h)

                    xmin, ymin, xmax, ymax = bbox

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

                    col = class_to_col[label]

                    csv_raw.append(raw_data)

                    if show_img or not save_raw:
                        if show_class == 2:
                            _label = f'{label}'
                        elif show_class == 1:
                            _label = f'{obj_id}: {label}'
                        else:
                            _label = f'{obj_id}'

                        drawBox(image, xmin, ymin, xmax, ymax, label=_label, font_size=0.5, box_color=col)
            else:
                n_empty += 1
                pbar.set_description(f'n_empty: {n_empty}')
                if not params.allow_empty:
                    raise AssertionError(f'No {data_type} found for frame {frame_id}')

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
            csv_file = linux_path(save_dir, save_seq_name, out_fname)
        else:
            csv_file = linux_path(seq_path, out_fname)

        if out_root_dir:
            csv_file = csv_file.replace(root_dir, out_root_dir, 1)

        print('saving csv file to {}'.format(csv_file))

        os.makedirs(os.path.dirname(csv_file), exist_ok=True)

        df = pd.DataFrame(csv_raw)
        df.to_csv(csv_file)
        if save_video:
            video_out.release()
        save_path = ''

        if show_img:
            try:
                cv2.destroyWindow(seq_name)
            except cv2.error:
                pass
        # total_n_frames += out_frame_id
        # print('out_n_frames: ', out_frame_id)

    print('total_n_frames: ', total_n_frames)
    print('total_unique_obj_ids: ', total_unique_obj_ids)


if __name__ == '__main__':
    main()
