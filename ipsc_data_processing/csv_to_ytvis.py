import pandas as pd
import os
import copy
import sys, time, random
# import glob
from random import shuffle
from tqdm import tqdm

from PIL import Image
from pprint import pprint, pformat
from eval_utils import sortKey
import math
import paramparse
import ast
import numpy as np

from subprocess import Popen, PIPE


# image_names = []
def checkImage(fn):
    proc = Popen(['identify', '-verbose', fn], stdout=PIPE, stderr=PIPE)
    out, err = proc.communicate()
    exitcode = proc.returncode
    return exitcode, out, err


# Function to convert csv to TFRecord
def csv_to_json(df_bboxes, file_path, class_dict, min_size, check_images=1):
    # Get metadata
    try:
        filename = df_bboxes.iloc[0].loc['filename']
    except IndexError:
        raise IOError('No annotations found for {}'.format(file_path))

    # image_name = os.path.join(seq_path, filename)
    image_name = file_path

    if check_images:
        code, output, error = checkImage(image_name)
        if str(code) != "0" or str(error) != "":
            raise IOError("Damaged image found: {} :: {}".format(image_name, error))

    # if image_name in image_names:
    #     raise SystemError('Image {} already exists'.format(image_name))

    assert filename == os.path.basename(file_path), (
        'DF filename: {} does not match the file_path: {}'.format(filename, file_path))

    # print('Adding {}'.format(image_name))
    # image_names.append(image_name)

    # print(image_name)

    width = float(df_bboxes.iloc[0].loc['width'])
    height = float(df_bboxes.iloc[0].loc['height'])

    xmins = []
    ymins = []

    xmaxs = []
    ymaxs = []

    classes_text = []
    class_ids = []

    # # Mapping name to id
    # class_dict = {
    #     'bear': 1,
    #     'moose': 2,
    #     'coyote': 3,
    #     'deer': 4,
    #     'person': 5
    # }
    n_df_bboxes = len(df_bboxes.index)

    bboxes = []

    for i in range(n_df_bboxes):
        df_bbox = df_bboxes.iloc[i]
        xmin = df_bbox.loc['xmin']
        ymin = df_bbox.loc['ymin']
        xmax = df_bbox.loc['xmax']
        ymax = df_bbox.loc['ymax']
        class_name = df_bbox.loc['class']

        try:
            class_id = class_dict[class_name]
        except KeyError:
            # print('\nIgnoring box with unknown class {} in image {} '.format(class_name, image_name))
            continue

        w = xmax - xmin
        h = ymax - ymin

        if w < 0 or h < 0:
            print('\nInvalid box in image {} with dimensions {} x {}\n'.format(image_name, w, h))
            xmin, xmax = xmax, xmin
            ymin, ymax = ymax, ymin
            w = xmax - xmin
            h = ymax - ymin
            if w < 0 or h < 0:
                raise IOError('\nSwapping corners still giving invalid dimensions {} x {}\n'.format(w, h))

        if w < min_size or h < min_size:
            print('\nIgnoring image {} with too small {} x {} box '.format(image_name, w, h))
            return None, None, None
            # continue

        def clamp(x, min_value=0.0, max_value=1.0):
            return max(min(x, max_value), min_value)

        bboxes.append((class_id, xmin, ymin, xmax, ymax))

        xmin = clamp(xmin / width)
        xmax = clamp(xmax / width)
        ymin = clamp(ymin / height)
        ymax = clamp(ymax / height)

        xmins.append(xmin)
        xmaxs.append(xmax)
        ymins.append(ymin)
        ymaxs.append(ymax)

        classes_text.append(class_name)
        class_ids.append(class_id)

    return bboxes, (height, width)


def str_to_list(_str, _type=str, _sep=','):
    if _sep not in _str:
        _str += _sep
    k = list(map(_type, _str.split(_sep)))
    k = [_k for _k in k if _k]
    return k


class Params:

    def __init__(self):
        self.cfg = ''
        self.allow_missing_annotations = 0
        self.allow_seq_skipping = 1
        self.min_samples_per_seq = 0
        self.annotations_list_path = ''
        self.check_images = 0
        self.class_names_path = ''
        self.csv_paths = ''
        self.even_sampling = 0.0
        self.exclude_loaded_samples = 0
        self.fixed_ar = 0
        self.inverted_sampling = 0
        self.load_samples = []
        self.load_samples_root = ''
        self.min_size = 1
        self.n_frames = 0
        self.only_sampling = 0
        self.output_path = ''
        self.random_sampling = 0
        self.root_dir = ''
        self.samples_per_class = 0
        self.samples_per_seq = 0
        self.sampling_ratio = 1.0
        self.sample_entire_seq = 0
        self.n_sample_permutations = 100

        self.seq_paths = ''
        self.shuffle_files = 1
        self.write_annotations_list = 2
        self.annotations_list_sep = ' '

        self.start_id = -1
        self.end_id = -1

        self.class_from_seq = 1


def main():
    flags = Params()
    paramparse.process(flags)

    output_path = flags.output_path
    seq_paths = flags.seq_paths
    csv_paths = flags.csv_paths
    shuffle_files = flags.shuffle_files
    n_frames = flags.n_frames
    class_names_path = flags.class_names_path
    min_size = flags.min_size
    root_dir = flags.root_dir
    allow_missing_annotations = flags.allow_missing_annotations
    inverted_sampling = flags.inverted_sampling
    load_samples = flags.load_samples
    load_samples_root = flags.load_samples_root
    exclude_loaded_samples = flags.exclude_loaded_samples
    check_images = flags.check_images

    sampling_ratio = flags.sampling_ratio
    random_sampling = flags.random_sampling
    even_sampling = flags.even_sampling
    samples_per_class = flags.samples_per_class
    samples_per_seq = flags.samples_per_seq
    sample_entire_seq = flags.sample_entire_seq
    n_sample_permutations = flags.n_sample_permutations
    min_samples_per_seq = flags.min_samples_per_seq

    only_sampling = flags.only_sampling
    write_annotations_list = flags.write_annotations_list
    ann_sep = flags.annotations_list_sep
    annotations_list_path = flags.annotations_list_path
    allow_seq_skipping = flags.allow_seq_skipping

    start_id = flags.start_id
    end_id = flags.end_id

    class_from_seq = flags.class_from_seq

    sample_from_end = 0
    variable_sampling_ratio = 0

    if ann_sep == '0':
        ann_sep = ' '
    elif ann_sep == '1':
        ann_sep = '\t'

    if samples_per_seq != 0:
        sampling_ratio = 0
        samples_per_class = 0
        variable_sampling_ratio = 1
        if samples_per_seq < 0:
            samples_per_seq = -samples_per_seq
            print('Sampling from end')
            sample_from_end = 1
        print('Using variable sampling ratio to include {} samples per sequence'.format(samples_per_seq))

    if samples_per_class != 0:
        sampling_ratio = 0
        if sample_entire_seq:
            variable_sampling_ratio = 3
        else:
            variable_sampling_ratio = 2

        if samples_per_class < 0:
            samples_per_class = -samples_per_class
            print('Sampling from end')
            sample_from_end = 1
        print('Using variable sampling ratio to include {} samples per class'.format(samples_per_class))

    if sampling_ratio < 0:
        print('Sampling from end')
        sample_from_end = 1
        sampling_ratio = -sampling_ratio
        print('sampling_ratio: ', sampling_ratio)

    if even_sampling != 0:
        print('Using evenly spaced sampling')
        random_sampling = 0
        if even_sampling < 0:
            even_sampling = -even_sampling
            inverted_sampling = 1

    if inverted_sampling:
        print('Inverting the sampling')

    assert sampling_ratio <= 1.0 and sampling_ratio >= 0.0, 'sampling_ratio must be between 0 and 1'

    class_names = open(class_names_path, 'r').readlines()

    class_names = [x.strip() for x in class_names]
    class_dict = {x: i for (i, x) in enumerate(class_names)}

    # Writer object for TFRecord creation
    output_dir = os.path.dirname(output_path)
    output_name = os.path.basename(output_path)
    output_name_no_ext = os.path.splitext(output_name)[0]
    print('output_path: ', output_path)
    print('output_dir: ', output_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    if not annotations_list_path:
        annotations_list_path = os.path.join(output_dir, output_name_no_ext + '.txt')

    if seq_paths:
        if os.path.isfile(seq_paths):
            seq_paths = [x.strip() for x in open(seq_paths).readlines() if x.strip()]
        else:
            seq_paths = seq_paths.split(',')
        if root_dir:
            seq_paths = [os.path.join(root_dir, k) for k in seq_paths]

    elif root_dir:
        seq_paths = [os.path.join(root_dir, name) for name in os.listdir(root_dir) if
                     os.path.isdir(os.path.join(root_dir, name))]
        seq_paths.sort(key=sortKey)
    else:
        raise IOError('Either seq_paths or root_dir must be provided')

    seq_paths = [k.replace(os.sep, '/') for k in seq_paths]

    if csv_paths:
        with open(csv_paths) as f:
            csv_paths = f.readlines()
        csv_paths = [x.strip() for x in csv_paths if x.strip()]
    else:
        csv_paths = [os.path.join(img_path, 'annotations.csv') for img_path in seq_paths]
    n_seq = len(seq_paths)
    if len(csv_paths) != n_seq:
        raise IOError('Mismatch between image and csv paths counts')

    if start_id < 0:
        start_id = 0
    if end_id < start_id:
        end_id = n_seq - 1

    assert end_id >= start_id, "end_id: {} is < start_id: {}".format(end_id, start_id)

    if start_id > 0 or end_id < n_seq - 1:
        print('curtailing sequences to between IDs {} and {}'.format(start_id, end_id))
        seq_paths = seq_paths[start_id:end_id + 1]
        csv_paths = csv_paths[start_id:end_id + 1]
        n_seq = len(seq_paths)

    if not load_samples:
        exclude_loaded_samples = 0

    seq_to_samples = {}

    if len(load_samples) == 1:
        if load_samples[0] == '1' or load_samples[0] == 1:
            load_samples = ['seq_to_samples.txt', ]
        elif load_samples[0] == '0' or load_samples[0] == 0:
            load_samples = []

    if load_samples:
        print('load_samples: {}'.format(pformat(load_samples)))
        if load_samples_root:
            load_samples = [os.path.join(load_samples_root, k) for k in load_samples if k]
        print('Loading samples from : {}'.format(pformat(load_samples)))
        for _f in load_samples:
            if os.path.isdir(_f):
                _f = os.path.join(_f, 'seq_to_samples.txt')
            with open(_f, 'r') as fid:
                curr_seq_to_samples = ast.literal_eval(fid.read())
                for _seq in curr_seq_to_samples:
                    if _seq in seq_to_samples:
                        seq_to_samples[_seq] += curr_seq_to_samples[_seq]
                    else:
                        seq_to_samples[_seq] = curr_seq_to_samples[_seq]

        if exclude_loaded_samples:
            print('Excluding loaded samples from source files')

    img_ext = 'jpg'
    total_samples = 0
    total_files = 0

    seq_to_src_files = {}
    class_to_n_files = {_class: 0 for _class in class_names}
    class_to_n_seq = {_class: 0 for _class in class_names}
    class_to_n_files_per_seq = {_class: {} for _class in class_names}

    seq_to_n_files = {}

    def get_class(seq_path):
        for _class in class_names:
            if _class in os.path.basename(seq_path):
                return _class
        raise IOError('No class found for {}'.format(seq_path))

    seq_to_sampling_ratio = {k: sampling_ratio for k in seq_paths}
    if class_from_seq:
        seq_to_class = {k: get_class(k) for k in seq_paths}
    else:
        print('assigning class {} to all sequences'.format(class_names[0]))
        seq_to_class = {k: class_names[0] for k in seq_paths}

    empty_seqs = []
    non_empty_seq_ids = []
    for idx, seq_path in enumerate(seq_paths):
        src_files = [os.path.join(seq_path, k).replace(os.sep, '/') for k in os.listdir(seq_path) if
                     os.path.splitext(k.lower())[1][1:] == img_ext]

        if exclude_loaded_samples:
            loaded_samples = seq_to_samples[seq_path]
            src_files = [k for k in src_files if k not in loaded_samples]

        if not src_files:
            if allow_seq_skipping:
                print('Skipping empty sequence: {}'.format(seq_path))
                empty_seqs.append(seq_path)
                continue
            else:
                raise IOError('Empty sequence found: {}'.format(seq_path))

        non_empty_seq_ids.append(idx)

        src_files.sort(key=sortKey)
        n_files = len(src_files)

        seq_to_n_files[seq_path] = n_files
        seq_to_src_files[seq_path] = src_files

        _class = seq_to_class[seq_path]
        class_to_n_files_per_seq[_class][seq_path] = n_files
        class_to_n_seq[_class] += 1
        class_to_n_files[_class] += n_files

        total_files += n_files

    print('class_to_n_files:')
    pprint(class_to_n_files)

    n_empty_seqs = len(empty_seqs)
    if n_empty_seqs:
        print('Found {} empty sequences:\n{}'.format(n_empty_seqs, pformat(empty_seqs)))
        seq_paths = [k for k in seq_paths if k not in empty_seqs]
        csv_paths = [csv_paths[i] for i in non_empty_seq_ids]

        n_seq = len(seq_paths)
        if len(csv_paths) != n_seq:
            raise IOError('Mismatch between image and csv paths counts')

    if variable_sampling_ratio == 1:
        seq_to_sampling_ratio = {
            k: float(samples_per_seq) / float(seq_to_n_files[k]) for k in seq_paths
        }
    elif variable_sampling_ratio == 2:
        class_to_sampling_ratio = {k: float(samples_per_class) / class_to_n_files[k] for k in class_to_n_files}

        print('class_to_sampling_ratio:')
        pprint(class_to_sampling_ratio)

        seq_to_sampling_ratio = {
            k: class_to_sampling_ratio[seq_to_class[k]] for k in seq_paths
        }
    elif variable_sampling_ratio == 3:
        seq_to_sampling_ratio = {}
        n_sampled_seq = n_unsampled_seq = 0
        for _class in class_to_n_files:
            n_files_per_seq = class_to_n_files_per_seq[_class]
            class_seqs = list(n_files_per_seq.keys())
            _n_seq = class_to_n_seq[_class]

            min_diff = None
            min_diff_id = None
            for permute_id in range(n_sample_permutations):
                shuffle(class_seqs)
                _n_files_per_seq = [n_files_per_seq[_seq] for _seq in class_seqs]
                _n_files_per_seq_cum = np.cumsum(_n_files_per_seq)

                diff = np.abs(_n_files_per_seq_cum - samples_per_class)
                curr_min_diff_id = np.argmin(diff)
                curr_min_diff = diff[curr_min_diff_id]

                if min_diff is None or curr_min_diff < min_diff:
                    min_diff = curr_min_diff
                    min_diff_id = curr_min_diff_id
                    min_diff_class_seqs = copy.deepcopy(class_seqs)
                    min_diff_n_files_per_seq = copy.deepcopy(_n_files_per_seq)
                    min_diff_n_files_per_seq_cum = copy.deepcopy(_n_files_per_seq_cum)

                    if min_diff == 0:
                        break

            min_diff_seqs = ['{:5d} {} {:10d} {:10d}'.format(
                i, min_diff_class_seqs[i], min_diff_n_files_per_seq[i], min_diff_n_files_per_seq_cum[i])
                for i in range(_n_seq)]

            print(pformat(min_diff_seqs))
            print('_class: {}'.format(_class))
            print('min_diff: {}'.format(min_diff))
            print('min_diff_id: {}'.format(min_diff_id))

            n_sampled_seq += min_diff_id + 1
            n_unsampled_seq += _n_seq - min_diff_id - 1

            _seq_to_sampling_ratio = {
                min_diff_class_seqs[i]: 1 if i <= min_diff_id else 0 for i in range(_n_seq)
            }
            seq_to_sampling_ratio.update(_seq_to_sampling_ratio)

            print()
        print('n_sampled_seq: {}'.format(n_sampled_seq))
        print('n_unsampled_seq: {}'.format(n_unsampled_seq))
        print()

    class_to_n_samples = {_class: 0 for _class in class_names}
    all_sampled_files = []

    n_sampled_seq = n_unsampled_seq = 0

    valid_seq_to_samples = {}

    seq_to_df_data = {}

    for idx, seq_path in enumerate(seq_paths):
        src_files = seq_to_src_files[seq_path]
        n_files = seq_to_n_files[seq_path]
        sampling_ratio = seq_to_sampling_ratio[seq_path]

        if sampling_ratio > 1.0:
            raise IOError('Invalid sampling_ratio: {} for sequence: {} with {} files'.format(
                sampling_ratio, seq_path, n_files))
        if load_samples and not exclude_loaded_samples:
            try:
                loaded_samples = seq_to_samples[seq_path]
            except KeyError:
                loaded_samples = []

            if inverted_sampling:
                loaded_samples = [k for k in src_files if k not in loaded_samples]

            n_loaded_samples = len(loaded_samples)

            if sampling_ratio == 1.0 or n_loaded_samples == 0:
                sampled_files = loaded_samples
            else:
                n_samples = int(n_loaded_samples * sampling_ratio)
                if n_samples < min_samples_per_seq:
                    sampling_ratio = float(min_samples_per_seq) / float(n_files)
                    n_samples = min_samples_per_seq
                    print('\nSetting sampling_ratio for {} to {} to be able to have {} samples\n'.format(
                        seq_path, sampling_ratio, min_samples_per_seq))

                if random_sampling:
                    sampled_files = random.sample(loaded_samples, n_samples)
                elif even_sampling:
                    if sampling_ratio > even_sampling:
                        raise SystemError('{} :: sampling_ratio: {} is greater than even_sampling: {}'.format(
                            seq_path, sampling_ratio, even_sampling))
                    sample_1_of_n = int(math.ceil(even_sampling / sampling_ratio))
                    end_file = int(n_files * even_sampling)

                    if sample_from_end:
                        sub_loaded_samples = loaded_samples[slice(-1, -end_file)]
                    else:
                        sub_loaded_samples = loaded_samples[slice(0, end_file)]

                    sampled_files = sub_loaded_samples[::sample_1_of_n]

                    more_samples_needed = n_samples - len(sampled_files)
                    if more_samples_needed > 0:
                        unsampled_files = [k for k in sub_loaded_samples if k not in sampled_files]
                        sampled_files += unsampled_files[:more_samples_needed]
                else:
                    if sample_from_end:
                        sampled_files = loaded_samples[-n_samples:]
                    else:
                        sampled_files = loaded_samples[:n_samples]
            n_samples = len(sampled_files)
        else:
            n_samples = int(round(n_files * sampling_ratio))
            if n_samples < min_samples_per_seq:
                sampling_ratio = float(min_samples_per_seq) / float(n_files)
                n_samples = min_samples_per_seq
                print('\nSetting sampling_ratio for {} to {} to be able to have {} samples\n'.format(
                    seq_path, sampling_ratio, min_samples_per_seq))

            if sampling_ratio != 1.0:
                if n_samples == 0:
                    sampled_files = []
                else:
                    if random_sampling:
                        sampled_files = random.sample(src_files, n_samples)
                    elif even_sampling:
                        if sampling_ratio > even_sampling:
                            raise SystemError('{} :: sampling_ratio: {} is greater than even_sampling: {}'.format(
                                seq_path, sampling_ratio, even_sampling))
                        sample_1_of_n = int(math.ceil(even_sampling / sampling_ratio))
                        end_file = int(round(n_files * even_sampling))

                        if sample_from_end:
                            sub_src_files = src_files[slice(-1, -end_file)]
                        else:
                            sub_src_files = src_files[slice(0, end_file)]

                        sampled_files = sub_src_files[::sample_1_of_n]

                        more_samples_needed = n_samples - len(sampled_files)
                        if more_samples_needed > 0:
                            unsampled_files = [k for k in sub_src_files if k not in sampled_files]
                            sampled_files += unsampled_files[:more_samples_needed]
                    else:
                        if sample_from_end:
                            sampled_files = src_files[-n_samples:]
                        else:
                            sampled_files = src_files[:n_samples]

                if inverted_sampling:
                    sampled_files = [k for k in src_files if k not in sampled_files]
            else:
                sampled_files = src_files

        valid_seq_to_samples[seq_path] = sampled_files
        actual_samples = len(sampled_files)
        class_to_n_samples[seq_to_class[seq_path]] += actual_samples
        total_samples += actual_samples
        csv_path = csv_paths[idx]

        all_sampled_files += sampled_files

        if not sampled_files:
            msg = 'No sampled files found for {} with {} source files'.format(seq_path, n_files)
            if allow_seq_skipping:
                print('\n{}\n'.format(msg))
                n_unsampled_seq += 1
            else:
                raise IOError(msg)
        else:
            print('Processing sequence {}/{}: reading {}({})/{} images from {} and csv from {}'
                  ' (total images: {}/{})'.format(
                idx + 1, n_seq, actual_samples, n_samples, n_files, seq_path, csv_path, total_samples, total_files))
            n_sampled_seq += 1

        if only_sampling:
            continue

        df = pd.read_csv(csv_path)

        seq_df_data = []

        for file_path in sampled_files:
            filename = os.path.basename(file_path)

            if not os.path.isfile(file_path):
                raise IOError('Image file not found: {}'.format(file_path))

            try:
                df_multiple_instance = df.loc[df['filename'] == filename]
            except KeyError:
                if allow_missing_annotations:
                    print('\nNo annotations found for {}\n'.format(file_path))
                    continue
                else:
                    raise IOError('No annotations found for {}'.format(file_path))
            try:
                _ = df_multiple_instance.iloc[0].loc['filename']
            except IndexError:
                if allow_missing_annotations:
                    print('\nNo annotations found for {}\n'.format(file_path))
                    continue
                else:
                    raise IOError('No annotations found for {}'.format(file_path))

            num_instances = len(df_multiple_instance.index)

            df = df.drop(df_multiple_instance.index[:num_instances])
            seq_df_data.append((df_multiple_instance, file_path))

        seq_to_df_data[seq_path] = seq_df_data

    seq_to_samples = valid_seq_to_samples

    print('')
    print('Total files: {}'.format(total_files))
    print('Total samples: {}'.format(total_samples))
    print('class_to_n_samples:')

    print('n_sampled_seq: {}'.format(n_sampled_seq))
    print('n_unsampled_seq: {}'.format(n_unsampled_seq))
    print('n_seq: {}'.format(len(seq_paths)))

    pprint(class_to_n_samples)

    out_dir = os.path.dirname(output_path)
    out_name = os.path.splitext(os.path.basename(output_path))[0]
    log_dir = os.path.join(out_dir, out_name)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    print('\nWriting sampling log to: {}\n'.format(log_dir))
    with open(os.path.join(log_dir, 'seq_to_src_files.txt'), 'w') as logFile:
        pprint(seq_to_src_files, logFile)
    with open(os.path.join(log_dir, 'seq_to_samples.txt'), 'w') as logFile:
        pprint(seq_to_samples, logFile)
    with open(os.path.join(log_dir, 'class_to_n_samples.txt'), 'w') as logFile:
        pprint(class_to_n_samples, logFile)
    with open(os.path.join(log_dir, 'class_to_n_files.txt'), 'w') as logFile:
        pprint(class_to_n_files, logFile)
    with open(os.path.join(log_dir, 'class_to_n_samples.txt'), 'w') as logFile:
        pprint(class_to_n_samples, logFile)
    with open(os.path.join(log_dir, 'all_sampled_files.txt'), 'w') as logFile:
        logFile.write('\n'.join(all_sampled_files))
        # pprint(class_to_n_samples, logFile)

    if only_sampling:
        return

    if shuffle_files:
        print('Shuffling data...')
        shuffle(seq_df_data)

    _n_frames = len(seq_df_data)
    print('Total frames: {}'.format(_n_frames))

    print('class_names: ', class_names)
    print('class_dict: ', class_dict)

    if n_frames > 0 and n_frames < _n_frames:
        _n_frames = n_frames

    if write_annotations_list:
        annotations_list_fid = open(annotations_list_path, 'w')
        print('Writing annotations_list to {} in format {}'.format(annotations_list_path, write_annotations_list))

    if check_images:
        print('Image checking is enabled')

    file_path_id = 0
    for frame_id in tqdm(range(_n_frames)):
        df_multiple_instance, file_path = seq_df_data[frame_id]
        tf_example, bboxes, img_shape = csv_to_json(df_multiple_instance, file_path, class_dict, min_size,
                                                    check_images=check_images)
        if write_annotations_list and bboxes:
            img_h, img_w = img_shape
            file_path_id += 1
            for bbox in bboxes:
                obj_info = [file_path_id, file_path]
                class_id, xmin, ymin, xmax, ymax = bbox
                if write_annotations_list == 1:
                    bbox_info = [class_id, xmin, ymin, xmax, ymax]
                else:
                    x_center, y_center, w, h = (xmin + xmax) / (2.0 * img_w), (
                            ymin + ymax) / (2.0 * img_h), (xmax - xmin) / img_w, (ymax - ymin) / img_h

                    x_center = min(max(0, x_center), 1)
                    y_center = min(max(0, y_center), 1)
                    w = min(max(0, w), 1)
                    h = min(max(0, h), 1)

                    bbox_info = [class_id, x_center, y_center, w, h]

                obj_info += bbox_info
                annotations_list_fid.write(ann_sep.join(map(str, obj_info)) + '\n')
    print()

    if write_annotations_list:
        annotations_list_fid.close()


if __name__ == '__main__':
    main()
