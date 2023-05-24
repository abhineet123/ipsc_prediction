import os, glob, re
import pandas as pd
import ast
import paramparse
from pprint import pformat
from tqdm import tqdm

from pascal_voc_io import PascalVocReader
from eval_utils import sortKey, add_suffix, linux_path

class Params:
    def __init__(self):
        self.cfg = ()
        self.batch_size = 1
        self.codec = 'H264'
        self.csv_file_name = ''
        self.enable_mask = 0
        self.fps = 20
        self.recursive = 0
        self.img_ext = 'png'
        self.load_path = ''
        self.load_samples = []
        self.load_samples_root = ''
        self.map_folder = ''
        self.n_classes = 4
        self.n_frames = 0
        self.root_dir = ''
        self.save_file_name = ''
        self.save_video = 1
        self.seq_paths = ''
        self.show_img = 0
        self.sources_to_include = []
        self.xml_dir = 'annotations'
        self.xml_suffix = ''
        self.csv_name = 'annotations.csv'
        self.start_id = -1
        self.end_id = -1


def save_boxes_csv(seq_path, voc_path, sources_to_include, enable_mask, recursive,
                   start_id, end_id, csv_name,
                   samples, img_ext='jpg'):
    if not voc_path or not os.path.isdir(voc_path):
        raise IOError(f'Folder containing the xml files does not exist: {voc_path}')
        # return None

    # src_files = [os.path.join(seq_path, k) for k in os.listdir(seq_path) if
    #              os.path.splitext(k.lower())[1][1:] == img_ext]
    # src_files.sort(key=sortKey)

    # seq_name = os.path.basename(img_path)
    print(f'looking for xml files in {voc_path}')
    if recursive:
        print(f'searching recursively')
        files = glob.glob(os.path.join(voc_path, '**/*.xml'), recursive=True)
    else:
        files = glob.glob(os.path.join(voc_path, '*.xml'))
    n_files = len(files)
    if n_files == 0:
        raise AssertionError('No xml files found')

    def getint(fn):
        basename = os.path.basename(fn)
        num = re.sub("\D", "", basename)
        try:
            return int(num)
        except:
            return 0

    files = sorted(files, key=getint)

    if samples:
        n_samples = len(samples)

        print('Using {} samples'.format(n_samples))
        sampled_files_no_ext = [os.path.splitext(os.path.basename(sample))[0] for sample in samples]
        files = [k for k in files if os.path.splitext(os.path.basename(k))[0] in sampled_files_no_ext]

        n_files = len(files)

        if n_samples != n_files:
            raise IOError('Mismatch between n_samples: {} and n_files: {} in seq: {}'.format(
                n_samples, n_files, seq_path))

    if start_id <= 0:
        start_id = 0

    if end_id <= 0:
        end_id = n_files - 1

    assert start_id <= end_id, "end_id cannot be less than start_id"

    print(f'start_id: {start_id}')
    print(f'end_id: {end_id}')

    files = files[start_id:end_id + 1]
    n_files = len(files)

    print(f'Loading annotations from {n_files:d} files at {voc_path:s}...')

    n_boxes = 0
    csv_raw = []

    sources_to_exclude = [k[1:] for k in sources_to_include if k.startswith('!')]
    sources_to_include = [k for k in sources_to_include if not k.startswith('!')]

    if sources_to_include:
        print('Including only boxes from following sources: {}'.format(sources_to_include))
    if sources_to_exclude:
        print('Excluding boxes from following sources: {}'.format(sources_to_exclude))

    pbar = tqdm(files)
    for file in pbar:
        xml_reader = PascalVocReader(file)

        filename = os.path.splitext(os.path.basename(file))[0] + '.{}'.format(img_ext)
        filename_from_xml = xml_reader.filename

        # file_path = os.path.join(seq_path, filename)
        # if samples and file_path not in samples:
        #     print('Skipping {} not in samples'.format(file_path))
        #     continue

        # if filename != filename_from_xml:
        #     sys.stdout.write('\n\nImage file name from xml: {} does not match the expected one: {}'.format(
        #         filename_from_xml, filename))
        #     sys.stdout.flush()

        shapes = xml_reader.getShapes()
        for shape in shapes:
            label, points, _, _, difficult, bbox_source, id_number, score, mask_pts, _ = shape

            if sources_to_include and bbox_source not in sources_to_include:
                continue

            if sources_to_exclude and bbox_source in sources_to_exclude:
                continue

            if id_number is None:
                id_number = -1

            xmin, ymin = points[0]
            xmax, ymax = points[2]
            raw_data = {
                'target_id': int(id_number),
                'filename': filename,
                'width': xml_reader.width,
                'height': xml_reader.height,
                'class': label,
                'xmin': int(xmin),
                'ymin': int(ymin),
                'xmax': int(xmax),
                'ymax': int(ymax)
            }
            if enable_mask:
                mask_txt = ''
                if mask_pts is not None and mask_pts:
                    mask_txt = f'{mask_pts[0][0]},{mask_pts[0][1]};'
                    for _pt in mask_pts[1:]:
                        mask_txt = f'{mask_txt}{_pt[0]},{_pt[1]};'
                raw_data.update({'mask': mask_txt})
            csv_raw.append(raw_data)
            n_boxes += 1

        pbar.set_description(f'n_boxes: {n_boxes:d}')

    df = pd.DataFrame(csv_raw)
    out_dir = os.path.dirname(voc_path)
    out_file_path = os.path.join(out_dir, csv_name)

    csv_columns = ['target_id', 'filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    if enable_mask:
        csv_columns.append('mask')

    df.to_csv(out_file_path, columns=csv_columns, index=False)

    return out_file_path


def main():
    params = Params()
    paramparse.process(params)

    seq_paths = params.seq_paths
    root_dir = params.root_dir
    sources_to_include = params.sources_to_include
    enable_mask = params.enable_mask
    load_samples = params.load_samples
    load_samples_root = params.load_samples_root

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
        raise AssertionError('Either seq_paths or root_dir must be provided')

    seq_to_samples = {}

    if len(load_samples) == 1:
        if load_samples[0] == 1:
            load_samples = ['seq_to_samples.txt', ]
        elif load_samples[0] == 0:
            load_samples = []

    if load_samples:
        # if load_samples == '1':
        #     load_samples = 'seq_to_samples.txt'
        print('load_samples: {}'.format(pformat(load_samples)))
        if load_samples_root:
            load_samples = [os.path.join(load_samples_root, k) for k in load_samples]
        print('Loading samples from : {}'.format(load_samples))
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

    n_seq = len(seq_paths)

    if params.xml_suffix:
        params.xml_dir = f'{params.xml_dir}_{params.xml_suffix}'
        params.csv_name = add_suffix(params.csv_name, params.xml_suffix)

    for seq_id, seq_path in enumerate(seq_paths):
        seq_path = seq_path.replace('\\', '/')
        print(f'{seq_id + 1} / {n_seq}: {seq_path}')
        if seq_to_samples:
            samples = seq_to_samples[seq_path]
        else:
            samples = []
        voc_path = linux_path(seq_path, params.xml_dir)
        save_boxes_csv(seq_path, voc_path, sources_to_include, enable_mask,
                       params.recursive, params.start_id, params.end_id,
                       params.csv_name,
                       samples)


if __name__ == '__main__':
    main()
