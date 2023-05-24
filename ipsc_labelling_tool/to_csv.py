import os, sys, glob, re
import pandas as pd
import ast
from pprint import pformat

from libs.pascal_voc_io import PascalVocReader

sys.path.append('..')
from tf_api.utilities import processArguments, sortKey


def saveBoxesCSV(seq_path, voc_path, sources_to_include, enable_mask, samples, img_ext='jpg', verbose=True):
    if not voc_path or not os.path.isdir(voc_path):
        raise IOError('Folder containing the loaded boxes does not exist: {}'.format(
            voc_path
        ))
        # return None

    # src_files = [os.path.join(seq_path, k) for k in os.listdir(seq_path) if
    #              os.path.splitext(k.lower())[1][1:] == img_ext]
    # src_files.sort(key=sortKey)

    # seq_name = os.path.basename(img_path)
    files = glob.glob(os.path.join(voc_path, '*.xml'))
    n_files = len(files)
    if n_files == 0:
        print('No loaded boxes found')
        return None

    def getint(fn):
        basename = os.path.basename(fn)
        num = re.sub("\D", "", basename)
        try:
            return int(num)
        except:
            return 0

    if len(files) > 0:
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

    print('Loading annotations from {:d} files at {:s}...'.format(n_files, voc_path))

    file_id = 0
    n_boxes = 0
    csv_raw = []

    sources_to_exclude = [k[1:] for k in sources_to_include if k.startswith('!')]
    sources_to_include = [k for k in sources_to_include if not k.startswith('!')]

    if sources_to_include:
        print('Including only boxes from following sources: {}'.format(sources_to_include))
    if sources_to_exclude:
        print('Excluding boxes from following sources: {}'.format(sources_to_exclude))

    for file in files:
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
                    mask_txt = '{},{};'.format(*(mask_pts[0][:2]))
                    for _pt in mask_pts[1:]:
                        mask_txt = '{}{},{};'.format(mask_txt, _pt[0], _pt[1])
                raw_data.update({'mask': mask_txt})
            csv_raw.append(raw_data)
            n_boxes += 1

        file_id += 1
        if verbose:
            sys.stdout.write('\rDone {:d}/{:d} files with {:d} boxes'.format(
                file_id, n_files, n_boxes))
            sys.stdout.flush()

    if verbose:
        sys.stdout.write('\n')
        sys.stdout.flush()

    df = pd.DataFrame(csv_raw)
    out_dir = os.path.dirname(voc_path)
    out_file_path = os.path.join(out_dir, 'annotations.csv')

    csv_columns = ['target_id', 'filename', 'width', 'height',
               'class', 'xmin', 'ymin', 'xmax', 'ymax']
    if enable_mask:
        csv_columns.append('mask')

    df.to_csv(out_file_path, columns=csv_columns)

    return out_file_path


def main():
    params = {
        'seq_paths': '',
        'root_dir': '',
        'class_names_path': '../labelling_tool/data//predefined_classes_orig.txt',
        'save_file_name': '',
        'csv_file_name': '',
        'map_folder': '',
        'load_path': '',
        'n_classes': 4,
        'sources_to_include': [],
        'img_ext': 'png',
        'batch_size': 1,
        'show_img': 0,
        'save_video': 1,
        'enable_mask': 0,
        'n_frames': 0,
        'codec': 'H264',
        'fps': 20,
        'load_samples_root': '',
        'load_samples': [],
    }
    processArguments(sys.argv[1:], params)
    seq_paths = params['seq_paths']
    root_dir = params['root_dir']
    sources_to_include = params['sources_to_include']
    enable_mask = params['enable_mask']
    load_samples = params['load_samples']
    load_samples_root = params['load_samples_root']

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

    for seq_path in seq_paths:
        seq_path = seq_path.replace('\\', '/')
        if seq_to_samples:
            samples = seq_to_samples[seq_path]
        else:
            samples = []
        voc_path = os.path.join(seq_path, 'annotations')
        saveBoxesCSV(seq_path, voc_path, sources_to_include, enable_mask, samples)


if __name__ == '__main__':
    main()
