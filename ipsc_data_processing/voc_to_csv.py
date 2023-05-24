import pandas as pd
import numpy as np
import os, sys, cv2, glob, re

import paramparse

from pascal_voc_io import PascalVocReader

from eval_utils import ImageSequenceWriter as ImageWriter
from eval_utils import sortKey, resize_ar, drawBox

params = {
    'file_name': '',
    'save_path': '',
    'save_file_name': '',
    'csv_file_name': '',
    'map_folder': '',
    'root_dir': 'N:\Datasets\VOC2012\VOCdevkit\VOC2012\JPEGImages',
    'list_file_name': '',
    'n_classes': 4,
    'img_ext': 'jpg',
    'only_person': 0,
    'show_img': 1,
    'min_vis': 0.5,
    'resize_factor': 1.0,
    'save_raw': 0,
    'n_frames': 0,
    'vis_width': 0,
    'vis_height': 0,
    'ext': 'mkv',
    'codec': 'H264',
    'fps': 30,
    'vis_root': '',
}

paramparse.process_dict(params)

file_name = params['file_name']
root_dir = params['root_dir']
list_file_name = params['list_file_name']
img_ext = params['img_ext']
only_person = params['only_person']
show_img = params['show_img']
resize_factor = params['resize_factor']
ext = params['ext']
codec = params['codec']
fps = params['fps']
_vis_width = params['vis_width']
_vis_height = params['vis_height']
save_path = params['save_path']
min_vis = params['min_vis']
save_raw = params['save_raw']
vis_root = params['vis_root']

image_exts = ['jpg', 'bmp', 'png', 'tif']
excluded_classes = ['bear', 'deer', 'coyote', 'moose']

if not vis_root:
    vis_root = 'vis' if not save_raw else 'raw'

pause_after_frame = 1

img_path = root_dir

src_path = img_path
src_files = [f for f in os.listdir(src_path) if
             os.path.isfile(os.path.join(src_path, f)) and f.endswith(img_ext)]
src_files.sort(key=sortKey)
n_frames = len(src_files)
print('n_frames: ', n_frames)

ann_path = os.path.join(root_dir, 'annotations')
# seq_name = os.path.basename(img_path)
ann_files = glob.glob(os.path.join(ann_path, '*.xml'))
n_ann_frames = len(ann_files)
if n_ann_frames == 0:
    raise IOError('No annotation xml files found')

def getint(fn):
    basename = os.path.basename(fn)
    num = re.sub("\D", "", basename)
    try:
        return int(num)
    except:
        return 0

if n_frames != n_ann_frames:
    print('n_frames: ', n_frames)
    print('n_ann_frames: ', n_ann_frames)
    raise IOError('Mismatch detected')

if _vis_height <= 0 or _vis_width <= 0:
    temp_img = cv2.imread(os.path.join(img_path, src_files[0]))
    vis_height, vis_width, _ = temp_img.shape
else:
    vis_height, vis_width = _vis_height, _vis_width


if not save_path:
    save_path = os.path.join(os.path.dirname(img_path), vis_root, os.path.basename(img_path) + '.' + ext)
elif not save_path.endswith(ext):
    save_path += '.{}'.format(ext)

save_dir = os.path.dirname(save_path)
if save_dir and not os.path.isdir(save_dir):
    os.makedirs(save_dir)

if ext in image_exts:
    video_out = ImageWriter(save_path)
    print('Saving {}x{} visualization sequence to {}'.format(
        vis_width, vis_height, save_path))
    csv_path = os.path.join(os.path.dirname(save_path),  os.path.splitext(os.path.basename(save_path))[0])
else:
    fourcc = cv2.VideoWriter_fourcc(*codec)
    video_out = cv2.VideoWriter(save_path, fourcc, fps, (vis_width, vis_height))
    if video_out is None:
        raise IOError('Output video file could not be opened: {}'.format(save_path))
    print('Saving {}x{} visualization video to {}'.format(
        vis_width, vis_height, save_path))
    csv_path = os.path.dirname(save_path)
csv_file = os.path.join(csv_path, 'annotations.csv')
print('saving csv file to {}'.format(csv_file))


print('Loading annotations from {:d} files at {:s}...'.format(n_ann_frames, ann_path))

file_id = 0
n_boxes = 0
csv_raw = []

enable_vis = show_img or not save_raw

out_frame_id = 0
for frame_id in range(n_frames):

    ann_file_path = os.path.join(ann_path, os.path.splitext(src_files[frame_id])[0] + '.xml')

    if not os.path.isfile(ann_file_path):
        raise IOError('xml file {} does not exist'.format(ann_file_path))

    xml_reader = PascalVocReader(ann_file_path)
    shapes = xml_reader.getShapes()
    curr_csv_raw = []

    filename = xml_reader.filename
    width = xml_reader.width
    height = xml_reader.height


    for shape in shapes:
        label, points, _, _, difficult, bbox_source, id_number, score = shape

        if id_number is None:
            id_number = -1

        _xmin, _ymin = points[0]
        _xmax, _ymax = points[2]

        # Bounding box sanity check
        if _xmin > _xmax or _ymin > _ymax:
            print('Invalid box {}\n\n'.format([xmin, ymin, xmax, ymax]))

        xmin = min(_xmin,_xmax)
        xmax = max(_xmin,_xmax)

        ymin = min(_ymin,_ymax)
        ymax = max(_ymin,_ymax)

        out_filename = 'image{:06d}.jpg'.format(out_frame_id + 1)

        raw_data = {
            'target_id': int(id_number),
            'filename': out_filename,
            'width': width,
            'height': height,
            'class': label,
            'xmin': int(xmin),
            'ymin': int(ymin),
            'xmax': int(xmax),
            'ymax': int(ymax)
        }
        curr_csv_raw.append(raw_data)
        n_boxes += 1

    if only_person:
        if np.all([obj['class'] != 'person' for obj in curr_csv_raw]):
            continue

        if np.any([obj['class'] in excluded_classes for obj in curr_csv_raw]):
            continue

        if np.count_nonzero([obj['class'] == 'person' for obj in curr_csv_raw]) < only_person:
            continue

        curr_csv_raw = [obj for obj in curr_csv_raw if obj['class'] == 'person']

    src_file_path = os.path.join(src_path, src_files[frame_id])
    if not os.path.exists(src_file_path):
        raise SystemError('Image file {} does not exist'.format(src_file_path))

    image = cv2.imread(src_file_path)
    if save_raw:
        video_out.write(image)

    _height, _width = image.shape[:2]

    if width != _width:
        print('width: ', width)
        print('_width: ', _width)
        raise IOError('Mismatch detected')

    if height != _height:
        print('height: ', height)
        print('_height: ', _height)
        raise IOError('Mismatch detected')

    csv_raw += curr_csv_raw

    out_frame_id += 1
    file_id += 1

    if enable_vis:
        for _data in curr_csv_raw:
            xmin = _data['xmin']
            ymin = _data['ymin']
            xmax = _data['xmax']
            ymax = _data['ymax']
            label = _data['class']

            box_color = (0, 255, 0)
            drawBox(image, xmin, ymin, xmax, ymax, box_color, label)

        image = resize_ar(image, vis_width, vis_height)

        if not save_raw:
            video_out.write(image)

        if show_img:
            cv2.imshow('VOC12', image)
            k = cv2.waitKey(1 - pause_after_frame) & 0xFF
            if k == ord('q') or k == 27:
                break
            elif k == 32:
                pause_after_frame = 1 - pause_after_frame
    sys.stdout.write('\rDone {:d}/{:d} files with {:d} boxes'.format(
        file_id, n_ann_frames, n_boxes))
    sys.stdout.flush()

sys.stdout.write('\n')
sys.stdout.flush()

if show_img:
    cv2.destroyWindow('VOC12')

df = pd.DataFrame(csv_raw)
df.to_csv(csv_file)
video_out.release()

print('out_n_frames: ', out_frame_id)
