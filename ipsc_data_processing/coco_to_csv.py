import pandas as pd
import numpy as np
import os, sys, cv2

import paramparse

from eval_utils import ImageSequenceWriter, resize_ar, drawBox

from cocoapi.PythonAPI.pycocotools.coco import COCO

params = {
    'data_type': 'train2017',
    'save_path': '',
    'save_file_name': '',
    'csv_file_name': '',
    'map_folder': '',
    'root_dir': 'N:\Datasets\COCO17',
    'list_file_name': '',
    'n_classes': 4,
    'img_ext': 'jpg',
    'only_person': 0,
    'write_img': 1,
    'show_img': 1,
    'min_vis': 0.5,
    'resize_factor': 1.0,
    'save_raw': 0,
    'start_id': 0,
    'n_frames': 0,
    'frame_gap': 1,
    'vis_width': 1280,
    'vis_height': 960,
    'ext': 'mkv',
    'codec': 'H264',
    'fps': 30,
    'vis_root': '',
}

paramparse.process_dict(params)

root_dir = params['root_dir']
data_type = params['data_type']
list_file_name = params['list_file_name']
img_ext = params['img_ext']
only_person = params['only_person']
write_img = params['write_img']
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
start_id = params['start_id']
n_frames = params['n_frames']
frame_gap = params['frame_gap']

image_exts = ['jpg', 'bmp', 'png', 'tif']
excluded_classes = ['bear', 'deer', 'coyote', 'moose']

if not vis_root:
    vis_root = 'vis' if not save_raw else 'raw'

pause_after_frame = 1

img_path = os.path.join(root_dir, data_type)

# src_path = img_path
# src_files = [f for f in os.listdir(src_path) if
#              os.path.isfile(os.path.join(src_path, f)) and f.endswith(img_ext)]
# src_files.sort(key=sortKey)
# n_frames = len(src_files)
# print('n_frames: ', n_frames)

ann_path = os.path.join(root_dir, 'annotations', 'instances_{}.json'.format(data_type))
coco = COCO(ann_path)
cats = coco.loadCats(coco.getCatIds())
cat_names = [cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(cat_names)))

sup_cat = set([cat['supercategory'] for cat in cats])
print('COCO super categories: \n{}'.format(' '.join(sup_cat)))

catIds = coco.getCatIds(catNms=['person'])
imgIds = coco.getImgIds(catIds=catIds)

n_ann_frames = len(imgIds)
print('n_ann_frames: ', n_ann_frames)

vis_height, vis_width = _vis_height, _vis_width

if not save_path:
    save_path = os.path.join(os.path.dirname(img_path), vis_root, os.path.basename(img_path) + '.' + ext)
elif not save_path.endswith(ext):
    save_path += '.{}'.format(ext)

save_dir = os.path.dirname(save_path)
if save_dir and not os.path.isdir(save_dir):
    os.makedirs(save_dir)


video_out = None
if ext in image_exts:
    if write_img:
        video_out = ImageSequenceWriter(save_path)
        print('Saving {}x{} visualization sequence to {}'.format(
            vis_width, vis_height, save_path))
    csv_path = os.path.join(os.path.dirname(save_path), os.path.splitext(os.path.basename(save_path))[0])
else:
    if write_img:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        video_out = cv2.VideoWriter(save_path, fourcc, fps, (vis_width, vis_height))
        if video_out is None:
            raise IOError('Output video file could not be opened: {}'.format(save_path))
        print('Saving {}x{} visualization video to {}'.format(
            vis_width, vis_height, save_path))
    csv_path = os.path.dirname(save_path)
csv_file = os.path.join(csv_path, 'annotations.csv')
print('saving csv file to {}'.format(csv_file))

file_id = 0
n_boxes = 0
csv_raw = []

_ids = range(start_id, n_ann_frames, frame_gap)
n_out_frames = len(_ids)

if n_frames > 0 and n_frames < n_out_frames:
    _ids = _ids[:n_frames]
    n_out_frames = len(_ids)

print('n_out_frames: ', n_out_frames)

enable_vis = show_img or not save_raw

out_frame_id = 0
for _id in _ids:

    img_id = imgIds[_id]

    img = coco.loadImgs(img_id)[0]
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)

    curr_csv_raw = []
    filename = img['file_name']

    src_file_path = os.path.join(root_dir, data_type, filename)

    if not os.path.exists(src_file_path):
        raise SystemError('Image file {} does not exist'.format(src_file_path))
    image = cv2.imread(src_file_path)

    height, width = image.shape[:2]

    for ann in anns:
        bbox = ann['bbox']
        category_id = ann['category_id']

        label = cat_names[category_id - 1]

        bbox = [int(x) for x in bbox]
        x, y, w, h = bbox

        xmin = x
        ymin = y

        xmax = xmin + w
        ymax = ymin + h

        out_filename = 'image{:06d}.jpg'.format(out_frame_id + 1)

        raw_data = {
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

    if write_img and save_raw:
        video_out.write(image)

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
        if write_img and not save_raw:
            video_out.write(image)

        if show_img:
            cv2.imshow('coco17', image)
            k = cv2.waitKey(1 - pause_after_frame) & 0xFF
            if k == ord('q') or k == 27:
                break
            elif k == 32:
                pause_after_frame = 1 - pause_after_frame
    sys.stdout.write('\rDone {:d}/{:d} files with {:d} boxes'.format(
        file_id, n_out_frames, n_boxes))
    sys.stdout.flush()


sys.stdout.write('\n')
sys.stdout.flush()

if show_img:
    cv2.destroyWindow('coco17')
if write_img:
    video_out.release()

df = pd.DataFrame(csv_raw)
df.to_csv(csv_file)

print('out_n_frames: ', out_frame_id)
