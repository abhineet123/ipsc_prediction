import os

from tqdm import tqdm
import imagesize
import pandas as pd

import paramparse

from datetime import datetime

from pascal_voc_io import PascalVocWriter
from eval_utils import linux_path


class Params:
    """
    Convert CSV annotations to VOC XMZ

    :ivar csv_paths: list of one or more CSV files or a single directory containing all of the XML files to be processed
    :type csv_paths: list[str]

    :ivar sizes: specify which values for the size attribute to allow in the rocks included in the output
    labels;
    all of the rocks would be excluded from the labels;should be specified as string with all size values to
    be included separated by commas;
    valid size values: small, large and huge;
    enabling this also writes a text file with size and area of all annotated objects;
    :type sizes: list

    :ivar min_area: minimum bounding box area in pixels for the corresponding rock to be included
    in the output labels
    :type min_area: int

    :ivar labels_name: name of the output labels directory;
    defaults to "labels" with an optional suffix if "filter_by_size"  and/or min_area are  provided;
    e.g. labels_name = labels_large_huge if filter_by_size=large,huge;
    similarly if min_area=200, labels_name = labels_min_200;
    they can also be combined, e.g. if filter_by_size=large and min_area=500, labels_name = labels_large_min_200;
    :type labels_name: str

    :ivar output: directory in which to save yolov5 output
    :type output: str

    """

    def __init__(self):
        self.cfg = ()
        self.csv_paths = ['', ]
        self.root_dir = ''
        self.csv_name = 'annotations.csv'
        self.sizes = []
        self.min_area = 0
        self.allow_missing_images = 0
        self.labels_name = ''
        self.rename_unused = 0
        self.rename_useless = 0
        self.allow_target_id = 1
        self.write_empty = 0
        self.name_from_title = False
        self.img_dir_name = 'images'
        self.output = ''
        self.img_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']


def to_epoch(dst_fname_noext, fmt='%Y_%m_%d-%H_%M_%S-%f', is_path=True):
    if is_path:
        dst_fname_noext = os.path.splitext(os.path.basename(dst_fname_noext))[0]

    timestamp_ms, _id = dst_fname_noext.split('__')
    timestamp_str = timestamp_ms + '000'
    timestamp = datetime.strptime(timestamp_str, fmt)
    epoch_sec = datetime.timestamp(timestamp)
    epoch = str(int(float(epoch_sec) * 1000))

    return epoch, _id


def main():
    params = Params()
    paramparse.process(params)

    db_root_path = params.output

    if not db_root_path:
        db_root_path = params.root_dir

    labels_name = params.labels_name
    write_empty = params.write_empty
    sizes = params.sizes
    size_to_area_file = None
    if sizes:
        print(f'including only rocks with sizes: {sizes}')

        size_to_area_path = linux_path(db_root_path, "size_to_area.txt")
        size_to_area_file = open(size_to_area_path, "w")

        print('writing size to area data to: {}'.format(str(size_to_area_path)))

    if not labels_name:
        labels_name = 'annotations'
        if sizes:
            labels_name += '_' + '_'.join(sizes)

        if params.min_area > 0:
            labels_name += '_min_{:d}'.format(params.min_area)

    csv_paths = params.csv_paths
    if len(csv_paths) == 1:
        csv_dir_path = csv_list_path = csv_paths[0]
        if params.root_dir:
            csv_dir_path = linux_path(params.root_dir, csv_dir_path)

        if os.path.isdir(csv_dir_path):
            print(f'looking for CSV files in {csv_dir_path}')
            # csv_paths = glob.glob(f'{csv_dir_path}/*.csv')
            csv_paths_gen = [[os.path.join(dirpath, f).replace(os.sep, '/') for f in filenames if
                              f.lower().endswith('.csv')]
                             for (dirpath, dirnames, filenames) in os.walk(csv_dir_path, followlinks=True)]
            csv_paths = [item for sublist in csv_paths_gen for item in sublist]
            csv_paths.sort()
        elif csv_list_path.endswith('.txt'):
            print(f'reading XML file list from {csv_list_path}')
            csv_paths = open(csv_list_path, "r").readlines()
            csv_paths = [linux_path(k.strip(), params.csv_name) for k in csv_paths if k.strip()]
        else:
            print(f'singular csv_path is neither a directory nor a list text file: {csv_list_path}')

    if params.root_dir:
        csv_paths = [linux_path(params.root_dir, k) for k in csv_paths]

    n_csv_paths = len(csv_paths)

    csv_paths_str = '\n'.join(csv_paths)
    print(f'found {n_csv_paths} csv_paths:\n{csv_paths_str}')

    if params.name_from_title:
        print('getting names from titles')

    n_total_images = 0
    n_skipped_images = 0
    n_skipped_boxes = 0
    n_total_boxes = 0

    missing_images = []
    skipped_images = []
    used_img_paths = []
    for csv_id, csv_path in enumerate(csv_paths):

        assert os.path.exists(csv_path), f'csv path does not exist: {csv_path}'

        csv_parent_path = os.path.dirname(csv_path)

        img_dir = linux_path(csv_parent_path, params.img_dir_name)

        assert os.path.isdir(img_dir), f"img_dir does not exist: {img_dir}"

        img_file_gen = [[os.path.join(dirpath, f) for f in filenames if
                         os.path.splitext(f.lower())[1] in params.img_exts]
                        for (dirpath, dirnames, filenames) in os.walk(img_dir, followlinks=True)]
        img_paths = [item for sublist in img_file_gen for item in sublist]

        n_img_paths = len(img_paths)
        print(f'found {n_img_paths} images in {img_dir}')

        img_title_to_path = {}
        for old_img_path in img_paths:
            if params.name_from_title:
                img_title, _ = to_epoch(old_img_path, is_path=True)
                # img_title = get_title_from_exif(old_img_path)
            else:
                img_title = os.path.basename(old_img_path)

            img_title_to_path[img_title] = old_img_path

        seq_name = os.path.basename(csv_parent_path)

        seq_xml_path = linux_path(csv_parent_path, labels_name)
        os.makedirs(seq_xml_path, exist_ok=True)

        print(f'\n{csv_id + 1} / {n_csv_paths}: {csv_path} --> {seq_xml_path}')

        n_missing_images = 0

        df = pd.read_csv(csv_path)
        n_predictions = len(df)

        df['ImageID'] = df['ImageID'].astype(str)

        grouped_predictions = df.groupby("ImageID")
        n_grouped_predictions = len(grouped_predictions.groups)

        print(f'{csv_path} --> {n_predictions} labels for {n_grouped_predictions} images')

        pbar = tqdm(grouped_predictions.groups.items(), total=len(grouped_predictions.groups))

        for img_id, row_ids in pbar:

            n_total_images += 1

            img_id = os.path.basename(img_id)

            img_df = df.loc[row_ids]
            n_boxes = len(img_df)

            img_path = img_title_to_path[img_id]

            used_img_paths.append(img_path)

            img_w, img_h = imagesize.get(img_path)

            img_name = os.path.basename(img_path)

            # relative to the folder containing images in this sequence
            img_path_rel = os.path.relpath(img_path, img_dir).rstrip(
                '.' + os.sep).replace(os.sep, '/')
            """path of the output XML relative to the folder containing all the XMLs for the sequence 
            should be the same as the path of the image relative to the folder containing all the 
            images for the sequence so that any sorting present in the latter is retained in the former"""
            img_path_rel_noext = os.path.splitext(img_path_rel)[0]

            """img_path_rel_db should be relative to the json file for swin detection so assume that 
            db_root_path is same as the path where json would be created, i.e. the root folder 
            containing all sequences"""
            img_path_rel_db = os.path.relpath(img_path, db_root_path).rstrip('.' + os.sep).replace(os.sep, '/')

            n_valid_bboxes = 0

            image_shape = [img_h, img_w, 3]

            xml_writer = PascalVocWriter(foldername=seq_xml_path,
                                         filename=img_name,
                                         imgSize=image_shape,
                                         localImgPath=img_path_rel_db,
                                         )

            for _, row in img_df.iterrows():
                n_total_boxes += 1

                skipped_box_pc = n_skipped_boxes / n_total_boxes * 100
                skipped_image_pc = n_skipped_images / n_total_images * 100

                pbar.set_description(
                    f'{seq_name} :: '
                    f'skipped boxes: {n_skipped_boxes} / {n_total_boxes} ({skipped_box_pc:.2f}%) '
                    f'skipped images: {n_skipped_images} / {n_total_images} ({skipped_image_pc:.2f}%) '
                    f'missing_images: {n_missing_images}')

                size = row["Size"]

                if params.sizes and size not in params.sizes:
                    n_skipped_boxes += 1
                    continue

                try:
                    is_syn = row["Synthetic"]
                except KeyError:
                    is_syn = 0

                if is_syn:
                    label = 'syn'
                else:
                    label = 'rock'

                x_max = float(row["XMax"])
                y_max = float(row["YMax"])
                x_min = float(row["XMin"])
                y_min = float(row["YMin"])
                mask_str = str(row["MaskXY"]).strip('"')
                x_coordinates = []
                y_coordinates = []
                for point_string in mask_str.split(";"):
                    if not point_string:
                        continue
                    x_coordinate, y_coordinate = point_string.split(",")
                    x_coordinate = float(x_coordinate)
                    y_coordinate = float(y_coordinate)
                    # if not absolute:
                    #     x_coordinate /= width
                    #     y_coordinate /= height

                    x_coordinates.append(x_coordinate)
                    y_coordinates.append(y_coordinate)

                mask = [(x, y, 1) for x, y in zip(x_coordinates, y_coordinates)]

                bbox_width = x_max - x_min
                bbox_height = y_max - y_min

                bbox_area = round(bbox_width * bbox_height)

                if bbox_area < params.min_area > 0:
                    n_skipped_boxes += 1
                    continue

                xml_dict = dict(
                    xmin=x_min,
                    ymin=y_min,
                    xmax=x_max,
                    ymax=y_max,
                    name=label,
                    difficult=False,
                    bbox_source='ground_truth',
                    id_number=0,
                    score=1.0,
                    mask=mask,
                    mask_img=None
                )

                xml_writer.addBndBox(**xml_dict)

                n_valid_bboxes += 1

            if n_valid_bboxes > 0 or write_empty:

                img_xml_file_path = linux_path(seq_xml_path, img_path_rel_noext + ".xml")
                img_xml_file_dir = os.path.dirname(img_xml_file_path)
                os.makedirs(img_xml_file_dir, exist_ok=1)

                xml_writer.save(targetFile=img_xml_file_path, verbose=False)
            else:
                # print(f"no valid boxes found out of total {n_boxes} boxes for {old_img_path}")
                n_skipped_images += 1
                skipped_images.append((old_img_path, img_name))

        # not in xml
        unused_img_paths = list(set(img_paths) - set(used_img_paths))

        if not unused_img_paths:
            continue

        n_unused_img_paths = len(unused_img_paths)

        if not write_empty:
            continue

        print(f'writing xml files for {n_unused_img_paths} unused images')
        for unused_img_path in tqdm(unused_img_paths):
            img_name = os.path.basename(unused_img_path)

            # relative to the folder containing images in this sequence
            img_path_rel = os.path.relpath(unused_img_path, img_dir).rstrip(
                '.' + os.sep).replace(os.sep, '/')

            # relative to the output root folder for json compatibility
            img_path_rel_db = os.path.relpath(unused_img_path, db_root_path).rstrip(
                '.' + os.sep).replace(os.sep, '/')

            img_w, img_h = imagesize.get(unused_img_path)

            image_shape = [img_h, img_w, 3]

            image_path_rel_noext = os.path.splitext(img_path_rel)[0]
            img_xml_file_path = linux_path(seq_xml_path, image_path_rel_noext + ".xml")
            img_xml_file_dir = os.path.dirname(img_xml_file_path)
            os.makedirs(img_xml_file_dir, exist_ok=1)

            xml_writer = PascalVocWriter(foldername=seq_xml_path,
                                         filename=img_name,
                                         imgSize=image_shape,
                                         localImgPath=img_path_rel_db,
                                         )
            xml_writer.save(targetFile=img_xml_file_path, verbose=False)

    if missing_images:
        with open('missing_images.txt', 'w') as fid:
            fid.write('\n'.join(missing_images))

    if skipped_images:
        with open('skipped_images.txt', 'w') as fid:
            fid.write('\n'.join('\t'.join(k) for k in skipped_images))

    if size_to_area_file is not None:
        size_to_area_file.close()


if __name__ == "__main__":
    main()
