import os
import glob
import sys
import paramparse
import shutil

import cv2

from tqdm import tqdm
from lxml import etree
import imagesize
import numpy as np

from datetime import datetime

from eval_utils import mask_str_to_pts, mask_pts_to_str, linux_path, drawBox
from pascal_voc_io import PascalVocWriter


class Params:
    """
    Convert CVAT XML annotations to YoloV5. Note: To copy images to --output directory use --copy_images flag.

    :ivar input: list of one or more XML files or a single directory containing all of the XML files to be processed
    :type input: list[str]

    :ivar move_images: move images to --output directory
    :type move_images: int

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
        self.input = ['', ]
        self.root_dir = ''
        self.xml_name = 'annotations.xml'
        self.move_images = False
        self.sizes = []
        self.min_area = 0
        self.allow_missing_images = 0
        self.labels_name = ''
        self.write_img = 0
        self.vert_flip = 0
        self.horz_flip = 0
        self.rename_unused = 0
        self.rename_useless = 0
        self.allow_target_id = 1
        self.write_empty = 0
        self.allow_missing_ann = 0
        self.name_from_title = True
        self.img_dir_name = 'images'
        self.output = ''
        self.img_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']


def parse_image(tag):
    """
    Parse image XML tag and store all information in the python dict
    Note: Invalid bounding boxes (e.g. area == 0) will be removed.
    For attributes and documentations about CVAT image 1.1 see:
    https://openvinotoolkit.github.io/cvat/docs/manual/advanced/xml_format/
    :param tag: image tag in XML file
    :type tag: lxml.ElementTree
    :param absolute: use absolute image pixel coordinates as opposed to fractional coordinates
    :type absolute: bool
    :return XML contents in python dict {image_attributes, shapes: [{polygon_attributes}]}

                from https://github.com/newfarms/2d-rock-detection/blob/development/src/preprocessing/cvat_to_csv.py
    """
    image_tag = {'shapes': []}
    for attr, value in tag.items():  # iterate all the attributes in the tag
        image_tag[attr] = value

    for poly_tag in tag.iter('polygon'):
        polygon = {}

        # extract attributes from annotation
        # for example: attributes size and easy
        for attribute_tag in poly_tag.iter('attribute'):
            polygon[attribute_tag.attrib['name']] = attribute_tag.text

        for key, value in poly_tag.items():  # for example: label
            polygon[key] = value

        # calculate bounding boxes
        points_string = poly_tag.get("points")
        width = float(image_tag["width"])
        height = float(image_tag["height"])
        bbox, mask = get_bbox_from_poly_points_string(points_string)
        polygon["bbox"] = bbox
        polygon["mask"] = mask

        if bbox[0] == bbox[1] or bbox[2] == bbox[3]:  # remove bbox with 0 area
            continue

        image_tag['width'] = width
        image_tag['height'] = height

        image_tag['shapes'].append(polygon)

    return image_tag


def get_bbox_from_poly_points_string(points_string):
    """
    Parse the polygon points string and return a list of coordinates
    :param points_string: a string of polygon points in the format of "x1,y1;x2,y2;...;"
    :type points_string: string
    :param absolute: use absolute image pixel coordinates as opposed to fractional coordinates
    :type absolute: bool
    :param width: width of the image
    :type width: float
    :param height: height of the image
    :type height: float
    :return a list of bounding box coordinates in [x_min, x_max, y_min, y_max]

                from https://github.com/newfarms/2d-rock-detection/blob/development/src/preprocessing/cvat_to_csv.py
    """
    # parse points from string
    x_coordinates = []
    y_coordinates = []
    for point_string in points_string.split(";"):
        x_coordinate, y_coordinate = point_string.split(",")
        x_coordinate = float(x_coordinate)
        y_coordinate = float(y_coordinate)
        # if not absolute:
        #     x_coordinate /= width
        #     y_coordinate /= height

        x_coordinates.append(x_coordinate)
        y_coordinates.append(y_coordinate)

    # calculate two diagonal points
    x_min = min(x_coordinates)
    x_max = max(x_coordinates)
    y_min = min(y_coordinates)
    y_max = max(y_coordinates)

    bbox = [x_min, x_max, y_min, y_max]

    mask = [(x, y, 1) for x, y in zip(x_coordinates, y_coordinates)]

    return bbox, mask


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
    rename_unused = params.rename_unused
    rename_useless = params.rename_useless
    write_empty = params.write_empty
    allow_missing_ann = params.allow_missing_ann
    allow_target_id = params.allow_target_id
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

    xml_paths = params.input
    if len(xml_paths) == 1:
        xml_dir_path = xml_list_path = xml_paths[0]
        if params.root_dir:
            xml_dir_path = linux_path(params.root_dir, xml_dir_path)

        if os.path.isdir(xml_dir_path):
            print(f'looking for XML files in {xml_dir_path}')
            xml_paths = glob.glob(f'{xml_dir_path}/*.xml')
            xml_paths.sort()

            # xml_paths_gen = [[os.path.join(dirpath, f).replace(os.sep, '/') for f in filenames if
            #                   any(f.lower().endswith(ext) for ext in ['.xml'])]
            #                  for (dirpath, dirnames, filenames) in os.walk(xml_path, followlinks=True)]
            # xml_paths = [item for sublist in xml_paths_gen for item in sublist]
        elif xml_list_path.endswith('.txt'):
            print(f'reading XML file list from {xml_list_path}')
            xml_paths = open(xml_list_path, "r").readlines()
            xml_paths = [linux_path(k.strip(), params.xml_name) for k in xml_paths if k.strip()]
        else:
            print(f'singular xml_path is neither a directory nor a list text file: {xml_list_path}')

    if params.root_dir:
        xml_paths = [linux_path(params.root_dir, k) for k in xml_paths]

    n_xml_paths = len(xml_paths)

    xml_paths_str = '\n'.join(xml_paths)
    print(f'found {n_xml_paths} xml_paths:\n{xml_paths_str}')

    if params.name_from_title:
        print('getting names from titles')

    n_total_images = 0
    n_skipped_images = 0
    n_skipped_boxes = 0
    n_total_boxes = 0

    missing_images = []
    skipped_images = []
    used_img_paths = []

    for xml_id, xml_path in enumerate(xml_paths):

        if not os.path.exists(xml_path):
            msg = f'xml path does not exist: {xml_path}'

            if allow_missing_ann and write_empty:
                print(msg)
                image_tags = []
            else:
                raise AssertionError(msg)
        else:
            # load and parse the XML file by lxml library
            xml_tree = etree.parse(xml_path)
            xml_root = xml_tree.getroot()
            image_tags = xml_root.xpath(".//image")  # get all the image tags

        xml_parent_path = os.path.dirname(xml_path)

        img_dir = linux_path(xml_parent_path, params.img_dir_name)

        if not os.path.isdir(img_dir):
            msg = f"img_dir does not exist: {img_dir}"
            if allow_missing_ann and write_empty:
                img_dir = xml_parent_path
                assert os.path.isdir(img_dir), f"img_dir does not exist: {img_dir}"
            else:
                raise AssertionError(msg)

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

        seq_name = os.path.basename(xml_parent_path)

        seq_root_path = linux_path(db_root_path, seq_name)
        seq_xml_path = linux_path(seq_root_path, labels_name)
        os.makedirs(seq_xml_path, exist_ok=True)

        if params.move_images:
            new_images_dir = linux_path(seq_root_path, "images")
            os.makedirs(new_images_dir, exist_ok=True)

        print(f'\n{xml_id + 1} / {n_xml_paths}: {xml_path} --> {seq_xml_path}')

        pbar = tqdm(image_tags)

        n_missing_images = 0

        # print('params.move_images: {}'.format(params.move_images))

        flip_img = params.vert_flip or params.horz_flip

        for image_tag in pbar:
            image_dict = parse_image(image_tag)

            image_title_path = image_dict["name"]

            image_title = os.path.splitext(os.path.basename(image_title_path))[0]
            try:
                old_img_path = img_title_to_path[image_title]
            except KeyError:
                msg = 'image not found: {}'.format(image_title)
                missing_images.append(image_title)
                if params.allow_missing_images:
                    # print(msg)
                    n_missing_images += 1
                    continue
                else:
                    raise AssertionError(msg)

            n_total_images += 1

            img_path = old_img_path

            used_img_paths.append(old_img_path)

            n_boxes = len(image_dict['shapes'])

            if n_boxes == 0:
                print(f'no annotated boxes in xml for image: {old_img_path}')
                if rename_useless:
                    dst_path = old_img_path + '.useless'
                    shutil.move(old_img_path, dst_path)

                if not write_empty:
                    continue

            if params.move_images:
                # old_img_path = in_path.parent / "images" / Path(image_title_path)
                new_img_path = linux_path(new_images_dir, os.path.basename(old_img_path))
                # print('{} --> {}'.format(old_img_path, new_img_path))

                shutil.move(old_img_path, new_img_path)

                img_path = new_img_path

            img_w = int(image_dict["width"])
            img_h = int(image_dict["height"])

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

            valid_bboxes = []

            image_shape = [img_h, img_w, 3]

            xml_writer = PascalVocWriter(foldername=seq_xml_path,
                                         filename=img_name,
                                         imgSize=image_shape,
                                         localImgPath=img_path_rel_db,
                                         )

            if flip_img and params.write_img:
                src_img = cv2.imread(img_path)
                if params.vert_flip:
                    src_img = np.flipud(src_img)
                if params.horz_flip:
                    src_img = np.fliplr(src_img)

                cv2.imwrite(img_path, src_img)

            for shape in image_dict['shapes']:
                n_total_boxes += 1

                skipped_box_percent = n_skipped_boxes / n_total_boxes * 100
                skipped_image_percent = n_skipped_images / n_total_images * 100

                pbar.set_description('{} :: skipped boxes: {} / {} ({:.2f}%) '
                                     'skipped images: {} / {} ({:.2f}%) '
                                     'missing_images: {}'.format(
                    seq_name,
                    n_skipped_boxes, n_total_boxes, skipped_box_percent,
                    n_skipped_images, n_total_images, skipped_image_percent,
                    n_missing_images))

                bbox = shape["bbox"]
                mask = shape["mask"]
                label = shape["label"]
                bbox_source = shape["source"]

                try:
                    obj_id = shape["id"]
                except KeyError:
                    obj_id = 0
                else:
                    if not allow_target_id:
                        raise AssertionError('target id found')

                x_min, x_max, y_min, y_max = bbox

                if flip_img:
                    if params.vert_flip:
                        y_min = img_h - y_min - 1
                        y_max = img_h - y_max - 1

                        if y_min > y_max:
                            y_min, y_max = y_max, y_min

                        assert y_min < y_max, "invalid y_min, y_max"

                        mask = [(x, img_h - y - 1, f) for x, y, f in mask]

                    if params.horz_flip:
                        x_min = img_w - x_min - 1
                        x_max = img_w - x_max - 1

                        if x_min > x_max:
                            x_min, x_max = x_max, x_min

                        assert x_min < x_max, "invalid x_min, x_max"

                        mask = [(img_w - x - 1, y, f) for x, y, f in mask]

                    # mask_pts =[(x, y) for x, y, f in mask]
                    # drawBox(src_img, x_min, y_min, x_max, y_max, mask=mask_pts)

                cx = (x_min + x_max) / 2.0
                cy = (y_min + y_max) / 2.0
                w = (x_max - x_min)
                h = (y_max - y_min)

                area = int(round(w * img_w * h * img_h))

                if sizes:
                    try:
                        size = shape["size"]
                    except KeyError:
                        raise AssertionError('size attribute missing for object')
                    else:
                        size_to_area_file.write('{:s}\t{:s}\t{:s}\t{:d}\n'.format(
                            seq_name, img_path_rel_noext, size, area))
                        if size not in sizes:
                            n_skipped_boxes += 1
                            continue

                if (params.min_area > 0) and (area < params.min_area):
                    n_skipped_boxes += 1
                    continue

                xml_dict = dict(
                    xmin=x_min,
                    ymin=y_min,
                    xmax=x_max,
                    ymax=y_max,
                    name=label,
                    difficult=False,
                    bbox_source=bbox_source,
                    id_number=obj_id,
                    score=1.0,
                    mask=mask,
                    mask_img=None
                )

                xml_writer.addBndBox(**xml_dict)

                valid_bboxes.append([cx, cy, w, h])

            if write_empty or valid_bboxes:

                img_xml_file_path = linux_path(seq_xml_path, img_path_rel_noext + ".xml")
                img_xml_file_dir = os.path.dirname(img_xml_file_path)
                os.makedirs(img_xml_file_dir, exist_ok=1)

                # if flip_img:
                #     cv2.imshow('src_img', src_img)
                #     cv2.waitKey(0)

                xml_writer.save(targetFile=img_xml_file_path, verbose=False)
            else:
                print(f"no valid boxes found out of total {n_boxes} boxes for {old_img_path}")
                n_skipped_images += 1
                skipped_images.append((old_img_path, img_name))

        # not in xml
        unused_img_paths = list(set(img_paths) - set(used_img_paths))

        if not unused_img_paths:
            continue

        n_unused_img_paths = len(unused_img_paths)
        # print('found {} unused images:\n{}'.format(n_unused_img_paths, '\n'.join(unused_img_paths)))
        if rename_unused:
            for unused_img_path in unused_img_paths:
                dst_path = unused_img_path + '.unused'
                shutil.move(unused_img_path, dst_path)

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
