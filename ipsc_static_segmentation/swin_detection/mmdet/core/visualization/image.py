import matplotlib.pyplot as plt
import os
import cv2
import PIL
import mmcv
import numpy as np
import pycocotools.mask as mask_util
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

from ..utils import mask2ndarray

from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import codecs

XML_EXT = '.xml'
ENCODE_METHOD = 'utf-8'


class PascalVocWriter:

    def __init__(self, foldername, filename, imgSize, databaseSrc='Unknown', localImgPath=None):
        self.foldername = foldername
        self.filename = filename
        self.out_fname = None
        if self.filename is not None:
            self.out_fname = self.filename + XML_EXT
        self.databaseSrc = databaseSrc
        self.imgSize = imgSize
        self.boxlist = []
        self.localImgPath = localImgPath
        self.verified = False

    def prettify(self, elem):
        """
            Return a pretty-printed XML string for the Element.
        """
        rough_string = ElementTree.tostring(elem, 'utf8')
        root = etree.fromstring(rough_string)
        return etree.tostring(root, pretty_print=True, encoding=ENCODE_METHOD).replace("  ".encode(), "\t".encode())
        # minidom does not support UTF-8
        '''reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="\t", encoding=ENCODE_METHOD)'''

    def genXML(self):
        """
            Return XML root
        """
        # Check conditions
        if self.filename is None or \
                self.foldername is None or \
                self.imgSize is None:
            return None

        top = Element('annotation')
        if self.verified:
            top.set('verified', 'yes')

        folder = SubElement(top, 'folder')
        folder.text = self.foldername

        filename = SubElement(top, 'filename')
        filename.text = self.filename

        if self.localImgPath is not None:
            localImgPath = SubElement(top, 'path')
            localImgPath.text = self.localImgPath

        source = SubElement(top, 'source')
        database = SubElement(source, 'database')
        database.text = self.databaseSrc

        size_part = SubElement(top, 'size')
        width = SubElement(size_part, 'width')
        height = SubElement(size_part, 'height')
        depth = SubElement(size_part, 'depth')
        width.text = str(self.imgSize[1])
        height.text = str(self.imgSize[0])
        if len(self.imgSize) == 3:
            depth.text = str(self.imgSize[2])
        else:
            depth.text = '1'

        segmented = SubElement(top, 'segmented')
        segmented.text = '0'
        return top

    def addBndBox(self, xmin, ymin, xmax, ymax, name, difficult,
                  bbox_source, id_number, score,
                  mask, mask_img=None):
        bndbox = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
        bndbox['name'] = name
        bndbox['difficult'] = difficult
        bndbox['bbox_source'] = bbox_source
        bndbox['id_number'] = id_number
        bndbox['score'] = score
        bndbox['mask'] = mask
        bndbox['mask_img'] = mask_img
        self.boxlist.append(bndbox)

    def appendObjects(self, top):

        # self.mask_images = {}

        for obj_id, each_object in enumerate(self.boxlist):
            object_item = SubElement(top, 'object')
            name = SubElement(object_item, 'name')
            name.text = each_object['name']
            pose = SubElement(object_item, 'pose')
            pose.text = "Unspecified"
            truncated = SubElement(object_item, 'truncated')
            if int(each_object['ymax']) == int(self.imgSize[0]) or (int(each_object['ymin']) == 1):
                truncated.text = "1"  # max == height or min
            elif (int(each_object['xmax']) == int(self.imgSize[1])) or (int(each_object['xmin']) == 1):
                truncated.text = "1"  # max == width or min
            else:
                truncated.text = "0"
            difficult = SubElement(object_item, 'difficult')
            difficult.text = str(bool(each_object['difficult']) & 1)
            bndbox = SubElement(object_item, 'bndbox')
            xmin = SubElement(bndbox, 'xmin')
            xmin.text = str(each_object['xmin'])
            ymin = SubElement(bndbox, 'ymin')
            ymin.text = str(each_object['ymin'])
            xmax = SubElement(bndbox, 'xmax')
            xmax.text = str(each_object['xmax'])
            ymax = SubElement(bndbox, 'ymax')
            ymax.text = str(each_object['ymax'])
            bbox_source = SubElement(object_item, 'bbox_source')
            bbox_source.text = str(each_object['bbox_source'])
            id_number = SubElement(object_item, 'id_number')
            id_number.text = str(each_object['id_number'])
            score = SubElement(object_item, 'score')
            score.text = str(each_object['score'])
            mask_pts = each_object['mask']

            if mask_pts is not None and mask_pts:
                mask_txt = '{},{},{};'.format(*mask_pts[0])
                for _pt in mask_pts[1:]:
                    mask_txt = '{} {},{},{};'.format(mask_txt, _pt[0], _pt[1], _pt[2])

                mask = SubElement(object_item, 'mask')
                mask.text = mask_txt

    def save(self, targetFile=None, _filename=None, _imgSize=None):

        if _filename is not None:
            self.filename = _filename
        if _imgSize is not None:
            self.imgSize = _imgSize

        if targetFile is None:
            if self.out_fname is None:
                raise IOError('targetFile must be provided when out_fname is None')
            self.out_fname = self.filename + XML_EXT
        else:
            self.out_fname = targetFile

            # self.filename = os.path.basename(self.out_fname)
            # self.foldername = os.path.dirname(self.out_fname)

        root = self.genXML()

        self.appendObjects(root)
        out_file = codecs.open(self.out_fname, 'w', encoding=ENCODE_METHOD)

        prettifyResult = self.prettify(root)
        out_file.write(prettifyResult.decode('utf8'))
        out_file.close()


EPS = 1e-2


def contour_pts_from_mask(mask_img, allow_multi=1):
    # print('Getting contour pts from mask...')
    if len(mask_img.shape) == 3:
        mask_img_gs = np.squeeze(mask_img[:, :, 0]).copy()
    else:
        mask_img_gs = mask_img.copy()

    _contour_pts, _ = cv2.findContours(mask_img_gs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]

    _contour_pts = list(_contour_pts)

    n_contours = len(_contour_pts)
    # print('n_contours: {}'.format(n_contours))
    # print('_contour_pts: {}'.format(_contour_pts))
    # print('contour_pts: {}'.format(type(contour_pts)))

    if allow_multi:
        if not _contour_pts:
            return [], []

        all_mask_pts = []

        for i in range(n_contours):
            _contour_pts[i] = list(np.squeeze(_contour_pts[i]))
            mask_pts = [[x, y, 1] for x, y in _contour_pts[i]]

            all_mask_pts.append(mask_pts)

        return all_mask_pts

    else:
        if not _contour_pts:
            return []
        contour_pts = list(np.squeeze(_contour_pts[0]))
        # return contour_pts

        if n_contours > 1:
            """get longest contour"""
            max_len = len(contour_pts)
            for _pts in _contour_pts[1:]:
                # print('_pts: {}'.format(_pts))
                _pts = np.squeeze(_pts)
                _len = len(_pts)
                if max_len < _len:
                    contour_pts = _pts
                    max_len = _len
        # print('contour_pts len: {}'.format(len(contour_pts)))
        mask_pts = [[x, y, 1] for x, y in contour_pts]

        return mask_pts


def color_val_matplotlib(color):
    """Convert various input in BGR order to normalized RGB matplotlib color
    tuples,

    Args:
        color (:obj:`Color`/str/tuple/int/ndarray): Color inputs

    Returns:
        tuple[float]: A tuple of 3 normalized floats indicating RGB channels.
    """
    color = mmcv.color_val(color)
    color = [color / 255 for color in color[::-1]]
    return tuple(color)


def imshow_det_bboxes(img,
                      bboxes,
                      labels,
                      segms=None,
                      score_thr=0,
                      out_mask_dir=None,
                      out_xml_dir=None,
                      out_filename=None,
                      out_dir=None,
                      write_masks=True,
                      write_xml=True,
                      classes=None,
                      palette_flat=None,
                      ):
    assert out_mask_dir is not None, "out_mask_dir must be provided"
    assert out_xml_dir is not None, "out_xml_dir must be provided"
    assert out_filename is not None, "out_filename must be provided"
    assert out_dir is not None, "out_dir must be provided"
    assert classes is not None, "classes must be provided"
    assert palette_flat is not None, "palette_flat must be provided"

    assert bboxes.ndim == 2, \
        f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
    assert labels.ndim == 1, \
        f' labels ndim should be 1, but its ndim is {labels.ndim}.'
    assert bboxes.shape[0] == labels.shape[0], \
        'bboxes.shape[0] and labels.shape[0] should have the same length.'
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5, \
        f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'
    img = mmcv.imread(img).astype(np.uint8)
    width, height = img.shape[1], img.shape[0]

    # font_size = 20

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        if segms is not None:
            segms = segms[inds, ...]

    # mask_colors = []
    # if labels.shape[0] > 0:
    #     if mask_color is None:
    #         # random color
    #         np.random.seed(42)
    #         mask_colors = [
    #             np.random.randint(0, 256, (1, 3), dtype=np.uint8)
    #             for _ in range(max(labels) + 1)
    #         ]
    #     else:
    #         # specify  color
    #         mask_colors = [np.array(mmcv.color_val(mask_color)[::-1], dtype=np.uint8)] * (max(labels) + 1)

    # bbox_color = color_val_matplotlib(bbox_color)
    # text_color = color_val_matplotlib(text_color)

    # img = mmcv.bgr2rgb(img)
    # img = np.ascontiguousarray(img)

    raw_mask = np.zeros((height, width), dtype=np.uint8)

    # fig = plt.figure(win_name, frameon=False)
    # plt.title(win_name)
    # canvas = fig.canvas
    # dpi = fig.get_dpi()
    # # add a small EPS to avoid precision lost due to matplotlib's truncation
    # # (https://github.com/matplotlib/matplotlib/issues/15363)
    # fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)
    #
    # # remove white edges by set subplot margin
    # plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # ax = plt.gca()
    # ax.axis('off')

    # polygons = []
    # color = []

    image_shape = [height, width, 3]

    out_filename_noext = os.path.splitext(out_filename)[0]

    csv_rows = []
    xml_dict = None
    xml_writer = None

    if write_xml:
        xml_writer = PascalVocWriter(out_dir, out_filename, image_shape)
        out_xml_path = os.path.join(out_xml_dir, out_filename_noext + '.xml')

    if write_masks:
        out_mask_file = os.path.join(out_mask_dir, out_filename_noext + '.png')

    is_valid = False

    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        # bbox_int = bbox.astype(np.int32)
        try:
            xmin, ymin, xmax, ymax, conf = bbox
        except ValueError:
            xmin, ymin, xmax, ymax = bbox
            conf = 1.0

        xmin, ymin, xmax, ymax = [int(x) for x in [xmin, ymin, xmax, ymax]]

        class_id = label + 1
        # if classes[0] == 'background':
        #     class_id += 1

        # bbox_list = list(bbox.squeeze())
        # conf = bbox_list[-1]

        # poly = [[xmin, ymin], [xmin, ymax],
        #         [xmax, ymax], [xmax, ymin]]
        # np_poly = np.array(poly).reshape((4, 2))
        # polygons.append(Polygon(np_poly))

        # if class_to_color is not None:
        #     bbox_color = [c / 255. for c in class_to_color[label][::-1]]
        #     text_color = [c / 255. for c in class_to_color[label][::-1]]

        # color.append(bbox_color)

        class_name = classes[class_id]

        label_text = class_name
        if len(bbox) > 4:
            label_text += f'|{bbox[-1]:.02f}'

        if write_xml:
            xml_dict = dict(
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
                name=class_name,
                difficult=False,
                bbox_source='object_detector',
                id_number=0,
                score=conf,
                mask=None,
                mask_img=None
            )

        row = {
            "ImageID": out_filename,
            "LabelName": class_name,
            "XMin": xmin,
            "XMax": xmax,
            "YMin": ymin,
            "YMax": ymax,
            "Confidence": conf,
        }

        if segms is not None:
            segm = segms[i]
            # if class_to_color is not None:
            #     color_mask = np.array(class_to_color[label][::-1], dtype=np.uint8).reshape((1, 3))
            # else:
            #     color_mask = mask_colors[label]
            if write_xml:
                mask_uint8 = segm.astype(np.uint8)
                seg_pts = contour_pts_from_mask(mask_uint8, allow_multi=False)
                xml_dict['mask'] = seg_pts

            mask_bool = segm.astype(bool)
            binary_seg = np.asfortranarray(mask_bool, dtype="uint8")
            rle = mask_util.encode(binary_seg)
            mask_h, mask_w = rle['size']
            mask_counts = rle['counts'].decode('utf-8')

            row.update(
                {
                    "mask_w": mask_w,
                    "mask_h": mask_h,
                    "mask_counts": mask_counts,

                }
            )
            # img[mask_bool] = img[mask_bool] * 0.5 + color_mask * 0.5
            raw_mask[mask_bool] = class_id

        csv_rows.append(row)
        if write_xml:
            xml_writer.addBndBox(**xml_dict)

        is_valid = True

    if is_valid:
        if write_masks:
            seg_pil = PIL.Image.fromarray(raw_mask)
            seg_pil = seg_pil.convert('P')
            seg_pil.putpalette(palette_flat)
            seg_pil.save(out_mask_file)

        if write_xml:
            xml_writer.save(targetFile=out_xml_path)

    # plt.imshow(img)

    # p = PatchCollection(
    #     polygons, facecolor='none', edgecolors=color, linewidths=thickness)
    # ax.add_collection(p)
    #
    # stream, _ = canvas.print_to_buffer()
    # buffer = np.frombuffer(stream, dtype='uint8')
    # img_rgba = buffer.reshape(height, width, 4)
    # rgb, alpha = np.split(img_rgba, [3], axis=2)
    # img = rgb.astype('uint8')
    # img = mmcv.rgb2bgr(img)

    # raw_mask = mmcv.rgb2bgr(raw_mask)

    # if show:
    #     # We do not use cv2 for display because in some cases, opencv will
    #     # conflict with Qt, it will output a warning: Current thread
    #     # is not the object's thread. You can refer to
    #     # https://github.com/opencv/opencv-python/issues/46 for details
    #     if wait_time == 0:
    #         plt.show()
    #     else:
    #         plt.show(block=False)
    #         plt.pause(wait_time)
    #
    #     plt.close()

    # mmcv.imwrite(img, out_file)
    # out_mask_dir = os.path.join(out_dir, 'masks')
    # os.makedirs(out_mask_dir, exist_ok=1)
    # out_mask_file = os.path.join(out_mask_dir, out_filename)
    # mmcv.imwrite(raw_mask, out_mask_file)
    #
    # masked_img[np.logical_not(raw_mask)] = 0
    # masked_img_out_dir = os.path.join(out_dir, 'masked_img')
    # os.makedirs(masked_img_out_dir, exist_ok=1)
    # masked_img_out_file = os.path.join(masked_img_out_dir, out_filename)
    # mmcv.imwrite(masked_img, masked_img_out_file)

    return csv_rows


def imshow_gt_det_bboxes(img,
                         annotation,
                         result,
                         class_names=None,
                         score_thr=0,
                         gt_bbox_color=(255, 102, 61),
                         gt_text_color=(255, 102, 61),
                         gt_mask_color=(255, 102, 61),
                         det_bbox_color=(72, 101, 241),
                         det_text_color=(72, 101, 241),
                         det_mask_color=(72, 101, 241),
                         thickness=2,
                         font_size=13,
                         win_name='',
                         show=True,
                         wait_time=0,
                         out_file=None):
    """General visualization GT and result function.

    Args:
      img (str or ndarray): The image to be displayed.)
      annotation (dict): Ground truth annotations where contain keys of
          'gt_bboxes' and 'gt_labels' or 'gt_masks'
      result (tuple[list] or list): The detection result, can be either
          (bbox, segm) or just bbox.
      class_names (list[str]): Names of each classes.
      score_thr (float): Minimum score of bboxes to be shown.  Default: 0
      gt_bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
           The tuple of color should be in BGR order. Default: (255, 102, 61)
      gt_text_color (str or tuple(int) or :obj:`Color`):Color of texts.
           The tuple of color should be in BGR order. Default: (255, 102, 61)
      gt_mask_color (str or tuple(int) or :obj:`Color`, optional):
           Color of masks. The tuple of color should be in BGR order.
           Default: (255, 102, 61)
      det_bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
           The tuple of color should be in BGR order. Default: (72, 101, 241)
      det_text_color (str or tuple(int) or :obj:`Color`):Color of texts.
           The tuple of color should be in BGR order. Default: (72, 101, 241)
      det_mask_color (str or tuple(int) or :obj:`Color`, optional):
           Color of masks. The tuple of color should be in BGR order.
           Default: (72, 101, 241)
      thickness (int): Thickness of lines. Default: 2
      font_size (int): Font size of texts. Default: 13
      win_name (str): The window name. Default: ''
      show (bool): Whether to show the image. Default: True
      wait_time (float): Value of waitKey param. Default: 0.
      out_file (str, optional): The filename to write the image.
         Default: None

    Returns:
        ndarray: The image with bboxes or masks drawn on it.
    """
    assert 'gt_bboxes' in annotation
    assert 'gt_labels' in annotation
    assert isinstance(
        result,
        (tuple, list)), f'Expected tuple or list, but get {type(result)}'

    gt_masks = annotation.get('gt_masks', None)
    if gt_masks is not None:
        gt_masks = mask2ndarray(gt_masks)

    img = mmcv.imread(img)

    img = imshow_det_bboxes(
        img,
        annotation['gt_bboxes'],
        annotation['gt_labels'],
        gt_masks,
    )

    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None

    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)

    segms = None
    if segm_result is not None and len(labels) > 0:  # non empty
        segms = mmcv.concat_list(segm_result)
        segms = mask_util.decode(segms)
        segms = segms.transpose(2, 0, 1)

    img = imshow_det_bboxes(
        img,
        bboxes,
        labels,
        segms=segms,
        score_thr=score_thr,
        out_file=out_file
    )
    return img
