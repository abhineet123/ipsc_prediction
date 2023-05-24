import os
import sys
import shutil
import glob
import re
import ast
import cv2
import functools
import copy
import paramparse
import threading
import numpy as np
from pprint import pformat
from tqdm import tqdm

from datetime import datetime

from libs.pascal_voc_io import PascalVocReader, PascalVocWriter
from libs.shape import Shape

from dmdp_utilities import draw_box, annotate_and_show, get_max_overlap_obj, \
    compute_overlaps_multi, find_associations, linux_path, draw_boxes, resize_ar, col_bgr, compute_overlap, CVText

"""
middle mouse : add new object or modify existing one
left mouse : select a single object
ctrl + left mouse : select object for cell division
    selects dets before track
shift + left mouse : select object for cell merging
    select tracks before dets

"""

DIRECT_TYPE = 0
DIVIDE_TYPE = 1
MERGE_TYPE = 2
NEW_TYPE = 3
CHANGED_TYPE = 4


class Params:
    class UI:
        def __init__(self):
            self.vis = 0
            self.vis_before_start = 0
            self.custom_fix = 0
            self.fmt = CVText(size=2, offset=(5,40))

            self.pause_after_frame = 0

    def __init__(self):
        """
        start_id
        """
        self.user_check = 0
        self.user_check_start_frame = 0

        self.start_id = -1
        self.sec_start_frame = -1
        self.sec_iou_thresh = 0.1
        self.max_start_frame = 0

        self.assoc_iou_threshold = 0.1
        self.det_iou_threshold = 0.5
        self.dim_thresholds = 0.5

        self.mask = Shape.Params.Mask()

        self.mask.border_size = (50, 50)

        self.ui = Params.UI()

        self.cfg = ()
        self.batch_size = 1
        self.class_names_path = 'data/predefined_classes_ipsc_2_class.txt'
        self.ignore_invalid_class = 0

        self.read_colors = True
        self.codec = 'mp4v'
        self.csv_file_name = ''
        self.enable_mask = 1
        self.fps = 20
        self.img_ext = 'jpg'
        self.load_path = ''
        self.load_samples = []
        self.load_samples_root = ''
        self.n_classes = 4

        """track from last to first frame"""
        self.inverted = 1
        self.root_dir = ''

        self.seq_paths = ''
        self.resize_factor = 0.4
        self.resize_to = (1890, 1050)
        self.neighborhood = 50
        self.method = 2
        self.cols = (
            'green_yellow', 'blue', 'orange_red', 'cyan', 'magenta', 'gold', 'brown', 'purple', 'peach_puff', 'azure',
            'dark_slate_gray', 'navy', 'turquoise', 'hot_pink', 'pink', 'maroon', 'forest_green')

        self.save_track = 1
        self.save_cell_lines = 0
        self.write_xml = 0
        # self.vis_size = 1280, 720
        self.vis_size = 1920, 1080

        self.sources_to_include = []

        self.n_proc = 1


class Track:
    def __init__(self, target_id, label, col):
        self.target_id = target_id
        self.label = label
        self.col = col
        self.nodes = []
        self.frame_id_to_node = {}
        self.parents = []
        self.children = []

    def remove_last_node(self):
        node = self.nodes.pop()
        del self.frame_id_to_node[node.frame_id]

    def get_node_as_det(self, frame_id):

        node = self.get_node(frame_id)  # type: TrackNode

        det = dict(
            difficult=node.difficult,
            source=node.source,
            score=node.score,
            target_id=self.target_id,
            filename=node.filename,
            col=self.col,
            label=self.label,
            mask=node.mask,
            parents=self.parents,
            children=self.children,
            bbox=node.bbox
        )

        return det

    def get_bbox(self, frame_id):
        node = self.get_node(frame_id)  # type: TrackNode
        if node is None:
            return None
        return node.bbox

    def get_mask(self, frame_id):
        node = self.get_node(frame_id)  # type: TrackNode
        if node is None:
            return None
        return node.mask

    def get_node(self, frame_id):
        try:
            node = self.frame_id_to_node[frame_id]
        except KeyError:
            # print(f'no node exists for track {self.target_id} in frame {frame_id}')
            node = None
        return node

    def add_node(self, node):
        self.frame_id_to_node[node.frame_id] = node
        self.nodes.append(node)

    def set_col(self, col):
        self.col = col

    def set_label(self, label):
        self.label = label

    def add_parent(self, parent):
        if parent not in self.parents:
            self.parents.append(parent)

    def add_child(self, child):
        if child not in self.children:
            self.children.append(child)

    def paths(self, targets, target_ids_in_paths):
        target_ids_in_paths.append(self.target_id)
        if not self.children:
            """no children so only one path made up of the nodes in this track itself"""
            return [self.nodes, ]
        paths = []
        for child_id in self.children:
            child = targets[child_id]
            for path in child.paths(targets, target_ids_in_paths):
                """one path corresponds to each child path appended to the track's own nodes"""
                paths.append(self.nodes + path)
        return paths


class TrackNode:
    def __init__(self, frame_id, bbox, mask, score, difficult, source, filename, dims, dims_ratio):
        self.frame_id = frame_id
        self.bbox = bbox
        self.mask = mask
        self.score = score
        self.difficult = difficult
        self.source = source
        self.filename = filename

        self.dims = dims
        self.dims_ratio = dims_ratio


def get_all_det_vis_frames(n_frames, det_vis_fn, n_proc):
    print('generating all det vis frames')
    if n_proc > 1:
        import multiprocessing
        import functools

        print('running in parallel over {} processes'.format(n_proc))
        pool = multiprocessing.Pool(n_proc)

        pool.map(det_vis_fn, list(range(n_frames)))
    else:
        for frame_id in range(n_frames):
            det_vis_fn(frame_id)


# def read_all_frames(seq_path, n_frames, frames, file_paths, file_names, all_dets, xml_files):
#     for frame_id in range(n_frames):
#         dets = get_dets(frame_id, all_dets, xml_files)
#
#         filename = dets[0]['filename']
#         filepath = linux_path(seq_path, filename)
#
#         file_names[frame_id] = filename
#         file_paths[frame_id] = filepath
#
#         get_frame(frame_id, frames, file_paths)


def get_nearest_bbox(x, y, det_centroids):
    if not det_centroids:
        return -1

    det_dists = [abs(cx - x) + abs(cy - y) if x1 <= x <= x2 and y1 <= y <= y2 else np.inf
                 for cx, cy, x1, y1, x2, y2 in det_centroids]
    min_dist_det = np.argmin(det_dists)

    # print(f'det_dists: {det_dists}')
    # print(f'min_dist_det: {min_dist_det}')

    if not np.isinf(det_dists[min_dist_det]):
        return min_dist_det

    return -1


def update_track_vis_frame(frame_id, frames, frame_paths, frame_id_to_tracks, seq_name, xml_files, cols,
                           track_vis_frames,
                           track_win_name, fmt, save_path=None, vis_size=None, show=False):
    frame = get_frame(frame_id, frames, frame_paths)

    try:
        tracks = list(frame_id_to_tracks[frame_id].values())
    except KeyError:
        tracks = []

    xml_file_name = os.path.basename(xml_files[frame_id])
    header = f'{seq_name} {frame_id}: {xml_file_name}'

    track_vis_frame = vis_tracks(frame, frame_id, tracks, header, all_cols=cols,
                                 size=vis_size, fmt=fmt
                                 )
    track_vis_frames[frame_id] = track_vis_frame

    if save_path is not None:
        cv2.imwrite(save_path, track_vis_frame)

    if show:
        cv2.imshow(track_win_name, track_vis_frame)

    return track_vis_frame


def get_dets(frame_id, all_dets, xml_files):
    """all dets in this frame"""
    try:
        dets = all_dets[frame_id]
    except IndexError:
        print(f'invalid frame_id: {frame_id}')
        return None, None, None

    if not dets:
        raise AssertionError(f'no detections in frame {frame_id}: {xml_files[frame_id]}')

    return dets


def get_frame(frame_id, frames, frame_paths):
    try:
        frame = frames[frame_id]
    except KeyError:
        filepath = frame_paths[frame_id]

        assert os.path.isfile(filepath), f"image file does not exist: {filepath}"

        frame = cv2.imread(filepath)

        if frame is None:
            raise AssertionError(f'frame {frame_id} could not be read: {filepath}')

        frames[frame_id] = frame

    return frame


def get_det_vis_frame(frame_id, all_dets, det_vis_frames, seq_name, seq_path, xml_files, frame_paths, frames,
                      enable_mask, vis_size, fmt,
                      # new_obj=None,
                      force_update=False):
    # if new_obj is not None:
    #     dets.append(new_obj)
    #     force_update = 1

    if not force_update:
        try:
            det_vis_frame = det_vis_frames[frame_id]
        except KeyError:
            pass
        else:
            return det_vis_frame

    else:
        print(f'force updating det_vis_frame {frame_id}')

    try:
        xml_file_name = os.path.basename(xml_files[frame_id])
    except IndexError:
        print(f'xml file not found for frame {frame_id}')
        return None

    dets = get_dets(frame_id, all_dets, xml_files)

    filename = dets[0]['filename']
    filepath = linux_path(seq_path, filename)

    frame_paths[frame_id] = filepath

    frame = get_frame(frame_id, frames, frame_paths)

    header = f'{seq_name} {frame_id}: {xml_file_name}'

    det_vis_frame = vis_dets(frame, dets, enable_mask, header, size=vis_size, fmt=fmt)

    det_vis_frames[frame_id] = det_vis_frame

    return det_vis_frame


# def get_nearest_track(frame_id, track, tracks):
#     track_bbox = track.get_node(frame_id).bbox
#     other_tracks = [_track for _track in tracks if _track.target_id != track.target_id]
#     other_bboxes = [_track.get_node(frame_id).bbox for _track in other_tracks]
#
#     track_bbox = np.asarray(track_bbox)
#     other_bboxes = np.asarray(other_bboxes)
#
#     max_iou, max_iou_idx = get_max_overlap_obj(other_bboxes, track_bbox)
#
#     return other_tracks[max_iou_idx]


def keyboard_handler(config):
    """

    :param Params.UI config:
    :return:
    """

    k = cv2.waitKey(1 - config.pause_after_frame)

    # print(f'k: {k}')
    # print(f'params.ui.pause_after_frame: {params.ui.pause_after_frame}')

    if k == ord('v'):
        config.vis = 1 - config.vis
    elif k == 32:
        config.pause_after_frame = 1 - config.pause_after_frame
    elif k == 27:
        cv2.destroyAllWindows()
        exit()
    elif k == ord('f'):
        config.custom_fix = 1 - config.custom_fix
        if config.custom_fix:
            print('turning on custom fix')
        else:
            print('turning off custom fix')


def process_user_input(det_sel_ids, track_sel_ids, dets, frame_id, frame_id_to_tracks,
                       target_id_to_tracks, det_to_target_id, unassociated_target_ids, unassociated_dets,
                       tracks, track_id, target_ids, track_vis_fn):
    new_target_ids = []

    prev_tracks_changed = False

    changed_dets = det_sel_ids[CHANGED_TYPE]
    new_dets = det_sel_ids[NEW_TYPE]
    direct_dets = det_sel_ids[DIRECT_TYPE]
    divide_dets = det_sel_ids[DIVIDE_TYPE]
    merge_dets = det_sel_ids[MERGE_TYPE]

    if direct_dets is not None:
        if track_id is not None:
            """tracking failure"""
            assert len(direct_dets) == 1, "a single object must be selected for direct association"

            _det_ids = [direct_dets[0], ]
            _track_ids = [track_id, ]

        elif track_sel_ids[DIRECT_TYPE] is not None:
            _det_ids = direct_dets
            _track_ids = track_sel_ids[DIRECT_TYPE]

            assert len(_det_ids) == len(_track_ids), \
                "Mismatch between the number of selected tracks and detections for association"
        else:
            _det_ids = []
            _track_ids = []

        if _track_ids:
            """add each detection to the corresponding track"""
            for _det_id, _track_id in zip(_det_ids, _track_ids):
                try:
                    old_target_id = det_to_target_id[_det_id]
                except KeyError:
                    """previously unassociated detection"""
                    try:
                        unassociated_dets.remove(_det_id)
                    except ValueError:
                        """associated to new object added by the user"""
                        assert _det_id in new_dets, \
                            f"weird det_id neither in unassociated_dets nor in new_dets: {_det_id}"
                else:
                    """previously associated to another track"""
                    unassociated_target_ids.append(old_target_id)
                    old_track = target_id_to_tracks[old_target_id]
                    old_track.remove_last_node()

                add_det_to_track(
                    frame_id, dets[_det_id], tracks[_track_id], frame_id_to_tracks, target_id_to_tracks, replace=False)
        else:
            """create a new track from each detection"""
            new_target_ids, target_ids = add_tracks(frame_id, dets, direct_dets, target_ids, frame_id_to_tracks,
                                                    target_id_to_tracks, det_to_target_id,
                                                    allow_no_id=True, force_new_tracks=True)

        track_vis_fn(frame_id)

    elif divide_dets is not None:
        """cell division"""

        divide_track_to_dets = {}

        if track_sel_ids[DIVIDE_TYPE] is None:
            assert track_id is not None, "track_id must be provided when parent track is not selected for cell division"
            assert len(divide_dets) >= 2, "at least two child cells must be selected for cell division"

            print('no parent cell selected for cell division so using the current one by default')

            divide_track_to_dets[track_id] = [det for det, ts in divide_dets]
        else:
            det_start_id = 0
            for divide_track, track_ts in track_sel_ids[DIVIDE_TYPE]:
                """all dets selected right before this track and after any previous tracks"""
                curr_divide_dets = [det for det, ts in divide_dets[det_start_id:] if ts < track_ts]
                assert len(
                    curr_divide_dets) >= 2, f"less than two child cells selected for division of track {divide_track}"
                divide_track_to_dets[divide_track] = curr_divide_dets
                det_start_id += len(curr_divide_dets)

        assert len(divide_track_to_dets) >= 1, "no parent tracks selected for cell division"

        for parent_track_id in divide_track_to_dets:
            child_dets = divide_track_to_dets[parent_track_id]
            parent_track = tracks[parent_track_id]

            new_target_ids, target_ids = add_tracks(frame_id, dets, child_dets, target_ids,
                                                    frame_id_to_tracks, target_id_to_tracks, det_to_target_id,
                                                    parent_ids=[parent_track.target_id, ], allow_no_id=True,
                                                    force_new_tracks=True)

            for det_id in child_dets:
                try:
                    unassociated_dets.remove(det_id)
                except ValueError:
                    """previously associated det"""
                    pass

            print(f'target {parent_track.target_id} divided into new targets {new_target_ids}')

            if parent_track.target_id in frame_id_to_tracks[frame_id]:
                """the parent track had been associated to a det in the current frame so remove that det 
                to end this track"""
                parent_track.remove_last_node()
                """this track no longer exists in the current frame"""
                frame_id_to_tracks[frame_id].pop(parent_track.target_id)

                prev_tracks_changed = True

            track_vis_fn(frame_id)

    elif merge_dets is not None:
        """cell merging"""

        merge_tracks = track_sel_ids[MERGE_TYPE]
        assert merge_tracks is not None, "no parent tracks selected"

        merge_det_to_tracks = {}

        track_start_id = 0
        for merge_det, det_ts in merge_dets:
            """all dets selected right before this track and after any previous tracks"""
            curr_merge_tracks = [track for track, ts in merge_tracks[track_start_id:] if ts < det_ts]
            assert len(curr_merge_tracks) >= 2, f"< two tracks selected for merging of det {merge_det}"

            merge_det_to_tracks[merge_det] = curr_merge_tracks
            track_start_id += len(curr_merge_tracks)

        for child_det in merge_det_to_tracks:
            parent_track_ids = merge_det_to_tracks[child_det]
            parent_target_ids = [tracks[k].target_id for k in parent_track_ids]

            print(f'got parents {parent_target_ids} for merged child {child_det}')

            new_target_ids, target_ids = add_tracks(frame_id, dets, [child_det, ], target_ids,
                                                    frame_id_to_tracks, target_id_to_tracks, det_to_target_id,
                                                    parent_ids=parent_target_ids, allow_no_id=True,
                                                    force_new_tracks=True)

            print(f'created new child targets {new_target_ids} from merged parents {parent_target_ids}')

            if child_det in unassociated_dets:
                unassociated_dets.remove(child_det)

            for parent_track_id in parent_track_ids:
                parent_track = tracks[parent_track_id]
                if parent_track.target_id in frame_id_to_tracks[frame_id]:
                    """parent track had been associated to a det in the current frame so 
                    remove that det to end this track"""
                    parent_track.remove_last_node()
                    """this track no longer exists in the current frame"""
                    frame_id_to_tracks[frame_id].pop(parent_track.target_id)
            track_vis_fn(frame_id)

            prev_tracks_changed = True

    elif new_dets is not None:
        if track_id is not None and len(new_dets) == 1:
            """one new object added by user - assume it fills in false negative"""
            new_obj = dets[new_dets[0]]
            print('adding the new object to the track')
            add_det_to_track(
                frame_id, new_obj, tracks[track_id], frame_id_to_tracks, target_id_to_tracks, replace=False)
        else:
            """add a new track for each added object"""
            new_objs = [dets[new_det] for new_det in new_dets]
            print(f'creating new tracks from {len(new_objs)} added object(s)')

            new_target_ids, target_ids = add_tracks(frame_id, dets, new_dets, target_ids,
                                                    frame_id_to_tracks, target_id_to_tracks, det_to_target_id,
                                                    allow_no_id=False,
                                                    force_new_tracks=True)

        track_vis_fn(frame_id)

    elif changed_dets is not None:
        for det_id in changed_dets:
            target_id = det_to_target_id[det_id]
            det = dets[det_id]
            track = target_id_to_tracks[target_id]  # type: Track
            print(f'updating track {track.target_id} corresponding to updated det {det_id}')
            node = track.get_node(frame_id)
            node.bbox = det['bbox']
            node.mask = det['mask']

        track_vis_fn(frame_id)

        prev_tracks_changed = True
    else:
        print('no changes made by user')

    return new_target_ids, target_ids, prev_tracks_changed


def show_boxes(frame, bboxes, masks, win_name, draw_only=False, ids=None, text_cols=None, labels=None,
               resize_factor=None, cols='green', ids_col=None, vis_frame=None, frame_res=None, border=0,
               incremental=False, in_place=False):
    """

    :param frame:
    :param bboxes:
    :param masks:
    :param win_name:
    :param draw_only:
    :param ids:
    :param text_cols:
    :param labels:
    :param resize_factor:
    :param cols:
    :param str|None ids_col:
    :param vis_frame:
    :return:
    """
    n_bboxes = len(bboxes)

    if not isinstance(cols, (list, tuple)):
        cols = [cols, ] * n_bboxes
    else:
        cols = cols[:]

    if ids is not None and ids_col is not None:
        for _id in ids:
            cols[_id] = ids_col

    if resize_factor is not None and resize_factor != 1:
        bboxes = [bbox * resize_factor for bbox in bboxes]
        if masks is not None:
            masks = [mask * resize_factor for mask in masks]

    # print(f'showing {n_bboxes} boxes')

    if incremental:

        assert vis_frame is not None, "vis_frame must be provided for incremental drawing"
        assert ids is not None, "ids must be provided for incremental drawing"
        assert frame_res is not None, "frame_res must be provided for incremental drawing"

        """only draw boxes specified in ids assuming that other boxes have already been drawn and don't need changing"""

        bboxes = [bboxes[_id] for _id in ids]
        masks = [masks[_id] for _id in ids]

        cols = [cols[_id] for _id in ids]

        # if isinstance(labels, (list, tuple)):
        #     labels = [labels[_id] for _id in ids]
        #
        # if isinstance(text_cols, (list, tuple)):
        #     text_cols = [text_cols[_id] for _id in ids]

        img_h, img_w = vis_frame.shape[:2]

        for bbox in bboxes:
            """undraw these boxes by restoring pixes from original frame"""
            x, y, w, h = bbox

            min_x = int(max(0, x - border))
            min_y = int(max(0, y - border))
            max_x = int(min(img_w, x + w + border))
            max_y = int(min(img_h, y + h + border))

            vis_frame[min_y:max_y, min_x:max_x, ...] = frame_res[min_y:max_y, min_x:max_x, ...]

        """don't write any ID or label text so changed area does not go beyond the bbox"""
        draw_boxes(vis_frame, bboxes, font_size=1.5 * resize_factor, colors=cols, masks=masks)
    else:
        """draw all boxes"""

        if frame_res is None:
            frame_res = frame

            if resize_factor is not None and resize_factor != 1:
                """no existing vis_frame so frame is in original size"""
                frame_res = resize_ar(frame_res, resize_factor=resize_factor)

        if in_place:
            assert vis_frame is not None, "vis_frame must be provided for in_place drawing"
            vis_frame[:] = frame_res
        else:
            vis_frame = np.copy(frame_res)

        draw_boxes(vis_frame, bboxes, font_size=1.5 * resize_factor, colors=cols,
                   ids=[str(k) for k in range(n_bboxes)],
                   labels=labels,
                   text_cols=text_cols,
                   masks=masks,
                   )

        # cv2.imshow('vis_frame', vis_frame)
        # cv2.imshow('frame_res', frame_res)
        # cv2.imshow('frame', frame)

        # print()

    if not draw_only:
        cv2.imshow(win_name, vis_frame)
        # cv2.waitKey(0)

    return vis_frame, frame_res


def mouse_handler(event, x, y, flags, param, config):
    sel_ids, cols, box_selected, vis_frame, frame_res = param

    frame, frame_id, objs, bboxes, masks, id_image, new_det_info, \
    sel_cols, win_name, resize_factor, obj_type, allow_new_objects, assoc_obj, temporal_fn, det_vis_fn = config

    if resize_factor is not None and resize_factor != 1:
        x, y = x / resize_factor, y / resize_factor

    if event == cv2.EVENT_MBUTTONDOWN:
        print(f'flags: {flags}')

        """add new object or modify existing one"""
        try:
            min_dist_det = id_image[int(y), int(x)]
        except IndexError:
            min_dist_det = -1

            if not allow_new_objects:
                print('new objects are not allowed')
                return

        # vis_frame = show_boxes(frame, bboxes, masks, win_name, draw_only=True,
        #                        ids=None, resize_factor=None, **kwargs)

        mask_params = Shape.Params.Mask()
        shape = Shape()

        if min_dist_det >= 0:
            bbox, mask = bboxes[min_dist_det], masks[min_dist_det]

            shape.set_bbox(bbox, mask)

            mask_params.border_size = (300, 300)

        elif flags == 12:
            """ctrl"""
            if assoc_obj is not None:
                print('copying associated object to modify')
                bbox, mask = assoc_obj['bbox'], assoc_obj['mask']
                shape.set_bbox(bbox, mask)
                mask_params.border_size = (300, 300)
            else:
                print("no associated object found to copy")

        added_masks = shape.addMask(frame, mask_params, generic=True)

        n_added_masks = len(added_masks)
        if not n_added_masks:
            print('no new objects added')
        else:
            if min_dist_det >= 0:
                assert n_added_masks == 1, "only one mask can be added when modifying an existing object"
                print(f'modified mask for object {min_dist_det}')
            else:
                print(f'added {n_added_masks} new objects')

            filename, label, col = new_det_info
            changed_obj_ids = []
            for mask in added_masks:
                mask_pts = [(x, y) for x, y, _ in mask]
                mask_arr = np.asarray(mask_pts)
                xmin, ymin = np.min(mask_arr, axis=0)
                xmax, ymax = np.max(mask_arr, axis=0)

                w, h = xmax - xmin, ymax - ymin

                bbox = np.asarray([xmin, ymin, w, h])

                if min_dist_det >= 0:

                    """modify an existing object"""

                    obj = objs[min_dist_det]
                    obj['bbox'] = bbox
                    obj['mask'] = mask_arr

                    bboxes[min_dist_det] = bbox
                    masks[min_dist_det] = mask_arr

                    obj_id = min_dist_det

                    if sel_ids[CHANGED_TYPE] is None:
                        sel_ids[CHANGED_TYPE] = []

                    sel_ids[CHANGED_TYPE].append(obj_id)

                    if det_vis_fn is not None:
                        det_vis_fn(frame_id, force_update=True)
                else:
                    """add a new object"""

                    obj = dict(
                        difficult=0,
                        source='ground_truth',
                        score=1,
                        target_id=0,
                        filename=filename,
                        col=col,
                        label=label,
                        mask=mask_arr,
                        bbox=bbox,
                        parents=None,
                        children=None,
                    )

                    obj_id = len(bboxes)
                    bboxes.append(bbox)
                    masks.append(mask_arr)
                    # centroids.append((xmin + w / 2, ymin + h / 2, xmin, ymin, xmin + w, ymin + h))
                    cols.append(sel_cols[NEW_TYPE])
                    objs.append(obj)

                    """add new objects to objs vis frame"""
                    if det_vis_fn is not None:
                        det_vis_fn(frame_id, force_update=True)

                    if sel_ids[NEW_TYPE] is None:
                        sel_ids[NEW_TYPE] = []

                    """don't add changed objects to new object IDs to avoid spurious track creation or association
                    if no other object is selected"""
                    sel_ids[NEW_TYPE].append(obj_id)

                """add both new and changed objects for incremental visualization"""
                changed_obj_ids.append(obj_id)

                temporal_fn(-1, frame_id)

                print(f'{bbox}')

            get_id_image(frame, bboxes, id_image=id_image)

            if changed_obj_ids:
                show_boxes(frame, bboxes, masks, win_name,
                           resize_factor=resize_factor,
                           cols=cols,
                           vis_frame=vis_frame,
                           frame_res=frame_res,
                           incremental=False,
                           in_place=True,
                           )

    elif event == cv2.EVENT_LBUTTONDOWN:

        try:
            min_dist_det = id_image[int(y), int(x)]
        except IndexError:
            min_dist_det = -1

        if min_dist_det >= 0:
            if flags == 1:
                # none
                sel_type = DIRECT_TYPE
                print(f'selected box: {min_dist_det}')
                sel_data = min_dist_det
            elif flags == 9:
                # ctrl
                sel_type = DIVIDE_TYPE
                print(f'selected division {obj_type} box: {min_dist_det}')
                sel_data = (min_dist_det, datetime.now())
            elif flags == 17:
                # shift
                sel_type = MERGE_TYPE
                print(f'selected merging {obj_type} box: {min_dist_det}')
                sel_data = (min_dist_det, datetime.now())
            else:
                raise AssertionError(f'Unsupported flags: {flags}')

            cols[min_dist_det] = sel_cols[sel_type]

            if sel_ids[sel_type] is None:
                sel_ids[sel_type] = []

            sel_ids[sel_type].append(sel_data)

            show_boxes(frame, bboxes, masks, win_name, resize_factor=resize_factor, cols=cols,
                       vis_frame=vis_frame,
                       frame_res=frame_res,
                       incremental=False,
                       ids=[min_dist_det, ])

    elif event == cv2.EVENT_MOUSEMOVE:
        # print(f'x: {x}, y: {y}')
        if box_selected[0] >= 0:
            show_boxes(frame, bboxes, masks, win_name,
                       vis_frame=vis_frame,
                       frame_res=frame_res,
                       incremental=False,
                       ids=[box_selected[0], ],
                       resize_factor=resize_factor, cols=cols)
            box_selected[0] = -1

        try:
            min_dist_det = id_image[int(y), int(x)]
        except IndexError:
            min_dist_det = -1

        if min_dist_det >= 0:
            show_boxes(frame, bboxes, masks, win_name,
                       vis_frame=vis_frame,
                       frame_res=frame_res,
                       incremental=False,
                       ids=[min_dist_det, ],
                       resize_factor=resize_factor, cols=cols, ids_col='white')
            box_selected[0] = min_dist_det


def get_id_image(frame, bboxes, id_image=None):
    img_h, img_w = frame.shape[:2]
    """sort by area so smaller objects get their IDs filled in later to avoid being overwritten 
    by larger overlapping objects"""
    sort_idx = sorted(range(len(bboxes)), key=lambda k: bboxes[k][-1] * bboxes[k][-2], reverse=True)

    if id_image is None:
        id_image = np.full((img_h, img_w), -1, dtype=np.int32)
    else:
        """in-place"""
        id_image.fill(-1)

    for bbox_id in sort_idx:
        x, y, w, h = bboxes[bbox_id]
        id_image[int(y):int(y + h), int(x):int(x + w)] = bbox_id

    return id_image


def select_boxes_interactively(det_data, new_det_info, assoc_obj, track_data=None, all_cols=None,
                               ids=None, resize_factor=1., temporal_fn=None, det_vis_fn=None):
    det_frame, det_frame_id, dets, det_win_name, det_cols = det_data

    n_types = 5

    det_bboxes = [det['bbox'] for det in dets]
    det_masks = [det['mask'] for det in dets]

    det_id_image = get_id_image(det_frame, det_bboxes)
    # det_centroids = [(x + w / 2, y + h / 2, x, y, x + w, y + h) for x, y, w, h in det_bboxes]

    n_det_bboxes = len(det_bboxes)

    default_col = 'green'
    sel_cols = ['gold', 'medium_spring_green', 'indian_red', 'olive_drab', 'slate_gray']

    det_box_selected = [-1, ]
    if det_cols is None:
        det_cols = [default_col, ] * n_det_bboxes
    else:
        det_cols = det_cols[:]
    det_sel_ids = [None, ] * n_types
    det_vis_frame, det_frame_res = show_boxes(det_frame, det_bboxes, det_masks, det_win_name, cols=det_cols,
                                              resize_factor=resize_factor)

    det_param = [det_sel_ids, det_cols, det_box_selected, det_vis_frame, det_frame_res]
    det_config = [det_frame, det_frame_id, dets, det_bboxes, det_masks, det_id_image, new_det_info,
                  sel_cols, det_win_name, resize_factor, 'child', True, assoc_obj, temporal_fn, det_vis_fn]
    det_mouse_handler = functools.partial(mouse_handler, config=det_config)
    cv2.setMouseCallback(det_win_name, det_mouse_handler, param=det_param)

    track_win_name = track_sel_ids = None

    if track_data is not None:
        track_frame, track_frame_id, tracks, track_win_name, track_cols = track_data

        tracks_as_dets = [track.get_node_as_det(track_frame_id) for track in tracks]

        track_bboxes = [track.get_bbox(track_frame_id) for track in tracks]
        track_masks = [track.get_mask(track_frame_id) for track in tracks]

        track_id_image = get_id_image(track_frame, track_bboxes)
        # track_centroids = [(x + w / 2, y + h / 2, x, y, x + w, y + h) for x, y, w, h in track_bboxes]

        track_box_selected = [-1, ]
        if track_cols is None:
            target_ids = [track.target_id for track in tracks]
            track_cols = [
                all_cols[target_id % len(all_cols)] for target_id in target_ids
            ]
        else:
            track_cols = track_cols[:]
        track_sel_ids = [None, ] * n_types
        track_vis_frame, track_frame_res = show_boxes(track_frame, track_bboxes, track_masks, track_win_name,
                                                      cols=track_cols, resize_factor=resize_factor)

        track_param = [track_sel_ids, track_cols, track_box_selected, track_vis_frame, track_frame_res]
        track_config = [track_frame, track_frame_id, tracks_as_dets, track_bboxes, track_masks, track_id_image,
                        new_det_info, sel_cols, track_win_name, resize_factor, 'parent', False, None, temporal_fn, None]
        track_mouse_handler = functools.partial(mouse_handler, config=track_config)
        cv2.setMouseCallback(track_win_name, track_mouse_handler, param=track_param)

    vis_frame_id = det_frame_id

    while True:
        k = cv2.waitKeyEx(1)
        if k == 27:
            return None

        if k == 13:
            break

        if ids is not None:
            if k == ord('1'):
                det_sel_ids[DIRECT_TYPE] = [ids[0], ]
                break
            elif k == ord('2'):
                det_sel_ids[DIRECT_TYPE] = [ids[1], ]
                break

        if temporal_fn is not None:
            vis_frame_id = temporal_fn(k, vis_frame_id)

    cv2.destroyWindow(det_win_name)
    if track_data is not None:
        cv2.destroyWindow(track_win_name)
        return det_sel_ids, track_sel_ids

    return det_sel_ids


def handle_duplicate_dets(frame, frame_id, dets, det_ids, new_det_info, temporal_fn, det_vis_fn, header,
                          det_iou_threshold,
                          resize_factor, enable_mask, vis_size):
    dets_to_remove = []

    n_dets = len(det_ids)

    if n_dets == 0:
        return dets_to_remove

    det_bboxes = [dets[_id]['bbox'] for _id in det_ids]
    det_bboxes_np = np.asarray(det_bboxes)

    """check for duplicate detections"""
    iou = np.empty((n_dets, n_dets))
    compute_overlaps_multi(iou, None, None, det_bboxes_np, det_bboxes_np)

    """upper triangular"""
    iou_triu = np.triu(iou)
    np.fill_diagonal(iou_triu, 0)

    close_dets = np.nonzero(iou_triu > det_iou_threshold)

    close_dets_ = np.c_[close_dets]

    win_name = 'Select FP detection. Press Enter to select the first one and Escape to skip.'

    for _id1, _id2 in close_dets_:
        id1, id2 = det_ids[_id1], det_ids[_id2]
        msg = f'frame {frame_id} :: detections {id1} and {id2} are too close: {iou_triu[_id1, _id2]:.3f}. ' \
            f'Select FP detection'
        print(msg)
        _header = f'{header} :: {msg}'

        det_vis_frame = vis_dets(frame, dets, enable_mask, _header, size=vis_size)

        det_win_name = 'problematic detections'

        cv2.imshow(det_win_name, det_vis_frame)

        # sel_ids = handle_overlapping_dets(frame, frame_id, dets, new_det_info, id1, id2, temporal_fn,
        #                                   win_name, resize_factor=resize_factor)

        cols = ['green', ] * len(dets)

        cols[id1] = 'red'
        cols[id2] = 'blue'

        det_data = [frame, frame_id, dets, win_name, cols]

        sel_ids = select_boxes_interactively(det_data=det_data, track_data=None,
                                             assoc_obj=None,
                                             new_det_info=new_det_info, ids=[id1, id2],
                                             resize_factor=resize_factor,
                                             det_vis_fn=det_vis_fn,
                                             temporal_fn=temporal_fn,
                                             all_cols=None)

        if sel_ids is None:
            print(f'leaving both detections intact')
            continue

        if sel_ids[DIRECT_TYPE] is None:
            print(f'selecting detection {id1} by default')
            """user doesn't care so assume first one is FP"""
            sel_ids[DIRECT_TYPE] = [id1, ]

        # if fp_det_id is not None:
        #     assert fp_det_id in [id1, id2], f"invalid detection selected {fp_det_id}"
        #     print(f'removing detection {fp_det_id}')

        fp_det_id = sel_ids[DIRECT_TYPE][0]

        dets_to_remove.append(fp_det_id)

        cv2.destroyWindow(det_win_name)

    return dets_to_remove


def vis_tracks(frame, frame_id, tracks, header,
               width=0, height=0,
               cols=None, all_cols=None,
               track_nodes=None, target_ids=None,
               text_cols=None,
               labels=None,
               size=None,
               header_col=None,
               fmt=None,
               ):
    if size is not None:
        width, height = size

    if track_nodes is None:
        track_nodes = [track.get_node(frame_id) for track in tracks]
        target_ids = [track.target_id for track in tracks]
        labels = [track.label for track in tracks]
        text_cols = [track.col for track in tracks]

    track_bboxes = [node.bbox for node in track_nodes if node is not None]
    track_masks = [node.mask for node in track_nodes if node is not None]

    if cols is None:
        cols = [
            all_cols[target_id % len(all_cols)] for target_id in target_ids
        ]

    tracks_vis_frame = np.copy(frame)

    draw_boxes(tracks_vis_frame, track_bboxes, colors=cols, labels=labels, text_cols=text_cols, ids=target_ids,
               font_size=1.5, masks=track_masks)

    if height > 0 or width > 0:
        tracks_vis_frame = resize_ar(tracks_vis_frame, width=width, height=height, only_shrink=1)

    header_fmt = CVText(fmt=fmt)

    if header is not None:
        if header_col is not None:
            header_fmt.color = header_col

        tracks_vis_frame = annotate_and_show(header, tracks_vis_frame, only_annotate=1, n_modules=0, fmt=header_fmt)

    return tracks_vis_frame


def temporal_interaction(k, vis_frame_id, max_frame_id, min_frame_id, track_win_name, det_win_name,
                         track_vis_frames, det_vis_frames, mode, det_vis_fn, track_vis_fn, custom_frame_id, show=True):
    # if k is None or k < 0:
    #     return vis_frame_id

    # print(f'k: {k}')

    _vis_frame_id = vis_frame_id

    if mode == 0:
        if k == 2555904 or k == 39:
            # right
            vis_frame_id += 1
            show = True

        elif k == 2424832 or k == 40:
            # left
            vis_frame_id = max(vis_frame_id - 1, min_frame_id)
            show = True

            if det_win_name is not None:
                det_vis_frame = det_vis_fn(vis_frame_id)

                cv2.imshow(det_win_name, det_vis_frame)
        elif k == ord('m'):
            custom_frame_id[0] = vis_frame_id

        if show:
            if det_win_name is not None:

                det_vis_frame = det_vis_fn(vis_frame_id)

                if det_vis_frame is not None:
                    det_vis_frames[vis_frame_id] = det_vis_frame
                    cv2.imshow(det_win_name, det_vis_frame)
                else:
                    vis_frame_id = _vis_frame_id

            if track_win_name is not None:
                if min_frame_id <= vis_frame_id <= max_frame_id:
                    try:
                        track_vis_frame = track_vis_frames[vis_frame_id]
                    except KeyError:
                        track_vis_frame = track_vis_fn(vis_frame_id, show=False)
                    cv2.imshow(track_win_name, track_vis_frame)
    else:
        raise AssertionError(f'invalid mode: {mode}')

    return vis_frame_id


def vis_dets(frame, dets, enable_mask, header, fmt, width=0, height=0, size=None):
    det_bboxes = [obj['bbox'] for obj in dets]
    if enable_mask:
        det_masks = [obj['mask'] for obj in dets]
    else:
        det_masks = None
    det_labels = [obj['label'] for obj in dets]
    det_cols = [obj['col'] for obj in dets]

    det_ids = [obj['target_id'] if obj['target_id'] is not None and obj['target_id'] > 0 else None for obj in dets]

    dets_vis_frame = np.copy(frame)

    # cols = [
    #     'green' for _ in dets
    # ]

    draw_boxes(dets_vis_frame, det_bboxes, colors=det_cols, ids=det_ids, font_size=1.5,
               masks=det_masks,
               text_cols=det_cols,
               labels=det_labels)

    dets_vis_frame = resize_ar(dets_vis_frame, width=width, height=height, size=size, only_shrink=1)

    dets_vis_frame = annotate_and_show(header, dets_vis_frame, only_annotate=1, n_modules=0, fmt=fmt)

    return dets_vis_frame


def add_det_to_track(frame_id, det, track, frame_id_to_tracks, target_id_to_tracks, replace):
    """

    :param int frame_id:
    :param dict det:
    :param Track track:
    :param dict frame_id_to_tracks:
    :param dict target_id_to_tracks:
    :return:
    """
    bbox = det['bbox']
    xmin, ymin, w, h = bbox
    area = w * h

    w_old, h_old, area_old = track.get_node(frame_id - 1).dims

    w_ratio, h_ratio, area_ratio = abs((w - w_old) / w_old), abs((h - h_old) / h_old), abs(
        (area - area_old) / area_old)

    dims_ratio = [w_ratio, h_ratio, area_ratio]

    track_node = TrackNode(
        frame_id=frame_id,
        bbox=bbox,
        mask=det['mask'],
        score=det['score'],
        source=det['source'],
        difficult=det['difficult'],
        dims_ratio=dims_ratio,
        dims=[w, h, area],
        filename=det['filename'],
    )

    if track.get_node(frame_id) is not None:
        """a previously added node already exists for this frame"""

        if replace:
            track.remove_last_node()
        else:
            assert track.get_node(frame_id) is None, f"track node for frame {frame_id} " \
                f"already exists in target {track.target_id}"

    track.add_node(track_node)

    parents = det['parents']
    children = det['children']

    if parents is not None:
        for parent in parents:
            track.add_parent(parent)

    if children is not None:
        for child in children:
            track.add_child(child)

    if frame_id not in frame_id_to_tracks:
        frame_id_to_tracks[frame_id] = {}

    frame_id_to_tracks[frame_id][track.target_id] = track

    assert target_id_to_tracks[track.target_id] is track, "mismatch between target_id_to_tracks and track"


def find_next_ambiguous_track(frame_id, tracks, processed_tracks, dim_thresholds):
    for track_id, track in enumerate(tracks):

        target_id = track.target_id

        if target_id in processed_tracks:
            continue

        processed_tracks.append(target_id)

        if len(track.nodes) == 1:
            """ambiguity is based on the setting change in size from one frame to the next 
            so it cannot be determined for a track that only exists in a single claim"""
            continue

        curr_node = track.get_node(frame_id)  # type: TrackNode
        prev_node = track.get_node(frame_id - 1)  # type: TrackNode

        w, h, area = curr_node.dims
        w_ratio, h_ratio, area_ratio = curr_node.dims_ratio
        w_old, h_old, area_old = prev_node.dims

        iou = np.empty((1, 1))
        ioa_1 = np.empty((1, 1))
        ioa_2 = np.empty((1, 1))

        curr_bbox = np.asarray(curr_node.bbox).reshape((1, 4))
        prev_bbox = np.asarray(prev_node.bbox).reshape((1, 4))

        compute_overlap(iou, ioa_1, ioa_2, curr_bbox, prev_bbox)

        iou = iou.item()
        ioa_1 = ioa_1.item()
        ioa_2 = ioa_2.item()

        msg = ''
        if w_ratio > dim_thresholds[0]:
            msg = f'Excessive change in width in target {target_id}:: ' \
                f'{w:.0f} to {w_old:.0f} : {w_ratio:.1f}'

        if h_ratio > dim_thresholds[1]:
            msg = f'Excessive change in height in target {target_id}:: ' \
                f'{h:.0f} to {h_old:.0f} : {h_ratio:.1f}'

        if area_ratio > dim_thresholds[2]:
            msg = f'Excessive change in area in target {target_id}:: ' \
                f'{area:.0f} to {area_old:.0f} : {area_ratio:.1f}'

        if msg:
            msg += f' (iou: {iou:.2f} ioa_1: {ioa_1:.2f} ioa_2: {ioa_2:.2f})'

            return track_id, msg

    return None


def add_tracks(frame_id, dets, det_ids, existing_target_ids, frame_id_to_tracks, target_id_to_tracks,
               det_to_target_id, parent_ids=None, allow_no_id=False, force_new_tracks=False):
    next_target_id = max(existing_target_ids) + 1
    new_target_ids = []

    target_to_det_id = {v: k for k, v in det_to_target_id.items()}

    for det_id in det_ids:
        det = dets[det_id]

        if force_new_tracks:
            target_id = next_target_id
            next_target_id += 1
        else:
            target_id = det['target_id']

            if target_id is None or target_id <= 0:
                assert allow_no_id, 'detection without ID not allowed to be added as a track'
                target_id = next_target_id
                next_target_id += 1

        new_target_ids.append(target_id)

        assert target_id not in target_to_det_id, f"target {target_id} already " \
            f"associated with det {target_to_det_id[target_id]}"

        target_to_det_id[target_id] = det_id
        det_to_target_id[det_id] = target_id

        bbox = det['bbox']
        xmin, ymin, w, h = bbox
        area = w * h

        if frame_id not in frame_id_to_tracks:
            frame_id_to_tracks[frame_id] = {}

        parents = det['parents']
        children = det['children']

        try:
            """try adding node to existing track"""
            track = target_id_to_tracks[target_id]
        except KeyError:
            """create new track"""
            if parents is not None:
                parent_cols = [target_id_to_tracks[parent].col for parent in parents]
                parent_labels = [target_id_to_tracks[parent].label for parent in parents]
                """set this track's label to be the same as its parents"""
                unique_labels = list(set(parent_labels))
                assert len(unique_labels) == 1, f"mismatching parent labels : {parent_labels}"

                label = parent_labels[0]
                col = parent_cols[0]
            else:
                label, col = det['label'], det['col']

            print(f'creating new track with target_id {target_id}')

            track = Track(target_id, label, col)
            target_id_to_tracks[target_id] = track

            frame_id_to_tracks[frame_id][target_id] = track
            dims_ratio = [0, 0, 0]
        else:
            """since track exists in frame_id_to_tracks, it should also exist in target_id_to_tracks"""
            assert target_id in target_id_to_tracks and \
                   target_id_to_tracks[target_id] is track, \
                "mismatch between the states of frame_id_to_tracks and target_id_to_tracks"

            w_old, h_old, area_old = track.get_node(frame_id - 1).dims

            w_ratio, h_ratio, area_ratio = abs((w - w_old) / w_old), abs((h - h_old) / h_old), abs(
                (area - area_old) / area_old)

            dims_ratio = [w_ratio, h_ratio, area_ratio]

        if target_id not in frame_id_to_tracks[frame_id]:
            frame_id_to_tracks[frame_id][target_id] = track

        if parents is not None:
            assert target_id not in parents, "a target cannot be its own parent"

            for parent in parents:
                track.add_parent(parent)

                parent_track = target_id_to_tracks[parent]
                if target_id not in parent_track.children:
                    print(f'adding missing child track {target_id} to track {parent}')
                    parent_track.add_child(target_id)

        if children is not None:
            assert target_id not in children, "a target cannot be its own child"

            for child in children:
                track.add_child(child)

                try:
                    child_track = target_id_to_tracks[child]
                except KeyError:
                    """child_track does not exist yet - parent info has been read directly from xml"""
                    pass
                else:
                    if target_id not in child_track.parents:
                        print(f'adding missing parent track {target_id} to track {child}')
                        child_track.add_parent(target_id)

        track_node = TrackNode(
            frame_id=frame_id,
            bbox=bbox,
            mask=det['mask'],
            score=det['score'],
            source=det['source'],
            difficult=det['difficult'],
            dims_ratio=dims_ratio,
            dims=[w, h, area],
            filename=det['filename'],
        )
        track.add_node(track_node)

        if parent_ids is not None:
            """add each parent to this track and vice versa"""
            parent_labels = []
            parent_tracks = []
            for parent_id in parent_ids:
                track.add_parent(parent_id)
                parent_track = target_id_to_tracks[parent_id]  # type: Track
                parent_track.add_child(target_id)

                parent_tracks.append(parent_track)
                parent_labels.append(parent_track.label)

            """set this track's label to be the same as its parents"""
            unique_labels = list(set(parent_labels))
            assert len(unique_labels) == 1, f"mismatching parent labels for track {target_id}: {parent_labels}"

            track.set_col(parent_tracks[0].col)
            track.set_label(unique_labels[0])

    existing_target_ids = sorted(list(set(existing_target_ids + new_target_ids)))
    return new_target_ids, existing_target_ids


def associate_objects(frame_id, tracks, all_dets, det_ids, method, iou_threshold, vis, frame, fmt):
    """

    :param list tracks:
    :param list all_dets:
    :param float iou_threshold:
    :param int vis:
    :param np.ndarray frame:

    :param int method:  0: associate each detection to the max overlapping annotation
                    1: same as 0 except that the unique GT association constraint is also
                        applied, i.e. an annotation is associated with a detection only if it has not already
                        been associated with some other detection
                    2: iterative mutual maximum overlap between detections and annotations,
                    3: associate using Hungarian algorithm
    :return:
    """

    """all detections to be associated with tracks"""
    dets = [all_dets[det_id] for det_id in det_ids]

    det_bboxes = [det['bbox'] for det in dets]
    # det_masks = [det['mask'] for det in dets]

    """last box in each existing track"""
    track_bboxes = [track.get_node(frame_id).bbox for track in tracks]

    track_bboxes = np.asarray(track_bboxes)
    det_bboxes = np.asarray(det_bboxes)

    n_det = len(dets)
    n_ann = len(tracks)

    if method < 2:
        """associate each det to max overlapping GT"""
        associated_gts = []
        det_to_gt = {}
        gt_to_det = {}

        unassociated_dets = list(range(n_det))
        unassociated_gts = list(range(n_ann))

        for _det_id in range(n_det):

            curr_det_bbox = det_bboxes[_det_id]

            # curr_ann_data = _curr_ann_data.squeeze().reshape((-1, 10))

            """**************************************************************************"""
            """would not work"""
            """**************************************************************************"""
            max_iou, max_iou_idx = get_max_overlap_obj(track_bboxes, curr_det_bbox)

            if (method == 0 or max_iou_idx not in associated_gts) and \
                    max_iou >= iou_threshold:
                det_to_gt[_det_id] = max_iou_idx
                gt_to_det[max_iou_idx] = _det_id
                associated_gts.append(max_iou_idx)

                unassociated_dets.remove(_det_id)
                unassociated_gts.remove(max_iou_idx)

            if vis == 2:
                frame_disp = np.copy(frame)
                draw_box(frame_disp, curr_det_bbox, color='blue')

                for _idx in range(track_bboxes.shape[0]):
                    col = 'green' if _idx == max_iou_idx else 'red'
                    draw_box(frame_disp, track_bboxes[_idx, :], color=col)

                annotate_and_show('_get_target_status', frame_disp, text='max_iou: {:.2f}'.format(max_iou), fmt=fmt)
    else:
        if method == 3:
            use_hungarian = 1
        else:
            use_hungarian = 0

        det_to_gt, gt_to_det, unassociated_dets, unassociated_gts = find_associations(
            frame, det_bboxes, track_bboxes, iou_threshold, use_hungarian=use_hungarian, vis=vis)

    """convert relative to absolute det indexing"""
    unassociated_dets = [det_ids[k] for k in unassociated_dets]
    det_to_gt = {det_ids[k]: v for k, v in det_to_gt.items()}
    gt_to_det = {k: det_ids[v] for k, v in gt_to_det.items()}

    return det_to_gt, gt_to_det, unassociated_dets, unassociated_gts


def write_xml(frame, frame_id, tracks, xml_file_path, seq_path, filename):
    image_shape = list(frame.shape)
    writer = PascalVocWriter(seq_path, filename, image_shape, enable_hierarchy=True)
    # print()

    for track_id, track in enumerate(tracks):
        track_node = track.get_node(frame_id)  # type: TrackNode

        xmin, ymin, w, h = track_node.bbox

        xmax = xmin + w
        ymax = ymin + h

        mask = track_node.mask

        writer.addBndBox(xmin, ymin, xmax, ymax,
                         mask=mask,
                         name=track.label,
                         id_number=track.target_id,
                         difficult=track_node.difficult,
                         bbox_source=track_node.source,
                         score=track_node.score,
                         parents=track.parents,
                         children=track.children,
                         )

    writer.save(targetFile=xml_file_path)


def read_xml(file, enable_mask, sources_to_include, sources_to_exclude, img_ext, class_to_cols, ignore_invalid_class):
    xml_reader = PascalVocReader(file, enable_hierarchy=True)

    filename = os.path.splitext(os.path.basename(file))[0] + '.{}'.format(img_ext)
    # filename_from_xml = xml_reader.filename

    boxes = []
    shapes = xml_reader.getShapes()
    for shape in shapes:

        label, points, _, _, difficult, bbox_source, target_id, score, mask_pts, _, parents, children = shape

        try:
            col = class_to_cols[label]
        except KeyError:
            msg = f'invalid class: {label}'
            if ignore_invalid_class:
                print(msg)
                continue
            else:
                raise AssertionError(msg)

        if target_id is not None and target_id <= 0:
            target_id = None

        if sources_to_include and bbox_source not in sources_to_include:
            continue

        if sources_to_exclude and bbox_source in sources_to_exclude:
            continue

        xmin, ymin = points[0]
        xmax, ymax = points[2]

        if enable_mask:
            try:
                mask_pts = np.asarray([[x, y] for x, y, _ in mask_pts])
            except TypeError:
                if enable_mask == 1:
                    print('ignoring object without mask')
                    continue
                else:
                    raise AssertionError('mask not found in object')
            else:
                xmin, ymin = np.amin(mask_pts, axis=0)
                xmax, ymax = np.amax(mask_pts, axis=0)
        else:
            mask_pts = None

        w, h = xmax - xmin, ymax - ymin

        bbox = np.asarray([xmin, ymin, w, h])

        box = dict(
            difficult=difficult,
            source=bbox_source,
            score=score,
            target_id=target_id,
            filename=filename,
            label=label,
            col=col,
            mask=mask_pts,
            parents=parents,
            children=children,
            bbox=bbox
        )
        boxes.append(box)

    return boxes


def add_secondary_dets(all_dets, all_sec_dets, sec_iou_thresh, start_frame):
    n_frames = len(all_dets)

    assert len(all_dets) == n_frames, "Mismatch between the lengths of primary and secondary detections"

    for frame_id in range(start_frame, n_frames):
        dets, sec_dets = all_dets[frame_id], all_sec_dets[frame_id]

        n_dets = len(dets)
        n_sec_dets = len(sec_dets)

        det_bboxes = [det['bbox'] for det in dets]
        sec_det_bboxes = [sec_det['bbox'] for sec_det in sec_dets]

        """check for duplicate detections"""
        iou = np.empty((n_dets, n_sec_dets))
        compute_overlaps_multi(iou, None, None, np.asarray(det_bboxes), np.asarray(sec_det_bboxes))

        max_sec_iou = np.max(iou, axis=0)

        n_added_dets = 0

        for sec_det_id, sec_det in enumerate(sec_dets):
            if max_sec_iou[sec_det_id] < sec_iou_thresh:
                print(f'\tframe {frame_id} :: adding a secondary det with max iou: {max_sec_iou[sec_det_id]}')

                dets.append(sec_det)
                n_added_dets += 1

        if n_added_dets:
            print(f'\nframe {frame_id} :: added {n_added_dets} secondary dets\n')


def get_xml_files(seq_path, inverted, xml_dir='annotations', allow_empty=False, backup=1):
    xml_dir_path = os.path.join(seq_path, xml_dir)

    if not xml_dir_path or not os.path.isdir(xml_dir_path):
        msg = f'Folder containing the annotation XML files does not exist: {xml_dir_path}'
        if not allow_empty:
            raise IOError(msg)
        print(msg)
        return None

    xml_files = glob.glob(os.path.join(xml_dir_path, '*.xml'))
    n_files = len(xml_files)
    if n_files == 0:
        msg = f'No annotation XML files found in {xml_dir_path}'
        if not allow_empty:
            raise IOError(msg)

        print(msg)
        return None

    if backup:
        time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
        backup_annotations = f'{xml_dir_path}__{time_stamp}'
        print(f'backing up annotations to {backup_annotations}')
        shutil.make_archive(backup_annotations, 'zip', root_dir=seq_path, base_dir='annotations')

    def getint(fn):
        basename = os.path.basename(fn)
        num = re.sub("\D", "", basename)
        try:
            return int(num)
        except:
            return 0

    if len(xml_files) > 0:
        xml_files = sorted(xml_files, key=getint)

    if inverted:
        print('inverting xml_files')
        xml_files = xml_files[::-1]

    print(f'Found {n_files} xml files at {xml_dir_path}')

    return xml_files, xml_dir_path, n_files


def run(seq_path, params, class_to_color):
    """

    :param str seq_path:
    :param Params params:
    :param dict class_to_color:
    :return:
    """

    xml_files, xml_dir_path, n_files = get_xml_files(seq_path, params.inverted,
                                                     xml_dir='annotations', allow_empty=False,
                                                     backup=params.write_xml)

    if not params.write_xml:
        print('write_xml is disabled')

    if params.start_id == -1:
        params.start_id = len(xml_files) - 1

    if params.start_id > 0:
        print(f'starting from file {params.start_id}')
        # xml_files = xml_files[params.start_id:]

    sources_to_exclude = [k[1:] for k in params.sources_to_include if k.startswith('!')]
    sources_to_include = [k for k in params.sources_to_include if not k.startswith('!')]

    if sources_to_include:
        print('Including only boxes from following sources: {}'.format(sources_to_include))

    if sources_to_exclude:
        print('Excluding boxes from following sources: {}'.format(sources_to_exclude))

    print('reading all xml_files...')
    all_dets = [
        read_xml(file, params.enable_mask, sources_to_include, sources_to_exclude, params.img_ext,
                 class_to_color, params.ignore_invalid_class)
        for file in tqdm(xml_files)]

    n_frames = len(all_dets)

    assert n_frames == len(xml_files), "mismatch between n_frames and xml_files"

    if params.sec_start_frame >= 0:

        sec_start_frame = max(params.sec_start_frame, params.start_id)

        annotations_id = 0

        while True:
            annotations_id += 1
            xml_dir = f'annotations_{annotations_id}'
            ret = get_xml_files(seq_path, params.inverted, xml_dir=xml_dir, allow_empty=True, backup=params.write_xml)
            if ret is None:
                break

            sec_xml_files, _, _ = ret

            assert n_frames == len(sec_xml_files), "mismatch between n_frames and sec_xml_files"

            print(f'adding secondary detections from {xml_dir} starting with frame {sec_start_frame}')

            all_sec_dets = [
                read_xml(file, params.enable_mask, sources_to_include, sources_to_exclude, params.img_ext,
                         class_to_color, params.ignore_invalid_class)
                for file in tqdm(sec_xml_files)]

            add_secondary_dets(all_dets, all_sec_dets, params.sec_iou_thresh, sec_start_frame)

    track_win_name = 'tracks'
    det_win_name = 'detections'

    backup_dets = copy.deepcopy(all_dets)

    """track dets from the previous frame into the current frame"""
    file_names = {}
    file_paths = {}
    frames = {}
    track_vis_frames = {}
    det_vis_frames = {}

    frame_id_to_tracks = {
        frame_id: {} for frame_id in range(n_frames)
    }
    target_id_to_tracks = {}

    existing_ids = [0, ]

    existing_ids += [det['target_id'] for frame_id in range(n_frames) for det in all_dets[frame_id]
                     if det['target_id'] is not None and det['target_id'] > 0]

    existing_ids_unique = sorted(list(set(existing_ids)))

    seq_name = os.path.basename(seq_path)

    out_dir_path = linux_path('log', seq_name, 'track')
    os.makedirs(out_dir_path, exist_ok=1)

    if params.save_track:
        print(f'saving tracking images to {out_dir_path}')

    save_path = None

    if params.start_id < 0:
        params.start_id = n_frames

    if params.max_start_frame <= 0:
        params.max_start_frame = n_frames

    custom_frame_id = [None, ]
    frame_id = -1

    # thread = threading.Thread(target=read_all_frames,
    #                           args=(seq_path, n_frames, frames, file_paths, file_names, all_dets, xml_files))
    # thread.start()

    det_vis_fn = functools.partial(
        get_det_vis_frame,
        all_dets=all_dets,
        det_vis_frames=det_vis_frames,
        seq_name=seq_name,
        seq_path=seq_path,
        xml_files=xml_files,
        frame_paths=file_paths,
        frames=frames,
        enable_mask=params.enable_mask,
        vis_size=params.vis_size,
        fmt=params.ui.fmt,
    )

    track_vis_fn = functools.partial(
        update_track_vis_frame,
        seq_name=seq_name,
        frames=frames,
        frame_paths=file_paths,
        frame_id_to_tracks=frame_id_to_tracks,
        xml_files=xml_files,
        cols=params.cols,
        track_vis_frames=track_vis_frames,
        track_win_name=track_win_name,
        vis_size=params.vis_size,
        fmt=params.ui.fmt,
    )

    thread = threading.Thread(target=get_all_det_vis_frames,
                              args=(n_frames, det_vis_fn, params.n_proc))
    thread.start()

    while frame_id < n_frames - 1:

        if custom_frame_id[0] is not None:
            retro_frame_id = custom_frame_id[0]  # type: int
            assert retro_frame_id < frame_id, "retrospective modification can only be done for a past frame"

            for _frame_id in range(retro_frame_id, frame_id):
                all_dets[_frame_id] = copy.copy(backup_dets[_frame_id])

            frame_id = custom_frame_id[0]

            custom_frame_id[0] = None
        else:
            frame_id += 1

        new_target_ids = []
        det_to_target_id = {}

        xml_file_path = xml_files[frame_id]
        xml_file_name = os.path.basename(xml_file_path)
        header = f'{seq_name} {frame_id}: {xml_file_name}'

        prev_tracks_changed = False

        dets = get_dets(frame_id, all_dets, xml_files)

        n_dets = len(dets)

        filename = dets[0]['filename']
        filepath = linux_path(seq_path, filename)

        file_names[frame_id] = filename
        file_paths[frame_id] = filepath

        save_path = linux_path(out_dir_path, file_names[frame_id])

        if frame_id < params.start_id:
            print(f'adding tracks from frame {frame_id}: {xml_file_name}')

            if params.ui.vis_before_start:
                det_vis_frame = det_vis_fn(frame_id)
            else:
                det_vis_frame = None

            det_ids = list(range(len(dets)))
            new_target_ids, existing_ids_unique = add_tracks(
                frame_id, dets, det_ids, existing_ids_unique, frame_id_to_tracks,
                target_id_to_tracks, det_to_target_id, allow_no_id=False)

            if params.save_track or params.ui.vis_before_start:
                track_vis_fn(frame_id, save_path=save_path, show=params.ui.vis_before_start)

            if params.ui.vis_before_start:
                cv2.imshow(det_win_name, det_vis_frame)
                keyboard_handler(params.ui)
            continue

        det_vis_frame = det_vis_fn(frame_id)

        frame = get_frame(frame_id, frames, file_paths)

        if params.resize_to:
            frame_h, frame_w = frame.shape[:2]
            res_h, res_w = params.resize_to[:2]
            params.resize_factor = min(res_h / frame_h, res_w / frame_w)

        label = dets[0]['label']
        col = dets[0]['col']

        new_det_info = [filename, label, col]

        if params.ui.vis:
            cv2.imshow(det_win_name, det_vis_frame)

        if not isinstance(params.dim_thresholds, (tuple, list)):
            params.dim_thresholds = [params.dim_thresholds, ] * 3

        perform_association = True

        unassociated_det_ids = []

        non_target_det_ids = []
        target_det_ids = []

        """filter out detections with existing IDs"""
        for det_id, det in enumerate(dets):
            if det['target_id'] is None:
                non_target_det_ids.append(det_id)
            else:
                target_det_ids.append(det_id)

        n_non_target_dets = len(non_target_det_ids)

        if n_non_target_dets == 0:
            print(f'No detections without target IDs found for frame {frame_id}: {xml_file_path}')
            perform_association = False
        else:
            temporal_fn = functools.partial(
                temporal_interaction,
                max_frame_id=frame_id,
                min_frame_id=0,
                track_win_name=None,
                det_win_name=det_win_name,
                track_vis_frames=None,
                det_vis_frames=det_vis_frames,
                mode=0,
                det_vis_fn=det_vis_fn,
                track_vis_fn=track_vis_fn,
                custom_frame_id=custom_frame_id,
            )

            if params.user_check and frame_id > params.user_check_start_frame:
                dets_to_remove = handle_duplicate_dets(frame, frame_id, dets, non_target_det_ids,
                                                       new_det_info,
                                                       temporal_fn,
                                                       det_vis_fn,
                                                       header,
                                                       params.det_iou_threshold,
                                                       params.resize_factor,
                                                       params.enable_mask,
                                                       params.vis_size,
                                                       )
                if dets_to_remove:
                    """filter out duplicate detections"""
                    assert all(k in non_target_det_ids for k in dets_to_remove), \
                        "only non-target dets can be removed as duplicates"

                    det_shift_factors = [len([removed_id for removed_id in dets_to_remove if removed_id < det_id])
                                         for det_id in range(n_dets)]

                    non_target_det_ids = [k - det_shift_factors[k] for k in non_target_det_ids
                                          if k not in dets_to_remove]
                    target_det_ids = [k - det_shift_factors[k] for k in target_det_ids]

                    dets = [obj for k, obj in enumerate(dets) if k not in dets_to_remove]

                    n_non_target_dets = len(non_target_det_ids)

                    if n_non_target_dets == 0:
                        print(f'No valid detections without target IDs found for frame {frame_id}: {xml_file_path}')

        """all tracks from previous frame, if any"""
        try:
            prev_tracks = list(frame_id_to_tracks[frame_id - 1].values())
        except KeyError:
            prev_tracks = []

        if not prev_tracks:
            """no existing tracks for association so add a new track for each valid detection without ID"""

            print(f'adding {n_non_target_dets} detections without IDs from frame {frame_id} as new tracks')
            new_target_ids, existing_ids_unique = add_tracks(
                frame_id, dets, non_target_det_ids, existing_ids_unique,
                frame_id_to_tracks, target_id_to_tracks, det_to_target_id, allow_no_id=True)
            perform_association = False

        target_to_track_id = {
            track.target_id: track_id for track_id, track in enumerate(prev_tracks)
        }

        if target_det_ids:
            """add detections with existing IDs to corresponding tracks or as new tracks if needed"""
            print(f'adding {len(target_det_ids)} detections with IDs from frame {frame_id} as new tracks')
            new_target_ids, existing_ids_unique = add_tracks(
                frame_id,
                dets, target_det_ids,
                existing_ids_unique,
                frame_id_to_tracks,
                target_id_to_tracks,
                det_to_target_id,
                allow_no_id=False)

        manually_associated_track_ids = [target_to_track_id[target_id] for target_id in new_target_ids
                                         if target_id in target_to_track_id]

        all_associated_target_ids = [prev_tracks[track_id].target_id for track_id in manually_associated_track_ids]
        all_associated_tracks = [prev_tracks[track_id] for track_id in manually_associated_track_ids]

        if perform_association:
            tracks_to_be_associated = [track for track_id, track in enumerate(prev_tracks)
                                       if track_id not in manually_associated_track_ids]

            if tracks_to_be_associated and non_target_det_ids:
                """associate unassociated tracks with remaining detections"""
                det_to_gt, gt_to_det, unassociated_det_ids, unassociated_track_ids = associate_objects(
                    frame_id - 1, tracks_to_be_associated, dets, non_target_det_ids,
                    params.method, params.assoc_iou_threshold, 0, frame, fmt=params.ui.fmt)

                det_to_target_id.update(
                    {
                        det_id: tracks_to_be_associated[track_id].target_id for det_id, track_id in det_to_gt.items()
                    }
                )
                for track_id, track in enumerate(tracks_to_be_associated):

                    try:
                        det_id = gt_to_det[track_id]
                    except KeyError:
                        print(f'no associated detection found for track {track_id}')
                        assert track_id in unassociated_track_ids, \
                            "track without associated detection not found in the list of unassociated tracks"
                        continue

                    all_associated_target_ids.append(track.target_id)
                    all_associated_tracks.append(track)

                    add_det_to_track(
                        frame_id, dets[det_id], track, frame_id_to_tracks, target_id_to_tracks, replace=False)

            elif non_target_det_ids:
                """no tracks to associate to"""
                unassociated_det_ids = non_target_det_ids[:]

        all_unassociated_target_ids = [track.target_id for track in prev_tracks
                                       if track.target_id not in all_associated_target_ids]

        if params.ui.vis:
            track_vis_fn(frame_id)
            keyboard_handler(params.ui)

        temporal_fn = functools.partial(
            temporal_interaction,
            max_frame_id=frame_id,
            min_frame_id=0,
            track_win_name=track_win_name,
            det_win_name=det_win_name,
            track_vis_frames=track_vis_frames,
            det_vis_frames=det_vis_frames,
            mode=0,
            det_vis_fn=det_vis_fn,
            track_vis_fn=track_vis_fn,
            custom_frame_id=custom_frame_id,
        )

        fix_det_win_name = 'add or select detections'
        fix_track_win_name = 'select tracks'
        unassoc_fix_win_name = 'unassociated track'
        ambiguous_assoc_win_name = 'ambiguous association'
        unassoc_det_win_name = 'select unassociated dets to start new tracks'

        fix_unassoc_track = ((params.user_check == 2) or
                             (params.user_check and all_unassociated_target_ids)) and \
                            (frame_id >= params.user_check_start_frame)

        if fix_unassoc_track:
            """fix unassociated tracks from previous frame"""
            n_tracks = len(prev_tracks)

            try:
                curr_existing_tracks = list(frame_id_to_tracks[frame_id].values())
            except KeyError:
                curr_existing_tracks = []

            unassoc_id = -1
            while True:
                unassoc_id += 1

                if unassoc_id >= len(all_unassociated_target_ids):
                    break

                target_id = all_unassociated_target_ids[unassoc_id]

                track_id = target_to_track_id[target_id]

                """highlight the current problematic track in magenta"""
                cols = ['cyan', ] * n_tracks
                cols[track_id] = 'magenta'

                track = prev_tracks[track_id]
                assoc_obj = track.get_node_as_det(frame_id - 1)  # type: dict

                child_tracks = [child.target_id for child in curr_existing_tracks if track.target_id in child.parents]

                if child_tracks:
                    print(f'target {track.target_id} divided into targets {child_tracks}')
                    continue

                _header = f'{header} :: unassociated target {track.target_id} from previous frame'

                prev_frame = get_frame(frame_id - 1, frames, file_paths)

                track_vis_frame = vis_tracks(prev_frame, frame_id - 1, prev_tracks, _header, cols=cols,
                                             size=params.vis_size, header_col='hot_pink', fmt=params.ui.fmt)

                cv2.imshow(unassoc_fix_win_name, track_vis_frame)

                """detections to be shown are from the current frame though the track is from the previous frame"""
                det_data = [frame, frame_id, dets, fix_det_win_name, None]

                prev_frame = get_frame(frame_id - 1, frames, file_paths)
                track_data = [prev_frame, frame_id - 1, prev_tracks, fix_track_win_name, None]

                ret = select_boxes_interactively(
                    det_data=det_data, track_data=track_data,
                    assoc_obj=assoc_obj,
                    new_det_info=new_det_info,
                    resize_factor=params.resize_factor,
                    temporal_fn=temporal_fn, all_cols=params.cols, det_vis_fn=det_vis_fn)

                if ret is None:
                    exit()

                det_sel_ids, track_sel_ids = ret

                new_target_ids, existing_ids_unique, prev_tracks_changed = process_user_input(
                    det_sel_ids, track_sel_ids, dets, frame_id,
                    frame_id_to_tracks, target_id_to_tracks, det_to_target_id,
                    all_unassociated_target_ids, unassociated_det_ids, prev_tracks, track_id,
                    existing_ids_unique, track_vis_fn)

            try:
                cv2.destroyWindow(unassoc_fix_win_name)
            except cv2.error:
                """non-existent window"""
                pass

        fix_ambiguous_assoc = ((params.user_check == 2) or
                               # only if association was performed, i.e. if there were dets without IDs
                               (params.user_check and perform_association)) and \
                              (frame_id >= params.user_check_start_frame)
        if fix_ambiguous_assoc:

            """fix ambiguous tracks from current frame one by one to be able to handle possible changes in the tracks
            needed to handle each"""
            processed_tracks = []
            while True:
                curr_tracks = list(frame_id_to_tracks[frame_id].values())
                ret = find_next_ambiguous_track(frame_id, curr_tracks, processed_tracks, params.dim_thresholds)

                if ret is None:
                    """no more ambiguous tracks in this frame"""
                    break

                track_id, msg = ret

                """highlight the current problematic track in magenta"""
                cols = ['cyan', ] * len(curr_tracks)
                cols[track_id] = 'magenta'

                _header = f'{header} :: {msg}'

                track_vis_frame = vis_tracks(frame, frame_id, curr_tracks, _header, cols=cols,
                                             size=params.vis_size, header_col='yellow', fmt=params.ui.fmt)

                cv2.imshow(ambiguous_assoc_win_name, track_vis_frame)

                print(msg)

                det_data = [frame, frame_id, dets, fix_det_win_name, None]

                prev_frame = get_frame(frame_id - 1, frames, file_paths)
                track_data = [prev_frame, frame_id - 1, prev_tracks, fix_track_win_name, None]

                track = prev_tracks[track_id]
                assoc_obj = track.get_node_as_det(frame_id - 1)  # type: TrackNode

                ret = select_boxes_interactively(
                    det_data=det_data, track_data=track_data, new_det_info=new_det_info,
                    assoc_obj=assoc_obj,
                    resize_factor=params.resize_factor,
                    det_vis_fn=det_vis_fn,
                    temporal_fn=temporal_fn,
                    all_cols=params.cols)

                if ret is None:
                    exit()

                det_sel_ids, track_sel_ids = ret

                new_target_ids, existing_ids_unique, prev_tracks_changed = process_user_input(
                    det_sel_ids, track_sel_ids, dets, frame_id,
                    frame_id_to_tracks, target_id_to_tracks, det_to_target_id,
                    all_unassociated_target_ids, unassociated_det_ids, prev_tracks, track_id,
                    existing_ids_unique, track_vis_fn)

                cv2.destroyWindow(ambiguous_assoc_win_name)

        fix_unassoc_dets = params.user_check and \
                           params.max_start_frame > frame_id >= params.user_check_start_frame and \
                           unassociated_det_ids
        if fix_unassoc_dets:
            dets_to_show = [dets[k] for k in unassociated_det_ids]

            vis_win_name = 'unassociated detections'
            fix_win_name = unassoc_det_win_name

            cols = ['cyan', ] * len(dets)
            for _id in unassociated_det_ids:
                cols[_id] = 'magenta'

            _header = f'{header} :: unassociated detections found: {unassociated_det_ids}'

            det_vis_frame = vis_dets(frame, dets_to_show, params.enable_mask, _header, size=params.vis_size)

            cv2.imshow(vis_win_name, det_vis_frame)

            det_data = [frame, frame_id, dets, fix_win_name, cols]
            prev_frame = get_frame(frame_id - 1, frames, file_paths)
            track_data = [prev_frame, frame_id - 1, prev_tracks, fix_track_win_name, None]

            ret = select_boxes_interactively(
                det_data=det_data, track_data=track_data, new_det_info=new_det_info,
                resize_factor=params.resize_factor,
                assoc_obj=None,
                det_vis_fn=det_vis_fn,
                temporal_fn=temporal_fn,
                all_cols=params.cols)

            if ret is None:
                exit()

            det_sel_ids, track_sel_ids = ret

            new_target_ids, existing_ids_unique, prev_tracks_changed = process_user_input(
                det_sel_ids, track_sel_ids, dets, frame_id,
                frame_id_to_tracks, target_id_to_tracks, det_to_target_id,
                all_unassociated_target_ids, unassociated_det_ids, prev_tracks, None,
                existing_ids_unique, track_vis_fn)

            try:
                cv2.destroyWindow(vis_win_name)
            except cv2.error:
                """non-existent window"""
                pass

        custom_fix = params.ui.custom_fix and frame_id > 0 and \
                     not fix_unassoc_dets  # fix_unassoc_dets allows all actions in custom_fix so no need to do both

        if custom_fix:
            _header = f'{header} :: custom fix needed'

            vis_win_name = 'custom_fix'

            cols = ['cyan', ] * len(dets)

            det_vis_frame = vis_dets(frame, dets, params.enable_mask, _header, size=params.vis_size)

            cv2.imshow(vis_win_name, det_vis_frame)

            det_data = [frame, frame_id, dets, fix_det_win_name, cols]

            prev_frame = get_frame(frame_id - 1, frames, file_paths)
            track_data = [prev_frame, frame_id - 1, prev_tracks, fix_track_win_name, None]

            ret = select_boxes_interactively(
                det_data=det_data, track_data=track_data, new_det_info=new_det_info,
                assoc_obj=None,
                resize_factor=params.resize_factor,
                det_vis_fn=det_vis_fn,
                temporal_fn=temporal_fn,
                all_cols=params.cols)

            if ret is None:
                exit()

            det_sel_ids, track_sel_ids = ret

            new_target_ids, existing_ids_unique, prev_tracks_changed = process_user_input(
                det_sel_ids, track_sel_ids, dets, frame_id,
                frame_id_to_tracks, target_id_to_tracks, det_to_target_id,
                all_unassociated_target_ids, unassociated_det_ids, prev_tracks, None,
                existing_ids_unique, track_vis_fn)

            try:
                cv2.destroyWindow(vis_win_name)
            except cv2.error:
                """non-existent window"""
                pass

        curr_tracks = list(frame_id_to_tracks[frame_id].values())

        if prev_tracks_changed:
            prev_tracks = list(frame_id_to_tracks[frame_id - 1].values())

            prev_frame = get_frame(frame_id - 1, frames, file_paths)
            if params.write_xml:
                write_xml(prev_frame, frame_id - 1, prev_tracks, xml_files[frame_id - 1],
                          seq_path, file_names[frame_id - 1])

        if params.write_xml:
            write_xml(frame, frame_id, curr_tracks, xml_file_path, seq_path, filename)

    if params.save_track:
        print(f'saving track images to {out_dir_path}...')

        for frame_id in tqdm(range(params.start_id, n_frames)):
            save_path = linux_path(out_dir_path, file_names[frame_id].replace(params.img_ext, 'png'))
            try:
                track_vis_fn(frame_id, save_path=save_path)
            except KeyError:
                continue

    all_cell_lines = {}
    if params.save_cell_lines:
        target_ids_in_paths = []
        for start_frame_id in range(params.max_start_frame):
            start_tracks = frame_id_to_tracks[start_frame_id]
            for target_id in start_tracks:
                if target_id in target_ids_in_paths:
                    continue
                track = start_tracks[target_id]  # type: Track
                track_paths = track.paths(target_id_to_tracks, target_ids_in_paths)
                all_cell_lines[target_id] = (track, track_paths)

        all_target_ids = set(target_id_to_tracks.keys())
        target_ids_not_in_paths = list(all_target_ids - set(target_ids_in_paths))
        if target_ids_not_in_paths:
            print(f'found target_ids not in any paths: {target_ids_not_in_paths}')

    return all_cell_lines, frames, file_paths, track_vis_frames, det_vis_frames


def main():
    params = Params()

    paramparse.process(params)

    classes = [k.strip() for k in open(params.class_names_path, 'r').readlines() if k.strip()]
    if params.read_colors:
        classes, class_cols = zip(*[k.split('\t') for k in classes])
        class_to_color = {
            _class: class_cols[_class_id]
            for _class_id, _class in enumerate(classes)
        }
    else:
        class_cols = class_to_color = None

    seq_paths = params.seq_paths
    root_dir = params.root_dir
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
        seq_paths.sort()
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
    n_seq = len(seq_paths)
    for seq_path_id, seq_path in enumerate(seq_paths):
        seq_path = seq_path.replace('\\', '/')
        print(f'\n\n{seq_path_id + 1} / {n_seq}: {seq_path}\n\n')
        all_cell_lines, frames, frame_paths, track_vis_frames, det_vis_frames = run(seq_path, params, class_to_color)

        if params.save_cell_lines:
            vis_cell_lines(seq_path, all_cell_lines, frames, frame_paths, params.neighborhood,
                           params.ui.vis, params.ui.fmt)


def vis_cell_lines(seq_path, all_paths, frames, frame_paths, neighborhood, vis, fmt):
    seq_name = os.path.basename(seq_path)

    pause_after_frame = 0

    for target_id in all_paths:
        track, _paths = all_paths[target_id]  # type: Track, list[list[TrackNode]]
        for _path_id, _path in enumerate(_paths):
            out_dir = f'target_id_{target_id}_line_{_path_id}'
            out_dir_path = linux_path('log', seq_name, out_dir)

            out_frame_dir_path = linux_path(out_dir_path, 'frames')
            out_patch_dir_path = linux_path(out_dir_path, 'patches')

            print(f'writing cell lines images to {out_dir_path}')

            os.makedirs(out_frame_dir_path, exist_ok=True)
            os.makedirs(out_patch_dir_path, exist_ok=True)

            for _node in _path:  # type: TrackNode
                bbox = _node.bbox
                frame_id = _node.frame_id
                filename = _node.filename
                base_filename = os.path.basename(filename)

                # filepath = linux_path(seq_path, filename)

                xmin, ymin, w, h = bbox

                xmax, ymax = xmin + w, ymin + h

                frame = get_frame(frame_id, frames, frame_paths)

                # track_vis_frame = track_vis_frames[frame_id]

                header = f'{seq_name}\n\nframe {frame_id}: {base_filename} target {target_id} line {_path_id}'

                vis_frame = vis_tracks(frame, frame_id, None, header=None,
                                       cols=[track.col, ], labels=[track.label, ],
                                       track_nodes=[_node, ], target_ids=[target_id, ], fmt=fmt)

                img_h, img_w = frame.shape[:2]

                assert vis_frame.shape == frame.shape, "mismatch in vis_frame shape"

                start_row = int(max(ymin - neighborhood, 0))
                end_row = int(min(ymax + neighborhood, img_h))
                start_col = int(max(xmin - neighborhood, 0))
                end_col = int(min(xmax + neighborhood, img_w))
                vis_patch = vis_frame[start_row:end_row, start_col:end_col, ...]
                vis_patch = resize_ar(vis_patch, height=360, only_border=0, only_shrink=1)

                vis_frame = resize_ar(vis_frame, height=960)
                vis_frame = annotate_and_show(header, vis_frame, only_annotate=1, n_modules=0, fmt=fmt)

                out_img_path = linux_path(out_frame_dir_path, filename)
                out_patch_path = linux_path(out_patch_dir_path, filename)

                cv2.imwrite(out_img_path, vis_frame)
                cv2.imwrite(out_patch_path, vis_patch)

                if vis:
                    cv2.imshow('vis_frame', vis_frame)
                    cv2.imshow('vis_patch', vis_patch)

                    k = cv2.waitKey(1 - pause_after_frame)

                    # print(f'k: {k}')
                    # print(f'params.ui.pause_after_frame: {params.ui.pause_after_frame}')

                    if k == 32:
                        pause_after_frame = 1 - pause_after_frame
                    elif k == 27:
                        cv2.destroyAllWindows()
                        exit()
    # print()


if __name__ == '__main__':
    main()
