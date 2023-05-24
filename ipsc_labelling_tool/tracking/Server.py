# import tensorflow.python.util.deprecation as deprecation
#
# deprecation._PRINT_DEPRECATION_WARNINGS = False

import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(os.path.dirname(__file__))

import numpy as np
import socket, paramiko
import time
import cv2
import threading
import multiprocessing
import pandas as pd
from datetime import datetime

import logging
import paramparse


def profile(self, message, *args, **kws):
    if self.isEnabledFor(PROFILE_LEVEL_NUM):
        self._log(PROFILE_LEVEL_NUM, message, args, **kws)


from PatchTracker import PatchTracker, PatchTrackerParams
from Visualizer import Visualizer, VisualizerParams

from utils.netio import send_msg_to_connection, recv_from_connection

from libs.frames_readers import get_frames_reader, DirectoryReader, VideoReader
from libs.netio import bindToPort

sys.path.append('../..')


def send(curr_frame, out_bbox, label, request_path, frame_id, id_number,
         request_port, masks=None):
    # print('frame_id: {}, out_bbox: {}'.format(frame_id, out_bbox))

    if len(curr_frame.shape) == 3:
        height, width, channels = curr_frame.shape
    else:
        height, width = curr_frame.shape
        channels = 1

    tracking_result = dict(
        action="add_bboxes",
        path=request_path,
        frame_number=frame_id,
        width=width,
        height=height,
        channel=channels,
        bboxes=[out_bbox],
        scores=[0],
        labels=[label],
        id_numbers=[id_number],
        bbox_source="single_object_tracker",
        last_frame_number=frame_id - 1,
        trigger_tracking_request=False,
        num_frames=1,
        # port=request_port,
    )
    if masks is not None:
        tracking_result['masks'] = [masks, ]

    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(('localhost', request_port))
        except ConnectionRefusedError as e:
            print('connection refused: {}'.format(e))
            continue
        else:
            send_msg_to_connection(tracking_result, sock)
            sock.close()
            break


def patch_tracking(params, request, frames_reader, logger):
    """

    :param Server.Params params:
    :param dict request:
    :param frames_reader:
    :param logger:
    :return:
    """
    script_dir = os.path.dirname(__file__)
    # print('changing working directory to {}'.format(script_dir))
    os.chdir(script_dir)

    if params.mode == 2:
        sys.stdout.write('@@@ Starting tracker\n')
        sys.stdout.flush()

    assert request is not None, "request is None"

    request_path = request["path"]
    request_roi = request["roi"]
    id_number = request['id_number']
    init_frame_id = request["frame_number"]
    init_bbox = request["bbox"]
    init_bbox_list = [
        int(init_bbox['xmin']),
        int(init_bbox['ymin']),
        int(init_bbox['xmax']),
        int(init_bbox['ymax']),
    ]
    label = request['label']
    request_port = request["port"]

    show_only = (params.mode == 1)
    tracker = PatchTracker(params.patch_tracker, logger, id_number, label, show_only=show_only)
    if not tracker.is_created:
        return

    init_frame = frames_reader.get_frame(init_frame_id)
    tracker.initialize(init_frame, init_bbox)
    if not tracker.is_initialized:
        logger.error('Tracker initialization was unsuccessful')
        return

    n_frames = frames_reader.num_frames

    if params.end_frame_id >= init_frame_id:
        end_frame_id = params.end_frame_id
    else:
        end_frame_id = n_frames - 1

    save_path = ''
    if params.save_dir:
        file_path = frames_reader.get_file_path()

        assert file_path is not None, "file_path is None"

        if isinstance(frames_reader, VideoReader):
            seq_name = os.path.splitext(os.path.basename(file_path))[0]
        elif isinstance(frames_reader, DirectoryReader):
            seq_name = os.path.basename(os.path.dirname(file_path))

        time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")

        save_path = os.path.join(params.save_dir, '{}_target_{}_{}'.format(seq_name, id_number, time_stamp))

        os.makedirs(save_path, exist_ok=1)

        if params.save_csv:
            save_csv_path = os.path.join(save_path, 'annotations.csv')
            print('Saving results csv to {}'.format(save_csv_path))

    if params.track_init_frame:
        start_frame_id = init_frame_id
    else:
        start_frame_id = init_frame_id + 1

    csv_raw = []
    if label is None:
        label = 'generic'

    for frame_id in range(start_frame_id, end_frame_id + 1):
        try:
            curr_frame = frames_reader.get_frame(frame_id)
        except IOError as e:
            print('{}'.format(e))
            break

        file_path = frames_reader.get_file_path()
        filename = os.path.basename(file_path)

        try:
            fps = tracker.update(curr_frame, frame_id)
        except KeyboardInterrupt:
            break

        if tracker.out_bbox is None:
            # self.logger.error('Tracker update was unsuccessful')
            break

        save_mask_path_bin = None

        if save_path:

            if tracker.curr_mask_cropped is not None:

                filename_no_ext = os.path.splitext(filename)[0]

                out_fname = '{}_frame_{}'.format(filename_no_ext, frame_id)

                if params.save_mask_img:
                    mask_filename = out_fname + '.png'
                    save_mask_path = os.path.join(save_path, mask_filename)
                    curr_mask_norm = (tracker.curr_mask_cropped * 255.0).astype(np.uint8)
                    cv2.imwrite(save_mask_path, curr_mask_norm)

                mask_filename_bin = out_fname + '.npy'
                save_mask_path_bin = os.path.join(save_path, mask_filename_bin)
                np.save(save_mask_path_bin, tracker.curr_mask_cropped)

            if params.save_csv:
                orig_height, orig_width = curr_frame.shape[:2]
                xmin = tracker.out_bbox['xmin']
                xmax = tracker.out_bbox['xmax']
                ymin = tracker.out_bbox['ymin']
                ymax = tracker.out_bbox['ymax']

                raw_data = {
                    'filename': filename,
                    'width': orig_width,
                    'height': orig_height,
                    'class': label,
                    'xmin': int(xmin),
                    'ymin': int(ymin),
                    'xmax': int(xmax),
                    'ymax': int(ymax),
                    'confidence': tracker.score
                }
                csv_raw.append(raw_data)

        if request_port is not None:
            mask = None
            if save_mask_path_bin is not None:
                mask = os.path.abspath(save_mask_path_bin)
            # if tracker.curr_mask_cropped is not None:
            #     # mask = np.expand_dims(tracker.curr_mask, axis=0).tolist()
            #     mask = tracker.curr_mask_cropped.tolist()
            # else:
            #     mask = None

            send(curr_frame, tracker.out_bbox, label, request_path, frame_id,
                 id_number, request_port, masks=mask)
        # self.single_object_tracking_results.append(tracking_result)

        if tracker.is_terminated:
            break

    sys.stdout.write('Closing tracker...\n')
    sys.stdout.flush()

    tracker.close()

    if save_path and params.save_csv:
        df = pd.DataFrame(csv_raw)
        df.to_csv(save_csv_path)


def sortKey(fname):
    fname = os.path.splitext(os.path.basename(fname))[0]
    # print('fname: ', fname)
    # split_fname = fname.split('_')
    # print('split_fname: ', split_fname)

    # nums = [int(s) for s in fname.split('_') if s.isdigit()]
    # non_nums = [s for s in fname.split('_') if not s.isdigit()]

    split_list = fname.split('_')
    key = ''

    for s in split_list:
        if s.isdigit():
            if not key:
                key = '{:08d}'.format(int(s))
            else:
                key = '{}_{:08d}'.format(key, int(s))
        else:
            if not key:
                key = s
            else:
                key = '{}_{}'.format(key, s)

    # for non_num in non_nums:
    #     if not key:
    #         key = non_num
    #     else:
    #         key = '{}_{}'.format(key, non_num)
    # for num in nums:
    #     if not key:
    #         key = '{:08d}'.format(num)
    #     else:
    #         key = '{}_{:08d}'.format(key, num)

    # try:
    #     key = nums[-1]
    # except IndexError:
    #     return fname

    # print('fname: {}, key: {}'.format(fname, key))
    return key


# from utils.frames_readers import get_frames_reader


# from functools import partial

class InitInfo:
    """
    :type roi: list
    """

    def __init__(self):
        self.bbox = [0, 0, 0, 0]
        self.bbox_source = ''
        self.path = ''
        self.label = ''
        self.id_number = 0
        self.port = 3000
        self.frame_number = 0
        self.num_frames = 0
        self.roi = None


class Server:
    """
    :type params: Server.Params
    :type logger: logging.RootLogger
    """

    def __init__(self, params, _logger):
        """
        :type params: Server.Params
        :type _logger: logging.RootLogger
        :rtype: None
        """

        self.params = params
        self.logger = _logger

        self.request_dict = {}
        self.request_list = []

        self.current_path = None
        self.frames_reader = None
        self.trainer = None
        self.tester = None
        self.visualizer = None
        self.enable_visualization = False
        self.traj_data = []

        self.trained_target = None
        self.tracking_res = None
        self.index_to_name_map = None

        self.max_frame_id = -1
        self.frame_id = -1

        self.pid = os.getpid()

        self.request_lock = threading.Lock()

        # create parsers for real time parameter manipulation
        # self.parser = argparse.ArgumentParser()
        # addParamsToParser(self.parser, self.params)

        self.client = None
        self.channel = None
        self._stdout = None
        self.remote_output = None

        if self.params.mode == 0:
            self.logger.info('Running in local execution mode')
        elif self.params.mode == 1:
            self.logger.info('Running in remote execution mode')
            self.connectToExecutionServer()
        elif self.params.mode == 2:
            self.logger.info('Running patch tracker directly')

        # self.patch_tracking_results = []

    class Params:
        """
        :type mode: int
        :type load_path: str
        :type continue_training: int | bool
        :type gate: GateParams
        :type patch_tracker: PatchTrackerParams
        :type visualizer: VisualizerParams
        """

        def __init__(self):
            self.cfg = 'cfg/params.cfg'
            self.mode = 0
            self.wait_timeout = 3
            self.port = 3002
            self.verbose = 0
            self.save_as_bin = 1

            self.remote_path = '/home/abhineet/acamp_code_non_root/labelling_tool/tracking'
            self.remote_cfg = 'params.cfg'
            self.remote_img_root_path = '/home/abhineet/acamp/object_detection/videos'
            self.hostname = ''
            self.username = ''
            self.password = ''

            self.img_path = ''
            self.img_paths = ''
            self.root_dir = ''
            self.save_dir = 'log'
            self.save_csv = 0
            self.save_mask_img = 0
            self.track_init_frame = 1

            self.roi = ''
            self.id_number = 0
            self.init_frame_id = 0
            self.end_frame_id = -1
            self.init_bbox = ''

            self.init = InitInfo()
            self.patch_tracker = PatchTrackerParams()
            self.visualizer = VisualizerParams()
            self.help = {
                'cfg': 'optional ASCII text file from where parameter values can be read;'
                       'command line parameter values will override the values in this file',
                'mode': 'mode in which to run the server:'
                        ' 0: local execution'
                        ' 1: remote execution'
                        ' 2: output to terminal / GUI in local execution mode (non-server)',
                'port': 'port on which the server listens for requests',
                'save_as_bin': 'save images as binary files for faster reloading (may take a lot of disk space)',
                'img_path': 'single sequence on which patch tracker is to be run (mode=2); overriden by img_path',
                'img_paths': 'list of sequences on which patch tracker is to be run (mode=2); overrides img_path',
                'root_dir': 'optional root directory containing sequences on which patch tracker is to be run (mode=2)',

                'verbose': 'show detailed diagnostic messages',
                'patch_tracker': 'parameters for the patch tracker module',
                'visualizer': 'parameters for the visualizer module',
            }

    def connectToExecutionServer(self):
        self.logger.info('Executing on {}@{}'.format(self.params.username, self.params.hostname))

        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy)
        self.client.connect(
            self.params.hostname,
            username=self.params.username,
            password=self.params.password
        )
        self.channel = self.client.invoke_shell(width=1000, height=3000)
        self._stdout = self.channel.makefile()
        # self.flushChannel()

    def getRemoteOutput(self):
        self.remote_output = self._stdout.readline().replace("^C", "")

    def flushChannel(self):
        # while not self.channel.exit_status_ready():
        while True:
            # if not self.channel.recv_ready():
            #     continue

            # remote_output = self._stdout.readline().replace("^C", "")

            self.remote_output = None

            p = multiprocessing.Process(target=self.getRemoteOutput)
            p.start()
            # Wait for 1 second or until process finishes
            p.join(self.params.wait_timeout)

            if p.is_alive():
                p.terminate()
                p.join()

            if not self.remote_output:
                break

            # print('remote_output: ', remote_output)
            if not self.remote_output.startswith('###'):
                sys.stdout.write(self.remote_output)
                sys.stdout.flush()

    def visualize(self, request):
        request_path = request["path"]
        csv_path = request["csv_path"]
        class_dict = request["class_dict"]
        request_roi = request["roi"]
        init_frame_id = request["frame_number"]

        save_fname_templ = os.path.splitext(os.path.basename(request_path))[0]

        df = pd.read_csv(csv_path)

        if request_path != self.current_path:
            self.frames_reader = get_frames_reader(request_path, save_as_bin=self.params.save_as_bin)
            if request_roi is not None:
                self.frames_reader.setROI(request_roi)
            self.current_path = request_path
        class_labels = dict((v, k) for k, v in class_dict.items())

        # print('self.params.visualizer.save: ', self.params.visualizer.save)
        visualizer = Visualizer(self.params.visualizer, self.logger, class_labels)
        init_frame = self.frames_reader.get_frame(init_frame_id)

        height, width, _ = init_frame.shape
        frame_size = width, height
        visualizer.initialize(save_fname_templ, frame_size)

        n_frames = self.frames_reader.num_frames
        for frame_id in range(init_frame_id, n_frames):
            try:
                curr_frame = self.frames_reader.get_frame(frame_id)
            except IOError as e:
                print('{}'.format(e))
                break

            file_path = self.frames_reader.get_file_path()
            if file_path is None:
                print('Visualization is only supported on image sequence data')
                return

            filename = os.path.basename(file_path)

            multiple_instance = df.loc[df['filename'] == filename]
            # Total # of object instances in a file
            no_instances = len(multiple_instance.index)
            # Remove from df (avoids duplication)
            df = df.drop(multiple_instance.index[:no_instances])

            frame_data = []

            for instance in range(0, len(multiple_instance.index)):
                target_id = multiple_instance.iloc[instance].loc['target_id']
                xmin = multiple_instance.iloc[instance].loc['xmin']
                ymin = multiple_instance.iloc[instance].loc['ymin']
                xmax = multiple_instance.iloc[instance].loc['xmax']
                ymax = multiple_instance.iloc[instance].loc['ymax']
                class_name = multiple_instance.iloc[instance].loc['class']
                class_id = class_dict[class_name]

                width = xmax - xmin
                height = ymax - ymin

                frame_data.append([frame_id, target_id, xmin, ymin, width, height, class_id])

            frame_data = np.asarray(frame_data)
            if not visualizer.update(frame_id, curr_frame, frame_data):
                break

        visualizer.close()

    def run(self):
        if self.params.mode == 2:
            img_paths = self.params.img_paths
            root_dir = self.params.root_dir

            if img_paths:
                if os.path.isfile(img_paths):
                    img_paths = [x.strip() for x in open(img_paths).readlines() if x.strip()]
                else:
                    img_paths = img_paths.split(',')
                if root_dir:
                    img_paths = [os.path.join(root_dir, name) for name in img_paths]

            elif root_dir:
                img_paths = [os.path.join(root_dir, name) for name in os.listdir(root_dir) if
                             os.path.isdir(os.path.join(root_dir, name))]
                img_paths.sort(key=sortKey)

            else:
                img_paths = (self.params.img_path,)

            print('Running patch tracker on {} sequences'.format(len(img_paths)))
            for img_path in img_paths:
                self.patchTracking(img_path=img_path)
            return

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        bindToPort(sock, self.params.port, 'tracking')
        sock.listen(1)
        self.logger.info('Tracking server started')
        # if self.params.mode == 0:
        #     self.logger.info('Started tracking server in local execution mode')
        # else:
        #     self.logger.info('Started tracking server in remote execution mode')
        while True:
            try:
                connection, addr = sock.accept()
                connection.settimeout(None)
                msg = recv_from_connection(connection)
                connection.close()
                if isinstance(msg, list):
                    raw_requests = msg
                else:
                    raw_requests = [msg]
                for request in raw_requests:
                    # print('request: ', request)
                    request_type = request['request_type']
                    if request_type == 'patch_tracking':
                        # self.params.processArguments()
                        try:
                            self.patchTracking(request)
                        except KeyboardInterrupt:
                            continue
                    # elif request_type == 'stop':
                    #     break
                    elif request_type == 'visualize':
                        self.visualize(request)
                    else:
                        self.logger.error('Invalid request type: {}'.format(request_type))
            except KeyboardInterrupt:
                print('Exiting due to KeyboardInterrupt')
                if self.client is not None:
                    self.client.close()
                return
            except SystemExit:
                if self.client is not None:
                    self.client.close()
                return
        # self.logger.info('Stopped tracking server')

    # def run(self):
    # threading.Thread(target=self.request_loop).start()


if __name__ == '__main__':
    # get parameters
    _params = Server.Params()

    paramparse.process(_params)

    # setup logger
    PROFILE_LEVEL_NUM = 9
    logging.addLevelName(PROFILE_LEVEL_NUM, "PROFILE")
    logging.Logger.profile = profile

    logging_fmt = '%(levelname)s::%(module)s::%(funcName)s::%(lineno)s :  %(message)s'
    logging_level = logging.INFO
    # logging_level = logging.DEBUG
    # logging_level = PROFILE_LEVEL_NUM
    logging.basicConfig(level=logging_level, format=logging_fmt)
    _logger = logging.getLogger()
    _logger.setLevel(logging.INFO)

    if _params.mode == -1:

        _logger.info('Running in direct execution mode')
        _params.mode = 0
        _server = Server(_params, _logger)
        init = _params.init

        bbox = dict(
            xmin=init.bbox[0],
            ymin=init.bbox[1],
            xmax=init.bbox[2],
            ymax=init.bbox[3],
        )

        roi = None
        if init.roi is not None:
            roi = dict(
                xmin=init.roi[0],
                ymin=init.roi[1],
                xmax=init.roi[2],
                ymax=init.roi[3],
            )
        request = dict(
            request_type="patch_tracking",
            cmd_args='',
            path=init.path,
            frame_number=init.frame_number,
            port=init.port,
            trigger_tracking_request=False,
            bbox=bbox,
            label=init.label,
            id_number=init.id_number,
            roi=roi,
            bbox_source=init.bbox_source,
            num_frames=init.num_frames,
        )
        patch_tracking(_server, request, _server.frames_reader, _logger)
    else:
        _server = Server(_params, _logger)
        _server.run()
