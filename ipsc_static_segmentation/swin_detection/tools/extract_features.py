import os
import sys

swin_det_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(swin_det_dir)

home_path = os.path.expanduser('~')
deep_mdp_path = os.path.join(home_path, 'isl_labeling_tool', 'deep_mdp')
sys.path.append(deep_mdp_path)

from tqdm import tqdm
import time
import copy
import cv2
from datetime import datetime

import torch

import mmcv
from mmcv import Config
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import load_checkpoint, wrap_fp16_model

from mmdet.models import build_detector
from mmdet.utils.misc import read_class_info, linux_path

from input import Input
from objects import Annotations
from data import Data
from utilities import CustomLogger, linux_path, draw_box

import numpy as np

import paramparse

reductions = dict(
    flat=torch.nn.Flatten(),
    avg_8=torch.nn.AdaptiveAvgPool2d(output_size=(8, 8)),
    avg_4=torch.nn.AdaptiveAvgPool2d(output_size=(4, 4)),
    avg_2=torch.nn.AdaptiveAvgPool2d(output_size=(2, 2)),
    max_2=torch.nn.MaxPool2d(2, stride=2, return_indices=True),
    max_2_inv=torch.nn.MaxUnpool2d(2, stride=2),
    max_4=torch.nn.MaxPool2d(4, stride=4, return_indices=True),
    max_4_inv=torch.nn.MaxUnpool2d(4, stride=4),
    max_8=torch.nn.MaxPool2d(8, stride=8, return_indices=True),
    max_8_inv=torch.nn.MaxUnpool2d(8, stride=8),
    max_16=torch.nn.MaxPool2d(16, stride=16, return_indices=True),
    max_16_inv=torch.nn.MaxUnpool2d(16, stride=16),
    f0=lambda x: x[0],
    f1=lambda x: x[1],
    f2=lambda x: x[2],
    f3=lambda x: x[3],
)


class Params:
    class SlidingWindow:
        sample = 0
        size = 0
        stride = 0
        num = 0

    def __init__(self):
        self.cfg = ('',)

        self.set = ''
        self.seq = ()
        self.start_seq = ()
        self.end_seq = ()

        self.config = ''
        self.ckpt = ''
        self.ckpt_dir = ''
        self.ckpt_name = ''

        self.cfg_options = None
        # self.eval = ["bbox", "segm"]
        self.eval = []
        self.eval_options = {
            "classwise": True
        }
        self.format_only = False
        self.fuse_conv_bn = False
        self.gpu_collect = False
        self.launcher = 'none'
        self.local_rank = 0
        self.multi_run = 0
        self.options = None
        self.out = None
        self.show = 0
        self.batch_size = 1
        self.out_dir = ''
        self.out_name = ''
        self.out_suffix = ''
        self.tmpdir = ''
        self.n_proc = 1

        self.class_info_path = 'data/mnist_mot.txt'

        self.test_name = 'val'
        self.feat_name = 'neck'
        self.reduce = ['f0', 'max_16']

        self.vis = 0
        self.raw = 0
        self.pool = 0

        self.mean = [93.154564, 162.03416, 240.90062]
        self.std = [3.8680854, 2.779077, 2.8976252]

        self.slide = Params.SlidingWindow()

        self.input = Input.Params(source_type=-1, batch_mode=True)
        self.data = Data.Params()
        self.ann = Annotations.Params()


def run(seq_info,
        model,
        params: Params,
        classes,
        ):
    model.eval()

    seq_id, start_id, end_id = seq_info

    _logger = CustomLogger.setup(__name__)

    _data = Data(params.data, _logger)

    if not _data.initialize(params.set, seq_id, 0, _logger, silent=1):
        _logger.error('Data module could not be initialized')
        return None

    subset = "training" if _data.split == 'train' else "validation"

    input_params = copy.deepcopy(params.input)  # type: Input.Params

    """end_id is exclusive but Input expects inclusive"""
    input_params.frame_ids = (start_id, end_id - 1)

    _input = Input(input_params, _logger)
    seq_name = _data.seq_name

    print(f'\nseq {seq_id + 1}: {seq_name}\n')

    if not _input.initialize(_data):
        _logger.error('Input pipeline could not be initialized')
        return False

    # read detections and annotations
    if not _input.read_annotations():
        _logger.error('Annotations could not be read')
        return False

    # if not _input.read_detections():
    #     _logger.error('Detections could not be read')
    #     return False

    _frame_size = _input.frame_size
    _n_frames = _input.n_frames

    _annotations = _input.annotations  # type: Annotations

    n_images = _input.n_frames
    n_batches = int(n_images / params.batch_size)

    frame_iter = _input.read_iter(length=params.batch_size)

    feat_name = params.feat_name

    if params.raw:
        out_ext = 'npz'
    else:
        out_ext = 'npy'

    out_name = f'{seq_name}--{start_id}_{end_id}.{out_ext}'

    out_path = linux_path(params.out_dir, out_name)

    pbar = tqdm(frame_iter, total=n_batches)
    reduced_feat_list = []
    score_thr = 0.3
    raw_feat = {}

    if params.vis and params.raw:
        x_all = load_raw(out_path, params.pool)

    for batch_id, img_list in enumerate(pbar):
        imgs = []
        img_metas = []
        for img_id, img in enumerate(img_list):
            # cv2.imshow('img init', img)
            # k = cv2.waitKey(1)
            # if k == 27:
            #     exit()
            # print(f'img: {img_id}')

            img_norm = mmcv.imnormalize(img, np.asarray(params.mean), np.asarray(params.std), to_rgb=True)

            imgs.append(img_norm)

            global_img_id = int(batch_id * params.batch_size + img_id)
            img_meta = dict(
                filename=linux_path(_input.source_path, f'image{global_img_id + 1:06d}.jpg'),
                ori_filename=linux_path(_input.seq_name, f'image{global_img_id + 1:06d}.jpg'),
                ori_shape=img.shape,
                img_shape=img.shape,
                pad_shape=img.shape,
                scale_factor=[1., 1., 1., 1.],
                flip=False,
                flip_direction=None,
                img_norm_cfg=dict(
                    mean=params.mean,
                    std=params.std,
                    to_rgb=True
                ),
                batch_input_shape=img.shape[:2]
            )
            img_metas.append(img_meta)

        img = np.stack(imgs, axis=0)
        """bring channel to front"""
        img_reshaped = img.transpose([0, 3, 1, 2])
        img_tensor = torch.tensor(img_reshaped, dtype=torch.float32).cuda()

        if params.vis:
            x = None
            if params.raw:
                x = [k.cuda() for k in x_all[batch_id]]

            with torch.no_grad():
                results = model(return_loss=False, rescale=True, img=[img_tensor, ], img_metas=[img_metas, ], x=x)

            # print()

            for img_id, img in enumerate(img_list):

                img_show = np.copy(img)

                curr_result = results[img_id]

                bboxes = np.vstack(curr_result)
                labels = [
                    np.full(bbox.shape[0], i, dtype=np.int32)
                    for i, bbox in enumerate(curr_result)
                ]
                labels = np.concatenate(labels)

                if score_thr > 0:
                    assert bboxes.shape[1] == 5
                    scores = bboxes[:, -1]
                    inds = scores > score_thr
                    bboxes = bboxes[inds, :]
                    labels = labels[inds]

                for i, (bbox, label) in enumerate(zip(bboxes, labels)):
                    # bbox_int = bbox.astype(np.int32)
                    try:
                        xmin, ymin, xmax, ymax, conf = bbox
                    except ValueError:
                        xmin, ymin, xmax, ymax = bbox
                        conf = 1.0

                    xmin, ymin, xmax, ymax = [int(x) for x in [xmin, ymin, xmax, ymax]]

                    class_id = label
                    class_name = classes[class_id]

                    w, h = xmax - xmin, ymax - ymin

                    bbox_wh = np.asarray([xmin, ymin, w, h])

                    draw_box(img_show, bbox_wh, _id=None, color='black', thickness=2,
                             is_dotted=0, transparency=0.,
                             text_col=None,
                             font=cv2.FONT_HERSHEY_TRIPLEX, font_size=0.5, label=class_name)

                cv2.imshow('img_show', img_show)
                k = cv2.waitKey(1)
                if k == 27:
                    exit()
                # print(f'img_show: {img_id}')


        else:
            with torch.no_grad():
                final_feat = model.extract_feat(img_tensor)

            if params.raw:
                save_raw(final_feat, raw_feat, batch_id, params.pool)
                continue

            feat_list = model.features[feat_name]
            reduced_feat = feat_list
            for r in params.reduce:
                try:
                    reduced_feat = reductions[r](reduced_feat)
                except KeyError:
                    raise AssertionError(f'invalid reduction type: {r}')

                if isinstance(reduced_feat, (tuple, list)):
                    reduced_feat = reduced_feat[0]

            reduced_feat_np = reduced_feat.cpu().numpy()
            reduced_feat_list.append(reduced_feat_np)

    if not params.vis:
        if params.raw:
            print(f'Saving raw features to {out_path}')
            np.savez_compressed(out_path, **raw_feat)
        else:
            reduced_feat_all = np.concatenate(reduced_feat_list, axis=0)
            print(f'\nsaving feature of size {reduced_feat_all.shape} to {out_path}\n')
            np.save(out_path, reduced_feat_all)


def load_raw(out_path, pool):
    # if not out_path.endswith('.npz'):
    #     out_path += '.npz'

    print(f'loading raw features from {out_path}')

    feat_dict = {}
    loaded_feat = np.load(out_path)
    for _id, feat in tqdm(loaded_feat.items()):

        if pool > 0:
            assert feat.shape[0] % 2 == 0, f"invalid feature shape for pooling: {feat.shape}"
            batch_size = int(feat.shape[0] / 2)
            feat_pooled = feat[:batch_size, ...].astype(np.float32)
            indices = feat[batch_size:, ...].astype(np.int64)

            feat_pooled = torch.from_numpy(feat_pooled).float().cuda()
            indices = torch.from_numpy(indices).cuda()
            if pool == 2:
                feat_unpooled = max_2_inv(feat_pooled, indices)
            elif pool == 4:
                feat_unpooled = max_4_inv(feat_pooled, indices)
            elif pool == 8:
                feat_unpooled = max_8_inv(feat_pooled, indices)
            elif pool == 16:
                feat_unpooled = max_16_inv(feat_pooled, indices)
            else:
                raise AssertionError(f'invalid pool: {pool}')

            feat_pt = feat_unpooled.cpu().float()
        else:
            feat_pt = torch.from_numpy(feat)

        batch_id, feat_id = _id.split('_')
        batch_id, feat_id = int(batch_id), int(feat_id)
        if batch_id not in feat_dict:
            feat_dict[batch_id] = {}
        feat_dict[batch_id][feat_id] = feat_pt

    n_batches = len(feat_dict)
    feat_list = [None, ] * n_batches
    for batch_id in feat_dict:
        n_feat = len(feat_dict[batch_id])
        feat_list[batch_id] = [None, ] * n_feat
        for feat_id in feat_dict[batch_id]:
            feat_list[batch_id][feat_id] = feat_dict[batch_id][feat_id]
    return feat_list


def save_raw(feat_list, raw_feat, batch_id, pool):
    for feat_id, feat in enumerate(feat_list):

        if pool == 0:
            feat_np = feat.cpu().numpy()
        else:
            if pool == 2:
                feat_pooled, indices = max_2(feat)
            elif pool == 4:
                feat_pooled, indices = max_4(feat)
            elif pool == 8:
                feat_pooled, indices = max_8(feat)
            elif pool == 16:
                feat_pooled, indices = max_16(feat)
            else:
                raise AssertionError(f'invalid pool: {pool}')

            feat_pooled_np = feat_pooled.cpu().numpy().astype(np.float32)
            indices_np = indices.cpu().numpy().astype(np.float32)

            feat_np = np.concatenate((feat_pooled_np, indices_np), axis=0)

        raw_feat[f'{batch_id}_{feat_id}'] = feat_np


def avg_all(feat_list):
    avg_pool_feats = []
    for feat in feat_list:
        avg_pool_feat = reductions['avg_2'](feat, (2, 2))
        # avg_pool_feat_flat = torch.flatten(avg_pool_feat)
        avg_pool_feats.append(avg_pool_feat)
    print()
    return avg_pool_feats


def main():
    params = Params()
    paramparse.process(params)

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(params.local_rank)

    classes, composite_classes = read_class_info(params.class_info_path)
    class_to_color = {i: k[1] for i, k in enumerate(classes)}
    classes = {i: k[0] for i, k in enumerate(classes)}

    config_name = os.path.splitext(os.path.basename(params.config))[0]

    if not params.ckpt:
        if not params.ckpt_dir:
            params.ckpt_dir = linux_path('work_dirs', config_name)
        if not params.ckpt_name:
            params.ckpt_name = 'latest.pth'

        params.ckpt = linux_path(params.ckpt_dir, params.ckpt_name)

    _logger = CustomLogger.setup(__name__)
    _data = Data(params.data, _logger)

    try:
        params.set = int(params.set)
    except ValueError:
        params.set = params.data.name_to_id(params.set)

    set_name = _data.sets[params.set]
    n_sequences = len(_data.sequences[set_name])
    print(f'n_sequences: {n_sequences}')

    seq_ids = params.seq

    if not seq_ids:
        seq_ids = tuple(range(n_sequences))

    if params.start_seq or params.end_seq:
        assert len(params.start_seq) == len(params.end_seq), "mismatch between start_seq and end_seq lengths"
        temp_seq_ids = []
        for start_seq, end_seq in zip(params.start_seq, params.end_seq):
            if start_seq < 0:
                start_seq = 0

            if end_seq < 0:
                end_seq = len(seq_ids) - 1

            temp_seq_ids += list(seq_ids[start_seq:end_seq + 1])
        seq_ids = tuple(temp_seq_ids)

    sample = params.slide.sample
    if sample <= 0:
        sample = 1

    seq_info_list = []
    pbar = tqdm(seq_ids)
    print(f'seq_ids: {seq_ids}')

    for seq_id in pbar:

        if not _data.initialize(params.set, seq_id, 0, _logger, silent=1):
            _logger.error('Data module could not be initialized')
            return None

        start_id = 0
        seq_name = _data.seq_name
        seq_n_frames = _data.seq_n_frames

        assert seq_n_frames % sample == 0, f"sample size {sample} does not divide seq_n_frames {seq_n_frames} evenly"

        win_size = params.slide.size
        if win_size <= 0:
            win_size = int(_data.seq_n_frames / sample)

        win_stride = params.slide.stride
        if win_stride <= 0:
            win_stride = win_size

        win_id = 0
        while True:
            abs_start_id = int(start_id * sample)

            if abs_start_id >= seq_n_frames or win_id >= params.slide.num > 0:
                break

            end_id = start_id + win_size

            abs_end_id = int(end_id * sample)

            if abs_end_id >= seq_n_frames:
                abs_end_id = seq_n_frames
                end_id = int(abs_end_id / sample)

            # suffix = f'{abs_start_id}_{abs_end_id}'

            seq_info_list.append((seq_id, abs_start_id, abs_end_id))

            # print(f'{seq_name}--{suffix}: {abs_start_id} to {abs_end_id}')

            start_id += win_stride

            win_id += 1

    n_seq = len(seq_info_list)

    checkpoint_dir = os.path.dirname(params.ckpt)
    checkpoint_name = os.path.splitext(os.path.basename(params.ckpt))[0]
    if not params.out_name:
        params.out_name = 'features'

    if not params.out_suffix:
        if params.raw:
            params.out_suffix = f'raw'
            if params.pool > 0:
                params.out_suffix = f'{params.out_suffix}_{params.pool}'
        else:
            reduce_str = '_'.join(params.reduce)
            params.out_suffix = f'{params.feat_name}_{reduce_str}'

    params.out_name = f'{params.out_name}_{params.out_suffix}'

    if not params.out_dir:
        params.out_dir = linux_path(checkpoint_dir, f'{checkpoint_name}_on_{params.test_name}')

    params.out_dir = linux_path(params.out_dir, params.out_name)

    print(f'writing features to {params.out_dir}')

    os.makedirs(params.out_dir, exist_ok=1)

    print(f'n_seq: {n_seq}')

    # exit()

    if True:
        cfg = Config.fromfile(params.config)
        if params.cfg_options is not None:
            cfg.merge_from_dict(params.cfg_options)
        # import modules from string list.
        if cfg.get('custom_imports', None):
            from mmcv.utils import import_modules_from_strings
            import_modules_from_strings(**cfg['custom_imports'])
        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        cfg.model.pretrained = None
        if cfg.model.get('neck'):
            if isinstance(cfg.model.neck, list):
                for neck_cfg in cfg.model.neck:
                    if neck_cfg.get('rfp_backbone'):
                        if neck_cfg.rfp_backbone.get('pretrained'):
                            neck_cfg.rfp_backbone.pretrained = None
            elif cfg.model.neck.get('rfp_backbone'):
                if cfg.model.neck.rfp_backbone.get('pretrained'):
                    cfg.model.neck.rfp_backbone.pretrained = None

        # build the model and load checkpoint
        cfg.model.train_cfg = None
        model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        checkpoint = load_checkpoint(model, params.ckpt, map_location='cpu')
        if params.fuse_conv_bn:
            model = fuse_conv_bn(model)
        model = model.cuda()

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")
    out_dir = linux_path('log', f'mot_to_dnc', f'{set_name}_{timestamp}')
    os.makedirs(out_dir, exist_ok=1)

    traj_lengths_out_dir = linux_path(out_dir, 'traj_lengths')
    os.makedirs(traj_lengths_out_dir, exist_ok=1)

    n_proc = min(params.n_proc, n_seq)

    import functools
    func = functools.partial(
        run,
        params=params,
        model=model,
        classes=classes,
    )

    if n_proc > 1:
        import multiprocessing

        print(f'running in parallel over {n_proc} processes')
        with multiprocessing.Pool(n_proc) as pool:
            results = pool.map(func, seq_info_list)

        results.sort(key=lambda x: x[0])
    else:
        results = []
        for seq_info in tqdm(seq_info_list):
            result = func(seq_info)

            results.append(result)


if __name__ == '__main__':
    main()
