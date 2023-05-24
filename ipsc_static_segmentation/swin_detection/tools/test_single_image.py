import os
import cv2

import numpy as np
import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.datasets import replace_ImageToTensor
from mmdet.models import build_detector
from mmdet.utils.misc import read_class_info

import paramparse


def single_gpu_test(model,
                    img_path,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3,
                    class_to_color=None):
    model.eval()
    results = []

    img = cv2.imread(img_path)
    img_reshaped = img.transpose([2, 0, 1])
    img_expanded = np.expand_dims(img_reshaped, 0)

    img_tensor = torch.tensor(img_expanded, dtype=torch.float32)

    with torch.no_grad():
        result = model(return_loss=False, rescale=True, img=img_tensor)

    batch_size = len(result)
    if show or out_dir:
        imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])

        for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :]

            ori_h, ori_w = img_meta['ori_shape'][:-1]
            img_show = mmcv.imresize(img_show, (ori_w, ori_h))

            if out_dir:
                import os
                fname_no_ext = os.path.splitext(img_meta['ori_filename'])[0]
                out_file = os.path.join(out_dir, fname_no_ext + '.jpg')
            else:
                out_file = None

            model.module.show_result(
                img_show,
                result[i],
                show=show,
                out_file=out_file,
                score_thr=show_score_thr,
                class_to_color=class_to_color,
            )

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results


class Params:
    """
    MMDet test (and eval) a model
    :ivar cfg_options: override some settings in the used config, the key-value pair in xxx=yyy format will be merged
    into config file. If the value to be overwritten is a list, it should be like key="[a,b]" or key=a,
    b It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation marks are necessary
    and that no white space is allowed.
    :type cfg_options: list

    :ivar eval: evaluation metrics, which depends on the dataset, e.g., "bbox", "segm", "proposal" for COCO,
    and "mAP", "recall" for PASCAL VOC
    :type eval: list

    :ivar eval_options: custom options for evaluation, the key-value pair in xxx=yyy format will be kwargs for
    dataset.evaluate() function
    :type eval_options: list

    :ivar format_only: Format the output results without perform evaluation. It is useful when you want to format the
    result to a specific format and submit it to the test server
    :type format_only: bool

    :ivar fuse_conv_bn: Whether to fuse conv and bn, this will slightly increasethe inference speed
    :type fuse_conv_bn: bool

    :ivar gpu_collect: whether to use gpu to collect results.
    :type gpu_collect: bool

    :ivar launcher: job launcher
    :type launcher: str

    :ivar local_rank:
    :type local_rank: int

    :ivar options: custom options for evaluation, the key-value pair in xxx=yyy format will be kwargs for
    dataset.evaluate() function (deprecate), change to --eval-options instead.
    :type options: list

    :ivar out: output result file in pickle format
    :type out: str

    :ivar show: show results
    :type show: bool

    :ivar show_dir: directory where painted images will be saved
    :type show_dir: str

    :ivar show_score_thr: score threshold (default: 0.3)
    :type show_score_thr: float

    :ivar tmpdir: tmp directory used for collecting results from multiple workers, available when gpu-collect is not
    specified
    :type tmpdir: str

    """

    def __init__(self):
        self.cfg = ()
        self.config = ''
        self.checkpoint = ''
        self.cfg_options = None
        self.eval = ["bbox", "segm"]
        self.eval_options = None
        self.format_only = False
        self.fuse_conv_bn = False
        self.gpu_collect = False
        self.launcher = 'none'
        self.local_rank = 0
        self.options = None
        self.out = None
        self.show = False
        self.show_dir = ''
        self.show_score_thr = 0.3
        self.tmpdir = ''
        self.test_name = 'test'
        self.class_info_path = 'data/classes_ipsc_5_class.txt'


def parse_args():

    args = Params()
    paramparse.process(args)

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    # if args.options and args.eval_options:
    #     raise ValueError(
    #         '--options and --eval-options cannot be both '
    #         'specified, --options is deprecated in favor of --eval-options')

    # if args.options:
    #     warnings.warn('--options is deprecated in favor of --eval-options')
    #     args.eval_options = args.options

    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
           or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    classes, composite_classes = read_class_info(args.class_info_path)
    class_to_color = {i: k[1] for i, k in enumerate(classes)}

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
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

    # in case the test dataset is concatenated
    samples_per_gpu = 1
    test_data_cfg = cfg.data[args.test_name]
    if isinstance(test_data_cfg, dict):
        test_data_cfg.test_mode = True
        samples_per_gpu = test_data_cfg.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            test_data_cfg.pipeline = replace_ImageToTensor(
                test_data_cfg.pipeline)
    elif isinstance(test_data_cfg, list):
        for ds_cfg in test_data_cfg:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in test_data_cfg])
        if samples_per_gpu > 1:
            for ds_cfg in test_data_cfg:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    model = MMDataParallel(model, device_ids=[0])
    outputs = single_gpu_test(model, args.show, args.show_dir,
                              args.show_score_thr, class_to_color=class_to_color)


if __name__ == '__main__':
    main()
