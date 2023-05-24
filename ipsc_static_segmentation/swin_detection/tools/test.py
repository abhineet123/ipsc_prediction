import shutil
import os
import sys

swin_det_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f'swin_det_dir: {swin_det_dir}')
sys.path.append(swin_det_dir)

import time

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils.misc import read_class_info

import paramparse


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
        self.write_xml = 0
        self.write_masks = 0
        self.filter_objects = 0
        self.show_score_thr = 0.3
        self.tmpdir = ''
        self.test_name = 'test'
        # self.class_info = 'data/classes_ipsc_5_class.txt'


def main():
    params = Params()
    paramparse.process(params)

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(params.local_rank)

    checkpoint_dir = os.path.dirname(params.checkpoint)

    if not params.out_dir:
        params.out_dir = os.path.join(checkpoint_dir, params.test_name)

    os.makedirs(params.out_dir, exist_ok=1)

    assert params.out or params.eval or params.format_only or params.show \
           or params.out_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    # if params.eval and params.format_only:
    #     raise ValueError('--eval and --format_only cannot be both specified')

    # if params.out is not None and not params.out.endswith(('.pkl', '.pickle')):
    #     raise ValueError('The output file must be a pkl file.')

    # classes, composite_classes = read_class_info(params.class_info)
    # class_id_to_color = {i: k[1] for i, k in enumerate(classes)}
    # class_id_to_names = {i: k[0] for i, k in enumerate(classes)}

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

    # in case the test dataset is concatenated
    samples_per_gpu = params.batch_size
    test_data_cfg = cfg.data[params.test_name]
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
    if params.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(params.launcher, **cfg.dist_params)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, params.checkpoint, map_location='cpu')
    if params.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    # if 'CLASSES' in checkpoint.get('meta', {}):
    #     model.CLASSES = checkpoint['meta']['CLASSES']
    # else:
    #     model.CLASSES = list(class_id_to_color.keys())

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)

    workers_per_gpu = cfg.data.workers_per_gpu

    if workers_per_gpu < samples_per_gpu:
        workers_per_gpu = samples_per_gpu

    while True:
        start_t = time.time()
        dataset = build_dataset(test_data_cfg)
        model.CLASSES = dataset.CLASSES[:]

        print(f'samples_per_gpu: {samples_per_gpu}')
        print(f'workers_per_gpu: {workers_per_gpu}')

        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=workers_per_gpu,
            dist=distributed,
            shuffle=False)

        if not distributed:
            outputs = single_gpu_test(model, data_loader,
                                      out_dir=params.out_dir,
                                      show_score_thr=params.show_score_thr,
                                      filter_objects=params.filter_objects,
                                      write_masks=params.write_masks,
                                      write_xml=params.write_xml,
                                      )
        else:
            outputs = multi_gpu_test(model, data_loader, params.tmpdir,
                                     params.gpu_collect)

        json_file_info = dataset.format_results(outputs, jsonfile_prefix=params.out)

        json_file_dict = json_file_info[0]
        for eval_type in params.eval:
            src_path = json_file_dict[eval_type]

            src_name = os.path.basename(src_path)
            dst_path = os.path.join(params.out_dir, src_name)

            shutil.copy(src_path, dst_path)

            print(f'{eval_type} --> {dst_path}')

        kwargs = {} if params.eval_options is None else params.eval_options
        rank, _ = get_dist_info()
        if rank == 0:
            if params.out:
                print(f'\nwriting results to {params.out}')
                mmcv.dump(outputs, params.out)
            if params.format_only:
                dataset.format_results(outputs, **kwargs)
            if params.eval:
                eval_kwargs = cfg.get('evaluation', {}).copy()
                # hard-code way to remove EvalHook params
                for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
                ]:
                    eval_kwargs.pop(key, None)
                eval_kwargs.update(dict(metric=params.eval, **kwargs))
                eval_out = dataset.evaluate(outputs, **eval_kwargs)
                print(eval_out)

        end_t = time.time()

        time_taken = end_t - start_t
        print('time taken: {:.2f} sec'.format(time_taken))

        if params.multi_run:
            k = input('\npress Enter to run segmentation again or q + Enter to exit\n')
            k = k.strip()
            # print('k: {}'.format(k))
            if k.lower() == 'q':
                print('exiting...')
                break

            # try:
            #     import msvcrt as m
            # except ImportError:
            #     _ = input('press enter to run segmentation again')
            # else:
            #     print('press any key to run segmentation again')
            #     m.getch()
            # continue
        else:
            break


if __name__ == '__main__':
    main()
