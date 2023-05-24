# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import json
import random
import argparse
import datetime
import numpy as np

from tqdm import tqdm
from tensorboardX import SummaryWriter

import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, \
    auto_resume_helper, \
    reduce_tensor


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    # for acceleration
    parser.add_argument('--fused_window_process', action='store_true',
                        help='Fused window shift & window partition, similar for reversed part.')
    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')

    # overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    # logger.info(str(model))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model, 'flops'):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    model.cuda()
    model_without_ddp = model

    optimizer = build_optimizer(config, model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    loss_scalar = NativeScalerWithGradNormCount()

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, loss_scalar, logger)
        if config.EVAL_MODE:
            acc1, loss = validate(config, data_loader_val, model)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
            return

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model_without_ddp, logger)
        if config.EVAL_MODE:
            acc1, acc5, loss = validate(config, data_loader_val, model)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
            return

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    writer = SummaryWriter(logdir=config.OUTPUT)

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler,
                        loss_scalar, writer)
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scalar,
                            logger)

        acc1, loss = validate(config, data_loader_val, model)

        # logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        max_accuracy = max(max_accuracy, acc1)
        # logger.info(f'Max accuracy: {max_accuracy:.2f}%')

        writer.add_scalar('val/loss', loss, epoch)
        writer.add_scalar('val/acc', acc1, epoch)
        writer.add_scalar('val/max_accuracy', max_accuracy, epoch)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn,
                    lr_scheduler, loss_scalar, writer):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scalar_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(tqdm(data_loader, total=num_steps, ncols=50)):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples)
        loss = criterion(outputs, targets)
        loss = loss / config.TRAIN.ACCUMULATION_STEPS

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scalar(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        iter_id = epoch * num_steps + idx

        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update(iter_id // config.TRAIN.ACCUMULATION_STEPS)
        loss_scale_value = loss_scalar.state_dict()["scale"]

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
        scalar_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        # if idx % config.PRINT_FREQ == 0:
        lr = optimizer.param_groups[0]['lr']
        wd = optimizer.param_groups[0]['weight_decay']
        memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        etas = batch_time.avg * (num_steps - idx)

        writer.add_scalar('train/lr', lr, iter_id)
        writer.add_scalar('train/wd', wd, iter_id)
        writer.add_scalar('train/loss', loss_meter.val, iter_id)
        writer.add_scalar('train/grad_norm', norm_meter.val, iter_id)
        writer.add_scalar('train/memory_used', memory_used, iter_id)
        writer.add_scalar('train/batch_time/val', batch_time.val, iter_id)
        writer.add_scalar('train/batch_time/avg', batch_time.avg, iter_id)

        # logger.info(
        #     f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
        #     f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
        #     f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
        #     f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
        #     f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
        #     f'loss_scale {scalar_meter.val:.4f} ({scalar_meter.avg:.4f})\t'
        #     f'mem {memory_used:.0f}MB')

    # epoch_time = time.time() - start
    # logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


def linux_path(*args, **kwargs):
    return os.path.join(*args, **kwargs).replace(os.sep, '/')

@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()

    end = time.time()
    num_steps = len(data_loader)
    all_gt_labels = []
    all_preds = []
    all_probs = []
    all_probs2 = []
    all_src_names = []

    src_id = 0
    for idx, (images, target) in enumerate(tqdm(data_loader, total=num_steps, ncols=50)):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        n_images = target.shape[0]

        for _ in range(n_images):
            src_path = data_loader.dataset.imgs[src_id][0]
            src_name = os.path.basename(src_path)
            all_src_names.append(src_name)
            src_id += 1

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images)

        sigmoid = output.sigmoid()
        probs = output.softmax(dim=1)
        probs2 = sigmoid.softmax(dim=1)

        probs_np = probs.detach().cpu().numpy().squeeze()
        probs2_np = probs2.detach().cpu().numpy().squeeze()

        all_probs.append(probs_np)
        all_probs2.append(probs2_np)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1 = accuracy(output, target, topk=(1,))[0]

        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()

        target_np = target.cpu().detach().numpy().squeeze()
        pred_np = pred.cpu().detach().numpy().squeeze()

        all_gt_labels += list(target_np)
        all_preds += list(pred_np)

        acc1 = reduce_tensor(acc1)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    # logger.info(f' * Acc@1 {acc1_meter.avg:.3f}')

    n_test = len(all_gt_labels)

    all_gt_labels = np.asarray(all_gt_labels)
    all_preds = np.asarray(all_preds)
    all_probs = np.concatenate(all_probs, axis=0).reshape((n_test, 2))
    all_probs2 = np.concatenate(all_probs2, axis=0).reshape((n_test, 2))

    correct_preds = np.equal(all_gt_labels, all_preds)
    correct_preds = correct_preds.reshape((n_test, 1)).astype(np.int32)
    n_correct = np.count_nonzero(correct_preds)

    overall_accuracy = (n_correct / n_test) * 100
    print(f'overall accuracy: {n_correct} / {n_test} ({overall_accuracy:.4f}%)')

    inference_dir = config.INFERENCE
    if not inference_dir:
        inference_dir = 'inference'

    inference_path = linux_path(config.OUTPUT, inference_dir)
    os.makedirs(inference_path, exist_ok=1)

    print(f'saving inference output to {inference_path}')

    test_probs_out_csv = linux_path(inference_path, 'test-probs.csv')
    test_probs2_out_csv = linux_path(inference_path, 'test-probs2.csv')
    test_labels_out_csv = linux_path(inference_path, 'test-labels.csv')

    np.savetxt(test_probs_out_csv, all_probs, delimiter='\t', newline='\n', fmt='%f,%f')
    np.savetxt(test_probs2_out_csv, all_probs2, delimiter='\t', newline='\n', fmt='%f,%f')
    np.savetxt(test_labels_out_csv, all_gt_labels, delimiter='\t', newline='\n', fmt='%d')

    test_src_names_out_csv = linux_path(inference_path, 'test-all_src_names.csv')
    with open(test_src_names_out_csv, 'w') as fid:
        fid.write('\n'.join(all_src_names))

    n_classes = config.MODEL.NUM_CLASSES
    for class_id in range(n_classes):
        class_labels_mask = (all_gt_labels == class_id)
        class_val_labels = all_gt_labels[class_labels_mask]
        class_val_preds = all_preds[class_labels_mask]

        n_class_val_labels = len(class_val_labels)

        class_correct_preds = np.equal(class_val_labels, class_val_preds)
        class_correct_preds = class_correct_preds.reshape((n_class_val_labels, 1)).astype(np.int32)
        n_class_correct = np.count_nonzero(class_correct_preds)

        class_accuracy = (n_class_correct / n_class_val_labels) * 100

        print(f'\tclass {class_id}: {n_class_correct} / {n_class_val_labels} ({class_accuracy:.4f}%)')

    return acc1_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    args, config = parse_option()

    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
        os.environ["RANK"] = "-1"
        os.environ["WORLD_SIZE"] = "-1"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "5565"

    torch.cuda.set_device(config.LOCAL_RANK)
    # torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.init_process_group(backend='gloo', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    # logger.info(config.dump())
    # logger.info(json.dumps(vars(args)))

    main(config)
