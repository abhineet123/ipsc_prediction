import os
import os.path as osp
import pickle
import shutil
import tempfile

import cv2
import PIL

import mmcv
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from tqdm import tqdm

from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

import pycocotools.mask as mask_util


def np2tmp(array, temp_file_name=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.

    Returns:
        str: The numpy file name.
    """

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False).name
    np.save(temp_file_name, array)
    return temp_file_name


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False,
                    write_masks=False,
                    blended_vis=False,
                    collect_results=False,
                    ):
    """Test with single GPU.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during infernece. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.

    Returns:
        list: The prediction results.
    """
    assert out_dir is not None, "out_dir must be provided"

    model.eval()
    results = []
    dataset = data_loader.dataset
    palette = dataset.PALETTE
    palette_flat = [value for color in palette for value in color]



    csv_columns = [
        "ImageID", "LabelName",
        "XMin", "XMax", "YMin", "YMax",
        # "Confidence",
        'mask_w', 'mask_h', 'mask_counts'
    ]
    mask_out_dir = out_csv_dir = None

    if out_dir:
        os.makedirs(out_dir, exist_ok=1)

        mask_out_dir = os.path.join(out_dir, 'masks')

        os.makedirs(mask_out_dir, exist_ok=1)

        out_csv_dir = os.path.join(out_dir, "csv")
        os.makedirs(out_csv_dir, exist_ok=1)
        print(f'out_csv_dir: {out_csv_dir}')

        if write_masks:
            print(f'mask_out_dir: {mask_out_dir}')

    # prog_bar = mmcv.ProgressBar(len(dataset))

    n_empty = 0
    n_images = 0
    n_csv_rows = 0

    pbar = tqdm(data_loader, ncols=100)

    seq_to_csv_rows = {}
    seq_name_to_paths = {}

    for i, data in enumerate(pbar):
        with torch.no_grad():
            result = model(return_loss=False, **data)

        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                # print(f'api:test :: out_dir: {out_dir}')

                img_path = img_meta['ori_filename']

                img_dir = osp.dirname(img_path)
                img_name = osp.basename(img_path)
                seq_name = osp.basename(img_dir)

                img_name_noext = osp.splitext(img_name)[0]

                out_img_name = img_name_noext + '.png'

                # print(f'api:test :: out_img_name: {out_img_name}')

                is_empty = True

                try:
                    seq_out_dir, seq_mask_out_dir, out_csv_path = seq_name_to_paths[seq_name]
                except KeyError:
                    seq_out_dir = osp.join(out_dir, seq_name)
                    seq_mask_out_dir = osp.join(mask_out_dir, seq_name)
                    out_csv_path = os.path.join(out_csv_dir, f"{seq_name}.csv")

                    seq_name_to_paths[seq_name] = (seq_out_dir, seq_mask_out_dir, out_csv_path)

                    os.makedirs(seq_out_dir, exist_ok=1)
                    os.makedirs(seq_mask_out_dir, exist_ok=1)

                    seq_to_csv_rows[out_csv_path] = []
                    """write header to csv file"""
                    pd.DataFrame([], columns=csv_columns).to_csv(out_csv_path, index=False)

                out_file = osp.join(seq_out_dir, out_img_name)

                mask_out_file = osp.join(seq_mask_out_dir, out_img_name)

                seg = result[0].astype(np.uint8)

                for label, color in enumerate(palette):
                    label_name = model.module.CLASSES[label]
                    if label_name == "background":
                        continue

                    binary_seg = (seg == label)

                    if not np.any(binary_seg):
                        continue

                    is_empty = False

                    binary_seg = np.asfortranarray(binary_seg,  dtype="uint8")

                    rle = mask_util.encode(binary_seg)
                    bbox = mask_util.toBbox(rle)
                    xmin, ymin, w, h = np.squeeze(bbox)

                    xmax, ymax = xmin + w, ymin + h

                    mask_h, mask_w = rle['size']

                    mask_counts = rle['counts'].decode('utf-8')

                    # rle2 = dict(
                    #     size=[mask_h, mask_w],
                    #     counts=mask_counts.encode('utf-8')
                    # )
                    # binary_seg2 = mask_util.decode(rle2)
                    #
                    # is_equal = np.all(binary_seg == binary_seg2)

                    row = {
                        "ImageID": img_name,
                        "LabelName": label_name,
                        "XMin": xmin,
                        "XMax": xmax,
                        "YMin": ymin,
                        "YMax": ymax,
                        "mask_w": mask_w,
                        "mask_h": mask_h,
                        "mask_counts": mask_counts,
                    }
                    seq_to_csv_rows[out_csv_path].append(row)
                    n_csv_rows += 1

                if n_csv_rows >= 100:
                    n_csv_rows = 0
                    for out_csv_path, csv_rows in seq_to_csv_rows.items():
                        df = pd.DataFrame(csv_rows, columns=csv_columns)
                        df.to_csv(out_csv_path, index=False, mode='a', header=False)
                        seq_to_csv_rows[out_csv_path] = []

                if write_masks and not is_empty:
                    seg_pil = PIL.Image.fromarray(seg)
                    seg_pil = seg_pil.convert('P')
                    seg_pil.putpalette(palette_flat)
                    seg_pil.save(mask_out_file)

                # print(f'api:test :: out_file: {out_file}')

                if not is_empty:
                    if blended_vis:
                        model.module.show_result(
                        img_show,
                        result,
                        palette=palette,
                        show=show,
                        out_file=out_file)

                else:
                    n_empty += 1

                n_images += 1
                empty_percent = (float(n_empty) / n_images) * 100
                pbar.set_description(f'n_empty: {n_empty} ({empty_percent:.2f}%)')

        if collect_results:
            if isinstance(result, list):
                if efficient_test:
                    result = [np2tmp(_) for _ in result]
                results.extend(result)
            else:
                if efficient_test:
                    result = np2tmp(result)
                results.append(result)

    if n_csv_rows >= 100:
        for out_csv_path, csv_rows in seq_to_csv_rows.items():
            df = pd.DataFrame(csv_rows, columns=csv_columns)
            df.to_csv(out_csv_path, index=False, mode='a', header=False)

    return results


def multi_gpu_test(model,
                   data_loader,
                   tmpdir=None,
                   gpu_collect=False,
                   efficient_test=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.

    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if isinstance(result, list):
            if efficient_test:
                result = [np2tmp(_) for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result)
            results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    """Collect results with CPU."""
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    """Collect results with GPU."""
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
