import os.path as osp
import pickle
import shutil
import tempfile
import time

import os
import numpy as np
import pandas as pd
import cv2
import mmcv
import torch
import torch.distributed as dist
from tqdm import tqdm

from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results


def single_gpu_test(model,
                    data_loader,
                    write_masks,
                    write_xml,
                    out_dir,
                    show_score_thr,
                    filter_objects,
                    ):
    assert out_dir is not None, "out_dir must be provided"

    model.eval()
    results = []
    dataset = data_loader.dataset

    classes = dataset.CLASSES[:]
    palette = dataset.PALETTE[:]

    if 'background' not in classes:
        classes.insert(0, 'background')
        palette.insert(0, [0, 0, 0])

    palette_flat = [value for color in palette for value in color]

    csv_columns = [
        "ImageID", "LabelName",
        "XMin", "XMax", "YMin", "YMax",
        "Confidence",
        'mask_w', 'mask_h', 'mask_counts'
    ]
    mask_out_dir = os.path.join(out_dir, 'masks')
    out_csv_dir = os.path.join(out_dir, "csv")
    xml_out_dir = os.path.join(out_dir, 'annotations')

    os.makedirs(out_csv_dir, exist_ok=1)
    print(f'out_csv_dir: {out_csv_dir}')

    if write_masks:
        os.makedirs(mask_out_dir, exist_ok=1)
        print(f'mask_out_dir: {mask_out_dir}')

    if write_xml:
        os.makedirs(xml_out_dir, exist_ok=1)
        print(f'xml_out_dir: {xml_out_dir}')

    n_data_loader = len(data_loader)

    n_csv_rows = 0

    seq_to_csv_rows = {}
    seq_name_to_paths = {}

    non_empty = n_total = 0

    pbar = tqdm(data_loader, total=n_data_loader)
    for batch_id, data in enumerate(pbar):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        batch_size = len(result)
        if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
            img_tensor = data['img'][0]
        else:
            img_tensor = data['img'][0].data[0]

        img_metas = data['img_metas'][0].data[0]
        imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])

        assert len(imgs) == len(img_metas), "mismatch between imgs and img_metas"

        _pause = 1

        for img_id, (img, img_meta) in enumerate(zip(imgs, img_metas)):
            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :]

            ori_h, ori_w = img_meta['ori_shape'][:-1]
            img_show = mmcv.imresize(img_show, (ori_w, ori_h))

            img_path = img_meta['ori_filename']

            img_dir = osp.dirname(img_path)
            img_name = osp.basename(img_path)

            seq_name = osp.basename(img_dir)
            if seq_name == 'images':
                seq_name = osp.basename(osp.dirname(img_dir))

            try:
                seq_out_dir, seq_mask_out_dir, seq_xml_out_dir, out_csv_path = seq_name_to_paths[seq_name]
            except KeyError:
                seq_out_dir = osp.join(out_dir, seq_name)
                seq_mask_out_dir = osp.join(mask_out_dir, seq_name)
                seq_xml_out_dir = osp.join(xml_out_dir, seq_name)
                out_csv_path = os.path.join(out_csv_dir, f"{seq_name}.csv")

                seq_name_to_paths[seq_name] = (seq_out_dir, seq_mask_out_dir, seq_xml_out_dir, out_csv_path)

                os.makedirs(seq_out_dir, exist_ok=1)
                os.makedirs(seq_mask_out_dir, exist_ok=1)
                os.makedirs(seq_xml_out_dir, exist_ok=1)

                seq_to_csv_rows[out_csv_path] = []
                """write header to csv file"""
                pd.DataFrame([], columns=csv_columns).to_csv(out_csv_path, index=False)

            curr_result = result[img_id]

            if filter_objects:
                """Set confidence of all the objects except the first one to 0"""
                curr_result_array = curr_result[0][0]
                curr_result_masks = curr_result[1][0]

                # obj_areas = [int((k[2] - k[0]) * (k[3] - k[1])) for k in curr_result_array]

                nonzero_counts = np.asarray([np.count_nonzero(k) for k in curr_result_masks])

                most_counts_ids = np.argsort(nonzero_counts)

                curr_result_array[:, -1] = 0
                curr_result_array[most_counts_ids[-1], -1] = 1.0
                curr_result_array[most_counts_ids[-2], -1] = 1.0
                curr_result_array[most_counts_ids[-3], -1] = 1.0

            csv_rows = model.module.show_result(
                img_show,
                curr_result,
                show=False,
                out_dir=seq_out_dir,
                out_filename=img_name,
                out_xml_dir=seq_xml_out_dir,
                out_mask_dir=seq_mask_out_dir,
                score_thr=show_score_thr,
                classes=classes,
                palette_flat=palette_flat,
                write_masks=write_masks,
                write_xml=write_xml,
            )

            if csv_rows:
                non_empty += 1
                seq_to_csv_rows[out_csv_path] += csv_rows
                n_csv_rows += len(csv_rows)

                if n_csv_rows >= 100:
                    n_csv_rows = 0
                    for out_csv_path, csv_rows in seq_to_csv_rows.items():
                        df = pd.DataFrame(csv_rows, columns=csv_columns)
                        df.to_csv(out_csv_path, index=False, mode='a', header=False)
                        seq_to_csv_rows[out_csv_path] = []

            n_total += 1

        non_empty_pc = non_empty / n_total * 100.
        pbar.set_description(f'non_empty: {non_empty} / {n_total} ({non_empty_pc:.2f}%)')

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        results.extend(result)

        # for _ in range(batch_size):
        #     prog_bar.update()

    if n_csv_rows >= 100:
        for out_csv_path, csv_rows in seq_to_csv_rows.items():
            df = pd.DataFrame(csv_rows, columns=csv_columns)
            df.to_csv(out_csv_path, index=False, mode='a', header=False)

    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN,),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
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
