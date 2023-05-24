# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
import pycocotools.mask as mask_util
import torch
from pycocotools.ytvos import YTVOS
import torch.nn.functional as F

import detectron2.utils.comm as comm
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils.file_io import PathManager

from tqdm import tqdm


class YTVISEvaluator(DatasetEvaluator):
    """
    Evaluate AR for object proposals, AP for instance detection/segmentation, AP
    for keypoint detection outputs using COCO's metrics.
    See http://cocodataset.org/#detection-eval and
    http://cocodataset.org/#keypoints-eval to understand its metrics.

    In addition to COCO, this evaluator is able to support any bounding box detection,
    instance segmentation, or keypoint detection dataset.
    """

    def __init__(
            self,
            dataset_name,
            tasks=None,
            distributed=True,
            output_dir=None,
            use_probs=0,
            *,
            use_fast_impl=False,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation

                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            tasks (tuple[str]): tasks that can be evaluated under the given
                configuration. A task is one of "bbox", "segm", "keypoints".
                By default, will infer this automatically from predictions.
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will only evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:

                1. "instances_predictions.pth" a file in torch serialization
                   format that contains all the raw original predictions.
                2. "coco_instances_results.json" a json file in COCO's result
                   format.
            use_fast_impl (bool): use a fast but **unofficial** implementation to compute AP.
                Although the results should be very close to the official implementation in COCO
                API, it is still recommended to compute results with the official API for use in
                papers. The faster implementation also uses more RAM.
        """
        self._logger = logging.getLogger(__name__)
        self._distributed = distributed
        self._use_probs = use_probs

        if self._use_probs:
            print('\nusing probabilities as scores\n')
        else:
            print('\nusing sigmoids as scores\n')

        self.output_dir = output_dir
        self._use_fast_impl = use_fast_impl

        if tasks is not None and isinstance(tasks, CfgNode):
            self._logger.warning(
                "COCO Evaluator instantiated using config, this is deprecated behavior."
                " Please pass in explicit arguments instead."
            )
            self._tasks = None  # Infering it from predictions should be better
        else:
            self._tasks = tasks

        self._cpu_device = torch.device("cpu")

        self._metadata = MetadataCatalog.get(dataset_name)

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._ytvis_api = YTVOS(json_file)

        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = "annotations" in self._ytvis_api.dataset

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        prediction = instances_to_coco_json_video(inputs, outputs, self._use_probs)
        # self._predictions.extend(prediction)

        if self.output_dir:
            try:
                idx = inputs[0]['idx']
            except KeyError:
                raise AssertionError('batch idx not found in inputs')
            else:
                # csv_file_path = os.path.join(self.output_dir, "masks.csv")

                out_json_dir = os.path.join(self.output_dir, f"json_results")
                os.makedirs(out_json_dir, exist_ok=1)
                file_path = os.path.join(out_json_dir, f"batch_{idx}.json")
                self._logger.info("Saving results to {}".format(file_path))
                with PathManager.open(file_path, "w") as f:
                    f.write(json.dumps(prediction))
                    f.flush()

    def evaluate(self):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """
        return None

        # if self._distributed:
        #     comm.synchronize()
        #     predictions = comm.gather(self._predictions, dst=0)
        #     predictions = list(itertools.chain(*predictions))
        #
        #     if not comm.is_main_process():
        #         return {}
        # else:
        #     predictions = self._predictions
        #
        # if len(predictions) == 0:
        #     self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
        #     return {}
        #
        # if self.output_dir:
        #     PathManager.mkdirs(self.output_dir)
        #     file_path = os.path.join(self.output_dir, "instances_predictions.pth")
        #     with PathManager.open(file_path, "wb") as f:
        #         torch.save(predictions, f)
        #
        # self._results = OrderedDict()
        # self._eval_predictions(predictions)
        # return copy.deepcopy(self._results)

    def _eval_predictions(self, predictions):
        """
        Evaluate predictions. Fill self._results with the metrics of the tasks.
        """
        # self._logger.info("Preparing results for YTVIS format ...")
        #
        # # unmap the category ids for COCO
        # if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
        #     dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
        #     all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
        #     num_classes = len(all_contiguous_ids)
        #     assert min(all_contiguous_ids) == 0 and max(all_contiguous_ids) == num_classes - 1
        #
        #     reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
        #     for result in predictions:
        #         category_id = result["category_id"]
        #         assert category_id < num_classes, (
        #             f"A prediction has class={category_id}, "
        #             f"but the dataset only has {num_classes} classes and "
        #             f"predicted class id should be in [0, {num_classes - 1}]."
        #         )
        #         result["category_id"] = reverse_id_mapping[category_id]
        #
        # if self.output_dir:
        #     file_path = os.path.join(self.output_dir, "results.json")
        #     self._logger.info("Saving results to {}".format(file_path))
        #     with PathManager.open(file_path, "w") as f:
        #         f.write(json.dumps(predictions))
        #         f.flush()
        #
        # self._logger.info("Annotations are not available for evaluation.")
        return


def instances_to_coco_json_video(inputs, outputs, use_probs):
    """
    Dump an "Instances" object to a COCO-format json that'score used for evaluation.

    Args:
        instances (Instances):
        video_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    assert len(inputs) == 1, "More than one inputs are loaded for inference!"

    video_id = inputs[0]["video_id"]
    video_length = inputs[0]["length"]

    if use_probs:
        scores = outputs["pred_probs"]
    else:
        scores = outputs["pred_scores"]

    labels = outputs["pred_labels"]
    masks = outputs["pred_masks"]
    is_zero_list = outputs["is_zero"]

    height = inputs[0]['height']
    width = inputs[0]['width']

    zero_mask_encoded = dict(
        size=[height, width],
        counts='Xjil6',
    )

    ytvis_results = []
    for instance_id, (score, label, mask, is_zero) in enumerate(tqdm(
            zip(scores, labels, masks, is_zero_list), desc=f'video_id: {video_id}', total=video_length, ncols=75)):
        segms = []
        for _is_zero, _mask in zip(is_zero, mask):

            if _is_zero:
                segms.append(zero_mask_encoded)
                continue

            _mask_bool = _mask > 0.5
            if torch.count_nonzero(_mask_bool) == 0:
                segms.append(zero_mask_encoded)
                continue

            _mask = torch.unsqueeze(_mask, 0)
            _mask = torch.unsqueeze(_mask, 0)

            _mask_res = F.interpolate(_mask, size=(height, width), mode='nearest').squeeze()

            _mask_res_bool = _mask_res > 0.5

            _mask_res_bool = _mask_res_bool.cpu()

            mask_bool = np.array(_mask_res_bool[:, :, None], order="F", dtype="uint8")
            mask_encoded_out = mask_util.encode(mask_bool)
            mask_encoded = mask_encoded_out[0]

            rle_orig = mask_encoded["counts"]
            rle_decoded = rle_orig.decode("utf-8")

            mask_encoded["counts"] = rle_decoded

            segms.append(mask_encoded)

            # print()

        res = {
            "video_id": video_id,
            "score": score,
            "category_id": label,
            "segmentations": segms,
        }
        ytvis_results.append(res)

    return ytvis_results
