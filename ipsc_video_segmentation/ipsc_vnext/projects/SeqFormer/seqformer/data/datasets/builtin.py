# -*- coding: utf-8 -*-

import os

from .ytvis import (
    register_ytvis_instances,
    _get_ytvis_2019_ipsc_instances_meta,
    _get_ytvis_2019_mj_rocks_instances_meta,
    _get_ytvis_2019_instances_meta,
    _get_ytvis_2021_instances_meta,
)

# ==== Predefined splits for YTVIS 2019 ===========

# ==== Predefined splits for YTVIS 2019 ===========
_PREDEFINED_SPLITS_YTVIS_2019 = {
    # "ytvis_2019_train": ("ytvis_2019/train/JPEGImages",
    #                      "ytvis_2019/annotations/instances_train_sub.json"),
    # "ytvis_2019_val": ("ytvis_2019/val/JPEGImages",
    #                    "ytvis_2019/annotations/instances_val_sub.json"),
    # "ytvis_2019_test": ("ytvis_2019/test/JPEGImages",
    #                     "ytvis_2019/test.json"),
    # "ytvis_2019_dev": ("ytvis_2019/train/JPEGImages",
    #                    "ytvis_2019/instances_train_sub.json"),
    "ytvis-mj_rock-db3-part12-train": (
        "mojow_rock/rock_dataset3/ytvis19/JPEGImages",
        "mojow_rock/rock_dataset3/ytvis19/mj_rock-db3-part12-train.json"
    ),
    "ytvis-mj_rock-db3-part12-val": (
        "mojow_rock/rock_dataset3/ytvis19/JPEGImages",
        "mojow_rock/rock_dataset3/ytvis19/mj_rock-db3-part12-val.json"
    ),

    "ytvis-mj_rock-db3_2_to_17_except_6_with_syn-train": (
        "mojow_rock/rock_dataset3/ytvis19/JPEGImages",
        "mojow_rock/rock_dataset3/ytvis19/mj_rock-db3_2_to_17_except_6_with_syn-train.json"
    ),
    "ytvis-mj_rock-db3_2_to_17_except_6_with_syn-val": (
        "mojow_rock/rock_dataset3/ytvis19/JPEGImages",
        "mojow_rock/rock_dataset3/ytvis19/mj_rock-db3_2_to_17_except_6_with_syn-val.json"
    ),

    "ytvis-mj_rock-db3_2_to_17_except_6-train": (
        "mojow_rock/rock_dataset3/ytvis19/JPEGImages",
        "mojow_rock/rock_dataset3/ytvis19/mj_rock-db3_2_to_17_except_6-train.json"
    ),
    "ytvis-mj_rock-db3_2_to_17_except_6-val": (
        "mojow_rock/rock_dataset3/ytvis19/JPEGImages",
        "mojow_rock/rock_dataset3/ytvis19/mj_rock-db3_2_to_17_except_6-val.json"
    ),

    "ytvis-ipsc-all_frames_roi_g2_0_37-train": (
        "ipsc/well3/all_frames_roi",
        "ipsc/well3/all_frames_roi/ytvis19/ytvis-ipsc-all_frames_roi_g2_0_37-train.json"
    ),
    "ytvis-ipsc-all_frames_roi_g2_0_37-val": (
        "ipsc/well3/all_frames_roi",
        "ipsc/well3/all_frames_roi/ytvis19/ytvis-ipsc-all_frames_roi_g2_0_37-val.json"
    ),
    "ytvis-ipsc-all_frames_roi_g2_38_53": (
        "ipsc/well3/all_frames_roi",
        "ipsc/well3/all_frames_roi/ytvis19/ytvis-ipsc-all_frames_roi_g2_38_53.json"
    ),
    "ytvis-ipsc-all_frames_roi_g2_seq_1_38_53": (
        "ipsc/well3/all_frames_roi",
        "ipsc/well3/all_frames_roi/ytvis19/ytvis-ipsc-all_frames_roi_g2_seq_1_38_53.json"
    ),
}

for db in (
        'ext_reorg_roi_g2_0_37',
        'ext_reorg_roi_g2_38_53',
        'ext_reorg_roi_g2_16_53',
        'ext_reorg_roi_g2_0_53',
        'ext_reorg_roi_g2_0_15',
        'ext_reorg_roi_g2_54_126'):
    _PREDEFINED_SPLITS_YTVIS_2019.update(
        {
            f"ytvis-ipsc-{db}": (
                "ipsc/well3/all_frames_roi",
                f"ipsc/well3/all_frames_roi/ytvis19/ipsc-{db}.json"
            ),
            f"ytvis-ipsc-{db}-incremental": (
                "ipsc/well3/all_frames_roi",
                f"ipsc/well3/all_frames_roi/ytvis19/ipsc-{db}-incremental.json"
            ),

        }
    )
    for k in (1, 2, 4, 8, 10, 19, 20):
        _PREDEFINED_SPLITS_YTVIS_2019.update(
            {
                f"ytvis-ipsc-{db}-max_length-{k}": (
                    "ipsc/well3/all_frames_roi",
                    f"ipsc/well3/all_frames_roi/ytvis19/ipsc-{db}-max_length-{k}.json"
                ),
                f"ytvis-ipsc-{db}-max_length-{k}-incremental": (
                    "ipsc/well3/all_frames_roi",
                    f"ipsc/well3/all_frames_roi/ytvis19/ipsc-{db}-max_length-{k}-incremental.json"
                ),

            }
        )


# ==== Predefined splits for YTVIS 2021 ===========
_PREDEFINED_SPLITS_YTVIS_2021 = {
    "ytvis_2021_train": ("ytvis_2021/train/JPEGImages",
                         "ytvis_2021/annotations/instances_train_sub.json"),
    "ytvis_2021_val": ("ytvis_2021/val/JPEGImages",
                       "ytvis_2021/annotations/instances_val_sub.json"),
    "ytvis_2021_test": ("ytvis_2021/test/JPEGImages",
                        "ytvis_2021/test.json"),
    "ytvis_2021_dev": ("ytvis_2021/train/JPEGImages",
                       "ytvis_2021/instances_train_sub.json"),
    "ytvis_2022_val_full": ("ytvis_2022/val/JPEGImages",
                            "ytvis_2022/instances.json"),
    "ytvis_2022_val_sub": ("ytvis_2022/val/JPEGImages",
                           "ytvis_2022/instances_sub.json"),
}


def register_all_ytvis_2019(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2019.items():
        # Assume pre-defined datasets live in `./datasets`.
        if 'mj_rock' in key:
            register_ytvis_instances(
                key,
                _get_ytvis_2019_mj_rocks_instances_meta(),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )
        elif 'ipsc' in key:
            register_ytvis_instances(
                key,
                _get_ytvis_2019_ipsc_instances_meta(),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )
        else:
            register_ytvis_instances(
                key,
                _get_ytvis_2019_instances_meta(),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )


def register_all_ytvis_2021(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2021.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2021_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_ytvis_2019(_root)
    register_all_ytvis_2021(_root)
