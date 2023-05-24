import os

from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from detectron2.data.datasets.coco import register_coco_instances

from .ovis import _get_ovis_instances_meta
from vita.data.datasets.ytvis import (
    register_ytvis_instances,
    _get_ytvis_2019_instances_meta,
    _get_ytvis_2021_instances_meta,
    _get_ytvis_2019_ipsc_instances_meta,
    _get_ytvis_2019_mj_rocks_instances_meta,
)

# ==== Predefined splits for YTVIS 2019 ===========

# ==== Predefined splits for YTVIS 2019 ===========
_PREDEFINED_SPLITS_YTVIS_2019 = {
    "ytvis_2019_train": ("ytvis_2019/train/JPEGImages",
                         "ytvis_2019/train.json"),
    "ytvis_2019_val": ("ytvis_2019/valid/JPEGImages",
                       "ytvis_2019/valid.json"),
    "ytvis_2019_test": ("ytvis_2019/test/JPEGImages",
                        "ytvis_2019/test.json"),
    "ytvis_2019_val_all_frames": ("ytvis_2019/valid_all_frames/JPEGImages",
                                  "ytvis_2019/valid_all_frames.json"),

    # mj_rock
    "ytvis-mj_rock-db3-part12-large_huge-train": (
        "mojow_rock/rock_dataset3/ytvis19/JPEGImages",
        "mojow_rock/rock_dataset3/ytvis19/mj_rock-db3-part12-large_huge-train.json"
    ),
    "ytvis-mj_rock-db3-part12-large_huge-val": (
        "mojow_rock/rock_dataset3/ytvis19/JPEGImages",
        "mojow_rock/rock_dataset3/ytvis19/mj_rock-db3-part12-large_huge-val.json"
    ),

    "ytvis-mj_rock-db3_2_to_17_except_6_with_syn-large_huge-train": (
        "mojow_rock/rock_dataset3/ytvis19/JPEGImages",
        "mojow_rock/rock_dataset3/ytvis19/mj_rock-db3_2_to_17_except_6_with_syn-large_huge-train.json"
    ),
    "ytvis-mj_rock-db3_2_to_17_except_6_with_syn-large_huge-val": (
        "mojow_rock/rock_dataset3/ytvis19/JPEGImages",
        "mojow_rock/rock_dataset3/ytvis19/mj_rock-db3_2_to_17_except_6_with_syn-large_huge-val.json"
    ),
    "ytvis-mj_rock-september_5_2020": (
        "mojow_rock/rock_dataset3/ytvis19/JPEGImages",
        "mojow_rock/rock_dataset3/ytvis19/mj_rock-september_5_2020.json"
    ),
    "ytvis-mj_rock-september_5_2020-large_huge": (
        "mojow_rock/rock_dataset3/ytvis19/JPEGImages",
        "mojow_rock/rock_dataset3/ytvis19/mj_rock-september_5_2020-large_huge.json"
    ),
    "ytvis-mj_rock-september_5_2020-large_huge-max_length-200": (
        "mojow_rock/rock_dataset3/ytvis19/JPEGImages",
        "mojow_rock/rock_dataset3/ytvis19/mj_rock-september_5_2020-large_huge-max_length-200.json"
    ),

    "ytvis-mj_rock-db3_2_to_17_except_6-large_huge-train": (
        "mojow_rock/rock_dataset3/ytvis19/JPEGImages",
        "mojow_rock/rock_dataset3/ytvis19/mj_rock-db3_2_to_17_except_6-large_huge-train.json"
    ),
    "ytvis-mj_rock-db3_2_to_17_except_6-large_huge-val": (
        "mojow_rock/rock_dataset3/ytvis19/JPEGImages",
        "mojow_rock/rock_dataset3/ytvis19/mj_rock-db3_2_to_17_except_6-large_huge-val.json"
    ),

    # ipsc-all_frames_roi

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

    "ytvis-ipsc-ext_reorg_roi_g2_seq_1_38_53": (
        "ipsc/well3/all_frames_roi",
        "ipsc/well3/all_frames_roi/ytvis19/ipsc-ext_reorg_roi_g2_seq_1_38_53.json"
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
                         "ytvis_2021/train.json"),
    "ytvis_2021_val": ("ytvis_2021/valid/JPEGImages",
                       "ytvis_2021/valid.json"),
    "ytvis_2021_test": ("ytvis_2021/test/JPEGImages",
                        "ytvis_2021/test.json"),
}

# ====    Predefined splits for OVIS    ===========
_PREDEFINED_SPLITS_OVIS = {
    "ovis_train": ("ovis/train",
                   "ovis/annotations/train.json"),
    "ovis_val": ("ovis/valid",
                 "ovis/annotations/valid.json"),
    "ovis_test": ("ovis/test",
                  "ovis/annotations/test.json"),
}

_PREDEFINED_SPLITS_COCO_VIDEO = {
    "coco2ytvis2019_train": ("coco/train2017", "coco/annotations/coco2ytvis2019_train.json"),
    "coco2ytvis2019_val": ("coco/val2017", "coco/annotations/coco2ytvis2019_val.json"),
    "coco2ytvis2021_train": ("coco/train2017", "coco/annotations/coco2ytvis2021_train.json"),
    "coco2ytvis2021_val": ("coco/val2017", "coco/annotations/coco2ytvis2021_val.json"),
    "coco2ovis_train": ("coco/train2017", "coco/annotations/coco2ovis_train.json"),
    "coco2ovis_val": ("coco/val2017", "coco/annotations/coco2ovis_val.json"),
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


def register_all_coco_video(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_COCO_VIDEO.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_builtin_metadata("coco"),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_ovis(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_OVIS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ovis_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_ovis(_root)
    register_all_ytvis_2019(_root)
    register_all_ytvis_2021(_root)
    register_all_coco_video(_root)
