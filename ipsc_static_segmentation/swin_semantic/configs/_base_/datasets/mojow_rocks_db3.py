# dataset settings
dataset_type = 'MojowRocks'
data_root = '/data/mojow_rock/rock_dataset3/'
img_norm_cfg = dict(
    mean=[143.12, 137.32, 138.46],
    std=[38.39, 41.68, 41.96],
    to_rgb=True)

img_scale = (1040, 1040)
crop_size = (960, 960)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        # resave_masks=1,
        type=dataset_type,
        img_dir='images',
        ann_dir='masks_large_huge',
        data_root=data_root,
        split='db3_2_to_17_except_6_with_syn-all.txt',
        pipeline=train_pipeline),
    val=dict(
        # resave_masks=1,
        type=dataset_type,
        img_dir='images',
        ann_dir='masks_large_huge',
        data_root=data_root,
        split='db3_2_to_17_except_6_with_syn-val.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        img_dir='images',
        ann_dir='masks_large_huge',
        data_root=data_root,
        split='september_5_2020.txt',
        pipeline=test_pipeline),
)
# checkpoint_config = dict(interval=1, max_keep_ckpts=3)
# checkpoint_config = dict(by_epoch=False, interval=500, max_keep_ckpts=5)
