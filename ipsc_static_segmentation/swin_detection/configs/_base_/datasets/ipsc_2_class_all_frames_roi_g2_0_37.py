dataset_type = 'IPSC2Class'
data_root = '/data/ipsc/well3/all_frames_roi/'
img_norm_cfg = dict(
    mean=[118.52, 118.52, 118.52], std=[20.43, 20.43, 20.43], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(666, 400), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(666, 400),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'all_frames_roi_g2_0_37-train.json',
        img_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'all_frames_roi_g2_0_37-val.json',
        img_prefix=data_root,
        pipeline=test_pipeline),
    g2_39_53=dict(
        type=dataset_type,
        ann_file=data_root + 'all_frames_roi_g2_38_53.json',
        img_prefix=data_root,
        pipeline=test_pipeline),
    g2_seq_1_39_53=dict(
        type=dataset_type,
        ann_file=data_root + 'all_frames_roi_g2_seq_1_39_53.json',
        img_prefix=data_root,
        pipeline=test_pipeline),
)

evaluation = dict(metric=['bbox', 'segm'])
