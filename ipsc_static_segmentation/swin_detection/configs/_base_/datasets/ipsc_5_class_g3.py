dataset_type = 'IPSC5Class'
data_root = '/data/ipsc_5_class_raw/'
img_norm_cfg = dict(
    mean=[117.414, 117.414, 117.414], std=[22.640, 22.640, 22.640], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
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
        img_scale=(1333, 800),
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
        ann_file=data_root + 'ipsc_5_class_g3_train.json',
        img_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'ipsc_5_class_g3_val.json',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'ipsc_5_class_all.json',
        img_prefix=data_root,
        pipeline=test_pipeline),
    g3=dict(
        type=dataset_type,
        ann_file=data_root + 'ipsc_5_class_g3_all.json',
        img_prefix=data_root,
        pipeline=test_pipeline),
    g4s=dict(
        type=dataset_type,
        ann_file=data_root + 'ipsc_5_class_g4s_all.json',
        img_prefix=data_root,
        pipeline=test_pipeline),
    unlabeled=dict(
        type=dataset_type,
        ann_file=data_root + 'Test_211208.json',
        img_prefix=data_root,
        pipeline=test_pipeline),
    nd03=dict(
        type=dataset_type,
        ann_file=data_root + 'nd03.json',
        img_prefix=data_root,
        pipeline=test_pipeline),
    realtime_test_images=dict(
        type=dataset_type,
        ann_file=None,
        img_prefix=data_root,
        pipeline=test_pipeline),


)

evaluation = dict(metric=['bbox', 'segm'])
