data_root_db3 = '/data/mojow_rock/rock_dataset3/'
data_root_db4 = '/data/mojow_rock/rock_dataset4/'
data_root_db4_rockmaps = '/data/mojow_rock/rock_dataset4/rockmaps/'
data_root_db5 = '/data/mojow_rock/rock_dataset5/'

img_norm_cfg = dict(
    mean=[143.12, 137.32, 138.46], std=[38.39, 41.68, 41.96], to_rgb=True)
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
    workers_per_gpu=0,

    # train=dict(
    #     type=dataset_type,
    #     img_prefix=data_root,
    #     pipeline=train_pipeline),
    # val=dict(
    #     type=dataset_type,
    #     img_prefix=data_root,
    #     pipeline=test_pipeline),

    part1=dict(
        type='MojowRocks',
        ann_file=data_root_db3 + 'part1-large_huge.json',
        img_prefix=data_root_db3,
        pipeline=test_pipeline
    ),


    part6=dict(
        type='MojowRocks',
        ann_file=data_root_db3 + 'part6-large_huge.json',
        img_prefix=data_root_db3,
        pipeline=test_pipeline
    ),

    db3_2_to_17_except_6_large_huge_train=dict(
        type='MojowRocksSyn',
        ann_file=data_root_db3 + 'db3_2_to_17_except_6-large_huge-train.json',
        img_prefix=data_root_db3,
        pipeline=test_pipeline,
        samples_per_gpu=3,
    ),
    db3_2_to_17_except_6_sept5_2k_100_large_huge=dict(
        type='MojowRocksSyn',
        ann_file=data_root_db3 + 'db3_2_to_17_except_6_sept5_2k_100-large_huge.json',
        img_prefix=data_root_db3,
        pipeline=test_pipeline,
        samples_per_gpu=3,
    ),
    sept5_2k_100=dict(
        type='MojowRocks',
        ann_file=data_root_db3 + 'september_5_2020_2K_100-large_huge.json',
        img_prefix=data_root_db3,
        pipeline=test_pipeline,
        samples_per_gpu=3,
    ),
    september_5_2020=dict(
        type='MojowRocks',
        ann_file=data_root_db3 + 'september_5_2020-large_huge.json',
        img_prefix=data_root_db3,
        pipeline=test_pipeline,
        samples_per_gpu=3,
    ),
    september_5_2020_fps=dict(
        type='MojowRocksFPsToGT',
        ann_file=data_root_db3 + 'september_5_2020-large_huge.json',
        img_prefix=data_root_db3,
        pipeline=test_pipeline,
        samples_per_gpu=3,
    ),
    part14_on_part4_on_part5_on_september_5_2020=dict(
        type='MojowRocksSyn',
        ann_file=data_root_db3 + 'part14_on_part4_on_part5_on_september_5_2020-large_huge.json',
        img_prefix=data_root_db3,
        pipeline=test_pipeline,
        samples_per_gpu=3,
    ),
    part4_on_part5_on_september_5_2020=dict(
        type='MojowRocksSyn',
        ann_file=data_root_db3 + 'part4_on_part5_on_september_5_2020-large_huge.json',
        img_prefix=data_root_db3,
        pipeline=test_pipeline,
        samples_per_gpu=3,
    ),
    part5_on_september_5_2020=dict(
        type='MojowRocksSyn',
        ann_file=data_root_db3 + 'part5_on_september_5_2020-large_huge.json',
        img_prefix=data_root_db3,
        pipeline=test_pipeline,
        samples_per_gpu=3,
    ),
    db5_part1=dict(
        type='MojowRocks',
        ann_file=data_root_db5 + 'db5-part1.json',
        img_prefix=data_root_db5,
        pipeline=test_pipeline,
        samples_per_gpu=1,
    ),
    db5_part2=dict(
        type='MojowRocks',
        ann_file=data_root_db5 + 'db5-part2.json',
        img_prefix=data_root_db5,
        pipeline=test_pipeline,
        samples_per_gpu=1,
    ),

    db4=dict(
        type='MojowRocks',
        ann_file=data_root_db4 + 'db4.json',
        img_prefix=data_root_db4,
        pipeline=test_pipeline,
        samples_per_gpu=1,
    ),

    db4_rockmaps=dict(
        type='MojowRocks',
        ann_file=data_root_db4_rockmaps + 'db4-rockmaps.json',
        img_prefix=data_root_db4_rockmaps,
        pipeline=test_pipeline,
        samples_per_gpu=8,
    ),

    db4_rockmaps_syn=dict(
        type='MojowRocksSyn',
        ann_file=data_root_db4 + 'db4-rockmaps_syn.json',
        img_prefix=data_root_db4,
        pipeline=test_pipeline,
        samples_per_gpu=8,
    ),

)

evaluation = dict(metric=['bbox', 'segm'])
