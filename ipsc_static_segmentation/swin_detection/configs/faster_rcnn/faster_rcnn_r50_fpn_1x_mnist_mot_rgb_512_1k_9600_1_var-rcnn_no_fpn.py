_base_ = [
    '../_base_/models/faster_rcnn_r50_mnist_mot.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
dataset_type = 'MNIST_MOT'
data_root = '/data_ssd/MNIST_MOT_RGB_512x512_1_1000_9600_var_json/Images/'
img_norm_cfg = dict(
    mean=[93.154, 162.034, 240.900],
    std=[3.868, 2.779, 2.897],
    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            # dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'mnist_mot_rgb_512_1k_9600_1_var-train.json',
        img_prefix=data_root,
        pipeline=train_pipeline),

    val=dict(
        type=dataset_type,
        ann_file=data_root + 'mnist_mot_rgb_512_1k_9600_1_var-val.json',
        img_prefix=data_root,
        pipeline=test_pipeline),

    test=dict(
        type=dataset_type,
        ann_file=data_root + 'mnist_mot_rgb_512_1k_9600_1_var-test.json',
        img_prefix=data_root,
        pipeline=test_pipeline),

    test_dummy=dict(
        type=dataset_type,
        ann_file=data_root + 'mnist_mot_rgb_512_1k_9600_1_var-test_dummy.json',
        img_prefix=data_root,
        pipeline=test_pipeline),

    test_1_10=dict(
        type=dataset_type,
        ann_file=data_root + 'mnist_mot_rgb_512_1k_9600_1_var-test_1_10.json',
        img_prefix=data_root,
        pipeline=test_pipeline),
)

runner = dict(type='EpochBasedRunnerAmp', max_epochs=10000)
checkpoint_config = dict(interval=1, max_keep_ckpts=3)

evaluation = dict(
    metric=['bbox'],
    save_best='bbox_mAP',
)
