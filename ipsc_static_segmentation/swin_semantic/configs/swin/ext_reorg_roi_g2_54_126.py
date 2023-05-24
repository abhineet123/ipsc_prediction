_base_ = [
    '../_base_/models/upernet_swin.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k_ckpt500.py'
]
# dataset settings
dataset_type = 'IPSC2Class'
data_root = '/data/ipsc/well3/all_frames_roi/'
img_norm_cfg = dict(
    mean=[130.26, 130.26, 130.26],
    std=[16.86, 16.86, 16.86],
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
        type=dataset_type,
        img_dir='images',
        ann_dir='masks',
        data_root=data_root,
        split='ext_reorg_roi_g2_54_126.txt',
        pipeline=train_pipeline),
    test=dict(
        type=dataset_type,
        img_dir='images',
        ann_dir='masks',
        data_root=data_root,
        split='ext_reorg_roi_g2_0_15.txt',
        pipeline=test_pipeline),
)

model = dict(
    backbone=dict(
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=False
    ),
    decode_head=dict(
        in_channels=[128, 256, 512, 1024],
        num_classes=3
    ),
    auxiliary_head=dict(
        in_channels=512,
        num_classes=3
    ))

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
# data = dict(samples_per_gpu=4)
# evaluation = dict(interval=4000)

# checkpoint_config = dict(interval=1, max_keep_ckpts=3)
# checkpoint_config = dict(by_epoch=False, interval=500, max_keep_ckpts=5)
