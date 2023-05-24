
<!-- MarkdownTOC -->

- [ext_reorg_roi_g2_16_53      @ from_mojow_rocks](#ext_reorg_roi_g2_16_53___from_mojow_rocks_)
    - [on_g2_0_15       @ ext_reorg_roi_g2_16_53](#on_g2_0_15___ext_reorg_roi_g2_16_53_)
- [ext_reorg_roi_g2_54_126      @ from_mojow_rocks](#ext_reorg_roi_g2_54_126___from_mojow_rocks_)
    - [on_g2_0_15       @ ext_reorg_roi_g2_54_126](#on_g2_0_15___ext_reorg_roi_g2_54_12_6_)

<!-- /MarkdownTOC -->

<a id="ext_reorg_roi_g2_16_53___from_mojow_rocks_"></a>
#  ext_reorg_roi_g2_16_53      @ from_mojow_rocks-->swin_seg
tools/dist_train.sh configs/swin/ext_reorg_roi_g2_16_53.py 2  --load-from  pretrained/upernet_swin_base_patch4_window7_512x512.pth --options model.backbone.use_checkpoint=True data.samples_per_gpu=4 --no-validate

CUDA_VISIBLE_DEVICES=0,1 tools/dist_train.sh configs/swin/ext_reorg_roi_g2_16_53.py 2  --load-from  pretrained/upernet_swin_base_patch4_window7_512x512.pth --options model.backbone.use_checkpoint=True data.samples_per_gpu=2 --no-validate

python3 tools/train.py configs/swin/ext_reorg_roi_g2_16_53.py --load-from pretrained/upernet_swin_base_patch4_window7_512x512.pth --options model.backbone.use_checkpoint=True data.samples_per_gpu=2 --no-validate

<a id="on_g2_0_15___ext_reorg_roi_g2_16_53_"></a>
## on_g2_0_15       @ ext_reorg_roi_g2_16_53-->swin_seg
python3 tools/test.py configs/swin/ext_reorg_roi.py work_dirs/ext_reorg_roi_g2_16_53/latest.pth --eval mIoU --show --show-dir work_dirs/ext_reorg_roi_g2_16_53/vis --write_masks --blended_vis

<a id="ext_reorg_roi_g2_54_126___from_mojow_rocks_"></a>
#  ext_reorg_roi_g2_54_126      @ from_mojow_rocks-->swin_seg
python3 tools/train.py configs/swin/ext_reorg_roi_g2_54_126.py --load-from pretrained/upernet_swin_base_patch4_window7_512x512.pth --options model.backbone.use_checkpoint=True data.samples_per_gpu=2 --no-validate

<a id="on_g2_0_15___ext_reorg_roi_g2_54_12_6_"></a>
## on_g2_0_15       @ ext_reorg_roi_g2_54_126-->swin_seg
python3 tools/test.py configs/swin/ext_reorg_roi_g2_54_126.py work_dirs/ext_reorg_roi_g2_54_126/latest.pth --eval mIoU --show --show-dir work_dirs/ext_reorg_roi_g2_54_126/vis --write_masks --blended_vis



















