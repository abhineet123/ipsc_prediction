
<!-- MarkdownTOC -->

- [ext_reorg_roi_g2_16_53](#ext_reorg_roi_g2_16_5_3_)
    - [on_g2_0_15       @ ext_reorg_roi_g2_16_53](#on_g2_0_15___ext_reorg_roi_g2_16_53_)
- [ext_reorg_roi_g2_54_126](#ext_reorg_roi_g2_54_126_)
    - [on_g2_0_15       @ ext_reorg_roi_g2_54_126](#on_g2_0_15___ext_reorg_roi_g2_54_12_6_)

<!-- /MarkdownTOC -->

<a id="ext_reorg_roi_g2_16_5_3_"></a>
#  ext_reorg_roi_g2_16_53
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/swin/ext_reorg_roi_g2_16_53.py --load-from pretrained/upernet_swin_base_patch4_window7_512x512.pth --options model.backbone.use_checkpoint=True data.samples_per_gpu=2 --no-validate
`dist`
CUDA_VISIBLE_DEVICES=0,1,2 tools/dist_train.sh configs/swin/ext_reorg_roi_g2_16_53.py 2  --load-from  pretrained/upernet_swin_base_patch4_window7_512x512.pth --options model.backbone.use_checkpoint=True data.samples_per_gpu=2 --no-validate
<a id="on_g2_0_15___ext_reorg_roi_g2_16_53_"></a>
## on_g2_0_15       @ ext_reorg_roi_g2_16_53-->sws-ipsc
CUDA_VISIBLE_DEVICES=2 python tools/test.py configs/swin/ext_reorg_roi_g2_16_53.py work_dirs/ext_reorg_roi_g2_16_53 --eval mIoU --show --show-dir work_dirs/ext_reorg_roi_g2_16_53/vis --write_masks --blended_vis --write_empty

<a id="ext_reorg_roi_g2_54_126_"></a>
#  ext_reorg_roi_g2_54_126 
CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/swin/ext_reorg_roi_g2_54_126.py --load-from pretrained/upernet_swin_base_patch4_window7_512x512.pth --options model.backbone.use_checkpoint=True data.samples_per_gpu=2 --no-validate
<a id="on_g2_0_15___ext_reorg_roi_g2_54_12_6_"></a>
## on_g2_0_15       @ ext_reorg_roi_g2_54_126-->sws-ipsc
CUDA_VISIBLE_DEVICES=2 python tools/test.py configs/swin/ext_reorg_roi_g2_54_126.py work_dirs/ext_reorg_roi_g2_54_126 --eval mIoU --show --show-dir work_dirs/ext_reorg_roi_g2_54_126/vis --write_masks --blended_vis --write_empty



















