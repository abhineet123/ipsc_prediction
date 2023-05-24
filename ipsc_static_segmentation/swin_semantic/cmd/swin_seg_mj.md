
<!-- MarkdownTOC -->

- [db3_2_to_17_except_6_with_syn      @ from_mojow_rocks](#db3_2_to_17_except_6_with_syn___from_mojow_rocks_)
    - [on_september_5_2020       @ db3_2_to_17_except_6_with_syn](#on_september_5_2020___db3_2_to_17_except_6_with_sy_n_)
- [db3_2_to_17_except_6      @ from_mojow_rocks](#db3_2_to_17_except_6___from_mojow_rocks_)
    - [on_september_5_2020       @ db3_2_to_17_except_6](#on_september_5_2020___db3_2_to_17_except_6_)

<!-- /MarkdownTOC -->

<a id="db3_2_to_17_except_6_with_syn___from_mojow_rocks_"></a>
#  db3_2_to_17_except_6_with_syn      @ from_mojow_rocks-->swin_seg
./tools/dist_train.sh configs/swin/db3_2_to_17_except_6_with_syn.py 2  --load-from  pretrained/upernet_swin_base_patch4_window7_512x512.pth --options model.backbone.use_checkpoint=True data.samples_per_gpu=4 --no-validate

python3 tools/train.py configs/swin/db3_2_to_17_except_6_with_syn.py --load-from pretrained/upernet_swin_base_patch4_window7_512x512.pth --options model.backbone.use_checkpoint=True data.samples_per_gpu=4 --init file:///tmp/db3_2_to_17_except_6

<a id="on_september_5_2020___db3_2_to_17_except_6_with_sy_n_"></a>
## on_september_5_2020       @ db3_2_to_17_except_6_with_syn-->swin_seg_mj
python3 tools/test.py configs/swin/db3_2_to_17_except_6_with_syn.py work_dirs/db3_2_to_17_except_6_with_syn/latest.pth --eval mIoU --show --show-dir work_dirs/db3_2_to_17_except_6_with_syn/vis

<a id="db3_2_to_17_except_6___from_mojow_rocks_"></a>
#  db3_2_to_17_except_6      @ from_mojow_rocks-->swin_seg
./tools/dist_train.sh configs/swin/db3_2_to_17_except_6.py 2  --load-from  pretrained/upernet_swin_base_patch4_window7_512x512.pth --options model.backbone.use_checkpoint=True data.samples_per_gpu=4 --no-validate

<a id="on_september_5_2020___db3_2_to_17_except_6_"></a>
## on_september_5_2020       @ db3_2_to_17_except_6-->swin_seg_mj
python3 tools/test.py configs/swin/db3_2_to_17_except_6.py work_dirs/db3_2_to_17_except_6/latest.pth --eval mIoU --show --show-dir work_dirs/db3_2_to_17_except_6/vis --write_masks=0

















