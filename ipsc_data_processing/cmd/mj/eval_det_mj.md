<!-- MarkdownTOC -->

- [swi-db4](#swi_db4_)
- [swi-db3_sept5_syn3-large_huge](#swi_db3_sept5_syn3_large_huge_)
    - [sept5       @ swi-db3_sept5_syn3-large_huge](#sept5___swi_db3_sept5_syn3_large_hug_e_)
    - [sept5_syn        @ swi-db3_sept5_syn3-large_huge](#sept5_syn___swi_db3_sept5_syn3_large_hug_e_)

<!-- /MarkdownTOC -->


<a id="swi_db4_"></a>
# swi-db4 
python3 eval_det.py cfg=mj:db4 img_paths=lists/db4.txt det_paths=log/swi/db3_2_to_17_except_6-large_huge/db4/csv save_vis=1 show_vis=0  save_suffix==mj/swi-db4 vid_fmt=H264,2,mkv show_tp=0

<a id="swi_db3_sept5_syn3_large_huge_"></a>
# swi-db3_sept5_syn3-large_huge 
<a id="sept5___swi_db3_sept5_syn3_large_hug_e_"></a>
## sept5       @ swi-db3_sept5_syn3-large_huge-->eval_det_mj

python3 eval_det.py cfg=mj:db3:syn img_paths=september_5_2020 det_paths=log/swi/db3_sept5_syn3-large_huge/september_5_2020/csv  save_suffix=mj/swi-db3_sept5_syn-on_sept5 vid_fmt=mp4v,5,jpg  check_seq_name=0 combine_dets=1 fps_to_gt=1

python3 eval_det.py cfg=mj:db3:syn img_paths=september_5_2020 det_paths=log/swi/db3_sept5_syn3-large_huge/september_5_2020/csv  save_suffix=mj/swi-db3_sept5_syn-on_sept5-dbg vid_fmt=mp4v,5,jpg  check_seq_name=0 combine_dets=1 fps_to_gt=1 gt_paths=/data/mAP/mj/swi-db3_sept5_syn-on_sept5/september_5_2020_gt_with_fp_nex_whole-dbg.csv

<a id="sept5_syn___swi_db3_sept5_syn3_large_hug_e_"></a>
##sept5_syn        @ swi-db3_sept5_syn3-large_huge-->eval_det_mj
python3 eval_det.py cfg=mj:db3:syn img_paths=syn/part14_on_part4_on_part5_on_september_5_2020 det_paths=log/swi/db3_sept5_syn3-large_huge/part14_on_part4_on_part5_on_september_5_2020/csv save_suffix=mj/swi-db3_sept5_syn  save_vis=1 vid_fmt=mp4v,5,jpg


