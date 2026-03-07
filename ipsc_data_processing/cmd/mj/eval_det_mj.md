<!-- MarkdownTOC -->

- [swi-db3_2_to_17_except_6-large_huge-fps_to_gt](#swi_db3_2_to_17_except_6_large_huge_fps_to_gt_)
    - [on-train_sept5_2k_100       @ swi-db3_2_to_17_except_6-large_huge-fps_to_gt](#on_train_sept5_2k_100___swi_db3_2_to_17_except_6_large_huge_fps_to_g_t_)
- [swi-db3_2_to_17_except_6_sept5_2k_100-large_huge-fps_to_gt](#swi_db3_2_to_17_except_6_sept5_2k_100_large_huge_fps_to_g_t_)
    - [on-train_sept5_2k_100       @ swi-db3_2_to_17_except_6_sept5_2k_100-large_huge-fps_to_gt](#on_train_sept5_2k_100___swi_db3_2_to_17_except_6_sept5_2k_100_large_huge_fps_to_gt_)
    - [on-september_5_2020_fps       @ swi-db3_2_to_17_except_6_sept5_2k_100-large_huge-fps_to_gt](#on_september_5_2020_fps___swi_db3_2_to_17_except_6_sept5_2k_100_large_huge_fps_to_gt_)
- [swi-db3_2_to_17_except_6-large_huge](#swi_db3_2_to_17_except_6_large_huge_)
    - [on-train       @ swi-db3_2_to_17_except_6-large_huge](#on_train___swi_db3_2_to_17_except_6_large_hug_e_)
    - [on-train_sept5_2k_100       @ swi-db3_2_to_17_except_6-large_huge](#on_train_sept5_2k_100___swi_db3_2_to_17_except_6_large_hug_e_)
    - [on-sept5_2k_100       @ swi-db3_2_to_17_except_6-large_huge](#on_sept5_2k_100___swi_db3_2_to_17_except_6_large_hug_e_)
    - [on-db5       @ swi-db3_2_to_17_except_6-large_huge](#on_db5___swi_db3_2_to_17_except_6_large_hug_e_)
- [swi-db3_sept5_syn3-large_huge](#swi_db3_sept5_syn3_large_huge_)
    - [sept5       @ swi-db3_sept5_syn3-large_huge](#sept5___swi_db3_sept5_syn3_large_hug_e_)
    - [sept5_syn       @ swi-db3_sept5_syn3-large_huge](#sept5_syn___swi_db3_sept5_syn3_large_hug_e_)
- [swi-db4](#swi_db4_)

<!-- /MarkdownTOC -->

<a id="swi_db3_2_to_17_except_6_large_huge_fps_to_gt_"></a>
# swi-db3_2_to_17_except_6-large_huge-fps_to_gt  
<a id="on_train_sept5_2k_100___swi_db3_2_to_17_except_6_large_huge_fps_to_g_t_"></a>
## on-train_sept5_2k_100       @ swi-db3_2_to_17_except_6-large_huge-fps_to_gt-->eval_det_mj
python3 eval_det.py cfg=mj:db3 img_paths=lists/db3_2_to_17_except_6_sept5_2k_100.txt det_paths=log/swi/db3_2_to_17_except_6-large_huge-fps_to_gt/epoch_86_on_db3_2_to_17_except_6_sept5_2k_100_large_huge/csv  save_suffix=mj/swi-db3_2_to_17_except_6-large_huge-fps-on_train_sept5_2k_100 check_seq_name=1 combine_dets=0 gt_csv_suffix=large_huge ignore_invalid_class=1 load_gt=1

python3 eval_det.py cfg=mj:db3 img_paths=lists/db3_2_to_17_except_6_sept5_2k_100.txt det_paths=log/swi/db3_2_to_17_except_6-large_huge-fps_to_gt/epoch_505_on_db3_2_to_17_except_6_sept5_2k_100_large_huge/csv  save_suffix=mj/swi-db3_2_to_17_except_6-large_huge-fps-on_train_sept5_2k_100-505 check_seq_name=1 combine_dets=0 gt_csv_suffix=large_huge ignore_invalid_class=1 load_gt=1 gt_pkl=swi-db3_2_to_17_except_6-large_huge-fps-on_train_sept5_2k_100.pkl save_vis=1 vid_fmt=mp4v,5,mp4

<a id="swi_db3_2_to_17_except_6_sept5_2k_100_large_huge_fps_to_g_t_"></a>
# swi-db3_2_to_17_except_6_sept5_2k_100-large_huge-fps_to_gt  
<a id="on_train_sept5_2k_100___swi_db3_2_to_17_except_6_sept5_2k_100_large_huge_fps_to_gt_"></a>
## on-train_sept5_2k_100       @ swi-db3_2_to_17_except_6_sept5_2k_100-large_huge-fps_to_gt-->eval_det_mj
python3 eval_det.py cfg=mj:db3 img_paths=lists/db3_2_to_17_except_6_sept5_2k_100.txt det_paths=log/swi/db3_2_to_17_except_6_sept5_2k_100-large_huge-fps_to_gt/db3_2_to_17_except_6_sept5_2k_100_large_huge/csv  save_suffix=mj/swi-db3_2_to_17_except_6_sept5_2k_100-large_huge-fps-on_train_sept5_2k_100 check_seq_name=1 combine_dets=0 gt_csv_suffix=large_huge ignore_invalid_class=1

python3 eval_det.py cfg=mj:db3 img_paths=lists/db3_2_to_17_except_6_sept5_2k_100.txt det_paths=log/swi/db3_2_to_17_except_6_sept5_2k_100-large_huge-fps_to_gt/epoch_227_on_db3_2_to_17_except_6_sept5_2k_100_large_huge/csv  save_suffix=mj/swi-db3_2_to_17_except_6_sept5_2k_100-large_huge-fps-on_train_sept5_2k_100-227 check_seq_name=1 combine_dets=0 gt_csv_suffix=large_huge ignore_invalid_class=1 load_gt=1 gt_pkl=swi-db3_2_to_17_except_6_sept5_2k_100-large_huge-fps-on_train_sept5_2k_100.pkl
<a id="on_september_5_2020_fps___swi_db3_2_to_17_except_6_sept5_2k_100_large_huge_fps_to_gt_"></a>
## on-september_5_2020_fps       @ swi-db3_2_to_17_except_6_sept5_2k_100-large_huge-fps_to_gt-->eval_det_mj
python3 eval_det.py cfg=mj:db3 img_paths=september_5_2020 det_paths=log/swi/db3_2_to_17_except_6_sept5_2k_100-large_huge-fps_to_gt/epoch_227_on_september_5_2020_fps/csv save_suffix=mj/swi-db3_2_to_17_except_6_sept5_2k_100-large_huge-fps-on_september_5_2020_fps-227 combine_dets=1 gt_csv_suffix=large_huge ignore_invalid_class=1 check_seq_name=0  enable_mask=0

<a id="swi_db3_2_to_17_except_6_large_huge_"></a>
# swi-db3_2_to_17_except_6-large_huge 
<a id="on_train___swi_db3_2_to_17_except_6_large_hug_e_"></a>
## on-train       @ swi-db3_2_to_17_except_6-large_huge-->eval_det_mj
python3 eval_det.py cfg=mj:db3 img_paths=lists/db3_2_to_17_except_6.txt det_paths=log/swi/db3_2_to_17_except_6-large_huge/db3_2_to_17_except_6_large_huge_train/csv  save_suffix=mj/swi-db3_2_to_17_except_6-large_huge-on_train check_seq_name=1 combine_dets=0 fps_to_gt=1 gt_csv_suffix=large_huge

<a id="on_train_sept5_2k_100___swi_db3_2_to_17_except_6_large_hug_e_"></a>
## on-train_sept5_2k_100       @ swi-db3_2_to_17_except_6-large_huge-->eval_det_mj
python3 eval_det.py cfg=mj:db3 img_paths=lists/db3_2_to_17_except_6_sept5_2k_100.txt det_paths=log/swi/db3_2_to_17_except_6-large_huge/db3_2_to_17_except_6_sept5_2k_100_large_huge/csv  save_suffix=mj/swi-db3_2_to_17_except_6-large_huge-on_train_sept5_2k_100 check_seq_name=1 combine_dets=0 fps_to_gt=1 gt_csv_suffix=large_huge

<a id="on_sept5_2k_100___swi_db3_2_to_17_except_6_large_hug_e_"></a>
## on-sept5_2k_100       @ swi-db3_2_to_17_except_6-large_huge-->eval_det_mj
python3 eval_det.py cfg=mj:db3:vis img_paths=september_5_2020_2K_100 det_paths=log/swi/db3_2_to_17_except_6-large_huge/sept5_2k_100/csv  save_suffix=mj/swi-db3_2_to_17_except_6-large_huge-on_sept5_2k_100 vid_fmt=mp4v,5,jpg check_seq_name=0 combine_dets=0 fps_to_gt=1

<a id="on_db5___swi_db3_2_to_17_except_6_large_hug_e_"></a>
## on-db5       @ swi-db3_2_to_17_except_6-large_huge-->eval_det_mj
python3 eval_det.py cfg=mj:db5 img_paths=part1 det_paths=log/swi/db3_2_to_17_except_6-large_huge/epoch_231_on_db5_part1/csv  save_suffix=mj/swi-db3_2_to_17_except_6-large_huge-epoch_231_on_db5_part1 check_seq_name=1 combine_dets=0 fps_to_gt=0 enable_mask=0 save_vis=1 vid_fmt=mp4v,5,mp4

python3 eval_det.py cfg=mj:db5 img_paths=part2 det_paths=log/swi/db3_2_to_17_except_6-large_huge/epoch_231_on_db5_part2/csv  save_suffix=mj/swi-db3_2_to_17_except_6-large_huge-epoch_231_on_db5_part2 check_seq_name=1 combine_dets=0 fps_to_gt=0 enable_mask=0 save_vis=1 vid_fmt=mp4v,5,mp4

<a id="swi_db3_sept5_syn3_large_huge_"></a>
# swi-db3_sept5_syn3-large_huge 
<a id="sept5___swi_db3_sept5_syn3_large_hug_e_"></a>
## sept5       @ swi-db3_sept5_syn3-large_huge-->eval_det_mj
python3 eval_det.py cfg=mj:db3:syn img_paths=september_5_2020 det_paths=log/swi/db3_sept5_syn3-large_huge/september_5_2020/csv  save_suffix=mj/swi-db3_sept5_syn-on_sept5 vid_fmt=mp4v,5,jpg  check_seq_name=0 combine_dets=1 fps_to_gt=1

python3 eval_det.py cfg=mj:db3:syn img_paths=september_5_2020 det_paths=log/swi/db3_sept5_syn3-large_huge/september_5_2020/csv  save_suffix=mj/swi-db3_sept5_syn-on_sept5-dbg vid_fmt=mp4v,5,jpg  check_seq_name=0 combine_dets=1 fps_to_gt=1 gt_paths=/data/mAP/mj/swi-db3_sept5_syn-on_sept5/september_5_2020_gt_with_fp_nex_whole-dbg.csv

<a id="sept5_syn___swi_db3_sept5_syn3_large_hug_e_"></a>
## sept5_syn       @ swi-db3_sept5_syn3-large_huge-->eval_det_mj
python3 eval_det.py cfg=mj:db3:syn img_paths=syn/part14_on_part4_on_part5_on_september_5_2020 det_paths=log/swi/db3_sept5_syn3-large_huge/part14_on_part4_on_part5_on_september_5_2020/csv save_suffix=mj/swi-db3_sept5_syn  save_vis=1 vid_fmt=mp4v,5,jpg


<a id="swi_db4_"></a>
# swi-db4 
python3 eval_det.py cfg=mj:db4 img_paths=lists/db4.txt det_paths=log/swi/db3_2_to_17_except_6-large_huge/db4/csv save_vis=1 show_vis=0  save_suffix==mj/swi-db4 vid_fmt=H264,2,mkv show_tp=0

