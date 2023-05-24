<!-- MarkdownTOC -->

- [all_frames_roi_g2_38_53       @ all_frames_roi](#all_frames_roi_g2_38_53___all_frames_roi_)
- [all_frames_roi_g2_seq_1_38_53       @ all_frames_roi](#all_frames_roi_g2_seq_1_38_53___all_frames_roi_)
- [all_frames_roi_g3_53_92       @ all_frames_roi](#all_frames_roi_g3_53_92___all_frames_roi_)
- [ext_reorg_roi_g2_0_37       @ xml_to_ytvis](#ext_reorg_roi_g2_0_37___xml_to_ytvis_)
    - [max_length-20       @ ext_reorg_roi_g2_0_37](#max_length_20___ext_reorg_roi_g2_0_3_7_)
    - [max_length-10       @ ext_reorg_roi_g2_0_37](#max_length_10___ext_reorg_roi_g2_0_3_7_)
- [ext_reorg_roi_g2_38_53       @ xml_to_ytvis](#ext_reorg_roi_g2_38_53___xml_to_ytvis_)
    - [incremental       @ ext_reorg_roi_g2_38_53](#incremental___ext_reorg_roi_g2_38_53_)
- [ext_reorg_roi_g2_16_53       @ xml_to_ytvis](#ext_reorg_roi_g2_16_53___xml_to_ytvis_)
- [ext_reorg_roi_g2_0_15       @ xml_to_ytvis](#ext_reorg_roi_g2_0_15___xml_to_ytvis_)
    - [incremental       @ ext_reorg_roi_g2_0_15](#incremental___ext_reorg_roi_g2_0_1_5_)
- [ext_reorg_roi_g2_54_126       @ xml_to_ytvis](#ext_reorg_roi_g2_54_126___xml_to_ytvis_)
    - [subseq       @ ext_reorg_roi_g2_54_126](#subseq___ext_reorg_roi_g2_54_12_6_)
- [ext_reorg_roi_g2_0_53       @ xml_to_ytvis](#ext_reorg_roi_g2_0_53___xml_to_ytvis_)
    - [incremental       @ ext_reorg_roi_g2_0_53](#incremental___ext_reorg_roi_g2_0_5_3_)

<!-- /MarkdownTOC -->


<a id="all_frames_roi_g2_38_53___all_frames_roi_"></a>
# all_frames_roi_g2_38_53       @ all_frames_roi-->xml_to_ytvis
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=all_frames_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=38 end_frame_id=53 ignore_invalid_label=1 val_ratio=0 min_val=0 allow_missing_images=0 get_img_stats=0 description=ipsc-all_frames_roi_g2_38_53 save_masks=0 coco_rle=1

<a id="all_frames_roi_g2_seq_1_38_53___all_frames_roi_"></a>
# all_frames_roi_g2_seq_1_38_53       @ all_frames_roi-->xml_to_ytvis
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=all_frames_roi.txt n_seq=1 class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=38 end_frame_id=53 ignore_invalid_label=1 val_ratio=0 min_val=0 allow_missing_images=0 get_img_stats=0 description=ipsc-all_frames_roi_g2_seq_1_38_53 save_masks=0 coco_rle=1

<a id="all_frames_roi_g3_53_92___all_frames_roi_"></a>
# all_frames_roi_g3_53_92       @ all_frames_roi-->xml_to_ytvis
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=all_frames_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=53 end_frame_id=92 ignore_invalid_label=1 val_ratio=0 min_val=0 allow_missing_images=0 get_img_stats=1 description=ipsc-all_frames_roi_g3_53_92 save_masks=1

<a id="ext_reorg_roi_g2_0_37___xml_to_ytvis_"></a>
# ext_reorg_roi_g2_0_37       @ xml_to_ytvis-->xml_to_ytvis
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=0 end_frame_id=37 ignore_invalid_label=1 val_ratio=0 allow_missing_images=0 get_img_stats=0 description=ipsc-ext_reorg_roi_g2_0_37 save_masks=0

<a id="max_length_20___ext_reorg_roi_g2_0_3_7_"></a>
## max_length-20       @ ext_reorg_roi_g2_0_37-->xml_to_ytvis
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=0 end_frame_id=37 ignore_invalid_label=1 val_ratio=0 allow_missing_images=0 get_img_stats=0 description=ipsc-ext_reorg_roi_g2_0_37 save_masks=0 max_length=20 n_proc=12

<a id="max_length_10___ext_reorg_roi_g2_0_3_7_"></a>
## max_length-10       @ ext_reorg_roi_g2_0_37-->xml_to_ytvis
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=0 end_frame_id=37 ignore_invalid_label=1 val_ratio=0 allow_missing_images=0 get_img_stats=0 description=ipsc-ext_reorg_roi_g2_0_37 save_masks=0 max_length=10 n_proc=12

<a id="ext_reorg_roi_g2_38_53___xml_to_ytvis_"></a>
# ext_reorg_roi_g2_38_53       @ xml_to_ytvis-->xml_to_ytvis
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=38 end_frame_id=53 ignore_invalid_label=1 val_ratio=0 min_val=0 allow_missing_images=0 get_img_stats=0 description=ipsc-ext_reorg_roi_g2_38_53 save_masks=0 n_proc=12

<a id="incremental___ext_reorg_roi_g2_38_53_"></a>
## incremental       @ ext_reorg_roi_g2_38_53-->xml_to_ytvis
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=38 end_frame_id=53 ignore_invalid_label=1 val_ratio=0 min_val=0 allow_missing_images=0 get_img_stats=0 description=ipsc-ext_reorg_roi_g2_38_53 save_masks=0 n_proc=24 incremental=1

<a id="ext_reorg_roi_g2_16_53___xml_to_ytvis_"></a>
# ext_reorg_roi_g2_16_53       @ xml_to_ytvis-->xml_to_ytvis
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=16 end_frame_id=53 ignore_invalid_label=1 val_ratio=0 allow_missing_images=0 get_img_stats=0 description=ipsc-ext_reorg_roi_g2_16_53 save_masks=0 subseq=1 subseq_split_ids=21 n_proc=12

<a id="ext_reorg_roi_g2_0_15___xml_to_ytvis_"></a>
# ext_reorg_roi_g2_0_15       @ xml_to_ytvis-->xml_to_ytvis
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=0 end_frame_id=15 ignore_invalid_label=1 val_ratio=0 allow_missing_images=0 get_img_stats=0 description=ipsc-ext_reorg_roi_g2_0_15 save_masks=0 n_proc=12
__-max_length-1__
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=0 end_frame_id=15 ignore_invalid_label=1 val_ratio=0 allow_missing_images=0 get_img_stats=0 description=ipsc-ext_reorg_roi_g2_0_15 save_masks=0 n_proc=12 max_length=1
__-max_length-2__
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=0 end_frame_id=15 ignore_invalid_label=1 val_ratio=0 allow_missing_images=0 get_img_stats=0 description=ipsc-ext_reorg_roi_g2_0_15 save_masks=0 n_proc=12 max_length=2
__-max_length-4__
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=0 end_frame_id=15 ignore_invalid_label=1 val_ratio=0 allow_missing_images=0 get_img_stats=0 description=ipsc-ext_reorg_roi_g2_0_15 save_masks=0 n_proc=12 max_length=4
__-max_length-8-__
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=0 end_frame_id=15 ignore_invalid_label=1 val_ratio=0 allow_missing_images=0 get_img_stats=0 description=ipsc-ext_reorg_roi_g2_0_15 save_masks=0 n_proc=12 max_length=8

<a id="incremental___ext_reorg_roi_g2_0_1_5_"></a>
## incremental       @ ext_reorg_roi_g2_0_15-->xml_to_ytvis
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=0 end_frame_id=15 ignore_invalid_label=1 val_ratio=0 allow_missing_images=0 get_img_stats=0 description=ipsc-ext_reorg_roi_g2_0_15 save_masks=0 n_proc=24 incremental=1
__-max_length-2__
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=0 end_frame_id=15 ignore_invalid_label=1 val_ratio=0 allow_missing_images=0 get_img_stats=0 description=ipsc-ext_reorg_roi_g2_0_15 save_masks=0 n_proc=2 incremental=1 max_length=2
__-max_length-10__
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=0 end_frame_id=15 ignore_invalid_label=1 val_ratio=0 allow_missing_images=0 get_img_stats=0 description=ipsc-ext_reorg_roi_g2_0_15 save_masks=0 n_proc=2 incremental=1 max_length=10

<a id="ext_reorg_roi_g2_54_126___xml_to_ytvis_"></a>
# ext_reorg_roi_g2_54_126       @ xml_to_ytvis-->xml_to_ytvis
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=54 end_frame_id=126 ignore_invalid_label=1 val_ratio=0 allow_missing_images=0 get_img_stats=0 description=ipsc-ext_reorg_roi_g2_54_126 save_masks=0 n_proc=12
```
pix_vals_mean: [118.06, 118.06, 118.06]
pix_vals_std: [23.99, 23.99, 23.99]
```
<a id="subseq___ext_reorg_roi_g2_54_12_6_"></a>
## subseq       @ ext_reorg_roi_g2_54_126-->xml_to_ytvis
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=54 end_frame_id=126 ignore_invalid_label=1 val_ratio=0 allow_missing_images=0 get_img_stats=0 description=ipsc-ext_reorg_roi_g2_54_126 save_masks=0 n_proc=12 subseq=1 subseq_split_ids=15 n_proc=12

<a id="ext_reorg_roi_g2_0_53___xml_to_ytvis_"></a>
# ext_reorg_roi_g2_0_53       @ xml_to_ytvis-->xml_to_ytvis
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=0 end_frame_id=53 ignore_invalid_label=1 val_ratio=0 allow_missing_images=0 get_img_stats=0 description=ipsc-ext_reorg_roi_g2_0_53 save_masks=0 n_proc=12 subseq=1 subseq_split_ids=37
__-max_length-1-__
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=0 end_frame_id=53 ignore_invalid_label=1 val_ratio=0 allow_missing_images=0 get_img_stats=0 description=ipsc-ext_reorg_roi_g2_0_53 save_masks=0 n_proc=12 max_length=1
__-max_length-2-__
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=0 end_frame_id=53 ignore_invalid_label=1 val_ratio=0 allow_missing_images=0 get_img_stats=0 description=ipsc-ext_reorg_roi_g2_0_53 save_masks=0 n_proc=12 max_length=2
__-max_length-4-__
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=0 end_frame_id=53 ignore_invalid_label=1 val_ratio=0 allow_missing_images=0 get_img_stats=0 description=ipsc-ext_reorg_roi_g2_0_53 save_masks=0 n_proc=12 max_length=4
__-max_length-8-__
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=0 end_frame_id=53 ignore_invalid_label=1 val_ratio=0 allow_missing_images=0 get_img_stats=0 description=ipsc-ext_reorg_roi_g2_0_53 save_masks=0 n_proc=12 max_length=8
__-max_length-19-__
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=0 end_frame_id=53 ignore_invalid_label=1 val_ratio=0 allow_missing_images=0 get_img_stats=0 description=ipsc-ext_reorg_roi_g2_0_53 save_masks=0 n_proc=12 max_length=19 subseq=1 subseq_split_ids=37

<a id="incremental___ext_reorg_roi_g2_0_5_3_"></a>
## incremental       @ ext_reorg_roi_g2_0_53-->xml_to_ytvis
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=0 end_frame_id=53 ignore_invalid_label=1 val_ratio=0 allow_missing_images=0 get_img_stats=0 description=ipsc-ext_reorg_roi_g2_0_53 save_masks=0 n_proc=12 subseq=1 subseq_split_ids=38 incremental=1
__-max_length-2__
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=0 end_frame_id=53 ignore_invalid_label=1 val_ratio=0 allow_missing_images=0 get_img_stats=0 description=ipsc-ext_reorg_roi_g2_0_53 save_masks=0 n_proc=12 incremental=1 max_length=2
__-max_length-10__
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=0 end_frame_id=53 ignore_invalid_label=1 val_ratio=0 allow_missing_images=0 get_img_stats=0 description=ipsc-ext_reorg_roi_g2_0_53 save_masks=0 n_proc=12 subseq=1 subseq_split_ids=38 incremental=1 max_length=10
__-max_length-20__
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=0 end_frame_id=53 ignore_invalid_label=1 val_ratio=0 allow_missing_images=0 get_img_stats=0 description=ipsc-ext_reorg_roi_g2_0_53 save_masks=0 n_proc=12 subseq=1 subseq_split_ids=38 incremental=1 max_length=20
