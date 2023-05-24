<!-- MarkdownTOC -->

- [db4](#db4_)
    - [all       @ db4](#all___db_4_)
- [db3-part12](#db3_part1_2_)
    - [large_huge       @ db3-part12](#large_huge___db3_part12_)
- [db3-part4](#db3_part4_)
    - [all       @ db3-part4](#all___db3_part_4_)
    - [large_huge       @ db3-part4](#large_huge___db3_part_4_)
    - [large_huge_target_ids       @ db3-part4](#large_huge_target_ids___db3_part_4_)
- [db3_2_to_17_except_6](#db3_2_to_17_except_6_)
    - [all       @ db3_2_to_17_except_6](#all___db3_2_to_17_except_6_)
    - [large_huge       @ db3_2_to_17_except_6](#large_huge___db3_2_to_17_except_6_)
- [db3_2_to_17_except_6_with_syn](#db3_2_to_17_except_6_with_syn_)
    - [large_huge       @ db3_2_to_17_except_6_with_syn](#large_huge___db3_2_to_17_except_6_with_sy_n_)
- [september_5_2020](#september_5_202_0_)
    - [all       @ september_5_2020](#all___september_5_2020_)
    - [large_huge       @ september_5_2020](#large_huge___september_5_2020_)
        - [max_length-200       @ large_huge/september_5_2020](#max_length_200___large_huge_september_5_202_0_)
        - [max_length-50       @ large_huge/september_5_2020](#max_length_50___large_huge_september_5_202_0_)
        - [max_length-20       @ large_huge/september_5_2020](#max_length_20___large_huge_september_5_202_0_)

<!-- /MarkdownTOC -->


<a id="db4_"></a>
# db4
<a id="all___db_4_"></a>
## all       @ db4-->xml_to_ytvis_mj
python3 xml_to_ytvis.py root_dir=/data/mojow_rock/rock_dataset4 seq_paths=db4.txt class_names_path=lists/classes/predefined_classes_rock.txt val_ratio=0 allow_missing_images=0 remove_mj_dir_suffix=0 get_img_stats=1 description=mj_rock-db4 save_masks=0 subseq=1 infer_target_id=1 n_proc=1

pix_vals_mean: [128.24416458682276, 128.80643404991795, 129.93059506678998]
pix_vals_std: [24.073809895323944, 25.343111484178248, 25.39594219533212]


<a id="db3_part1_2_"></a>
# db3-part12
<a id="large_huge___db3_part12_"></a>
## large_huge       @ db3-part12-->xml_to_ytvis_mj
python3 xml_to_ytvis.py root_dir=/data/mojow_rock/rock_dataset3 seq_paths=part12 class_names_path=lists/classes/predefined_classes_rock.txt val_ratio=0.3 dir_suffix=large_huge allow_missing_images=0 remove_mj_dir_suffix=1 get_img_stats=0 description=mj_rock-db3-part12-large_huge save_masks=1

<a id="db3_part4_"></a>
# db3-part4
<a id="all___db3_part_4_"></a>
## all       @ db3-part4-->xml_to_ytvis_mj
python3 xml_to_ytvis.py root_dir=/data/mojow_rock/rock_dataset3 seq_paths=part4 class_names_path=lists/classes/predefined_classes_rock.txt val_ratio=0 allow_missing_images=0 remove_mj_dir_suffix=1 get_img_stats=0 description=mj_rock-db3_2_to_17_except_6 save_masks=0 subseq=1 infer_target_id=1 n_proc=1

<a id="large_huge___db3_part_4_"></a>
## large_huge       @ db3-part4-->xml_to_ytvis_mj
python3 xml_to_ytvis.py root_dir=/data/mojow_rock/rock_dataset3 seq_paths=part4 class_names_path=lists/classes/predefined_classes_rock.txt val_ratio=0 allow_missing_images=0 remove_mj_dir_suffix=1 get_img_stats=0 description=mj_rock-db3_2_to_17_except_6-large_huge save_masks=0 subseq=1 infer_target_id=1 n_proc=6 dir_suffix=large_huge

<a id="large_huge_target_ids___db3_part_4_"></a>
## large_huge_target_ids       @ db3-part4-->xml_to_ytvis_mj
python3 xml_to_ytvis.py root_dir=/data/mojow_rock/rock_dataset3 seq_paths=part4 class_names_path=lists/classes/predefined_classes_rock.txt val_ratio=0 allow_missing_images=0 remove_mj_dir_suffix=1 get_img_stats=0 description=mj_rock-db3_2_to_17_except_6-large_huge save_masks=0 subseq=1 infer_target_id=0 n_proc=6 dir_suffix=large_huge_target_ids

<a id="db3_2_to_17_except_6_"></a>
# db3_2_to_17_except_6
<a id="all___db3_2_to_17_except_6_"></a>
## all       @ db3_2_to_17_except_6-->xml_to_ytvis_mj
python3 xml_to_ytvis.py root_dir=/data/mojow_rock/rock_dataset3 seq_paths=db3_2_to_17_except_6.txt class_names_path=lists/classes/predefined_classes_rock.txt val_ratio=0 allow_missing_images=0 remove_mj_dir_suffix=1 get_img_stats=0 description=mj_rock-db3_2_to_17_except_6 save_masks=0 subseq=1 infer_target_id=1 n_proc=6

<a id="large_huge___db3_2_to_17_except_6_"></a>
## large_huge       @ db3_2_to_17_except_6-->xml_to_ytvis_mj
python3 xml_to_ytvis.py root_dir=/data/mojow_rock/rock_dataset3 seq_paths=db3_2_to_17_except_6.txt class_names_path=lists/classes/predefined_classes_rock.txt val_ratio=0 allow_missing_images=0 remove_mj_dir_suffix=1 get_img_stats=0 description=mj_rock-db3_2_to_17_except_6-large_huge save_masks=0 subseq=1 infer_target_id=1 dir_suffix=large_huge n_proc=24

<a id="db3_2_to_17_except_6_with_syn_"></a>
# db3_2_to_17_except_6_with_syn
<a id="large_huge___db3_2_to_17_except_6_with_sy_n_"></a>
## large_huge       @ db3_2_to_17_except_6_with_syn-->xml_to_ytvis_mj
python3 xml_to_ytvis.py root_dir=/data/mojow_rock/rock_dataset3 seq_paths=db3_2_to_17_except_6_with_syn.txt class_names_path=lists/classes/predefined_classes_rock.txt val_ratio=0.3 dir_suffix=large_huge allow_missing_images=0 remove_mj_dir_suffix=1 get_img_stats=0 description=mj_rock-db3_2_to_17_except_6_with_syn-large_huge save_masks=1

<a id="september_5_202_0_"></a>
# september_5_2020

<a id="all___september_5_2020_"></a>
## all       @ september_5_2020-->xml_to_ytvis_mj
python3 xml_to_ytvis.py root_dir=/data/mojow_rock/rock_dataset3 seq_paths=september_5_2020 class_names_path=lists/classes/predefined_classes_rock.txt val_ratio=0 allow_missing_images=0 get_img_stats=0 description=mj_rock-september_5_2020 save_masks=1 subseq=1 infer_target_id=1 min_length=1 n_proc=12 max_length=200

<a id="large_huge___september_5_2020_"></a>
## large_huge       @ september_5_2020-->xml_to_ytvis_mj
python3 xml_to_ytvis.py root_dir=/data/mojow_rock/rock_dataset3 seq_paths=september_5_2020 class_names_path=lists/classes/predefined_classes_rock.txt val_ratio=0 allow_missing_images=0 get_img_stats=0 description=mj_rock-september_5_2020-large_huge save_masks=0 subseq=1 infer_target_id=1 dir_suffix=large_huge min_length=1 n_proc=12

<a id="max_length_200___large_huge_september_5_202_0_"></a>
### max_length-200       @ large_huge/september_5_2020-->xml_to_ytvis_mj
python3 xml_to_ytvis.py root_dir=/data/mojow_rock/rock_dataset3 seq_paths=september_5_2020 class_names_path=lists/classes/predefined_classes_rock.txt val_ratio=0 allow_missing_images=0 get_img_stats=0 description=mj_rock-september_5_2020 save_masks=0 subseq=1 infer_target_id=1 dir_suffix=large_huge min_length=1 n_proc=12 max_length=200

<a id="max_length_50___large_huge_september_5_202_0_"></a>
### max_length-50       @ large_huge/september_5_2020-->xml_to_ytvis_mj
python3 xml_to_ytvis.py root_dir=/data/mojow_rock/rock_dataset3 seq_paths=september_5_2020 class_names_path=lists/classes/predefined_classes_rock.txt val_ratio=0 allow_missing_images=0 get_img_stats=0 description=mj_rock-september_5_2020 save_masks=0 subseq=1 infer_target_id=1 dir_suffix=large_huge min_length=1 n_proc=12 max_length=50

<a id="max_length_20___large_huge_september_5_202_0_"></a>
### max_length-20       @ large_huge/september_5_2020-->xml_to_ytvis_mj
python3 xml_to_ytvis.py root_dir=/data/mojow_rock/rock_dataset3 seq_paths=september_5_2020 class_names_path=lists/classes/predefined_classes_rock.txt val_ratio=0 allow_missing_images=0 get_img_stats=0 description=mj_rock-september_5_2020 save_masks=0 subseq=1 infer_target_id=1 dir_suffix=large_huge min_length=1 n_proc=12 max_length=20
