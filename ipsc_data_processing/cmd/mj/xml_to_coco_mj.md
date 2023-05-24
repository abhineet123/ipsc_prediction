
<!-- MarkdownTOC -->

- [db4](#db4_)
    - [rockmaps       @ db4](#rockmaps___db_4_)
        - [hub1       @ rockmaps/db4](#hub1___rockmaps_db4_)
        - [hub2       @ rockmaps/db4](#hub2___rockmaps_db4_)
        - [cs1       @ rockmaps/db4](#cs1___rockmaps_db4_)
        - [cs2       @ rockmaps/db4](#cs2___rockmaps_db4_)
    - [rockmaps_syn       @ db4](#rockmaps_syn___db_4_)
        - [hub1       @ rockmaps_syn/db4](#hub1___rockmaps_syn_db4_)
        - [hub2       @ rockmaps_syn/db4](#hub2___rockmaps_syn_db4_)
        - [cs1       @ rockmaps_syn/db4](#cs1___rockmaps_syn_db4_)
        - [cs2       @ rockmaps_syn/db4](#cs2___rockmaps_syn_db4_)
- [db3](#db3_)
    - [part12       @ db3](#part12___db_3_)
    - [part1       @ db3](#part1___db_3_)
        - [large_huge       @ part1/db3](#large_huge___part1_db_3_)
    - [part6       @ db3](#part6___db_3_)
        - [large_huge       @ part6/db3](#large_huge___part6_db_3_)
    - [september_5_2020       @ db3](#september_5_2020___db_3_)
        - [large_huge       @ september_5_2020/db3](#large_huge___september_5_2020_db3_)
    - [db3_2_to_17_except_6       @ db3](#db3_2_to_17_except_6___db_3_)
        - [large_huge       @ db3_2_to_17_except_6/db3](#large_huge___db3_2_to_17_except_6_db3_)
    - [db3_2_to_17_except_6_no_rocks       @ db3](#db3_2_to_17_except_6_no_rocks___db_3_)
        - [large_huge       @ db3_2_to_17_except_6_no_rocks/db3](#large_huge___db3_2_to_17_except_6_no_rocks_db_3_)
    - [db3_2_to_17_except_6_with_syn       @ db3](#db3_2_to_17_except_6_with_syn___db_3_)
        - [large_huge       @ db3_2_to_17_except_6_with_syn/db3](#large_huge___db3_2_to_17_except_6_with_syn_db_3_)
    - [part5_on_september_5_2020       @ db3](#part5_on_september_5_2020___db_3_)
        - [large_huge       @ part5_on_september_5_2020/db3](#large_huge___part5_on_september_5_2020_db_3_)
    - [part4_on_part5_on_september_5_2020       @ db3](#part4_on_part5_on_september_5_2020___db_3_)
        - [large_huge       @ part4_on_part5_on_september_5_2020/db3](#large_huge___part4_on_part5_on_september_5_2020_db3_)
    - [part14_on_part4_on_part5_on_september_5_2020       @ db3](#part14_on_part4_on_part5_on_september_5_2020___db_3_)
        - [large_huge       @ part14_on_part4_on_part5_on_september_5_2020/db3](#large_huge___part14_on_part4_on_part5_on_september_5_2020_db3_)

<!-- /MarkdownTOC -->

<a id="db4_"></a>
# db4
python3 xml_to_coco.py root_dir=/data/mojow_rock/rock_dataset4 seq_paths=db4.txt class_names_path=lists/classes/predefined_classes_rock.txt output_json=db4.json val_ratio=0 allow_missing_images=0 remove_mj_dir_suffix=0 get_img_stats=0

<a id="rockmaps___db_4_"></a>
## rockmaps       @ db4-->xml_to_coco_mj
python3 xml_to_coco.py root_dir=/data/mojow_rock/rock_dataset4/rockmaps seq_paths=db4.txt class_names_path=lists/classes/predefined_classes_rock.txt output_json=db4-rockmaps.json val_ratio=0 allow_missing_images=0 remove_mj_dir_suffix=0 get_img_stats=0
<a id="hub1___rockmaps_db4_"></a>
### hub1       @ rockmaps/db4-->xml_to_coco_mj
python3 xml_to_coco.py root_dir=/data/mojow_rock/rock_dataset4/rockmaps seq_paths=hub_cs210422100 class_names_path=lists/classes/predefined_classes_rock.txt output_json=db4-rockmaps-hub1.json val_ratio=0 allow_missing_images=0 remove_mj_dir_suffix=0 get_img_stats=0
<a id="hub2___rockmaps_db4_"></a>
### hub2       @ rockmaps/db4-->xml_to_coco_mj
python3 xml_to_coco.py root_dir=/data/mojow_rock/rock_dataset4/rockmaps seq_paths=hub_cs210422102 class_names_path=lists/classes/predefined_classes_rock.txt output_json=db4-rockmaps-hub2.json val_ratio=0 allow_missing_images=0 remove_mj_dir_suffix=0 get_img_stats=0
<a id="cs1___rockmaps_db4_"></a>
### cs1       @ rockmaps/db4-->xml_to_coco_mj
python3 xml_to_coco.py root_dir=/data/mojow_rock/rock_dataset4/rockmaps seq_paths=camera_stick_cs210422100 class_names_path=lists/classes/predefined_classes_rock.txt output_json=db4-rockmaps-cs1.json val_ratio=0 allow_missing_images=0 remove_mj_dir_suffix=0 get_img_stats=0
<a id="cs2___rockmaps_db4_"></a>
### cs2       @ rockmaps/db4-->xml_to_coco_mj
python3 xml_to_coco.py root_dir=/data/mojow_rock/rock_dataset4/rockmaps seq_paths=camera_stick_cs210422102 class_names_path=lists/classes/predefined_classes_rock.txt output_json=db4-rockmaps-cs2.json val_ratio=0 allow_missing_images=0 remove_mj_dir_suffix=0 get_img_stats=0

<a id="rockmaps_syn___db_4_"></a>
## rockmaps_syn       @ db4-->xml_to_coco_mj
python3 xml_to_coco.py root_dir=/data/mojow_rock/rock_dataset4/rockmaps/syn seq_paths=db4_syn.txt class_names_path=lists/classes/predefined_classes_rock_syn.txt output_json=/data/mojow_rock/rock_dataset4/db4-rockmaps_syn.json val_ratio=0 allow_missing_images=0 remove_mj_dir_suffix=0 get_img_stats=0
<a id="hub1___rockmaps_syn_db4_"></a>
### hub1       @ rockmaps_syn/db4-->xml_to_coco_mj
python3 xml_to_coco.py root_dir=/data/mojow_rock/rock_dataset4/rockmaps/syn seq_paths=part2_on_hub_cs210422100 class_names_path=lists/classes/predefined_classes_rock_syn.txt output_json=/data/mojow_rock/rock_dataset4/db4-rockmaps_syn-hub1.json val_ratio=0 allow_missing_images=0 remove_mj_dir_suffix=0 get_img_stats=0
<a id="hub2___rockmaps_syn_db4_"></a>
### hub2       @ rockmaps_syn/db4-->xml_to_coco_mj
python3 xml_to_coco.py root_dir=/data/mojow_rock/rock_dataset4/rockmaps/syn seq_paths=part2_on_hub_cs210422102 class_names_path=lists/classes/predefined_classes_rock_syn.txt output_json=/data/mojow_rock/rock_dataset4/db4-rockmaps_syn-hub2.json val_ratio=0 allow_missing_images=0 remove_mj_dir_suffix=0 get_img_stats=0
<a id="cs1___rockmaps_syn_db4_"></a>
### cs1       @ rockmaps_syn/db4-->xml_to_coco_mj
python3 xml_to_coco.py root_dir=/data/mojow_rock/rock_dataset4/rockmaps/syn seq_paths=part2_on_camera_stick_cs210422100 class_names_path=lists/classes/predefined_classes_rock_syn.txt output_json=/data/mojow_rock/rock_dataset4/db4-rockmaps_syn-cs1.json val_ratio=0 allow_missing_images=0 remove_mj_dir_suffix=0 get_img_stats=0
<a id="cs2___rockmaps_syn_db4_"></a>
### cs2       @ rockmaps_syn/db4-->xml_to_coco_mj
python3 xml_to_coco.py root_dir=/data/mojow_rock/rock_dataset4/rockmaps/syn seq_paths=part2_on_camera_stick_cs210422102 class_names_path=lists/classes/predefined_classes_rock_syn.txt output_json=/data/mojow_rock/rock_dataset4/db4-rockmaps_syn-cs2.json val_ratio=0 allow_missing_images=0 remove_mj_dir_suffix=0 get_img_stats=0

<a id="db3_"></a>
# db3 
<a id="part12___db_3_"></a>
## part12       @ db3-->xml_to_coco_mj
python3 xml_to_coco.py root_dir=/data/mojow_rock/rock_dataset3 seq_paths=part12 class_names_path=lists/classes/predefined_classes_rock.txt output_json=db3_part12-large_huge.json val_ratio=0.3 dir_suffix=large_huge allow_missing_images=0 remove_mj_dir_suffix=1 get_img_stats=0

<a id="part1___db_3_"></a>
## part1       @ db3-->xml_to_coco_mj
python3 xml_to_coco.py root_dir=/data/mojow_rock/rock_dataset3 seq_paths=part1 class_names_path=lists/classes/predefined_classes_rock.txt output_json=part1.json val_ratio=0 allow_missing_images=0 remove_mj_dir_suffix=0 get_img_stats=0

<a id="large_huge___part1_db_3_"></a>
### large_huge       @ part1/db3-->xml_to_coco_mj
python3 xml_to_coco.py root_dir=/data/mojow_rock/rock_dataset3 seq_paths=part1 class_names_path=lists/classes/predefined_classes_rock.txt output_json=part1-large_huge.json val_ratio=0 dir_suffix=large_huge allow_missing_images=0 remove_mj_dir_suffix=0 get_img_stats=0

<a id="part6___db_3_"></a>
## part6       @ db3-->xml_to_coco_mj
python3 xml_to_coco.py root_dir=/data/mojow_rock/rock_dataset3 seq_paths=part1 class_names_path=lists/classes/predefined_classes_rock.txt output_json=part6.json val_ratio=0 allow_missing_images=0 remove_mj_dir_suffix=0 get_img_stats=0

<a id="large_huge___part6_db_3_"></a>
### large_huge       @ part6/db3-->xml_to_coco_mj
python3 xml_to_coco.py root_dir=/data/mojow_rock/rock_dataset3 seq_paths=part6 class_names_path=lists/classes/predefined_classes_rock.txt output_json=part6-large_huge.json val_ratio=0 dir_suffix=large_huge allow_missing_images=0 remove_mj_dir_suffix=0 get_img_stats=0

<a id="september_5_2020___db_3_"></a>
## september_5_2020       @ db3-->xml_to_coco_mj
python3 xml_to_coco.py root_dir=/data/mojow_rock/rock_dataset3 seq_paths=september_5_2020 class_names_path=lists/classes/predefined_classes_rock.txt output_json=september_5_2020.json val_ratio=0 min_val=0 allow_missing_images=0 remove_mj_dir_suffix=0 get_img_stats=0

<a id="large_huge___september_5_2020_db3_"></a>
### large_huge       @ september_5_2020/db3-->xml_to_coco_mj
python3 xml_to_coco.py root_dir=/data/mojow_rock/rock_dataset3 seq_paths=september_5_2020 class_names_path=lists/classes/predefined_classes_rock.txt output_json=september_5_2020-large_huge.json val_ratio=0 min_val=0 dir_suffix=large_huge allow_missing_images=0 remove_mj_dir_suffix=0 get_img_stats=0

__no_annotations__
python3 xml_to_coco.py root_dir=/data/mojow_rock/rock_dataset3 seq_paths=september_5_2020 class_names_path=lists/classes/predefined_classes_rock.txt output_json=september_5_2020-large_huge.json val_ratio=0 min_val=0 dir_suffix=large_huge allow_missing_images=0 remove_mj_dir_suffix=0 get_img_stats=0 no_annotations=1

<a id="db3_2_to_17_except_6___db_3_"></a>
## db3_2_to_17_except_6       @ db3-->xml_to_coco_mj
python3 xml_to_coco.py root_dir=/data/mojow_rock/rock_dataset3 seq_paths=db3_2_to_17_except_6.txt class_names_path=lists/classes/predefined_classes_rock.txt output_json=db3_2_to_17_except_6.json val_ratio=0.3 allow_missing_images=1 remove_mj_dir_suffix=0 get_img_stats=0

python3 xml_to_coco.py root_dir=/data/mojow_rock/rock_dataset3 seq_paths=db3_2_to_17_except_6.txt class_names_path=lists/classes/predefined_classes_rock.txt output_json=db3_2_to_17_except_6.json val_ratio=0 allow_missing_images=1 remove_mj_dir_suffix=0 get_img_stats=0

<a id="large_huge___db3_2_to_17_except_6_db3_"></a>
### large_huge       @ db3_2_to_17_except_6/db3-->xml_to_coco_mj
python3 xml_to_coco.py root_dir=/data/mojow_rock/rock_dataset3 seq_paths=db3_2_to_17_except_6.txt class_names_path=lists/classes/predefined_classes_rock.txt output_json=db3_2_to_17_except_6-large_huge.json val_ratio=0.3 dir_suffix=large_huge allow_missing_images=1 remove_mj_dir_suffix=0 get_img_stats=0

<a id="db3_2_to_17_except_6_no_rocks___db_3_"></a>
## db3_2_to_17_except_6_no_rocks       @ db3-->xml_to_coco_mj
python3 xml_to_coco.py root_dir=/data/mojow_rock/rock_dataset3 seq_paths=db3_2_to_17_except_6_no_rocks.txt class_names_path=lists/classes/predefined_classes_rock.txt output_json=db3_2_to_17_except_6_no_rocks.json val_ratio=0 allow_missing_images=1 remove_mj_dir_suffix=0 get_img_stats=1
pix_vals_mean: [143.90739512604202, 137.85737889429524, 138.77791972367567]
pix_vals_std: [39.216714984858854, 43.18105474265608, 43.806309282665055]

<a id="large_huge___db3_2_to_17_except_6_no_rocks_db_3_"></a>
### large_huge       @ db3_2_to_17_except_6_no_rocks/db3-->xml_to_coco_mj
python3 xml_to_coco.py root_dir=/data/mojow_rock/rock_dataset3 seq_paths=db3_2_to_17_except_6_no_rocks.txt class_names_path=lists/classes/predefined_classes_rock.txt output_json=db3_2_to_17_except_6_no_rocks-large_huge.json val_ratio=0 dir_suffix=large_huge allow_missing_images=1 remove_mj_dir_suffix=0 get_img_stats=1
pix_vals_mean: [143.90739512604202, 137.85737889429524, 138.77791972367567]
pix_vals_std: [39.216714984858854, 43.18105474265608, 43.806309282665055]

<a id="db3_2_to_17_except_6_with_syn___db_3_"></a>
## db3_2_to_17_except_6_with_syn       @ db3-->xml_to_coco_mj
python xml_to_coco.py root_dir=/data/mojow_rock/rock_dataset3 seq_paths=db3_2_to_17_except_6_with_syn.txt class_names_path=lists/classes/predefined_classes_rock_syn.txt output_json=db3_2_to_17_except_6_with_syn.json val_ratio=0 allow_missing_images=0 remove_mj_dir_suffix=0 get_img_stats=0

<a id="large_huge___db3_2_to_17_except_6_with_syn_db_3_"></a>
### large_huge       @ db3_2_to_17_except_6_with_syn/db3-->xml_to_coco_mj
python xml_to_coco.py root_dir=/data/mojow_rock/rock_dataset3 seq_paths=db3_2_to_17_except_6_with_syn.txt class_names_path=lists/classes/predefined_classes_rock_syn.txt output_json=db3_2_to_17_except_6_with_syn-large_huge.json val_ratio=0 dir_suffix=large_huge allow_missing_images=0 remove_mj_dir_suffix=0 get_img_stats=0

<a id="part5_on_september_5_2020___db_3_"></a>
## part5_on_september_5_2020       @ db3-->xml_to_coco_mj
python xml_to_coco.py root_dir=/data/mojow_rock/rock_dataset3 seq_paths=syn/part5_on_september_5_2020 class_names_path=lists/classes/predefined_classes_rock_syn.txt output_json=part5_on_september_5_2020.json val_ratio=0 allow_missing_images=0 remove_mj_dir_suffix=0 get_img_stats=0

<a id="large_huge___part5_on_september_5_2020_db_3_"></a>
### large_huge       @ part5_on_september_5_2020/db3-->xml_to_coco_mj
python xml_to_coco.py root_dir=/data/mojow_rock/rock_dataset3 seq_paths=syn/part5_on_september_5_2020 class_names_path=lists/classes/predefined_classes_rock_syn.txt output_json=part5_on_september_5_2020-large_huge.json val_ratio=0 dir_suffix=large_huge allow_missing_images=0 remove_mj_dir_suffix=0 get_img_stats=0

<a id="part4_on_part5_on_september_5_2020___db_3_"></a>
## part4_on_part5_on_september_5_2020       @ db3-->xml_to_coco_mj
python xml_to_coco.py root_dir=/data/mojow_rock/rock_dataset3 seq_paths=syn/part4_on_part5_on_september_5_2020 class_names_path=lists/classes/predefined_classes_rock_syn.txt output_json=part4_on_part5_on_september_5_2020.json val_ratio=0 allow_missing_images=0 remove_mj_dir_suffix=0 get_img_stats=0

<a id="large_huge___part4_on_part5_on_september_5_2020_db3_"></a>
### large_huge       @ part4_on_part5_on_september_5_2020/db3-->xml_to_coco_mj
python xml_to_coco.py root_dir=/data/mojow_rock/rock_dataset3 seq_paths=syn/part4_on_part5_on_september_5_2020 class_names_path=lists/classes/predefined_classes_rock_syn.txt output_json=part4_on_part5_on_september_5_2020-large_huge.json val_ratio=0 dir_suffix=large_huge allow_missing_images=0 remove_mj_dir_suffix=0 get_img_stats=0

<a id="part14_on_part4_on_part5_on_september_5_2020___db_3_"></a>
## part14_on_part4_on_part5_on_september_5_2020       @ db3-->xml_to_coco_mj
python xml_to_coco.py root_dir=/data/mojow_rock/rock_dataset3 seq_paths=syn/part14_on_part4_on_part5_on_september_5_2020 class_names_path=lists/classes/predefined_classes_rock_syn.txt output_json=part14_on_part4_on_part5_on_september_5_2020.json val_ratio=0 allow_missing_images=0 remove_mj_dir_suffix=0 get_img_stats=0

<a id="large_huge___part14_on_part4_on_part5_on_september_5_2020_db3_"></a>
### large_huge       @ part14_on_part4_on_part5_on_september_5_2020/db3-->xml_to_coco_mj
python xml_to_coco.py root_dir=/data/mojow_rock/rock_dataset3 seq_paths=syn/part14_on_part4_on_part5_on_september_5_2020 class_names_path=lists/classes/predefined_classes_rock_syn.txt output_json=part14_on_part4_on_part5_on_september_5_2020-large_huge.json val_ratio=0 dir_suffix=large_huge allow_missing_images=0 remove_mj_dir_suffix=0 get_img_stats=0
