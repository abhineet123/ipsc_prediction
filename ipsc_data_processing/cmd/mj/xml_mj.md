<!-- MarkdownTOC -->

- [cvat_to_csv_with_masks](#cvat_to_csv_with_mask_s_)
- [epoch_to_time_and_id](#epoch_to_time_and_i_d_)
- [csv_to_yolov5](#csv_to_yolov5_)
    - [ipsc-5_class       @ csv_to_yolov5](#ipsc_5_class___csv_to_yolov_5_)
    - [ext_reorg_roi       @ csv_to_yolov5](#ext_reorg_roi___csv_to_yolov_5_)
- [xml_to_csv](#xml_to_cs_v_)
    - [db3_2_to_17_except_6       @ xml_to_csv](#db3_2_to_17_except_6___xml_to_csv_)
        - [large_huge       @ db3_2_to_17_except_6/xml_to_csv](#large_huge___db3_2_to_17_except_6_xml_to_cs_v_)
    - [sept5       @ xml_to_csv](#sept5___xml_to_csv_)
        - [large_huge       @ sept5/xml_to_csv](#large_huge___sept5_xml_to_csv_)
    - [sept5_2k_100       @ xml_to_csv](#sept5_2k_100___xml_to_csv_)
        - [large_huge       @ sept5_2k_100/xml_to_csv](#large_huge___sept5_2k_100_xml_to_cs_v_)
    - [db4       @ xml_to_csv](#db4___xml_to_csv_)
        - [rockmaps       @ db4/xml_to_csv](#rockmaps___db4_xml_to_csv_)
- [csv_to_xml](#csv_to_xm_l_)
    - [db3_2_to_17_except_6_with_syn       @ csv_to_xml](#db3_2_to_17_except_6_with_syn___csv_to_xml_)
        - [large,huge       @ db3_2_to_17_except_6_with_syn/csv_to_xml](#large_huge___db3_2_to_17_except_6_with_syn_csv_to_xml_)
    - [part5_on_september_5_2020       @ csv_to_xml](#part5_on_september_5_2020___csv_to_xml_)
        - [large,huge       @ part5_on_september_5_2020/csv_to_xml](#large_huge___part5_on_september_5_2020_csv_to_xml_)
    - [part4_on_part5_on_september_5_2020       @ csv_to_xml](#part4_on_part5_on_september_5_2020___csv_to_xml_)
        - [large,huge       @ part4_on_part5_on_september_5_2020/csv_to_xml](#large_huge___part4_on_part5_on_september_5_2020_csv_to_xm_l_)
    - [part14_on_part4_on_part5_on_september_5_2020       @ csv_to_xml](#part14_on_part4_on_part5_on_september_5_2020___csv_to_xml_)
        - [large,huge       @ part14_on_part4_on_part5_on_september_5_2020/csv_to_xml](#large_huge___part14_on_part4_on_part5_on_september_5_2020_csv_to_xm_l_)
    - [db4_rockmaps_syn       @ csv_to_xml](#db4_rockmaps_syn___csv_to_xml_)
- [cvat_to_xml](#cvat_to_xml_)
    - [db5       @ cvat_to_xml](#db5___cvat_to_xm_l_)
        - [part1       @ db5/cvat_to_xml](#part1___db5_cvat_to_xm_l_)
    - [db4       @ cvat_to_xml](#db4___cvat_to_xm_l_)
        - [rockmaps       @ db4/cvat_to_xml](#rockmaps___db4_cvat_to_xm_l_)
    - [db3       @ cvat_to_xml](#db3___cvat_to_xm_l_)
        - [large,huge       @ db3/cvat_to_xml](#large_huge___db3_cvat_to_xm_l_)
    - [part4       @ cvat_to_xml](#part4___cvat_to_xm_l_)
    - [db3_2_to_17_except_6       @ cvat_to_xml](#db3_2_to_17_except_6___cvat_to_xm_l_)
        - [large,huge       @ db3_2_to_17_except_6/cvat_to_xml](#large_huge___db3_2_to_17_except_6_cvat_to_xml_)
    - [db3_2_to_17_except_6_with_syn       @ cvat_to_xml](#db3_2_to_17_except_6_with_syn___cvat_to_xm_l_)
        - [large,huge       @ db3_2_to_17_except_6_with_syn/cvat_to_xml](#large_huge___db3_2_to_17_except_6_with_syn_cvat_to_xm_l_)
    - [db3_2_to_17_except_6_no_rocks       @ cvat_to_xml](#db3_2_to_17_except_6_no_rocks___cvat_to_xm_l_)
        - [large,huge       @ db3_2_to_17_except_6_no_rocks/cvat_to_xml](#large_huge___db3_2_to_17_except_6_no_rocks_cvat_to_xm_l_)
    - [september_5_2020       @ cvat_to_xml](#september_5_2020___cvat_to_xm_l_)
        - [large,huge       @ september_5_2020/cvat_to_xml](#large_huge___september_5_2020_cvat_to_xml_)
    - [db3_syn       @ cvat_to_xml](#db3_syn___cvat_to_xm_l_)
        - [large,huge       @ db3_syn/cvat_to_xml](#large_huge___db3_syn_cvat_to_xm_l_)
    - [all       @ cvat_to_xml](#all___cvat_to_xm_l_)
        - [large,huge       @ all/cvat_to_xml](#large_huge___all_cvat_to_xm_l_)
    - [september_5_2020_2K_100       @ cvat_to_xml](#september_5_2020_2k_100___cvat_to_xm_l_)
        - [large,huge       @ september_5_2020_2K_100/cvat_to_xml](#large_huge___september_5_2020_2k_100_cvat_to_xm_l_)
    - [part1       @ cvat_to_xml](#part1___cvat_to_xm_l_)
        - [large,huge       @ part1/cvat_to_xml](#large_huge___part1_cvat_to_xm_l_)

<!-- /MarkdownTOC -->

<a id="cvat_to_csv_with_mask_s_"></a>
# cvat_to_csv_with_masks
synthetic-data-generation/cmd/cvat_vis.md

<a id="epoch_to_time_and_i_d_"></a>
# epoch_to_time_and_id
mj_yolov5/cmd/yolov5_setup.md

<a id="csv_to_yolov5_"></a>
# csv_to_yolov5
<a id="ipsc_5_class___csv_to_yolov_5_"></a>
## ipsc-5_class       @ csv_to_yolov5-->xml_mj
python36 csv_to_yolov5.py root_dir=/data/ipsc_5_class_raw class_names_path=data/predefined_classes_ipsc_5_class.txt

<a id="ext_reorg_roi___csv_to_yolov_5_"></a>
## ext_reorg_roi       @ csv_to_yolov5-->xml_mj
python csv_to_yolov5.py root_dir=/data/ipsc/well3/images class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt enable_mask=1 ignore_invalid_class=1 consolidate_db=1

<a id="xml_to_cs_v_"></a>
# xml_to_csv
<a id="db3_2_to_17_except_6___xml_to_csv_"></a>
## db3_2_to_17_except_6       @ xml_to_csv-->xml_mj
<a id="large_huge___db3_2_to_17_except_6_xml_to_cs_v_"></a>
### large_huge       @ db3_2_to_17_except_6/xml_to_csv-->xml_mj
python3 xml_to_csv.py root_dir=/data/mojow_rock/rock_dataset3 seq_paths=lists/db3_2_to_17_except_6.txt enable_mask=1 csv_name=annotations.csv xml_suffix=large_huge recursive=0

<a id="sept5___xml_to_csv_"></a>
## sept5       @ xml_to_csv-->xml_mj
<a id="large_huge___sept5_xml_to_csv_"></a>
### large_huge       @ sept5/xml_to_csv-->xml_mj
python3 xml_to_csv.py seq_paths=/data/mojow_rock/rock_dataset3/september_5_2020 enable_mask=1 csv_name=annotations.csv xml_suffix=large_huge recursive=1

<a id="sept5_2k_100___xml_to_csv_"></a>
## sept5_2k_100       @ xml_to_csv-->xml_mj
<a id="large_huge___sept5_2k_100_xml_to_cs_v_"></a>
### large_huge       @ sept5_2k_100/xml_to_csv-->xml_mj
python3 xml_to_csv.py seq_paths=/data/mojow_rock/rock_dataset3/september_5_2020_2K_100 enable_mask=1 csv_name=annotations.csv xml_suffix=large_huge recursive=0

<a id="db4___xml_to_csv_"></a>
## db4       @ xml_to_csv-->xml_mj
python3 xml_to_csv.py root_dir=/data/mojow_rock/rock_dataset4 seq_paths=db4.txt enable_mask=1 csv_name=annotations.csv

<a id="rockmaps___db4_xml_to_csv_"></a>
### rockmaps       @ db4/xml_to_csv-->xml_mj
python3 xml_to_csv.py root_dir=/data/mojow_rock/rock_dataset4/rockmaps seq_paths=db4.txt enable_mask=1 csv_name=annotations.csv recursive=1

<a id="csv_to_xm_l_"></a>
# csv_to_xml
<a id="db3_2_to_17_except_6_with_syn___csv_to_xml_"></a>
## db3_2_to_17_except_6_with_syn       @ csv_to_xml-->xml_mj
python3 -m csv_to_xml csv_paths=db3_2_to_17_except_6_with_syn.txt root_dir=/data/mojow_rock/rock_dataset3 allow_missing_images=1

<a id="large_huge___db3_2_to_17_except_6_with_syn_csv_to_xml_"></a>
### large,huge       @ db3_2_to_17_except_6_with_syn/csv_to_xml-->xml_mj
python3 -m csv_to_xml csv_paths=db3_2_to_17_except_6_with_syn.txt root_dir=/data/mojow_rock/rock_dataset3  sizes=large,huge allow_missing_images=1

<a id="part5_on_september_5_2020___csv_to_xml_"></a>
## part5_on_september_5_2020       @ csv_to_xml-->xml_mj
python3 -m csv_to_xml csv_paths=syn/part5_on_september_5_2020 root_dir=/data/mojow_rock/rock_dataset3 allow_missing_images=0

<a id="large_huge___part5_on_september_5_2020_csv_to_xml_"></a>
### large,huge       @ part5_on_september_5_2020/csv_to_xml-->xml_mj
python3 -m csv_to_xml csv_paths=syn/part5_on_september_5_2020 root_dir=/data/mojow_rock/rock_dataset3 sizes=large,huge allow_missing_images=0

<a id="part4_on_part5_on_september_5_2020___csv_to_xml_"></a>
## part4_on_part5_on_september_5_2020       @ csv_to_xml-->xml_mj
python3 -m csv_to_xml csv_paths=syn/part4_on_part5_on_september_5_2020 root_dir=/data/mojow_rock/rock_dataset3 allow_missing_images=0

<a id="large_huge___part4_on_part5_on_september_5_2020_csv_to_xm_l_"></a>
### large,huge       @ part4_on_part5_on_september_5_2020/csv_to_xml-->xml_mj
python3 -m csv_to_xml csv_paths=syn/part4_on_part5_on_september_5_2020 root_dir=/data/mojow_rock/rock_dataset3 sizes=large,huge allow_missing_images=0

<a id="part14_on_part4_on_part5_on_september_5_2020___csv_to_xml_"></a>
## part14_on_part4_on_part5_on_september_5_2020       @ csv_to_xml-->xml_mj
python3 -m csv_to_xml csv_paths=syn/part14_on_part4_on_part5_on_september_5_2020 root_dir=/data/mojow_rock/rock_dataset3 allow_missing_images=0

<a id="large_huge___part14_on_part4_on_part5_on_september_5_2020_csv_to_xm_l_"></a>
### large,huge       @ part14_on_part4_on_part5_on_september_5_2020/csv_to_xml-->xml_mj
python3 -m csv_to_xml csv_paths=syn/part14_on_part4_on_part5_on_september_5_2020 root_dir=/data/mojow_rock/rock_dataset3 sizes=large,huge allow_missing_images=0

<a id="db4_rockmaps_syn___csv_to_xml_"></a>
## db4_rockmaps_syn       @ csv_to_xml-->xml_mj
python3 -m csv_to_xml csv_paths=rockmaps/syn/part2_on_camera_stick_cs210422100/annotations.csv root_dir=/data/mojow_rock/rock_dataset4/ allow_missing_images=0

python3 -m csv_to_xml csv_paths=rockmaps/syn/part2_on_camera_stick_cs210422102/annotations.csv root_dir=/data/mojow_rock/rock_dataset4/ allow_missing_images=0

python3 -m csv_to_xml csv_paths=rockmaps/syn/part2_on_hub_cs210422100/annotations.csv root_dir=/data/mojow_rock/rock_dataset4/ allow_missing_images=0

python3 -m csv_to_xml csv_paths=rockmaps/syn/part2_on_hub_cs210422102/annotations.csv root_dir=/data/mojow_rock/rock_dataset4/ allow_missing_images=0

<a id="cvat_to_xml_"></a>
# cvat_to_xml
<a id="db5___cvat_to_xm_l_"></a>
## db5       @ cvat_to_xml-->xml_mj
<a id="part1___db5_cvat_to_xm_l_"></a>
### part1       @ db5/cvat_to_xml-->xml_mj
python3 -m cvat_to_xml input=part1 root_dir=/data/mojow_rock/rock_dataset5 allow_missing_images=0 move_images=0 vert_flip=0 allow_empty=1 db5=1

python3 -m cvat_to_xml input=part2 root_dir=/data/mojow_rock/rock_dataset5 allow_missing_images=0 move_images=0 vert_flip=0 allow_empty=1 db5=1

<a id="db4___cvat_to_xm_l_"></a>
## db4       @ cvat_to_xml-->xml_mj
python3 -m cvat_to_xml input=db4.txt root_dir=/data/mojow_rock/rock_dataset4 allow_missing_images=0 move_images=0 vert_flip=1

<a id="rockmaps___db4_cvat_to_xm_l_"></a>
### rockmaps       @ db4/cvat_to_xml-->xml_mj
python3 -m cvat_to_xml input=db4.txt root_dir=/data/mojow_rock/rock_dataset4/rockmaps allow_missing_images=0 move_images=0 vert_flip=1 write_img=0 allow_missing_ann=1 write_empty=0

<a id="db3___cvat_to_xm_l_"></a>
## db3       @ cvat_to_xml-->xml_mj
python3 -m cvat_to_xml input=db3.txt root_dir=/data/mojow_rock/rock_dataset3 allow_missing_images=1

<a id="large_huge___db3_cvat_to_xm_l_"></a>
### large,huge       @ db3/cvat_to_xml-->xml_mj
python3 -m cvat_to_xml input=db3.txt root_dir=/data/mojow_rock/rock_dataset3 sizes=large,huge allow_missing_images=1

<a id="part4___cvat_to_xm_l_"></a>
## part4       @ cvat_to_xml-->xml_mj
python3 -m cvat_to_xml input=part4 root_dir=/data/mojow_rock/rock_dataset3 allow_missing_images=1

<a id="db3_2_to_17_except_6___cvat_to_xm_l_"></a>
## db3_2_to_17_except_6       @ cvat_to_xml-->xml_mj
python3 -m cvat_to_xml input=db3_2_to_17_except_6.txt root_dir=/data/mojow_rock/rock_dataset3  allow_missing_images=1

<a id="large_huge___db3_2_to_17_except_6_cvat_to_xml_"></a>
### large,huge       @ db3_2_to_17_except_6/cvat_to_xml-->xml_mj
python3 -m cvat_to_xml input=db3_2_to_17_except_6.txt root_dir=/data/mojow_rock/rock_dataset3  sizes=large,huge allow_missing_images=1

<a id="db3_2_to_17_except_6_with_syn___cvat_to_xm_l_"></a>
## db3_2_to_17_except_6_with_syn       @ cvat_to_xml-->xml_mj
python3 -m cvat_to_xml input=db3_2_to_17_except_6_with_syn.txt root_dir=/data/mojow_rock/rock_dataset3  allow_missing_images=1

<a id="large_huge___db3_2_to_17_except_6_with_syn_cvat_to_xm_l_"></a>
### large,huge       @ db3_2_to_17_except_6_with_syn/cvat_to_xml-->xml_mj
python3 -m cvat_to_xml input=db3_2_to_17_except_6_with_syn.txt root_dir=/data/mojow_rock/rock_dataset3  sizes=large,huge allow_missing_images=1

<a id="db3_2_to_17_except_6_no_rocks___cvat_to_xm_l_"></a>
## db3_2_to_17_except_6_no_rocks       @ cvat_to_xml-->xml_mj
python3 -m cvat_to_xml input=db3_2_to_17_except_6_no_rocks.txt root_dir=/data/mojow_rock/rock_dataset3  allow_missing_images=1 write_empty=1 allow_missing_ann=1

<a id="large_huge___db3_2_to_17_except_6_no_rocks_cvat_to_xm_l_"></a>
### large,huge       @ db3_2_to_17_except_6_no_rocks/cvat_to_xml-->xml_mj
python3 -m cvat_to_xml input=db3_2_to_17_except_6_no_rocks.txt root_dir=/data/mojow_rock/rock_dataset3  sizes=large,huge allow_missing_images=1 write_empty=1 allow_missing_ann=1

<a id="september_5_2020___cvat_to_xm_l_"></a>
## september_5_2020       @ cvat_to_xml-->xml_mj
python3 -m cvat_to_xml input=september_5_2020 root_dir=/data/mojow_rock/rock_dataset3  allow_missing_images=1 write_empty=1 allow_target_id=0

<a id="large_huge___september_5_2020_cvat_to_xml_"></a>
### large,huge       @ september_5_2020/cvat_to_xml-->xml_mj
python3 -m cvat_to_xml input=september_5_2020 root_dir=/data/mojow_rock/rock_dataset3  allow_missing_images=1 write_empty=1 sizes=large,huge

<a id="db3_syn___cvat_to_xm_l_"></a>
## db3_syn       @ cvat_to_xml-->xml_mj
<a id="large_huge___db3_syn_cvat_to_xm_l_"></a>
### large,huge       @ db3_syn/cvat_to_xml-->xml_mj
python3 -m cvat_to_xml input=db3_syn.txt root_dir=/data/mojow_rock/rock_dataset3  sizes=large,huge allow_missing_images=0

<a id="all___cvat_to_xm_l_"></a>
## all       @ cvat_to_xml-->xml_mj
<a id="large_huge___all_cvat_to_xm_l_"></a>
### large,huge       @ all/cvat_to_xml-->xml_mj
python3 -m cvat_to_xml input=/data/mojow_rock/rock_dataset3 sizes=large,huge

<a id="september_5_2020_2k_100___cvat_to_xm_l_"></a>
## september_5_2020_2K_100       @ cvat_to_xml-->xml_mj
python3 -m cvat_to_xml input=/data/mojow_rock/rock_dataset3/september_5_2020_2K_100/annotations.xml output=/data/mojow_rock/rock_dataset3 allow_missing_images=1

<a id="large_huge___september_5_2020_2k_100_cvat_to_xm_l_"></a>
### large,huge       @ september_5_2020_2K_100/cvat_to_xml-->xml_mj
python3 -m cvat_to_xml input=/data/mojow_rock/rock_dataset3/september_5_2020_2K_100/annotations.xml output=/data/mojow_rock/rock_dataset3 sizes=large,huge allow_missing_images=1

<a id="part1___cvat_to_xm_l_"></a>
## part1       @ cvat_to_xml-->xml_mj
<a id="large_huge___part1_cvat_to_xm_l_"></a>
### large,huge       @ part1/cvat_to_xml-->xml_mj
python3 -m cvat_to_xml input=/data/mojow_rock/rock_dataset3/part1/annotations.xml output=/data/mojow_rock/rock_dataset3 sizes=large,huge



