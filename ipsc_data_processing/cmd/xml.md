<!-- MarkdownTOC -->

- [csv_to_yolov5](#csv_to_yolov5_)
    - [ipsc-5_class       @ csv_to_yolov5](#ipsc_5_class___csv_to_yolov_5_)
    - [ext_reorg_roi       @ csv_to_yolov5](#ext_reorg_roi___csv_to_yolov_5_)
- [xml_to_csv](#xml_to_cs_v_)
    - [ipsc-5_class       @ xml_to_csv](#ipsc_5_class___xml_to_csv_)
    - [ext_reorg_roi       @ xml_to_csv](#ext_reorg_roi___xml_to_csv_)
        - [g2_38_53       @ ext_reorg_roi/xml_to_csv](#g2_38_53___ext_reorg_roi_xml_to_csv_)
        - [g2_0_15       @ ext_reorg_roi/xml_to_csv](#g2_0_15___ext_reorg_roi_xml_to_csv_)
        - [g2_0_53       @ ext_reorg_roi/xml_to_csv](#g2_0_53___ext_reorg_roi_xml_to_csv_)

<!-- /MarkdownTOC -->


<a id="csv_to_yolov5_"></a>
# csv_to_yolov5
<a id="ipsc_5_class___csv_to_yolov_5_"></a>
## ipsc-5_class       @ csv_to_yolov5-->xml
python36 csv_to_yolov5.py root_dir=/data/ipsc_5_class_raw class_names_path=data/predefined_classes_ipsc_5_class.txt

<a id="ext_reorg_roi___csv_to_yolov_5_"></a>
## ext_reorg_roi       @ csv_to_yolov5-->xml
python csv_to_yolov5.py root_dir=/data/ipsc/well3/images class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt enable_mask=1 ignore_invalid_class=1 consolidate_db=1

<a id="xml_to_cs_v_"></a>
# xml_to_csv
<a id="ipsc_5_class___xml_to_csv_"></a>
## ipsc-5_class       @ xml_to_csv-->xml
python36 xml_to_csv.py root_dir=/data/ipsc_5_class_raw class_names_path=data/predefined_classes_ipsc_5_class.txt enable_mask=1

<a id="ext_reorg_roi___xml_to_csv_"></a>
## ext_reorg_roi       @ xml_to_csv-->xml
python3 xml_to_csv.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=ext_reorg_roi.txt enable_mask=1 csv_name=annotations.csv

<a id="g2_38_53___ext_reorg_roi_xml_to_csv_"></a>
### g2_38_53       @ ext_reorg_roi/xml_to_csv-->xml
python3 xml_to_csv.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=ext_reorg_roi.txt enable_mask=1 start_id=38 end_id=53 csv_name=annotations_38_53.csv

<a id="g2_0_15___ext_reorg_roi_xml_to_csv_"></a>
### g2_0_15       @ ext_reorg_roi/xml_to_csv-->xml
python3 xml_to_csv.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=ext_reorg_roi.txt enable_mask=1 start_id=0 end_id=15 csv_name=annotations_0_15.csv

<a id="g2_0_53___ext_reorg_roi_xml_to_csv_"></a>
### g2_0_53       @ ext_reorg_roi/xml_to_csv-->xml
python3 xml_to_csv.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=ext_reorg_roi.txt enable_mask=1 start_id=0 end_id=53 csv_name=annotations_0_53.csv
