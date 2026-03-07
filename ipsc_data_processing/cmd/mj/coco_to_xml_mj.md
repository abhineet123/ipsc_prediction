<!-- MarkdownTOC -->

- [db3_2_to_17_except_6       @ coco_to_xml](#db3_2_to_17_except_6___coco_to_xm_l_)
    - [idol       @ db3_2_to_17_except_6](#idol___db3_2_to_17_except_6_)
    - [vita       @ db3_2_to_17_except_6](#vita___db3_2_to_17_except_6_)
        - [r50       @ vita/db3_2_to_17_except_6](#r50___vita_db3_2_to_17_except_6_)
        - [r101       @ vita/db3_2_to_17_except_6](#r101___vita_db3_2_to_17_except_6_)
- [yl8       @ coco_to_xml](#yl8___coco_to_xm_l_)
    - [db3_2_to_17_except_6_val_1       @ yl8](#db3_2_to_17_except_6_val_1___yl_8_)
        - [last_on_sept5       @ db3_2_to_17_except_6_val_1/yl8](#last_on_sept5___db3_2_to_17_except_6_val_1_yl8_)
        - [last_on_db4_rockmaps       @ db3_2_to_17_except_6_val_1/yl8](#last_on_db4_rockmaps___db3_2_to_17_except_6_val_1_yl8_)
        - [best_on_sept5       @ db3_2_to_17_except_6_val_1/yl8](#best_on_sept5___db3_2_to_17_except_6_val_1_yl8_)
        - [best_on_db4_rockmaps       @ db3_2_to_17_except_6_val_1/yl8](#best_on_db4_rockmaps___db3_2_to_17_except_6_val_1_yl8_)
    - [db3_2_to_17_except_6_with_syn_val_2k_100       @ yl8](#db3_2_to_17_except_6_with_syn_val_2k_100___yl_8_)
        - [last_on_sept5_syn       @ db3_2_to_17_except_6_with_syn_val_2k_100/yl8](#last_on_sept5_syn___db3_2_to_17_except_6_with_syn_val_2k_100_yl8_)
        - [last_on_sept5_syn_2       @ db3_2_to_17_except_6_with_syn_val_2k_100/yl8](#last_on_sept5_syn_2___db3_2_to_17_except_6_with_syn_val_2k_100_yl8_)
        - [last_on_sept5_syn_3       @ db3_2_to_17_except_6_with_syn_val_2k_100/yl8](#last_on_sept5_syn_3___db3_2_to_17_except_6_with_syn_val_2k_100_yl8_)
        - [last_on_db4_rockmaps_syn       @ db3_2_to_17_except_6_with_syn_val_2k_100/yl8](#last_on_db4_rockmaps_syn___db3_2_to_17_except_6_with_syn_val_2k_100_yl8_)
        - [best_on_sept5_syn       @ db3_2_to_17_except_6_with_syn_val_2k_100/yl8](#best_on_sept5_syn___db3_2_to_17_except_6_with_syn_val_2k_100_yl8_)
        - [best_on_sept5_syn_2       @ db3_2_to_17_except_6_with_syn_val_2k_100/yl8](#best_on_sept5_syn_2___db3_2_to_17_except_6_with_syn_val_2k_100_yl8_)
        - [best_on_sept5_syn_3       @ db3_2_to_17_except_6_with_syn_val_2k_100/yl8](#best_on_sept5_syn_3___db3_2_to_17_except_6_with_syn_val_2k_100_yl8_)
        - [best_on_db4_rockmaps_syn       @ db3_2_to_17_except_6_with_syn_val_2k_100/yl8](#best_on_db4_rockmaps_syn___db3_2_to_17_except_6_with_syn_val_2k_100_yl8_)

<!-- /MarkdownTOC -->
<a id="db3_2_to_17_except_6___coco_to_xm_l_"></a>
# db3_2_to_17_except_6       @ coco_to_xml-->coco_to_xml

<a id="idol___db3_2_to_17_except_6_"></a>
## idol       @ db3_2_to_17_except_6-->coco_to_xml_mj
python3 coco_to_xml.py root_dir=/data/mojow_rock/rock_dataset3/ytvis19 json=vnxt/idol-ytvis-mj_rock-db3_2_to_17_except_6_large_huge/inference/json_results  gt_json=mj_rock-september_5_2020-large_huge-max_length-50.json class_names_path=lists/classes/predefined_classes_rock.txt ytvis=1 eval=0 fix_category_id=1 nms_thresh=0

<a id="vita___db3_2_to_17_except_6_"></a>
## vita       @ db3_2_to_17_except_6-->coco_to_xml_mj
<a id="r50___vita_db3_2_to_17_except_6_"></a>
### r50       @ vita/db3_2_to_17_except_6-->coco_to_xml_mj
python3 coco_to_xml.py root_dir=/data/mojow_rock/rock_dataset3/ytvis19 json=vita/db3_2_to_17_except_6-vita_r50/inference/results.json  gt_json=mj_rock-september_5_2020-large_huge-max_length-200.json class_names_path=lists/classes/predefined_classes_rock.txt ytvis=1 eval=0 fix_category_id=0 nms_thresh=0

<a id="r101___vita_db3_2_to_17_except_6_"></a>
### r101       @ vita/db3_2_to_17_except_6-->coco_to_xml_mj
python3 coco_to_xml.py root_dir=/data/mojow_rock/rock_dataset3/ytvis19 json=vita/db3_2_to_17_except_6-vita_r101/inference/results.json  gt_json=mj_rock-september_5_2020-large_huge-max_length-200.json class_names_path=lists/classes/predefined_classes_rock.txt ytvis=1 eval=0 fix_category_id=0 nms_thresh=0

<a id="yl8___coco_to_xm_l_"></a>
# yl8       @ coco_to_xml-->coco_to_xml
<a id="db3_2_to_17_except_6_val_1___yl_8_"></a>
## db3_2_to_17_except_6_val_1       @ yl8-->coco_to_xml_mj
<a id="last_on_sept5___db3_2_to_17_except_6_val_1_yl8_"></a>
### last_on_sept5       @ db3_2_to_17_except_6_val_1/yl8-->coco_to_xml_mj
python3 coco_to_xml.py root_dir=/data/mojow_rock/rock_dataset3 json=yl8/db3_2_to_17_except_6_val_1/last_on_sept5/predictions.json  gt_json=september_5_2020.json class_names_path=lists/classes/predefined_classes_rock.txt nms_thresh=0 n_proc=1 base_id_in_preds=1
<a id="last_on_db4_rockmaps___db3_2_to_17_except_6_val_1_yl8_"></a>
### last_on_db4_rockmaps       @ db3_2_to_17_except_6_val_1/yl8-->coco_to_xml_mj
python3 coco_to_xml.py root_dir=/data/mojow_rock/rock_dataset4 json=yl8/db3_2_to_17_except_6_val_1/last_on_db4_rockmaps-hub1/predictions.json  gt_json=rockmaps/db4-rockmaps-hub1.json class_names_path=lists/classes/predefined_classes_rock.txt nms_thresh=0 n_proc=1 base_id_in_preds=1

python3 coco_to_xml.py root_dir=/data/mojow_rock/rock_dataset4 json=yl8/db3_2_to_17_except_6_val_1/last_on_db4_rockmaps-hub2/predictions.json  gt_json=rockmaps/db4-rockmaps-hub2.json class_names_path=lists/classes/predefined_classes_rock.txt nms_thresh=0 n_proc=1 base_id_in_preds=1

python3 coco_to_xml.py root_dir=/data/mojow_rock/rock_dataset4 json=yl8/db3_2_to_17_except_6_val_1/last_on_db4_rockmaps-cs1/predictions.json gt_json=rockmaps/db4-rockmaps-cs1.json class_names_path=lists/classes/predefined_classes_rock.txt nms_thresh=0 n_proc=1 base_id_in_preds=1

python3 coco_to_xml.py root_dir=/data/mojow_rock/rock_dataset4 json=yl8/db3_2_to_17_except_6_val_1/last_on_db4_rockmaps-cs2/predictions.json  gt_json=rockmaps/db4-rockmaps-cs2.json class_names_path=lists/classes/predefined_classes_rock.txt nms_thresh=0 n_proc=1 base_id_in_preds=1

<a id="best_on_sept5___db3_2_to_17_except_6_val_1_yl8_"></a>
### best_on_sept5       @ db3_2_to_17_except_6_val_1/yl8-->coco_to_xml_mj
python3 coco_to_xml.py root_dir=/data/mojow_rock/rock_dataset3 json=yl8/db3_2_to_17_except_6_val_1/best_on_sept5/predictions.json  gt_json=september_5_2020.json class_names_path=lists/classes/predefined_classes_rock.txt nms_thresh=0 n_proc=1 base_id_in_preds=1

<a id="best_on_db4_rockmaps___db3_2_to_17_except_6_val_1_yl8_"></a>
### best_on_db4_rockmaps       @ db3_2_to_17_except_6_val_1/yl8-->coco_to_xml_mj
python3 coco_to_xml.py root_dir=/data/mojow_rock/rock_dataset4 json=yl8/db3_2_to_17_except_6_val_1/best_on_db4_rockmaps-hub1/predictions.json  gt_json=rockmaps/db4-rockmaps-hub1.json class_names_path=lists/classes/predefined_classes_rock.txt nms_thresh=0 n_proc=1 base_id_in_preds=1

python3 coco_to_xml.py root_dir=/data/mojow_rock/rock_dataset4 json=yl8/db3_2_to_17_except_6_val_1/best_on_db4_rockmaps-hub2/predictions.json  gt_json=rockmaps/db4-rockmaps-hub2.json class_names_path=lists/classes/predefined_classes_rock.txt nms_thresh=0 n_proc=1 base_id_in_preds=1

python3 coco_to_xml.py root_dir=/data/mojow_rock/rock_dataset4 json=yl8/db3_2_to_17_except_6_val_1/best_on_db4_rockmaps-cs1/predictions.json gt_json=rockmaps/db4-rockmaps-cs1.json class_names_path=lists/classes/predefined_classes_rock.txt nms_thresh=0 n_proc=1 base_id_in_preds=1

python3 coco_to_xml.py root_dir=/data/mojow_rock/rock_dataset4 json=yl8/db3_2_to_17_except_6_val_1/best_on_db4_rockmaps-cs2/predictions.json  gt_json=rockmaps/db4-rockmaps-cs2.json class_names_path=lists/classes/predefined_classes_rock.txt nms_thresh=0 n_proc=1 base_id_in_preds=1

<a id="db3_2_to_17_except_6_with_syn_val_2k_100___yl_8_"></a>
## db3_2_to_17_except_6_with_syn_val_2k_100       @ yl8-->coco_to_xml_mj
<a id="last_on_sept5_syn___db3_2_to_17_except_6_with_syn_val_2k_100_yl8_"></a>
### last_on_sept5_syn       @ db3_2_to_17_except_6_with_syn_val_2k_100/yl8-->coco_to_xml_mj
python3 coco_to_xml.py root_dir=/data/mojow_rock/rock_dataset3 json=yl8/db3_2_to_17_except_6_with_syn_val_2k_100/last_on_sept5_syn/predictions.json  gt_json=part5_on_september_5_2020.json class_names_path=lists/classes/predefined_classes_rock_syn.txt nms_thresh=0 n_proc=1 base_id_in_preds=1
<a id="last_on_sept5_syn_2___db3_2_to_17_except_6_with_syn_val_2k_100_yl8_"></a>
### last_on_sept5_syn_2       @ db3_2_to_17_except_6_with_syn_val_2k_100/yl8-->coco_to_xml_mj
python3 coco_to_xml.py root_dir=/data/mojow_rock/rock_dataset3 json=yl8/db3_2_to_17_except_6_with_syn_val_2k_100/last_on_sept5_syn_2/predictions.json  gt_json=part4_on_part5_on_september_5_2020.json class_names_path=lists/classes/predefined_classes_rock_syn.txt nms_thresh=0 n_proc=1 base_id_in_preds=1
<a id="last_on_sept5_syn_3___db3_2_to_17_except_6_with_syn_val_2k_100_yl8_"></a>
### last_on_sept5_syn_3       @ db3_2_to_17_except_6_with_syn_val_2k_100/yl8-->coco_to_xml_mj
python3 coco_to_xml.py root_dir=/data/mojow_rock/rock_dataset3 json=yl8/db3_2_to_17_except_6_with_syn_val_2k_100/last_on_sept5_syn_3/predictions.json  gt_json=part14_on_part4_on_part5_on_september_5_2020.json class_names_path=lists/classes/predefined_classes_rock_syn.txt nms_thresh=0 n_proc=1 base_id_in_preds=1
<a id="last_on_db4_rockmaps_syn___db3_2_to_17_except_6_with_syn_val_2k_100_yl8_"></a>
### last_on_db4_rockmaps_syn       @ db3_2_to_17_except_6_with_syn_val_2k_100/yl8-->coco_to_xml_mj
python3 coco_to_xml.py root_dir=/data/mojow_rock/rock_dataset4 json=yl8/db3_2_to_17_except_6_with_syn_val_2k_100/last_on_db4_rockmaps_syn-hub1/predictions.json  gt_json=db4-rockmaps_syn-hub1.json class_names_path=lists/classes/predefined_classes_rock_syn.txt nms_thresh=0 n_proc=1 base_id_in_preds=1

python3 coco_to_xml.py root_dir=/data/mojow_rock/rock_dataset4 json=yl8/db3_2_to_17_except_6_with_syn_val_2k_100/last_on_db4_rockmaps_syn-hub2/predictions.json  gt_json=db4-rockmaps_syn-hub2.json class_names_path=lists/classes/predefined_classes_rock_syn.txt nms_thresh=0 n_proc=1 base_id_in_preds=1

python3 coco_to_xml.py root_dir=/data/mojow_rock/rock_dataset4 json=yl8/db3_2_to_17_except_6_with_syn_val_2k_100/last_on_db4_rockmaps_syn-cs1/predictions.json  gt_json=db4-rockmaps_syn-cs1.json class_names_path=lists/classes/predefined_classes_rock_syn.txt nms_thresh=0 n_proc=1 base_id_in_preds=1

python3 coco_to_xml.py root_dir=/data/mojow_rock/rock_dataset4 json=yl8/db3_2_to_17_except_6_with_syn_val_2k_100/last_on_db4_rockmaps_syn-cs2/predictions.json  gt_json=db4-rockmaps_syn-cs2.json class_names_path=lists/classes/predefined_classes_rock_syn.txt nms_thresh=0 n_proc=1 base_id_in_preds=1

<a id="best_on_sept5_syn___db3_2_to_17_except_6_with_syn_val_2k_100_yl8_"></a>
### best_on_sept5_syn       @ db3_2_to_17_except_6_with_syn_val_2k_100/yl8-->coco_to_xml_mj
python3 coco_to_xml.py root_dir=/data/mojow_rock/rock_dataset3 json=yl8/db3_2_to_17_except_6_with_syn_val_2k_100/best_on_sept5_syn/predictions.json  gt_json=part5_on_september_5_2020.json class_names_path=lists/classes/predefined_classes_rock_syn.txt nms_thresh=0 n_proc=1 base_id_in_preds=1
<a id="best_on_sept5_syn_2___db3_2_to_17_except_6_with_syn_val_2k_100_yl8_"></a>
### best_on_sept5_syn_2       @ db3_2_to_17_except_6_with_syn_val_2k_100/yl8-->coco_to_xml_mj
python3 coco_to_xml.py root_dir=/data/mojow_rock/rock_dataset3 json=yl8/db3_2_to_17_except_6_with_syn_val_2k_100/best_on_sept5_syn_2/predictions.json  gt_json=part4_on_part5_on_september_5_2020.json class_names_path=lists/classes/predefined_classes_rock_syn.txt nms_thresh=0 n_proc=1 base_id_in_preds=1
<a id="best_on_sept5_syn_3___db3_2_to_17_except_6_with_syn_val_2k_100_yl8_"></a>
### best_on_sept5_syn_3       @ db3_2_to_17_except_6_with_syn_val_2k_100/yl8-->coco_to_xml_mj
python3 coco_to_xml.py root_dir=/data/mojow_rock/rock_dataset3 json=yl8/db3_2_to_17_except_6_with_syn_val_2k_100/best_on_sept5_syn_3/predictions.json  gt_json=part14_on_part4_on_part5_on_september_5_2020.json class_names_path=lists/classes/predefined_classes_rock_syn.txt nms_thresh=0 n_proc=1 base_id_in_preds=1

<a id="best_on_db4_rockmaps_syn___db3_2_to_17_except_6_with_syn_val_2k_100_yl8_"></a>
### best_on_db4_rockmaps_syn       @ db3_2_to_17_except_6_with_syn_val_2k_100/yl8-->coco_to_xml_mj
python3 coco_to_xml.py root_dir=/data/mojow_rock/rock_dataset4 json=yl8/db3_2_to_17_except_6_with_syn_val_2k_100/best_on_db4_rockmaps_syn-hub1/predictions.json  gt_json=db4-rockmaps_syn-hub1.json class_names_path=lists/classes/predefined_classes_rock_syn.txt nms_thresh=0 n_proc=1 base_id_in_preds=1

python3 coco_to_xml.py root_dir=/data/mojow_rock/rock_dataset4 json=yl8/db3_2_to_17_except_6_with_syn_val_2k_100/best_on_db4_rockmaps_syn-hub2/predictions.json  gt_json=db4-rockmaps_syn-hub2.json class_names_path=lists/classes/predefined_classes_rock_syn.txt nms_thresh=0 n_proc=1 base_id_in_preds=1

python3 coco_to_xml.py root_dir=/data/mojow_rock/rock_dataset4 json=yl8/db3_2_to_17_except_6_with_syn_val_2k_100/best_on_db4_rockmaps_syn-cs1/predictions.json  gt_json=db4-rockmaps_syn-cs1.json class_names_path=lists/classes/predefined_classes_rock_syn.txt nms_thresh=0 n_proc=1 base_id_in_preds=1

python3 coco_to_xml.py root_dir=/data/mojow_rock/rock_dataset4 json=yl8/db3_2_to_17_except_6_with_syn_val_2k_100/best_on_db4_rockmaps_syn-cs2/predictions.json  gt_json=db4-rockmaps_syn-cs2.json class_names_path=lists/classes/predefined_classes_rock_syn.txt nms_thresh=0 n_proc=1 base_id_in_preds=1

