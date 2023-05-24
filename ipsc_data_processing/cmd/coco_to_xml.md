<!-- MarkdownTOC -->

- [all_frames_roi_g2_0_37       @ coco_to_xml](#all_frames_roi_g2_0_37___coco_to_xm_l_)
    - [swi       @ all_frames_roi_g2_0_37](#swi___all_frames_roi_g2_0_37_)
    - [idol       @ all_frames_roi_g2_0_37](#idol___all_frames_roi_g2_0_37_)
    - [seqformer       @ all_frames_roi_g2_0_37](#seqformer___all_frames_roi_g2_0_37_)
    - [vita-swin       @ all_frames_roi_g2_0_37](#vita_swin___all_frames_roi_g2_0_37_)
    - [vita-r50       @ all_frames_roi_g2_0_37](#vita_r50___all_frames_roi_g2_0_37_)
    - [vita-r101       @ all_frames_roi_g2_0_37](#vita_r101___all_frames_roi_g2_0_37_)
- [ext_reorg_roi_g2_0_37       @ coco_to_xml](#ext_reorg_roi_g2_0_37___coco_to_xm_l_)
    - [swi       @ ext_reorg_roi_g2_0_37](#swi___ext_reorg_roi_g2_0_3_7_)
        - [nms       @ swi/ext_reorg_roi_g2_0_37](#nms___swi_ext_reorg_roi_g2_0_3_7_)
    - [swi-no_validate-rcnn       @ ext_reorg_roi_g2_0_37](#swi_no_validate_rcnn___ext_reorg_roi_g2_0_3_7_)
    - [swi-no_validate-rcnn-win7       @ ext_reorg_roi_g2_0_37](#swi_no_validate_rcnn_win7___ext_reorg_roi_g2_0_3_7_)
    - [swi-rcnn-win7       @ ext_reorg_roi_g2_0_37](#swi_rcnn_win7___ext_reorg_roi_g2_0_3_7_)
    - [cvnxt-base       @ ext_reorg_roi_g2_0_37](#cvnxt_base___ext_reorg_roi_g2_0_3_7_)
        - [nms       @ cvnxt-base/ext_reorg_roi_g2_0_37](#nms___cvnxt_base_ext_reorg_roi_g2_0_37_)
    - [cvnxt-large       @ ext_reorg_roi_g2_0_37](#cvnxt_large___ext_reorg_roi_g2_0_3_7_)
        - [nms       @ cvnxt-large/ext_reorg_roi_g2_0_37](#nms___cvnxt_large_ext_reorg_roi_g2_0_3_7_)
    - [idol       @ ext_reorg_roi_g2_0_37](#idol___ext_reorg_roi_g2_0_3_7_)
    - [idol-incremental       @ ext_reorg_roi_g2_0_37](#idol_incremental___ext_reorg_roi_g2_0_3_7_)
        - [probs       @ idol-incremental/ext_reorg_roi_g2_0_37](#probs___idol_incremental_ext_reorg_roi_g2_0_37_)
    - [seqformer       @ ext_reorg_roi_g2_0_37](#seqformer___ext_reorg_roi_g2_0_3_7_)
    - [seqformer-incremental       @ ext_reorg_roi_g2_0_37](#seqformer_incremental___ext_reorg_roi_g2_0_3_7_)
        - [nms       @ seqformer-incremental/ext_reorg_roi_g2_0_37](#nms___seqformer_incremental_ext_reorg_roi_g2_0_3_7_)
        - [probs       @ seqformer-incremental/ext_reorg_roi_g2_0_37](#probs___seqformer_incremental_ext_reorg_roi_g2_0_3_7_)
    - [vita-swin       @ ext_reorg_roi_g2_0_37](#vita_swin___ext_reorg_roi_g2_0_3_7_)
        - [incremental       @ vita-swin/ext_reorg_roi_g2_0_37](#incremental___vita_swin_ext_reorg_roi_g2_0_3_7_)
            - [nms       @ incremental/vita-swin/ext_reorg_roi_g2_0_37](#nms___incremental_vita_swin_ext_reorg_roi_g2_0_3_7_)
    - [vita-r50       @ ext_reorg_roi_g2_0_37](#vita_r50___ext_reorg_roi_g2_0_3_7_)
- [ext_reorg_roi_g2_16_53       @ coco_to_xml](#ext_reorg_roi_g2_16_53___coco_to_xm_l_)
    - [yl8       @ ext_reorg_roi_g2_16_53](#yl8___ext_reorg_roi_g2_16_53_)
        - [val       @ yl8/ext_reorg_roi_g2_16_53](#val___yl8_ext_reorg_roi_g2_16_53_)
        - [seq-val       @ yl8/ext_reorg_roi_g2_16_53](#seq_val___yl8_ext_reorg_roi_g2_16_53_)
    - [swi       @ ext_reorg_roi_g2_16_53](#swi___ext_reorg_roi_g2_16_53_)
    - [swi-rcnn       @ ext_reorg_roi_g2_16_53](#swi_rcnn___ext_reorg_roi_g2_16_53_)
    - [cvnxt       @ ext_reorg_roi_g2_16_53](#cvnxt___ext_reorg_roi_g2_16_53_)
    - [idol       @ ext_reorg_roi_g2_16_53](#idol___ext_reorg_roi_g2_16_53_)
        - [probs       @ idol/ext_reorg_roi_g2_16_53](#probs___idol_ext_reorg_roi_g2_16_5_3_)
    - [idol-inc       @ ext_reorg_roi_g2_16_53](#idol_inc___ext_reorg_roi_g2_16_53_)
        - [nms       @ idol-inc/ext_reorg_roi_g2_16_53](#nms___idol_inc_ext_reorg_roi_g2_16_5_3_)
    - [idol-inc-probs       @ ext_reorg_roi_g2_16_53](#idol_inc_probs___ext_reorg_roi_g2_16_53_)
        - [nms-01       @ idol-inc-probs/ext_reorg_roi_g2_16_53](#nms_01___idol_inc_probs_ext_reorg_roi_g2_16_5_3_)
    - [seq       @ ext_reorg_roi_g2_16_53](#seq___ext_reorg_roi_g2_16_53_)
        - [probs       @ seq/ext_reorg_roi_g2_16_53](#probs___seq_ext_reorg_roi_g2_16_53_)
    - [seq-inc       @ ext_reorg_roi_g2_16_53](#seq_inc___ext_reorg_roi_g2_16_53_)
        - [nms       @ seq-inc/ext_reorg_roi_g2_16_53](#nms___seq_inc_ext_reorg_roi_g2_16_53_)
        - [probs       @ seq-inc/ext_reorg_roi_g2_16_53](#probs___seq_inc_ext_reorg_roi_g2_16_53_)
    - [vita       @ ext_reorg_roi_g2_16_53](#vita___ext_reorg_roi_g2_16_53_)
        - [0119999       @ vita/ext_reorg_roi_g2_16_53](#0119999___vita_ext_reorg_roi_g2_16_5_3_)
        - [0329999       @ vita/ext_reorg_roi_g2_16_53](#0329999___vita_ext_reorg_roi_g2_16_5_3_)
    - [vita-inc       @ ext_reorg_roi_g2_16_53](#vita_inc___ext_reorg_roi_g2_16_53_)
        - [0119999       @ vita-inc/ext_reorg_roi_g2_16_53](#0119999___vita_inc_ext_reorg_roi_g2_16_5_3_)
        - [0329999       @ vita-inc/ext_reorg_roi_g2_16_53](#0329999___vita_inc_ext_reorg_roi_g2_16_5_3_)
            - [nms       @ 0329999/vita-inc/ext_reorg_roi_g2_16_53](#nms___0329999_vita_inc_ext_reorg_roi_g2_16_5_3_)
    - [vita-inc-retrain       @ ext_reorg_roi_g2_16_53](#vita_inc_retrain___ext_reorg_roi_g2_16_53_)
        - [0004999       @ vita-inc-retrain/ext_reorg_roi_g2_16_53](#0004999___vita_inc_retrain_ext_reorg_roi_g2_16_5_3_)
        - [0079999       @ vita-inc-retrain/ext_reorg_roi_g2_16_53](#0079999___vita_inc_retrain_ext_reorg_roi_g2_16_5_3_)
        - [0104999       @ vita-inc-retrain/ext_reorg_roi_g2_16_53](#0104999___vita_inc_retrain_ext_reorg_roi_g2_16_5_3_)
- [ext_reorg_roi_g2_54_126       @ coco_to_xml](#ext_reorg_roi_g2_54_126___coco_to_xm_l_)
    - [yl8       @ ext_reorg_roi_g2_54_126](#yl8___ext_reorg_roi_g2_54_12_6_)
    - [swi       @ ext_reorg_roi_g2_54_126](#swi___ext_reorg_roi_g2_54_12_6_)
        - [g2_0_15       @ swi/ext_reorg_roi_g2_54_126](#g2_0_15___swi_ext_reorg_roi_g2_54_12_6_)
    - [cvnxt-large       @ ext_reorg_roi_g2_54_126](#cvnxt_large___ext_reorg_roi_g2_54_12_6_)
        - [g2_0_15       @ cvnxt-large/ext_reorg_roi_g2_54_126](#g2_0_15___cvnxt_large_ext_reorg_roi_g2_54_12_6_)
    - [cvnxt-base       @ ext_reorg_roi_g2_54_126](#cvnxt_base___ext_reorg_roi_g2_54_12_6_)
        - [g2_0_15       @ cvnxt-base/ext_reorg_roi_g2_54_126](#g2_0_15___cvnxt_base_ext_reorg_roi_g2_54_126_)
    - [idol       @ ext_reorg_roi_g2_54_126](#idol___ext_reorg_roi_g2_54_12_6_)
        - [g2_0_53       @ idol/ext_reorg_roi_g2_54_126](#g2_0_53___idol_ext_reorg_roi_g2_54_126_)
        - [g2_0_15       @ idol/ext_reorg_roi_g2_54_126](#g2_0_15___idol_ext_reorg_roi_g2_54_126_)
    - [idol-inc       @ ext_reorg_roi_g2_54_126](#idol_inc___ext_reorg_roi_g2_54_12_6_)
        - [g2_0_53       @ idol-inc/ext_reorg_roi_g2_54_126](#g2_0_53___idol_inc_ext_reorg_roi_g2_54_126_)
        - [g2_0_15       @ idol-inc/ext_reorg_roi_g2_54_126](#g2_0_15___idol_inc_ext_reorg_roi_g2_54_126_)
    - [seq       @ ext_reorg_roi_g2_54_126](#seq___ext_reorg_roi_g2_54_12_6_)
        - [g2_0_53       @ seq/ext_reorg_roi_g2_54_126](#g2_0_53___seq_ext_reorg_roi_g2_54_12_6_)
        - [g2_0_15       @ seq/ext_reorg_roi_g2_54_126](#g2_0_15___seq_ext_reorg_roi_g2_54_12_6_)
    - [seq-inc       @ ext_reorg_roi_g2_54_126](#seq_inc___ext_reorg_roi_g2_54_12_6_)
        - [g2_0_53       @ seq-inc/ext_reorg_roi_g2_54_126](#g2_0_53___seq_inc_ext_reorg_roi_g2_54_12_6_)
        - [g2_0_15       @ seq-inc/ext_reorg_roi_g2_54_126](#g2_0_15___seq_inc_ext_reorg_roi_g2_54_12_6_)
    - [vita       @ ext_reorg_roi_g2_54_126](#vita___ext_reorg_roi_g2_54_12_6_)
        - [g2_0_53       @ vita/ext_reorg_roi_g2_54_126](#g2_0_53___vita_ext_reorg_roi_g2_54_126_)
        - [g2_0_15       @ vita/ext_reorg_roi_g2_54_126](#g2_0_15___vita_ext_reorg_roi_g2_54_126_)
    - [vita-inc       @ ext_reorg_roi_g2_54_126](#vita_inc___ext_reorg_roi_g2_54_12_6_)
        - [g2_0_53       @ vita-inc/ext_reorg_roi_g2_54_126](#g2_0_53___vita_inc_ext_reorg_roi_g2_54_126_)
            - [max_length-2       @ g2_0_53/vita-inc/ext_reorg_roi_g2_54_126](#max_length_2___g2_0_53_vita_inc_ext_reorg_roi_g2_54_126_)
        - [g2_0_15       @ vita-inc/ext_reorg_roi_g2_54_126](#g2_0_15___vita_inc_ext_reorg_roi_g2_54_126_)
            - [max_length-2       @ g2_0_15/vita-inc/ext_reorg_roi_g2_54_126](#max_length_2___g2_0_15_vita_inc_ext_reorg_roi_g2_54_126_)

<!-- /MarkdownTOC -->


<a id="all_frames_roi_g2_0_37___coco_to_xm_l_"></a>
# all_frames_roi_g2_0_37       @ coco_to_xml-->coco_to_xml

<a id="swi___all_frames_roi_g2_0_37_"></a>
## swi       @ all_frames_roi_g2_0_37-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi json=swin_det/ipsc_2_class_all_frames_roi_g2_0_37/g2_38_53/results.segm.json  gt_json=all_frames_roi_g2_38_53.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt save_csv=1

python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi json=swin_det/ipsc_2_class_all_frames_roi_g2_0_37/g2_38_53/results.segm.json  gt_json=ytvis19/ytvis-ipsc-all_frames_roi_g2_38_53_coco.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt

<a id="idol___all_frames_roi_g2_0_37_"></a>
## idol       @ all_frames_roi_g2_0_37-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/idol-ipsc-all_frames_roi_g2_0_37/inference/results.json  gt_json=ytvis-ipsc-all_frames_roi_g2_38_53-.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1

<a id="seqformer___all_frames_roi_g2_0_37_"></a>
## seqformer       @ all_frames_roi_g2_0_37-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/seqformer-ipsc-all_frames_roi_g2_0_37/inference/results.json  gt_json=ytvis-ipsc-all_frames_roi_g2_38_53.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1

<a id="vita_swin___all_frames_roi_g2_0_37_"></a>
## vita-swin       @ all_frames_roi_g2_0_37-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vita/vita-ipsc-all_frames_roi_g2_0_37_swin/inference/results.json  gt_json=ytvis-ipsc-all_frames_roi_g2_38_53.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1

<a id="vita_r50___all_frames_roi_g2_0_37_"></a>
## vita-r50       @ all_frames_roi_g2_0_37-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vita/vita-ipsc-all_frames_roi_g2_0_37_r50/inference/results.json  gt_json=ytvis-ipsc-all_frames_roi_g2_38_53.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1

<a id="vita_r101___all_frames_roi_g2_0_37_"></a>
## vita-r101       @ all_frames_roi_g2_0_37-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vita/vita-ipsc-all_frames_roi_g2_0_37_r101/inference/results.json  gt_json=ytvis-ipsc-all_frames_roi_g2_38_53.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1


<a id="ext_reorg_roi_g2_0_37___coco_to_xm_l_"></a>
# ext_reorg_roi_g2_0_37       @ coco_to_xml-->coco_to_xml
<a id="swi___ext_reorg_roi_g2_0_3_7_"></a>
## swi       @ ext_reorg_roi_g2_0_37-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi json=swi/ipsc_2_class_ext_reorg_roi_g2_0_37-no_validate/g2_38_53/results.segm.json  gt_json=ext_reorg_roi_g2_38_53.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt 
<a id="nms___swi_ext_reorg_roi_g2_0_3_7_"></a>
### nms       @ swi/ext_reorg_roi_g2_0_37-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi json=swi/ipsc_2_class_ext_reorg_roi_g2_0_37-no_validate/g2_38_53/results.segm.json  gt_json=ext_reorg_roi_g2_38_53.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt nms_thresh=0.1:0.9:0.1 n_proc=12

<a id="swi_no_validate_rcnn___ext_reorg_roi_g2_0_3_7_"></a>
## swi-no_validate-rcnn       @ ext_reorg_roi_g2_0_37-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi json=swi/ipsc_2_class_ext_reorg_roi_g2_0_37-no_validate-rcnn/g2_38_53/results.bbox.json  gt_json=ext_reorg_roi_g2_38_53.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt nms_thresh=0.1:0.9:0.1 n_proc=12 enable_mask=0 save=0

<a id="swi_no_validate_rcnn_win7___ext_reorg_roi_g2_0_3_7_"></a>
## swi-no_validate-rcnn-win7       @ ext_reorg_roi_g2_0_37-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi json=swi/ipsc_2_class_ext_reorg_roi_g2_0_37-no_validate-rcnn-win7/g2_38_53/results.bbox.json  gt_json=ext_reorg_roi_g2_38_53.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt nms_thresh=0.1:0.9:0.1 n_proc=12 enable_mask=0 save=0

<a id="swi_rcnn_win7___ext_reorg_roi_g2_0_3_7_"></a>
## swi-rcnn-win7       @ ext_reorg_roi_g2_0_37-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi json=swi/ipsc_2_class_ext_reorg_roi_g2_0_37-rcnn-win7/g2_38_53/results.bbox.json  gt_json=ext_reorg_roi_g2_38_53.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt nms_thresh=0.1:0.9:0.1 n_proc=12 enable_mask=0 save=0

<a id="cvnxt_base___ext_reorg_roi_g2_0_3_7_"></a>
## cvnxt-base       @ ext_reorg_roi_g2_0_37-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi json=swi/ipsc_2_class_ext_reorg_roi_g2_0_37-no_validate-convnext_base_in22k/g2_38_53/results.segm.json  gt_json=ext_reorg_roi_g2_38_53.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt 
<a id="nms___cvnxt_base_ext_reorg_roi_g2_0_37_"></a>
### nms       @ cvnxt-base/ext_reorg_roi_g2_0_37-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi json=swi/ipsc_2_class_ext_reorg_roi_g2_0_37-no_validate-convnext_base_in22k/g2_38_53/results.segm.json  gt_json=ext_reorg_roi_g2_38_53.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt nms_thresh=0.1:0.9:0.1 n_proc=12

<a id="cvnxt_large___ext_reorg_roi_g2_0_3_7_"></a>
## cvnxt-large       @ ext_reorg_roi_g2_0_37-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi json=swi/ipsc_2_class_ext_reorg_roi_g2_0_37-convnext_large_in22k/g2_38_53/results.segm.json  gt_json=ext_reorg_roi_g2_38_53.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt 
<a id="nms___cvnxt_large_ext_reorg_roi_g2_0_3_7_"></a>
### nms       @ cvnxt-large/ext_reorg_roi_g2_0_37-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi json=swi/ipsc_2_class_ext_reorg_roi_g2_0_37-convnext_large_in22k/g2_38_53/results.segm.json  gt_json=ext_reorg_roi_g2_38_53.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt nms_thresh=0.1:0.9:0.1 n_proc=12

<a id="idol___ext_reorg_roi_g2_0_3_7_"></a>
## idol       @ ext_reorg_roi_g2_0_37-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/idol-ipsc-ext_reorg_roi_g2_0_37/inference/json_results  gt_json=ipsc-ext_reorg_roi_g2_38_53.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 fix_category_id=1 save=0

<a id="idol_incremental___ext_reorg_roi_g2_0_3_7_"></a>
## idol-incremental       @ ext_reorg_roi_g2_0_37-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/idol-ipsc-ext_reorg_roi_g2_0_37/inference_38_53-incremental/json_results  gt_json=ipsc-ext_reorg_roi_g2_38_53-incremental.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 fix_category_id=1 save=0 incremental=1
__nms__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/idol-ipsc-ext_reorg_roi_g2_0_37/inference_38_53-incremental/json_results  gt_json=ipsc-ext_reorg_roi_g2_38_53-incremental.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 fix_category_id=1 save=0 incremental=1 nms_thresh=0.1:0.9:0.1 n_proc=12

<a id="probs___idol_incremental_ext_reorg_roi_g2_0_37_"></a>
### probs       @ idol-incremental/ext_reorg_roi_g2_0_37-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/idol-ipsc-ext_reorg_roi_g2_0_37/inference_probs/json_results  gt_json=ipsc-ext_reorg_roi_g2_38_53-incremental.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 fix_category_id=1 save=0 incremental=1
__nms__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/idol-ipsc-ext_reorg_roi_g2_0_37/inference_probs/json_results  gt_json=ipsc-ext_reorg_roi_g2_38_53-incremental.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 fix_category_id=1 save=0 incremental=1 nms_thresh=0.1:0.9:0.1 n_proc=12

<a id="seqformer___ext_reorg_roi_g2_0_3_7_"></a>
## seqformer       @ ext_reorg_roi_g2_0_37-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/seqformer-ipsc-ext_reorg_roi_g2_0_37/inference/results.json  gt_json=ipsc-ext_reorg_roi_g2_38_53.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=1

<a id="seqformer_incremental___ext_reorg_roi_g2_0_3_7_"></a>
## seqformer-incremental       @ ext_reorg_roi_g2_0_37-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/seqformer-ipsc-ext_reorg_roi_g2_0_37/inference_38_53-incremental/results.json  gt_json=ipsc-ext_reorg_roi_g2_38_53-incremental.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 incremental=1
<a id="nms___seqformer_incremental_ext_reorg_roi_g2_0_3_7_"></a>
### nms       @ seqformer-incremental/ext_reorg_roi_g2_0_37-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/seqformer-ipsc-ext_reorg_roi_g2_0_37/inference_38_53-incremental/results.json  gt_json=ipsc-ext_reorg_roi_g2_38_53-incremental.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 incremental=1 nms_thresh=0.1:0.9:0.1 n_proc=12

<a id="probs___seqformer_incremental_ext_reorg_roi_g2_0_3_7_"></a>
### probs       @ seqformer-incremental/ext_reorg_roi_g2_0_37-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/seqformer-ipsc-ext_reorg_roi_g2_0_37/inference_probs/results.json  gt_json=ipsc-ext_reorg_roi_g2_38_53-incremental.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 incremental=1

<a id="vita_swin___ext_reorg_roi_g2_0_3_7_"></a>
## vita-swin       @ ext_reorg_roi_g2_0_37-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vita/vita-ipsc-ext_reorg_roi_g2_0_37_swin/inference/results.json  gt_json=ipsc-ext_reorg_roi_g2_38_53.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0

<a id="incremental___vita_swin_ext_reorg_roi_g2_0_3_7_"></a>
### incremental       @ vita-swin/ext_reorg_roi_g2_0_37-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vita/vita-ipsc-ext_reorg_roi_g2_0_37_swin/inference_38_53-incremental/results.json  gt_json=ipsc-ext_reorg_roi_g2_38_53-incremental.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 incremental=1
<a id="nms___incremental_vita_swin_ext_reorg_roi_g2_0_3_7_"></a>
#### nms       @ incremental/vita-swin/ext_reorg_roi_g2_0_37-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vita/vita-ipsc-ext_reorg_roi_g2_0_37_swin/inference_38_53-incremental/results.json  gt_json=ipsc-ext_reorg_roi_g2_38_53-incremental.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 incremental=1 nms_thresh=0.1:0.9:0.1 n_proc=12

<a id="vita_r50___ext_reorg_roi_g2_0_3_7_"></a>
## vita-r50       @ ext_reorg_roi_g2_0_37-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vita/vita-ipsc-ext_reorg_roi_g2_0_37_r50/inference/results.json  gt_json=ipsc-ext_reorg_roi_g2_38_53.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 eval=0

<a id="ext_reorg_roi_g2_16_53___coco_to_xm_l_"></a>
# ext_reorg_roi_g2_16_53       @ coco_to_xml-->coco_to_xml

<a id="yl8___ext_reorg_roi_g2_16_53_"></a>
## yl8       @ ext_reorg_roi_g2_16_53-->coco_to_xml
__last__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi json=yl8/ext_reorg_roi_g2_16_53/last/predictions.json  gt_json=ext_reorg_roi_g2_0_15.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt nms_thresh=0:0.9:0.1 n_proc=12 save=0
__best__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi json=yl8/ext_reorg_roi_g2_16_53/best/predictions.json  gt_json=ext_reorg_roi_g2_0_15.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt nms_thresh=0:0.9:0.1 n_proc=12 save=0
<a id="val___yl8_ext_reorg_roi_g2_16_53_"></a>
### val       @ yl8/ext_reorg_roi_g2_16_53-->coco_to_xml
__last__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi json=yl8/ext_reorg_roi_g2_16_53-val/last/predictions.json  gt_json=ext_reorg_roi_g2_0_15.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt nms_thresh=0:0.9:0.1 n_proc=12 save=0
__best__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi json=yl8/ext_reorg_roi_g2_16_53-val/best/predictions.json  gt_json=ext_reorg_roi_g2_0_15.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt nms_thresh=0:0.9:0.1 n_proc=12 save=0

<a id="seq_val___yl8_ext_reorg_roi_g2_16_53_"></a>
### seq-val       @ yl8/ext_reorg_roi_g2_16_53-->coco_to_xml
__last__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi json=yl8/ext_reorg_roi_g2_16_53-seq-val/last/predictions.json  gt_json=ext_reorg_roi_g2_0_15.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt nms_thresh=0:0.9:0.1 n_proc=12 save=0
__best__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi json=yl8/ext_reorg_roi_g2_16_53-seq-val/best/predictions.json  gt_json=ext_reorg_roi_g2_0_15.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt nms_thresh=0:0.9:0.1 n_proc=12 save=0

<a id="swi___ext_reorg_roi_g2_16_53_"></a>
## swi       @ ext_reorg_roi_g2_16_53-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi json=swi/ipsc_2_class_ext_reorg_roi_g2_16_53-no_validate/g2_0_15/results.segm.json  gt_json=ext_reorg_roi_g2_0_15.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt nms_thresh=0.1:0.9:0.1 n_proc=12

<a id="swi_rcnn___ext_reorg_roi_g2_16_53_"></a>
## swi-rcnn       @ ext_reorg_roi_g2_16_53-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi json=swi/ipsc_2_class_ext_reorg_roi_g2_16_53-no_validate-rcnn/g2_0_15/results.bbox.json  gt_json=ext_reorg_roi_g2_0_15.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt nms_thresh=0.1:0.9:0.1 n_proc=12 enable_mask=0 save=0

<a id="cvnxt___ext_reorg_roi_g2_16_53_"></a>
## cvnxt       @ ext_reorg_roi_g2_16_53-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi json=swi/ipsc_2_class_ext_reorg_roi_g2_16_53-convnext_large_in22k/g2_0_15/results.segm.json  gt_json=ext_reorg_roi_g2_0_15.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt nms_thresh=0.1:0.9:0.1 n_proc=12

<a id="idol___ext_reorg_roi_g2_16_53_"></a>
## idol       @ ext_reorg_roi_g2_16_53-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/idol-ipsc-ext_reorg_roi_g2_16_53/inference_model_0254999/json_results  gt_json=ipsc-ext_reorg_roi_g2_0_15.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 fix_category_id=1 save=0 nms_thresh=0:0.9:0.1 n_proc=2
__-max_length-1-__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/idol-ipsc-ext_reorg_roi_g2_16_53/inference_model_0254999_max_length-1/json_results  gt_json=ipsc-ext_reorg_roi_g2_0_15-max_length-1.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 fix_category_id=1 save=0 nms_thresh=0:0.9:0.1 n_proc=2
__-max_length-2-__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/idol-ipsc-ext_reorg_roi_g2_16_53/inference_model_0254999_max_length-2/json_results  gt_json=ipsc-ext_reorg_roi_g2_0_15-max_length-2.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 fix_category_id=1 save=0 nms_thresh=0:0.9:0.1 n_proc=2
__-max_length-4-__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/idol-ipsc-ext_reorg_roi_g2_16_53/inference_model_0254999_max_length-4/json_results  gt_json=ipsc-ext_reorg_roi_g2_0_15-max_length-4.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 fix_category_id=1 save=0 nms_thresh=0:0.9:0.1 n_proc=2
__-max_length-8-__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/idol-ipsc-ext_reorg_roi_g2_16_53/inference_model_0254999_max_length-8/json_results  gt_json=ipsc-ext_reorg_roi_g2_0_15-max_length-8.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 fix_category_id=1 save=0 nms_thresh=0:0.9:0.1 n_proc=2

<a id="probs___idol_ext_reorg_roi_g2_16_5_3_"></a>
### probs       @ idol/ext_reorg_roi_g2_16_53-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/idol-ipsc-ext_reorg_roi_g2_16_53/inference_model_0254999_probs/json_results  gt_json=ipsc-ext_reorg_roi_g2_0_15.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 fix_category_id=1 save=0 nms_thresh=0:0.9:0.1 n_proc=2

<a id="idol_inc___ext_reorg_roi_g2_16_53_"></a>
## idol-inc       @ ext_reorg_roi_g2_16_53-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/idol-ipsc-ext_reorg_roi_g2_16_53/inference_model_0254999_incremental/json_results  gt_json=ipsc-ext_reorg_roi_g2_0_15-incremental.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 fix_category_id=1 save=0 incremental=1

<a id="nms___idol_inc_ext_reorg_roi_g2_16_5_3_"></a>
### nms       @ idol-inc/ext_reorg_roi_g2_16_53-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/idol-ipsc-ext_reorg_roi_g2_16_53/inference_model_0254999_incremental/json_results  gt_json=ipsc-ext_reorg_roi_g2_0_15-incremental.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 fix_category_id=1 save=0 incremental=1 nms_thresh=0.1:0.9:0.1 n_proc=12

<a id="idol_inc_probs___ext_reorg_roi_g2_16_53_"></a>
## idol-inc-probs       @ ext_reorg_roi_g2_16_53-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/idol-ipsc-ext_reorg_roi_g2_16_53/inference_model_0254999_incremental_probs/json_results  gt_json=ipsc-ext_reorg_roi_g2_0_15-incremental.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 fix_category_id=1 save=1 incremental=1 nms_thresh=0.1 n_proc=12

<a id="nms_01___idol_inc_probs_ext_reorg_roi_g2_16_5_3_"></a>
### nms-01       @ idol-inc-probs/ext_reorg_roi_g2_16_53-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/idol-ipsc-ext_reorg_roi_g2_16_53/inference_model_0254999_incremental_probs/json_results  gt_json=ipsc-ext_reorg_roi_g2_0_15-incremental.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 fix_category_id=1 save=0 incremental=1 nms_thresh=0.01

<a id="seq___ext_reorg_roi_g2_16_53_"></a>
## seq       @ ext_reorg_roi_g2_16_53-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/seqformer-ipsc-ext_reorg_roi_g2_16_53/inference_model_0241999/results.json  gt_json=ipsc-ext_reorg_roi_g2_0_15.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 nms_thresh=0:0.9:0.1 n_proc=2
__-max_length-1-__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/seqformer-ipsc-ext_reorg_roi_g2_16_53/inference_model_02419999_max_length-1/results.json  gt_json=ipsc-ext_reorg_roi_g2_0_15-max_length-1.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 nms_thresh=0:0.9:0.1 n_proc=2
__-max_length-2-__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/seqformer-ipsc-ext_reorg_roi_g2_16_53/inference_model_02419999_max_length-2/results.json  gt_json=ipsc-ext_reorg_roi_g2_0_15-max_length-2.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 nms_thresh=0:0.9:0.1 n_proc=2
__-max_length-4-__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/seqformer-ipsc-ext_reorg_roi_g2_16_53/inference_model_02419999_max_length-4/results.json  gt_json=ipsc-ext_reorg_roi_g2_0_15-max_length-4.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 nms_thresh=0:0.9:0.1 n_proc=2
__-max_length-8-__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/seqformer-ipsc-ext_reorg_roi_g2_16_53/inference_model_02419999_max_length-8/results.json  gt_json=ipsc-ext_reorg_roi_g2_0_15-max_length-8.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 nms_thresh=0:0.9:0.1 n_proc=2

<a id="probs___seq_ext_reorg_roi_g2_16_53_"></a>
### probs       @ seq/ext_reorg_roi_g2_16_53-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/seqformer-ipsc-ext_reorg_roi_g2_16_53/inference_model_0241999_probs/results.json  gt_json=ipsc-ext_reorg_roi_g2_0_15.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0

<a id="seq_inc___ext_reorg_roi_g2_16_53_"></a>
## seq-inc       @ ext_reorg_roi_g2_16_53-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/seqformer-ipsc-ext_reorg_roi_g2_16_53/inference_model_0241999_incremental/results.json  gt_json=ipsc-ext_reorg_roi_g2_0_15-incremental.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 incremental=1

<a id="nms___seq_inc_ext_reorg_roi_g2_16_53_"></a>
### nms       @ seq-inc/ext_reorg_roi_g2_16_53-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/seqformer-ipsc-ext_reorg_roi_g2_16_53/inference_model_0241999_incremental/results.json  gt_json=ipsc-ext_reorg_roi_g2_0_15-incremental.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=1 incremental=1 nms_thresh=0.1 n_proc=12

<a id="probs___seq_inc_ext_reorg_roi_g2_16_53_"></a>
### probs       @ seq-inc/ext_reorg_roi_g2_16_53-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/seqformer-ipsc-ext_reorg_roi_g2_16_53/inference_model_0241999_incremental_probs/results.json  gt_json=ipsc-ext_reorg_roi_g2_0_15-incremental.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 incremental=1

<a id="vita___ext_reorg_roi_g2_16_53_"></a>
## vita       @ ext_reorg_roi_g2_16_53-->coco_to_xml
<a id="0119999___vita_ext_reorg_roi_g2_16_5_3_"></a>
### 0119999       @ vita/ext_reorg_roi_g2_16_53-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vita/vita-ipsc-ext_reorg_roi_g2_16_53_swin/inference_model_0119999/results.json  gt_json=ipsc-ext_reorg_roi_g2_0_15.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0
<a id="0329999___vita_ext_reorg_roi_g2_16_5_3_"></a>
### 0329999       @ vita/ext_reorg_roi_g2_16_53-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vita/vita-ipsc-ext_reorg_roi_g2_16_53_swin/inference_model_0329999/results.json  gt_json=ipsc-ext_reorg_roi_g2_0_15.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 nms_thresh=0.1 n_proc=12
__-max_length-1-__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vita/vita-ipsc-ext_reorg_roi_g2_16_53_swin/inference_model_0329999_max_length-1/results.json  gt_json=ipsc-ext_reorg_roi_g2_0_15-max_length-1.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 nms_thresh=0:0.9:0.1 n_proc=2
__-max_length-2-__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vita/vita-ipsc-ext_reorg_roi_g2_16_53_swin/inference_model_0329999_max_length-2/results.json  gt_json=ipsc-ext_reorg_roi_g2_0_15-max_length-2.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 nms_thresh=0:0.9:0.1 n_proc=2
__-max_length-4-__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vita/vita-ipsc-ext_reorg_roi_g2_16_53_swin/inference_model_0329999_max_length-4/results.json  gt_json=ipsc-ext_reorg_roi_g2_0_15-max_length-4.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 nms_thresh=0:0.9:0.1 n_proc=2
__-max_length-8-__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vita/vita-ipsc-ext_reorg_roi_g2_16_53_swin/inference_model_0329999_max_length-8/results.json  gt_json=ipsc-ext_reorg_roi_g2_0_15-max_length-8.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 nms_thresh=0:0.9:0.1 n_proc=2


<a id="vita_inc___ext_reorg_roi_g2_16_53_"></a>
## vita-inc       @ ext_reorg_roi_g2_16_53-->coco_to_xml
<a id="0119999___vita_inc_ext_reorg_roi_g2_16_5_3_"></a>
### 0119999       @ vita-inc/ext_reorg_roi_g2_16_53-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vita/vita-ipsc-ext_reorg_roi_g2_16_53_swin/inference_model_0119999_incremental/results.json  gt_json=ipsc-ext_reorg_roi_g2_0_15-incremental.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 incremental=1
<a id="0329999___vita_inc_ext_reorg_roi_g2_16_5_3_"></a>
### 0329999       @ vita-inc/ext_reorg_roi_g2_16_53-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vita/vita-ipsc-ext_reorg_roi_g2_16_53_swin/inference_model_0329999_incremental/results.json  gt_json=ipsc-ext_reorg_roi_g2_0_15-incremental.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=1 incremental=1
<a id="nms___0329999_vita_inc_ext_reorg_roi_g2_16_5_3_"></a>
#### nms       @ 0329999/vita-inc/ext_reorg_roi_g2_16_53-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vita/vita-ipsc-ext_reorg_roi_g2_16_53_swin/inference_model_0329999_incremental/results.json  gt_json=ipsc-ext_reorg_roi_g2_0_15-incremental.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 incremental=1 nms_thresh=0.1 n_proc=12

<a id="vita_inc_retrain___ext_reorg_roi_g2_16_53_"></a>
## vita-inc-retrain       @ ext_reorg_roi_g2_16_53-->coco_to_xml
<a id="0004999___vita_inc_retrain_ext_reorg_roi_g2_16_5_3_"></a>
### 0004999       @ vita-inc-retrain/ext_reorg_roi_g2_16_53-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vita/vita-ipsc-ext_reorg_roi_g2_16_53_swin_retrain/inference_model_0004999_incremental/results.json  gt_json=ipsc-ext_reorg_roi_g2_0_15-incremental.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 incremental=1
<a id="0079999___vita_inc_retrain_ext_reorg_roi_g2_16_5_3_"></a>
### 0079999       @ vita-inc-retrain/ext_reorg_roi_g2_16_53-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vita/vita-ipsc-ext_reorg_roi_g2_16_53_swin_retrain/inference_model_0079999_incremental/results.json  gt_json=ipsc-ext_reorg_roi_g2_0_15-incremental.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 incremental=1
<a id="0104999___vita_inc_retrain_ext_reorg_roi_g2_16_5_3_"></a>
### 0104999       @ vita-inc-retrain/ext_reorg_roi_g2_16_53-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vita/vita-ipsc-ext_reorg_roi_g2_16_53_swin_retrain/inference_model_0104999_incremental/results.json  gt_json=ipsc-ext_reorg_roi_g2_0_15-incremental.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 incremental=1

<a id="ext_reorg_roi_g2_54_126___coco_to_xm_l_"></a>
# ext_reorg_roi_g2_54_126       @ coco_to_xml-->coco_to_xml

<a id="yl8___ext_reorg_roi_g2_54_12_6_"></a>
## yl8       @ ext_reorg_roi_g2_54_126-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi json=yl8/ext_reorg_roi_g2_54_126/last/predictions.json  gt_json=ext_reorg_roi_g2_0_53.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt nms_thresh=0.1:0.9:0.1 n_proc=12

<a id="swi___ext_reorg_roi_g2_54_12_6_"></a>
## swi       @ ext_reorg_roi_g2_54_126-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi json=swi/ipsc_2_class_ext_reorg_roi_g2_54_126-no_validate/g2_0_53/results.segm.json  gt_json=ext_reorg_roi_g2_0_53.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt nms_thresh=0.1:0.9:0.1 n_proc=12
<a id="g2_0_15___swi_ext_reorg_roi_g2_54_12_6_"></a>
### g2_0_15       @ swi/ext_reorg_roi_g2_54_126-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi json=swi/ipsc_2_class_ext_reorg_roi_g2_54_126-no_validate/g2_0_15/results.segm.json  gt_json=ext_reorg_roi_g2_0_15.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt nms_thresh=0.1:0.9:0.1 n_proc=2

<a id="cvnxt_large___ext_reorg_roi_g2_54_12_6_"></a>
## cvnxt-large       @ ext_reorg_roi_g2_54_126-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi json=swi/ipsc_2_class_ext_reorg_roi_g2_54_126-convnext_large_in22k/g2_0_53/results.segm.json  gt_json=ext_reorg_roi_g2_0_53.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt nms_thresh=0.1:0.9:0.1 n_proc=1
<a id="g2_0_15___cvnxt_large_ext_reorg_roi_g2_54_12_6_"></a>
### g2_0_15       @ cvnxt-large/ext_reorg_roi_g2_54_126-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi json=swi/ipsc_2_class_ext_reorg_roi_g2_54_126-convnext_large_in22k/g2_0_15/results.segm.json  gt_json=ext_reorg_roi_g2_0_15.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt nms_thresh=0.1:0.9:0.1 n_proc=2

<a id="cvnxt_base___ext_reorg_roi_g2_54_12_6_"></a>
## cvnxt-base       @ ext_reorg_roi_g2_54_126-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi json=swi/ipsc_2_class_ext_reorg_roi_g2_54_126-convnext_base_in22k/g2_0_53/results.segm.json  gt_json=ext_reorg_roi_g2_0_53.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt nms_thresh=0.1:0.9:0.1 n_proc=2
<a id="g2_0_15___cvnxt_base_ext_reorg_roi_g2_54_126_"></a>
### g2_0_15       @ cvnxt-base/ext_reorg_roi_g2_54_126-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi json=swi/ipsc_2_class_ext_reorg_roi_g2_54_126-convnext_base_in22k/g2_0_15/results.segm.json  gt_json=ext_reorg_roi_g2_0_15.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt nms_thresh=0.1:0.9:0.1 n_proc=2

<a id="idol___ext_reorg_roi_g2_54_12_6_"></a>
## idol       @ ext_reorg_roi_g2_54_126-->coco_to_xml
<a id="g2_0_53___idol_ext_reorg_roi_g2_54_126_"></a>
### g2_0_53       @ idol/ext_reorg_roi_g2_54_126-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/idol-ipsc-ext_reorg_roi_g2_54_126/inference_model_0596999/json_results  gt_json=ipsc-ext_reorg_roi_g2_0_53.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 fix_category_id=1 save=0 nms_thresh=0:0.9:0.1 n_proc=12
__-max_length-1-__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/idol-ipsc-ext_reorg_roi_g2_54_126/inference_model_0596999_max_length-1/json_results  gt_json=ipsc-ext_reorg_roi_g2_0_53-max_length-1.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 fix_category_id=1 save=0 nms_thresh=0:0.9:0.1 n_proc=1
__-max_length-2-__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/idol-ipsc-ext_reorg_roi_g2_54_126/inference_model_0596999_max_length-2/json_results  gt_json=ipsc-ext_reorg_roi_g2_0_53-max_length-2.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 fix_category_id=1 save=0 nms_thresh=0:0.9:0.1 n_proc=1
__-max_length-4-__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/idol-ipsc-ext_reorg_roi_g2_54_126/inference_model_0596999_max_length-4/json_results  gt_json=ipsc-ext_reorg_roi_g2_0_53-max_length-4.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 fix_category_id=1 save=0 nms_thresh=0:0.9:0.1 n_proc=1
__-max_length-8-__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/idol-ipsc-ext_reorg_roi_g2_54_126/inference_model_0596999_max_length-8/json_results  gt_json=ipsc-ext_reorg_roi_g2_0_53-max_length-8.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 fix_category_id=1 save=0 nms_thresh=0:0.9:0.1 n_proc=1
__-max_length-19-__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/idol-ipsc-ext_reorg_roi_g2_54_126/inference_model_0596999_max_length-19/json_results  gt_json=ipsc-ext_reorg_roi_g2_0_53-max_length-19.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 fix_category_id=1 save=0 nms_thresh=0:0.9:0.1 n_proc=12

<a id="g2_0_15___idol_ext_reorg_roi_g2_54_126_"></a>
### g2_0_15       @ idol/ext_reorg_roi_g2_54_126-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/idol-ipsc-ext_reorg_roi_g2_54_126/inference_model_0596999_g2_0_15/json_results  gt_json=ipsc-ext_reorg_roi_g2_0_15.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 fix_category_id=1 save=0 nms_thresh=0:0.9:0.1 n_proc=1
__-max_length-1__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/idol-ipsc-ext_reorg_roi_g2_54_126/inference_model_0596999_g2_0_15-max_length-1/json_results  gt_json=ipsc-ext_reorg_roi_g2_0_15-max_length-1.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 fix_category_id=1 save=0 nms_thresh=0:0.9:0.1 n_proc=1
__-max_length-2-__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/idol-ipsc-ext_reorg_roi_g2_54_126/inference_model_0596999_g2_0_15-max_length-2/json_results  gt_json=ipsc-ext_reorg_roi_g2_0_15-max_length-2.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 fix_category_id=1 save=0 nms_thresh=0:0.9:0.1 n_proc=1
__-max_length-4-__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/idol-ipsc-ext_reorg_roi_g2_54_126/inference_model_0596999_g2_0_15-max_length-4/json_results  gt_json=ipsc-ext_reorg_roi_g2_0_15-max_length-4.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 fix_category_id=1 save=0 nms_thresh=0:0.9:0.1 n_proc=1
__-max_length-8-__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/idol-ipsc-ext_reorg_roi_g2_54_126/inference_model_0596999_g2_0_15-max_length-8/json_results  gt_json=ipsc-ext_reorg_roi_g2_0_15-max_length-8.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 fix_category_id=1 save=0 nms_thresh=0:0.9:0.1 n_proc=1


<a id="idol_inc___ext_reorg_roi_g2_54_12_6_"></a>
## idol-inc       @ ext_reorg_roi_g2_54_126-->coco_to_xml
<a id="g2_0_53___idol_inc_ext_reorg_roi_g2_54_126_"></a>
### g2_0_53       @ idol-inc/ext_reorg_roi_g2_54_126-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/idol-ipsc-ext_reorg_roi_g2_54_126/inference_model_0596999_incremental_probs/json_results  gt_json=ipsc-ext_reorg_roi_g2_0_53-incremental.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 fix_category_id=1 save=0 incremental=1 nms_thresh=0:0.9:0.1 n_proc=12
__-max_length-2-__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/idol-ipsc-ext_reorg_roi_g2_54_126/inference_model_0596999_max_length-2-incremental_probs/json_results  gt_json=ipsc-ext_reorg_roi_g2_0_53-max_length-2-incremental.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 fix_category_id=1 save=0 incremental=1 nms_thresh=0:0.9:0.1 n_proc=12

<a id="g2_0_15___idol_inc_ext_reorg_roi_g2_54_126_"></a>
### g2_0_15       @ idol-inc/ext_reorg_roi_g2_54_126-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/idol-ipsc-ext_reorg_roi_g2_54_126/inference_model_0596999_g2_0_15-incremental_probs/json_results  gt_json=ipsc-ext_reorg_roi_g2_0_15-incremental.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 fix_category_id=1 save=1 incremental=1 nms_thresh=0.1 n_proc=2
__-max_length-2-__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/idol-ipsc-ext_reorg_roi_g2_54_126/inference_model_0596999_g2_0_15-max_length-2-incremental_probs/json_results  gt_json=ipsc-ext_reorg_roi_g2_0_15-max_length-2-incremental.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 fix_category_id=1 save=0 incremental=1 nms_thresh=0:0.9:0.1 n_proc=12

<a id="seq___ext_reorg_roi_g2_54_12_6_"></a>
## seq       @ ext_reorg_roi_g2_54_126-->coco_to_xml
<a id="g2_0_53___seq_ext_reorg_roi_g2_54_12_6_"></a>
### g2_0_53       @ seq/ext_reorg_roi_g2_54_126-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/seqformer-ipsc-ext_reorg_roi_g2_54_126/inference_model_0495999/results.json  gt_json=ipsc-ext_reorg_roi_g2_0_53.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 nms_thresh=0:0.9:0.1 n_proc=1
__-max_length-1-__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/seqformer-ipsc-ext_reorg_roi_g2_54_126/inference_model_0495999_max_length-1/results.json gt_json=ipsc-ext_reorg_roi_g2_0_53-max_length-1.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 nms_thresh=0:0.9:0.1 n_proc=1
__-max_length-2-__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/seqformer-ipsc-ext_reorg_roi_g2_54_126/inference_model_0495999_max_length-2/results.json gt_json=ipsc-ext_reorg_roi_g2_0_53-max_length-2.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 nms_thresh=0:0.9:0.1 n_proc=1
__-max_length-4-__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/seqformer-ipsc-ext_reorg_roi_g2_54_126/inference_model_0495999_max_length-4/results.json gt_json=ipsc-ext_reorg_roi_g2_0_53-max_length-4.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 nms_thresh=0:0.9:0.1 n_proc=1
__-max_length-8-__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/seqformer-ipsc-ext_reorg_roi_g2_54_126/inference_model_0495999_max_length-8/results.json gt_json=ipsc-ext_reorg_roi_g2_0_53-max_length-8.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 nms_thresh=0:0.9:0.1 n_proc=1
__-max_length-19-__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/seqformer-ipsc-ext_reorg_roi_g2_54_126/inference_model_0495999_max_length-19/results.json gt_json=ipsc-ext_reorg_roi_g2_0_53-max_length-19.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 nms_thresh=0:0.9:0.1 n_proc=1

<a id="g2_0_15___seq_ext_reorg_roi_g2_54_12_6_"></a>
### g2_0_15       @ seq/ext_reorg_roi_g2_54_126-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/seqformer-ipsc-ext_reorg_roi_g2_54_126/inference_model_0495999_g2_0_15/results.json  gt_json=ipsc-ext_reorg_roi_g2_0_15.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 nms_thresh=0:0.9:0.1 n_proc=1
__-max_length-1-__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/seqformer-ipsc-ext_reorg_roi_g2_54_126/inference_model_0495999_g2_0_15-max_length-1/results.json  gt_json=ipsc-ext_reorg_roi_g2_0_15-max_length-1.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 nms_thresh=0:0.9:0.1 n_proc=1
__-max_length-2-__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/seqformer-ipsc-ext_reorg_roi_g2_54_126/inference_model_0495999_g2_0_15-max_length-2/results.json  gt_json=ipsc-ext_reorg_roi_g2_0_15-max_length-2.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 nms_thresh=0:0.9:0.1 n_proc=1
__-max_length-4-__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/seqformer-ipsc-ext_reorg_roi_g2_54_126/inference_model_0495999_g2_0_15-max_length-4/results.json  gt_json=ipsc-ext_reorg_roi_g2_0_15-max_length-4.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 nms_thresh=0:0.9:0.1 n_proc=1
__-max_length-8-__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/seqformer-ipsc-ext_reorg_roi_g2_54_126/inference_model_0495999_g2_0_15-max_length-8/results.json  gt_json=ipsc-ext_reorg_roi_g2_0_15-max_length-8.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 nms_thresh=0:0.9:0.1 n_proc=1

<a id="seq_inc___ext_reorg_roi_g2_54_12_6_"></a>
## seq-inc       @ ext_reorg_roi_g2_54_126-->coco_to_xml
<a id="g2_0_53___seq_inc_ext_reorg_roi_g2_54_12_6_"></a>
### g2_0_53       @ seq-inc/ext_reorg_roi_g2_54_126-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/seqformer-ipsc-ext_reorg_roi_g2_54_126/inference_model_0495999-incremental_probs/results.json  gt_json=ipsc-ext_reorg_roi_g2_0_53-incremental.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 incremental=1 nms_thresh=0:0.9:0.1 n_proc=2
__-max_length-2-__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/seqformer-ipsc-ext_reorg_roi_g2_54_126/inference_model_0495999_max_length-2-incremental_probs/results.json  gt_json=ipsc-ext_reorg_roi_g2_0_53-max_length-2-incremental.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 incremental=1 nms_thresh=0:0.9:0.1 n_proc=12

__-max_length-10-__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/seqformer-ipsc-ext_reorg_roi_g2_54_126/inference_model_0495999_max_length-10-incremental_probs/results.json  gt_json=ipsc-ext_reorg_roi_g2_0_53-max_length-10-incremental.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 incremental=1 nms_thresh=0:0.9:0.1 n_proc=12

<a id="g2_0_15___seq_inc_ext_reorg_roi_g2_54_12_6_"></a>
### g2_0_15       @ seq-inc/ext_reorg_roi_g2_54_126-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/seqformer-ipsc-ext_reorg_roi_g2_54_126/inference_model_0495999_g2_0_15-incremental_probs/results.json  gt_json=ipsc-ext_reorg_roi_g2_0_15-incremental.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 incremental=1 nms_thresh=0:0.9:0.1 n_proc=2
__-max_length-2-__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vnxt/seqformer-ipsc-ext_reorg_roi_g2_54_126/inference_model_0495999_g2_0_15-max_length-2-incremental_probs/results.json  gt_json=ipsc-ext_reorg_roi_g2_0_15-max_length-2-incremental.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 incremental=1 nms_thresh=0:0.9:0.1 n_proc=12

<a id="vita___ext_reorg_roi_g2_54_12_6_"></a>
## vita       @ ext_reorg_roi_g2_54_126-->coco_to_xml
<a id="g2_0_53___vita_ext_reorg_roi_g2_54_126_"></a>
### g2_0_53       @ vita/ext_reorg_roi_g2_54_126-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vita/vita-ipsc-ext_reorg_roi_g2_54_126_swin/inference_model_0194999/results.json  gt_json=ipsc-ext_reorg_roi_g2_0_53.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 nms_thresh=0:0.9:0.1 n_proc=12
__-max_length-1__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vita/vita-ipsc-ext_reorg_roi_g2_54_126_swin/inference_model_0194999_max_length-1/results.json  gt_json=ipsc-ext_reorg_roi_g2_0_53-max_length-1.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 nms_thresh=0:0.9:0.1 n_proc=1
__-max_length-2-__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vita/vita-ipsc-ext_reorg_roi_g2_54_126_swin/inference_model_0194999_max_length-2/results.json  gt_json=ipsc-ext_reorg_roi_g2_0_53-max_length-2.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 nms_thresh=0:0.9:0.1 n_proc=1
__-max_length-4-__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vita/vita-ipsc-ext_reorg_roi_g2_54_126_swin/inference_model_0194999_max_length-4/results.json  gt_json=ipsc-ext_reorg_roi_g2_0_53-max_length-4.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 nms_thresh=0:0.9:0.1 n_proc=1
__-max_length-8-__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vita/vita-ipsc-ext_reorg_roi_g2_54_126_swin/inference_model_0194999_max_length-8/results.json  gt_json=ipsc-ext_reorg_roi_g2_0_53-max_length-8.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 nms_thresh=0:0.9:0.1 n_proc=1
__-max_length-19-__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vita/vita-ipsc-ext_reorg_roi_g2_54_126_swin/inference_model_0194999_max_length-19/results.json  gt_json=ipsc-ext_reorg_roi_g2_0_53-max_length-19.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 nms_thresh=0:0.9:0.1 n_proc=12

<a id="g2_0_15___vita_ext_reorg_roi_g2_54_126_"></a>
### g2_0_15       @ vita/ext_reorg_roi_g2_54_126-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vita/vita-ipsc-ext_reorg_roi_g2_54_126_swin/inference_model_0194999_g2_0_15/results.json  gt_json=ipsc-ext_reorg_roi_g2_0_15.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 nms_thresh=0:0.9:0.1 n_proc=12
__-max_length-1__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vita/vita-ipsc-ext_reorg_roi_g2_54_126_swin/inference_model_0194999_g2_0_15-max_length-1/results.json  gt_json=ipsc-ext_reorg_roi_g2_0_15-max_length-1.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 nms_thresh=0:0.9:0.1 n_proc=1
__-max_length-2-__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vita/vita-ipsc-ext_reorg_roi_g2_54_126_swin/inference_model_0194999_g2_0_15-max_length-2/results.json  gt_json=ipsc-ext_reorg_roi_g2_0_15-max_length-2.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 nms_thresh=0:0.9:0.1 n_proc=1
__-max_length-4-__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vita/vita-ipsc-ext_reorg_roi_g2_54_126_swin/inference_model_0194999_g2_0_15-max_length-4/results.json  gt_json=ipsc-ext_reorg_roi_g2_0_15-max_length-4.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 nms_thresh=0:0.9:0.1 n_proc=1
__-max_length-8-__
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vita/vita-ipsc-ext_reorg_roi_g2_54_126_swin/inference_model_0194999_g2_0_15-max_length-8/results.json  gt_json=ipsc-ext_reorg_roi_g2_0_15-max_length-8.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 nms_thresh=0:0.9:0.1 n_proc=1

<a id="vita_inc___ext_reorg_roi_g2_54_12_6_"></a>
## vita-inc       @ ext_reorg_roi_g2_54_126-->coco_to_xml
<a id="g2_0_53___vita_inc_ext_reorg_roi_g2_54_126_"></a>
### g2_0_53       @ vita-inc/ext_reorg_roi_g2_54_126-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vita/vita-ipsc-ext_reorg_roi_g2_54_126_swin/inference_model_0194999_incremental/results.json  gt_json=ipsc-ext_reorg_roi_g2_0_53-incremental.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 incremental=1 nms_thresh=0:0.9:0.1 n_proc=12
<a id="max_length_2___g2_0_53_vita_inc_ext_reorg_roi_g2_54_126_"></a>
#### max_length-2       @ g2_0_53/vita-inc/ext_reorg_roi_g2_54_126-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vita/vita-ipsc-ext_reorg_roi_g2_54_126_swin/inference_model_0194999_max_length-2-incremental/results.json  gt_json=ipsc-ext_reorg_roi_g2_0_53-max_length-2-incremental.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 incremental=1 nms_thresh=0:0.9:0.1 n_proc=12

<a id="g2_0_15___vita_inc_ext_reorg_roi_g2_54_126_"></a>
### g2_0_15       @ vita-inc/ext_reorg_roi_g2_54_126-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vita/vita-ipsc-ext_reorg_roi_g2_54_126_swin/inference_model_0194999_g2_0_15-incremental/results.json  gt_json=ipsc-ext_reorg_roi_g2_0_15-incremental.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 incremental=1 nms_thresh=0.2 n_proc=12
<a id="max_length_2___g2_0_15_vita_inc_ext_reorg_roi_g2_54_126_"></a>
#### max_length-2       @ g2_0_15/vita-inc/ext_reorg_roi_g2_54_126-->coco_to_xml
python3 coco_to_xml.py root_dir=/data/ipsc/well3/all_frames_roi/ytvis19 json=vita/vita-ipsc-ext_reorg_roi_g2_54_126_swin/inference_model_0194999_g2_0_15-max_length-2-incremental/results.json  gt_json=ipsc-ext_reorg_roi_g2_0_15-max_length-2-incremental.json class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt ytvis=1 save=0 incremental=1 nms_thresh=0:0.9:0.1 n_proc=12

