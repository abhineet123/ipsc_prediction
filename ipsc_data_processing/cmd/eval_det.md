<!-- MarkdownTOC -->

- [all_frames_roi](#all_frames_ro_i_)
    - [swi       @ all_frames_roi](#swi___all_frames_roi_)
- [ext_reorg_roi_0_37](#ext_reorg_roi_0_3_7_)
    - [swi       @ ext_reorg_roi_0_37](#swi___ext_reorg_roi_0_37_)
    - [no_validate-rcnn       @ ext_reorg_roi_0_37](#no_validate_rcnn___ext_reorg_roi_0_37_)
    - [no_validate-rcnn-win7       @ ext_reorg_roi_0_37](#no_validate_rcnn_win7___ext_reorg_roi_0_37_)
    - [rcnn-win7       @ ext_reorg_roi_0_37](#rcnn_win7___ext_reorg_roi_0_37_)
    - [cvnxt-large       @ ext_reorg_roi_0_37](#cvnxt_large___ext_reorg_roi_0_37_)
    - [cvnxt-large-coco       @ ext_reorg_roi_0_37](#cvnxt_large_coco___ext_reorg_roi_0_37_)
    - [cvnxt-base       @ ext_reorg_roi_0_37](#cvnxt_base___ext_reorg_roi_0_37_)
    - [idol       @ ext_reorg_roi_0_37](#idol___ext_reorg_roi_0_37_)
        - [idol-inc-sigma       @ idol/ext_reorg_roi_0_37](#idol_inc_sigma___idol_ext_reorg_roi_0_3_7_)
        - [idol-inc-probs       @ idol/ext_reorg_roi_0_37](#idol_inc_probs___idol_ext_reorg_roi_0_3_7_)
        - [idol-inc-all       @ idol/ext_reorg_roi_0_37](#idol_inc_all___idol_ext_reorg_roi_0_3_7_)
    - [seq       @ ext_reorg_roi_0_37](#seq___ext_reorg_roi_0_37_)
        - [seq-inc-sigma       @ seq/ext_reorg_roi_0_37](#seq_inc_sigma___seq_ext_reorg_roi_0_37_)
        - [seq-inc-probs       @ seq/ext_reorg_roi_0_37](#seq_inc_probs___seq_ext_reorg_roi_0_37_)
    - [vita-swin-inc       @ ext_reorg_roi_0_37](#vita_swin_inc___ext_reorg_roi_0_37_)
    - [vita-r50       @ ext_reorg_roi_0_37](#vita_r50___ext_reorg_roi_0_37_)
- [ext_reorg_roi_16_53](#ext_reorg_roi_16_53_)
    - [yl8       @ ext_reorg_roi_16_53](#yl8___ext_reorg_roi_16_5_3_)
        - [val       @ yl8/ext_reorg_roi_16_53](#val___yl8_ext_reorg_roi_16_5_3_)
        - [seq-val       @ yl8/ext_reorg_roi_16_53](#seq_val___yl8_ext_reorg_roi_16_5_3_)
    - [swi       @ ext_reorg_roi_16_53](#swi___ext_reorg_roi_16_5_3_)
    - [swi-rcnn       @ ext_reorg_roi_16_53](#swi_rcnn___ext_reorg_roi_16_5_3_)
    - [cvnxt-large       @ ext_reorg_roi_16_53](#cvnxt_large___ext_reorg_roi_16_5_3_)
    - [idol       @ ext_reorg_roi_16_53](#idol___ext_reorg_roi_16_5_3_)
        - [probs       @ idol/ext_reorg_roi_16_53](#probs___idol_ext_reorg_roi_16_53_)
    - [idol-inc       @ ext_reorg_roi_16_53](#idol_inc___ext_reorg_roi_16_5_3_)
        - [probs       @ idol-inc/ext_reorg_roi_16_53](#probs___idol_inc_ext_reorg_roi_16_53_)
            - [nms-01       @ probs/idol-inc/ext_reorg_roi_16_53](#nms_01___probs_idol_inc_ext_reorg_roi_16_53_)
    - [seq       @ ext_reorg_roi_16_53](#seq___ext_reorg_roi_16_5_3_)
        - [probs       @ seq/ext_reorg_roi_16_53](#probs___seq_ext_reorg_roi_16_5_3_)
    - [seq-inc       @ ext_reorg_roi_16_53](#seq_inc___ext_reorg_roi_16_5_3_)
        - [probs       @ seq-inc/ext_reorg_roi_16_53](#probs___seq_inc_ext_reorg_roi_16_5_3_)
    - [vita       @ ext_reorg_roi_16_53](#vita___ext_reorg_roi_16_5_3_)
        - [0119999       @ vita/ext_reorg_roi_16_53](#0119999___vita_ext_reorg_roi_16_53_)
        - [0329999       @ vita/ext_reorg_roi_16_53](#0329999___vita_ext_reorg_roi_16_53_)
    - [vita-inc       @ ext_reorg_roi_16_53](#vita_inc___ext_reorg_roi_16_5_3_)
        - [0119999       @ vita-inc/ext_reorg_roi_16_53](#0119999___vita_inc_ext_reorg_roi_16_53_)
        - [0329999       @ vita-inc/ext_reorg_roi_16_53](#0329999___vita_inc_ext_reorg_roi_16_53_)
    - [vita-retrain-inc       @ ext_reorg_roi_16_53](#vita_retrain_inc___ext_reorg_roi_16_5_3_)
        - [0004999       @ vita-retrain-inc/ext_reorg_roi_16_53](#0004999___vita_retrain_inc_ext_reorg_roi_16_53_)
        - [0079999       @ vita-retrain-inc/ext_reorg_roi_16_53](#0079999___vita_retrain_inc_ext_reorg_roi_16_53_)
        - [0104999       @ vita-retrain-inc/ext_reorg_roi_16_53](#0104999___vita_retrain_inc_ext_reorg_roi_16_53_)
- [ext_reorg_roi_54_126](#ext_reorg_roi_54_12_6_)
    - [yl8       @ ext_reorg_roi_54_126](#yl8___ext_reorg_roi_54_126_)
        - [val       @ yl8/ext_reorg_roi_54_126](#val___yl8_ext_reorg_roi_54_126_)
        - [seq-val       @ yl8/ext_reorg_roi_54_126](#seq_val___yl8_ext_reorg_roi_54_126_)
    - [swi       @ ext_reorg_roi_54_126](#swi___ext_reorg_roi_54_126_)
        - [g2_0_15       @ swi/ext_reorg_roi_54_126](#g2_0_15___swi_ext_reorg_roi_54_126_)
    - [cvnxt-base       @ ext_reorg_roi_54_126](#cvnxt_base___ext_reorg_roi_54_126_)
        - [g2_0_15       @ cvnxt-base/ext_reorg_roi_54_126](#g2_0_15___cvnxt_base_ext_reorg_roi_54_12_6_)
    - [cvnxt-large       @ ext_reorg_roi_54_126](#cvnxt_large___ext_reorg_roi_54_126_)
        - [g2_0_15       @ cvnxt-large/ext_reorg_roi_54_126](#g2_0_15___cvnxt_large_ext_reorg_roi_54_126_)
    - [idol       @ ext_reorg_roi_54_126](#idol___ext_reorg_roi_54_126_)
        - [g2_0_53       @ idol/ext_reorg_roi_54_126](#g2_0_53___idol_ext_reorg_roi_54_12_6_)
        - [g2_0_15       @ idol/ext_reorg_roi_54_126](#g2_0_15___idol_ext_reorg_roi_54_12_6_)
    - [idol-inc       @ ext_reorg_roi_54_126](#idol_inc___ext_reorg_roi_54_126_)
        - [g2_0_15       @ idol-inc/ext_reorg_roi_54_126](#g2_0_15___idol_inc_ext_reorg_roi_54_12_6_)
    - [seq       @ ext_reorg_roi_54_126](#seq___ext_reorg_roi_54_126_)
        - [g2_0_53       @ seq/ext_reorg_roi_54_126](#g2_0_53___seq_ext_reorg_roi_54_126_)
        - [g2_0_15       @ seq/ext_reorg_roi_54_126](#g2_0_15___seq_ext_reorg_roi_54_126_)
    - [seq-inc       @ ext_reorg_roi_54_126](#seq_inc___ext_reorg_roi_54_126_)
        - [g2_0_15       @ seq-inc/ext_reorg_roi_54_126](#g2_0_15___seq_inc_ext_reorg_roi_54_126_)
    - [vita       @ ext_reorg_roi_54_126](#vita___ext_reorg_roi_54_126_)
        - [g2_0_53       @ vita/ext_reorg_roi_54_126](#g2_0_53___vita_ext_reorg_roi_54_12_6_)
        - [g2_0_15       @ vita/ext_reorg_roi_54_126](#g2_0_15___vita_ext_reorg_roi_54_12_6_)
    - [vita-inc       @ ext_reorg_roi_54_126](#vita_inc___ext_reorg_roi_54_126_)
        - [g2_0_53       @ vita-inc/ext_reorg_roi_54_126](#g2_0_53___vita_inc_ext_reorg_roi_54_12_6_)
        - [g2_0_15       @ vita-inc/ext_reorg_roi_54_126](#g2_0_15___vita_inc_ext_reorg_roi_54_12_6_)

<!-- /MarkdownTOC -->

<a id="all_frames_ro_i_"></a>
# all_frames_roi

<a id="swi___all_frames_roi_"></a>
## swi       @ all_frames_roi-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/all_frames_roi.txt det_paths=log/swi/ipsc_2_class_all_frames_roi_g2_0_37/g2_38_53/csv gt_csv_name=annotations_38_53.csv

<a id="ext_reorg_roi_0_3_7_"></a>
# ext_reorg_roi_0_37 
<a id="swi___ext_reorg_roi_0_37_"></a>
## swi       @ ext_reorg_roi_0_37-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/swi/ipsc_2_class_ext_reorg_roi_g2_0_37-no_validate/g2_38_53/csv gt_csv_name=annotations_38_53.csv save_suffix=swi gt_pkl=g2_38_53.pkl nms_thresh=0 n_proc=12
__iw__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/swi/ipsc_2_class_ext_reorg_roi_g2_0_37-no_validate/g2_38_53/csv gt_csv_name=annotations_38_53.csv  save_suffix=swi gt_pkl=g2_38_53.pkl n_proc=12 iw=1

python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/swi/ipsc_2_class_ext_reorg_roi_g2_0_37-no_validate/g2_38_53/csv gt_csv_name=annotations_38_53.csv save_suffix=swi gt_pkl=g2_38_53.pkl n_proc=12 iw=1 nms_thresh=0.1

<a id="no_validate_rcnn___ext_reorg_roi_0_37_"></a>
## no_validate-rcnn       @ ext_reorg_roi_0_37-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/swi/ipsc_2_class_ext_reorg_roi_g2_0_37-no_validate-rcnn/g2_38_53/csv gt_csv_name=annotations_38_53.csv save_suffix=swi-rcnn-no_validate gt_pkl=g2_38_53.pkl nms_thresh=0:0.9:0.1 n_proc=12 enable_mask=0

<a id="no_validate_rcnn_win7___ext_reorg_roi_0_37_"></a>
## no_validate-rcnn-win7       @ ext_reorg_roi_0_37-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/swi/ipsc_2_class_ext_reorg_roi_g2_0_37-no_validate-rcnn-win7/g2_38_53/csv gt_csv_name=annotations_38_53.csv save_suffix=swi-rcnn-no_validate-win7 gt_pkl=g2_38_53.pkl nms_thresh=0:0.9:0.1 n_proc=12 enable_mask=0

<a id="rcnn_win7___ext_reorg_roi_0_37_"></a>
## rcnn-win7       @ ext_reorg_roi_0_37-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/swi/ipsc_2_class_ext_reorg_roi_g2_0_37-rcnn-win7/g2_38_53/csv gt_csv_name=annotations_38_53.csv save_suffix=swi-rcnn-win7 gt_pkl=g2_38_53.pkl nms_thresh=0:0.9:0.1 n_proc=12 enable_mask=0

<a id="cvnxt_large___ext_reorg_roi_0_37_"></a>
## cvnxt-large       @ ext_reorg_roi_0_37-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/swi/ipsc_2_class_ext_reorg_roi_g2_0_37-convnext_large_in22k/g2_38_53/csv gt_csv_name=annotations_38_53.csv save_suffix=cvnxt-large gt_pkl=g2_38_53.pkl nms_thresh=0:0.9:0.1 
__iw__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/swi/ipsc_2_class_ext_reorg_roi_g2_0_37-convnext_large_in22k/g2_38_53/csv gt_csv_name=annotations_38_53.csv save_suffix=cvnxt-large gt_pkl=g2_38_53.pkl iw=1

python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/swi/ipsc_2_class_ext_reorg_roi_g2_0_37-convnext_large_in22k/g2_38_53/csv gt_csv_name=annotations_38_53.csv save_suffix=cvnxt-large gt_pkl=g2_38_53.pkl iw=1 nms_thresh=0.9

<a id="cvnxt_large_coco___ext_reorg_roi_0_37_"></a>
## cvnxt-large-coco       @ ext_reorg_roi_0_37-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/swi/ipsc_2_class_ext_reorg_roi_g2_0_37-convnext_large_in22k_coco_pretrained/g2_38_53/csv gt_csv_name=annotations_38_53.csv save_suffix=cvnxt-large-coco gt_pkl=g2_38_53.pkl

<a id="cvnxt_base___ext_reorg_roi_0_37_"></a>
## cvnxt-base       @ ext_reorg_roi_0_37-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/swi/ipsc_2_class_ext_reorg_roi_g2_0_37-no_validate-convnext_base_in22k/g2_38_53/csv gt_csv_name=annotations_38_53.csv save_suffix=cvnxt-base gt_pkl=g2_38_53.pkl nms_thresh=0:0.9:0.1 n_proc=12

<a id="idol___ext_reorg_roi_0_37_"></a>
## idol       @ ext_reorg_roi_0_37-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/idol-ipsc-ext_reorg_roi_g2_0_37/inference_38_53-incremental/csv gt_csv_name=annotations_38_53.csv save_suffix=idol gt_pkl=g2_38_53.pkl

<a id="idol_inc_sigma___idol_ext_reorg_roi_0_3_7_"></a>
### idol-inc-sigma       @ idol/ext_reorg_roi_0_37-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/idol-ipsc-ext_reorg_roi_g2_0_37/inference_38_53-incremental/csv_incremental gt_csv_name=annotations_38_53.csv save_suffix=idol-inc iw=0 gt_pkl=g2_38_53.pkl nms_thresh=0:0.9:0.1 n_proc=12

<a id="idol_inc_probs___idol_ext_reorg_roi_0_3_7_"></a>
### idol-inc-probs       @ idol/ext_reorg_roi_0_37-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/idol-ipsc-ext_reorg_roi_g2_0_37/inference_probs/csv_incremental gt_csv_name=annotations_38_53.csv save_suffix=idol-inc-probs iw=0 gt_pkl=g2_38_53.pkl nms_thresh=0:0.9:0.1
__iw__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/idol-ipsc-ext_reorg_roi_g2_0_37/inference_probs/csv_incremental gt_csv_name=annotations_38_53.csv save_suffix=idol-inc-probs gt_pkl=g2_38_53.pkl iw=1

<a id="idol_inc_all___idol_ext_reorg_roi_0_3_7_"></a>
### idol-inc-all       @ idol/ext_reorg_roi_0_37-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/idol-ipsc-ext_reorg_roi_g2_0_37/inference_38_53-incremental/csv gt_csv_name=annotations_38_53.csv save_suffix=idol-inc-all iw=0 end_id=0 gt_pkl=g2_38_53.pkl

<a id="seq___ext_reorg_roi_0_37_"></a>
## seq       @ ext_reorg_roi_0_37-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/seqformer-ipsc-ext_reorg_roi_g2_0_37/inference/csv gt_csv_name=annotations_38_53.csv save_suffix=seq gt_pkl=g2_38_53.pkl

<a id="seq_inc_sigma___seq_ext_reorg_roi_0_37_"></a>
### seq-inc-sigma       @ seq/ext_reorg_roi_0_37-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/seqformer-ipsc-ext_reorg_roi_g2_0_37/inference_38_53-incremental/csv_incremental gt_csv_name=annotations_38_53.csv save_suffix=seq-inc gt_pkl=g2_38_53.pkl nms_thresh=0:0.9:0.1 n_proc=12
__iw__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/seqformer-ipsc-ext_reorg_roi_g2_0_37/inference_38_53-incremental/csv_incremental gt_csv_name=annotations_38_53.csv save_suffix=seq-inc n_proc=12 iw=1

<a id="seq_inc_probs___seq_ext_reorg_roi_0_37_"></a>
### seq-inc-probs       @ seq/ext_reorg_roi_0_37-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/seqformer-ipsc-ext_reorg_roi_g2_0_37/inference_probs/csv_incremental gt_csv_name=annotations_38_53.csv save_suffix=seq-inc-probs gt_pkl=g2_38_53.pkl

<a id="vita_swin_inc___ext_reorg_roi_0_37_"></a>
## vita-swin-inc       @ ext_reorg_roi_0_37-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vita/vita-ipsc-ext_reorg_roi_g2_0_37_swin/inference_38_53-incremental/csv_incremental gt_csv_name=annotations_38_53.csv save_suffix=vita-swin-inc gt_pkl=g2_38_53.pkl nms_thresh=0:0.9:0.1 n_proc=12
__iw__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vita/vita-ipsc-ext_reorg_roi_g2_0_37_swin/inference_38_53-incremental/csv_incremental gt_csv_name=annotations_38_53.csv save_suffix=vita-swin-inc gt_pkl=g2_38_53.pkl n_proc=12 iw=1

<a id="vita_r50___ext_reorg_roi_0_37_"></a>
## vita-r50       @ ext_reorg_roi_0_37-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vita/vita-ipsc-ext_reorg_roi_g2_0_37_r50/inference/csv gt_csv_name=annotations_38_53.csv save_suffix=vita-r50 gt_pkl=g2_38_53.pkl

<a id="ext_reorg_roi_16_53_"></a>
# ext_reorg_roi_16_53
<a id="yl8___ext_reorg_roi_16_5_3_"></a>
## yl8       @ ext_reorg_roi_16_53-->eval_det
__last__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/yl8/ext_reorg_roi_g2_16_53/last/csv gt_csv_name=annotations_0_15.csv save_suffix=inv-yl8-last gt_pkl=g2_0_15.pkl iw=0 nms_thresh=0:0.9:0.1 n_proc=12
__best__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/yl8/ext_reorg_roi_g2_16_53/best/csv gt_csv_name=annotations_0_15.csv save_suffix=inv-yl8-best gt_pkl=g2_0_15.pkl iw=0 nms_thresh=0:0.9:0.1 n_proc=12
<a id="val___yl8_ext_reorg_roi_16_5_3_"></a>
### val       @ yl8/ext_reorg_roi_16_53-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/yl8/ext_reorg_roi_g2_16_53-val/last/csv gt_csv_name=annotations_0_15.csv save_suffix=inv-yl8-val-last gt_pkl=g2_0_15.pkl iw=0 nms_thresh=0:0.9:0.1 n_proc=12

python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/yl8/ext_reorg_roi_g2_16_53-val/best/csv gt_csv_name=annotations_0_15.csv save_suffix=inv-yl8-val-best gt_pkl=g2_0_15.pkl iw=0 nms_thresh=0:0.9:0.1 n_proc=12
<a id="seq_val___yl8_ext_reorg_roi_16_5_3_"></a>
### seq-val       @ yl8/ext_reorg_roi_16_53-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/yl8/ext_reorg_roi_g2_16_53-seq-val/last/csv gt_csv_name=annotations_0_15.csv save_suffix=inv-yl8-seq-val-last gt_pkl=g2_0_15.pkl iw=0 nms_thresh=0.1 n_proc=1

python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/yl8/ext_reorg_roi_g2_16_53-seq-val/best/csv gt_csv_name=annotations_0_15.csv save_suffix=inv-yl8-seq-val-best gt_pkl=g2_0_15.pkl iw=0 nms_thresh=0:0.9:0.1 n_proc=12

<a id="swi___ext_reorg_roi_16_5_3_"></a>
## swi       @ ext_reorg_roi_16_53-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/swi/ipsc_2_class_ext_reorg_roi_g2_16_53-no_validate/g2_0_15/csv gt_csv_name=annotations_0_15.csv save_suffix=inv-swi gt_pkl=g2_0_15.pkl iw=0 nms_thresh=0.1 n_proc=12
__iw__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/swi/ipsc_2_class_ext_reorg_roi_g2_16_53-no_validate/g2_0_15/csv gt_csv_name=annotations_0_15.csv save_suffix=inv-swi gt_pkl=g2_0_15.pkl iw=0 n_proc=12 iw=1

<a id="swi_rcnn___ext_reorg_roi_16_5_3_"></a>
## swi-rcnn       @ ext_reorg_roi_16_53-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/swi/ipsc_2_class_ext_reorg_roi_g2_16_53-no_validate-rcnn/g2_0_15/csv gt_csv_name=annotations_0_15.csv save_suffix=inv-swi-rcnn gt_pkl=g2_0_15.pkl iw=0 nms_thresh=0:0.9:0.1 n_proc=12 enable_mask=0

<a id="cvnxt_large___ext_reorg_roi_16_5_3_"></a>
## cvnxt-large       @ ext_reorg_roi_16_53-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/swi/ipsc_2_class_ext_reorg_roi_g2_16_53-convnext_large_in22k/g2_0_15/csv gt_csv_name=annotations_0_15.csv save_suffix=inv-cvnxt-large gt_pkl=g2_0_15.pkl iw=0 nms_thresh=0.9 n_proc=12
__iw__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/swi/ipsc_2_class_ext_reorg_roi_g2_16_53-convnext_large_in22k/g2_0_15/csv gt_csv_name=annotations_0_15.csv save_suffix=inv-cvnxt-large gt_pkl=g2_0_15.pkl iw=0 n_proc=12 iw=1

<a id="idol___ext_reorg_roi_16_5_3_"></a>
## idol       @ ext_reorg_roi_16_53-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/idol-ipsc-ext_reorg_roi_g2_16_53/inference_model_0254999/csv gt_csv_name=annotations_0_15.csv save_suffix=inv-idol gt_pkl=g2_0_15.pkl nms_thresh=0:0.9:0.1 n_proc=2 
__-max_length-1-__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/idol-ipsc-ext_reorg_roi_g2_16_53/inference_model_0254999_max_length-1/csv gt_csv_name=annotations_0_15.csv save_suffix=inv-idol-max_length-1 gt_pkl=g2_0_15.pkl nms_thresh=0:0.9:0.1 n_proc=2 
__-max_length-2-__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/idol-ipsc-ext_reorg_roi_g2_16_53/inference_model_0254999_max_length-2/csv gt_csv_name=annotations_0_15.csv save_suffix=inv-idol-max_length-2 gt_pkl=g2_0_15.pkl nms_thresh=0:0.9:0.1 n_proc=2 
__-max_length-4-__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/idol-ipsc-ext_reorg_roi_g2_16_53/inference_model_0254999_max_length-4/csv gt_csv_name=annotations_0_15.csv save_suffix=inv-idol-max_length-4 gt_pkl=g2_0_15.pkl nms_thresh=0:0.9:0.1 n_proc=2 
__-max_length-8-__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/idol-ipsc-ext_reorg_roi_g2_16_53/inference_model_0254999_max_length-8/csv gt_csv_name=annotations_0_15.csv save_suffix=inv-idol-max_length-8 gt_pkl=g2_0_15.pkl nms_thresh=0:0.9:0.1 n_proc=2 


<a id="probs___idol_ext_reorg_roi_16_53_"></a>
### probs       @ idol/ext_reorg_roi_16_53-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/idol-ipsc-ext_reorg_roi_g2_16_53/inference_model_0254999_probs/csv gt_csv_name=annotations_0_15.csv save_suffix=inv-idol-probs gt_pkl=g2_0_15.pkl

<a id="idol_inc___ext_reorg_roi_16_5_3_"></a>
## idol-inc       @ ext_reorg_roi_16_53-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/idol-ipsc-ext_reorg_roi_g2_16_53/inference_model_0254999_incremental/csv_incremental gt_csv_name=annotations_0_15.csv save_suffix=inv-idol-inc gt_pkl=g2_0_15.pkl nms_thresh=0:0.9:0.1 n_proc=12
__iw__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/idol-ipsc-ext_reorg_roi_g2_16_53/inference_model_0254999_incremental/csv_incremental gt_csv_name=annotations_0_15.csv save_suffix=inv-idol-inc iw=0 n_proc=1 iw=1 gt_pkl=g2_0_15.pkl

<a id="probs___idol_inc_ext_reorg_roi_16_53_"></a>
### probs       @ idol-inc/ext_reorg_roi_16_53-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/idol-ipsc-ext_reorg_roi_g2_16_53/inference_model_0254999_incremental_probs/csv_incremental gt_csv_name=annotations_0_15.csv save_suffix=inv-idol-inc-probs gt_pkl=g2_0_15.pkl nms_thresh=0.1 n_proc=12
__iw__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/idol-ipsc-ext_reorg_roi_g2_16_53/inference_model_0254999_incremental_probs/csv_incremental gt_csv_name=annotations_0_15.csv save_suffix=inv-idol-inc-probs gt_pkl=g2_0_15.pkl iw=1

<a id="nms_01___probs_idol_inc_ext_reorg_roi_16_53_"></a>
#### nms-01       @ probs/idol-inc/ext_reorg_roi_16_53-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/idol-ipsc-ext_reorg_roi_g2_16_53/inference_model_0254999_incremental_probs/csv_incremental_nms_01 gt_csv_name=annotations_0_15.csv save_suffix=inv-idol-inc-probs-nms-01 gt_pkl=g2_0_15.pkl iw=0 vid_fmt=H264,2,jpg

<a id="seq___ext_reorg_roi_16_5_3_"></a>
## seq       @ ext_reorg_roi_16_53-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/seqformer-ipsc-ext_reorg_roi_g2_16_53/inference_model_0241999/csv gt_csv_name=annotations_0_15.csv save_suffix=inv-seq gt_pkl=g2_0_15.pkl nms_thresh=0:0.9:0.1 n_proc=2 
__-max_length-1-__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/seqformer-ipsc-ext_reorg_roi_g2_16_53/inference_model_0241999_max_length-1/csv gt_csv_name=annotations_0_15.csv save_suffix=inv-seq-max_length-1 gt_pkl=g2_0_15.pkl nms_thresh=0:0.9:0.1 n_proc=2 
__-max_length-2-__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/seqformer-ipsc-ext_reorg_roi_g2_16_53/inference_model_0241999_max_length-2/csv gt_csv_name=annotations_0_15.csv save_suffix=inv-seq-max_length-2 gt_pkl=g2_0_15.pkl nms_thresh=0:0.9:0.1 n_proc=2 
__-max_length-4-__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/seqformer-ipsc-ext_reorg_roi_g2_16_53/inference_model_0241999_max_length-4/csv gt_csv_name=annotations_0_15.csv save_suffix=inv-seq-max_length-4 gt_pkl=g2_0_15.pkl nms_thresh=0:0.9:0.1 n_proc=2 
__-max_length-8-__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/seqformer-ipsc-ext_reorg_roi_g2_16_53/inference_model_0241999_max_length-8/csv gt_csv_name=annotations_0_15.csv save_suffix=inv-seq-max_length-8 gt_pkl=g2_0_15.pkl nms_thresh=0:0.9:0.1 n_proc=2 

<a id="probs___seq_ext_reorg_roi_16_5_3_"></a>
### probs       @ seq/ext_reorg_roi_16_53-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/seqformer-ipsc-ext_reorg_roi_g2_16_53/inference_model_0241999_probs/csv gt_csv_name=annotations_0_15.csv save_suffix=inv-seq-probs gt_pkl=g2_0_15.pkl

<a id="seq_inc___ext_reorg_roi_16_5_3_"></a>
## seq-inc       @ ext_reorg_roi_16_53-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/seqformer-ipsc-ext_reorg_roi_g2_16_53/inference_model_0241999_incremental/csv_incremental gt_csv_name=annotations_0_15.csv save_suffix=inv-seq-inc gt_pkl=g2_0_15.pkl nms_thresh=0.1 n_proc=12
__iw__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/seqformer-ipsc-ext_reorg_roi_g2_16_53/inference_model_0241999_incremental/csv_incremental gt_csv_name=annotations_0_15.csv save_suffix=inv-seq-inc iw=1 gt_pkl=g2_0_15.pkl

<a id="probs___seq_inc_ext_reorg_roi_16_5_3_"></a>
### probs       @ seq-inc/ext_reorg_roi_16_53-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/seqformer-ipsc-ext_reorg_roi_g2_16_53/inference_model_0241999_incremental_probs/csv_incremental gt_csv_name=annotations_0_15.csv save_suffix=inv-seq-inc-probs gt_pkl=g2_0_15.pkl

<a id="vita___ext_reorg_roi_16_5_3_"></a>
## vita       @ ext_reorg_roi_16_53-->eval_det
<a id="0119999___vita_ext_reorg_roi_16_53_"></a>
### 0119999       @ vita/ext_reorg_roi_16_53-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vita/vita-ipsc-ext_reorg_roi_g2_16_53_swin/inference_model_0119999/csv gt_csv_name=annotations_0_15.csv save_suffix=inv-vita-swin-0119999 gt_pkl=g2_0_15.pkl
<a id="0329999___vita_ext_reorg_roi_16_53_"></a>
### 0329999       @ vita/ext_reorg_roi_16_53-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vita/vita-ipsc-ext_reorg_roi_g2_16_53_swin/inference_model_0329999/csv gt_csv_name=annotations_0_15.csv save_suffix=inv-vita gt_pkl=g2_0_15.pkl nms_thresh=0:0.9:0.1 n_proc=2
__-max_length-1-__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vita/vita-ipsc-ext_reorg_roi_g2_16_53_swin/inference_model_0329999_max_length-1/csv gt_csv_name=annotations_0_15.csv save_suffix=inv-vita-max_length-1 gt_pkl=g2_0_15.pkl nms_thresh=0:0.9:0.1 n_proc=2
__-max_length-2-__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vita/vita-ipsc-ext_reorg_roi_g2_16_53_swin/inference_model_0329999_max_length-2/csv gt_csv_name=annotations_0_15.csv save_suffix=inv-vita-max_length-2 gt_pkl=g2_0_15.pkl nms_thresh=0:0.9:0.1 n_proc=2
__-max_length-4-__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vita/vita-ipsc-ext_reorg_roi_g2_16_53_swin/inference_model_0329999_max_length-4/csv gt_csv_name=annotations_0_15.csv save_suffix=inv-vita-max_length-4 gt_pkl=g2_0_15.pkl nms_thresh=0:0.9:0.1 n_proc=2
__-max_length-8-__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vita/vita-ipsc-ext_reorg_roi_g2_16_53_swin/inference_model_0329999_max_length-8/csv gt_csv_name=annotations_0_15.csv save_suffix=inv-vita-max_length-8 gt_pkl=g2_0_15.pkl nms_thresh=0:0.9:0.1 n_proc=2

<a id="vita_inc___ext_reorg_roi_16_5_3_"></a>
## vita-inc       @ ext_reorg_roi_16_53-->eval_det
<a id="0119999___vita_inc_ext_reorg_roi_16_53_"></a>
### 0119999       @ vita-inc/ext_reorg_roi_16_53-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vita/vita-ipsc-ext_reorg_roi_g2_16_53_swin/inference_model_0119999_incremental/csv_incremental gt_csv_name=annotations_0_15.csv save_suffix=inv-vita-inc-0119999 gt_pkl=g2_0_15.pkl
<a id="0329999___vita_inc_ext_reorg_roi_16_53_"></a>
### 0329999       @ vita-inc/ext_reorg_roi_16_53-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vita/vita-ipsc-ext_reorg_roi_g2_16_53_swin/inference_model_0329999_incremental/csv_incremental gt_csv_name=annotations_0_15.csv save_suffix=inv-vita-inc gt_pkl=g2_0_15.pkl iw=0 nms_thresh=0.1 n_proc=12
__iw__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vita/vita-ipsc-ext_reorg_roi_g2_16_53_swin/inference_model_0329999_incremental/csv_incremental gt_csv_name=annotations_0_15.csv save_suffix=inv-vita-inc gt_pkl=g2_0_15.pkl iw=0 n_proc=12 iw=1


<a id="vita_retrain_inc___ext_reorg_roi_16_5_3_"></a>
## vita-retrain-inc       @ ext_reorg_roi_16_53-->eval_det
<a id="0004999___vita_retrain_inc_ext_reorg_roi_16_53_"></a>
### 0004999       @ vita-retrain-inc/ext_reorg_roi_16_53-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vita/vita-ipsc-ext_reorg_roi_g2_16_53_swin_retrain/inference_model_0004999_incremental/csv_incremental gt_csv_name=annotations_0_15.csv save_suffix=inv-vita-retrain-inc-0004999 gt_pkl=g2_0_15.pkl
<a id="0079999___vita_retrain_inc_ext_reorg_roi_16_53_"></a>
### 0079999       @ vita-retrain-inc/ext_reorg_roi_16_53-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vita/vita-ipsc-ext_reorg_roi_g2_16_53_swin_retrain/inference_model_0079999_incremental/csv_incremental gt_csv_name=annotations_0_15.csv save_suffix=inv-vita-retrain-inc-0079999 gt_pkl=g2_0_15.pkl
<a id="0104999___vita_retrain_inc_ext_reorg_roi_16_53_"></a>
### 0104999       @ vita-retrain-inc/ext_reorg_roi_16_53-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vita/vita-ipsc-ext_reorg_roi_g2_16_53_swin_retrain/inference_model_0104999_incremental/csv_incremental gt_csv_name=annotations_0_15.csv save_suffix=inv-vita-retrain-inc-0104999 gt_pkl=g2_0_15.pkl

<a id="ext_reorg_roi_54_12_6_"></a>
# ext_reorg_roi_54_126
<a id="yl8___ext_reorg_roi_54_126_"></a>
## yl8       @ ext_reorg_roi_54_126-->eval_det
__last__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/yl8/ext_reorg_roi_g2_54_126/last/csv gt_csv_name=annotations_0_53.csv save_suffix=inv-yl8-last gt_pkl=g2_0_53.pkl iw=0 nms_thresh=0:0.9:0.1 n_proc=12
__best__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/yl8/ext_reorg_roi_g2_54_126/best/csv gt_csv_name=annotations_0_53.csv save_suffix=inv-yl8-best gt_pkl=g2_0_53.pkl iw=0 nms_thresh=0:0.9:0.1 n_proc=12
<a id="val___yl8_ext_reorg_roi_54_126_"></a>
### val       @ yl8/ext_reorg_roi_54_126-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/yl8/ext_reorg_roi_g2_54_126-val/last/csv gt_csv_name=annotations_0_53.csv save_suffix=inv-yl8-val-last gt_pkl=g2_0_53.pkl iw=0 nms_thresh=0:0.9:0.1 n_proc=12

python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/yl8/ext_reorg_roi_g2_54_126-val/best/csv gt_csv_name=annotations_0_53.csv save_suffix=inv-yl8-val-best gt_pkl=g2_0_53.pkl iw=0 nms_thresh=0:0.9:0.1 n_proc=12
<a id="seq_val___yl8_ext_reorg_roi_54_126_"></a>
### seq-val       @ yl8/ext_reorg_roi_54_126-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/yl8/ext_reorg_roi_g2_54_126-seq-val/last/csv gt_csv_name=annotations_0_53.csv save_suffix=inv-yl8-seq-val-last gt_pkl=g2_0_53.pkl iw=0 nms_thresh=0.1 n_proc=1

python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/yl8/ext_reorg_roi_g2_54_126-seq-val/best/csv gt_csv_name=annotations_0_53.csv save_suffix=inv-yl8-seq-val-best gt_pkl=g2_0_53.pkl iw=0 nms_thresh=0:0.9:0.1 n_proc=12

<a id="swi___ext_reorg_roi_54_126_"></a>
## swi       @ ext_reorg_roi_54_126-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/swi/ipsc_2_class_ext_reorg_roi_g2_54_126-no_validate/g2_0_53/csv gt_csv_name=annotations_0_53.csv save_suffix=full-swi iw=0 nms_thresh=0:0.9:0.1 n_proc=12 gt_pkl=g2_0_53.pkl
<a id="g2_0_15___swi_ext_reorg_roi_54_126_"></a>
### g2_0_15       @ swi/ext_reorg_roi_54_126-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/swi/ipsc_2_class_ext_reorg_roi_g2_54_126-no_validate/g2_0_15/csv gt_csv_name=annotations_0_15.csv save_suffix=full-swi-g2_0_15 save_vis=1 save_classes=ipsc nms_thresh=0.9 n_proc=12 gt_pkl=g2_0_15.pkl
__iw__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/swi/ipsc_2_class_ext_reorg_roi_g2_54_126-no_validate/g2_0_15/csv gt_csv_name=annotations_0_15.csv save_suffix=full-swi-g2_0_15 nms_thresh=0.9 n_proc=1 gt_pkl=g2_0_15.pkl iw=1

<a id="cvnxt_base___ext_reorg_roi_54_126_"></a>
## cvnxt-base       @ ext_reorg_roi_54_126-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/swi/ipsc_2_class_ext_reorg_roi_g2_54_126-convnext_base_in22k/g2_0_53/csv gt_csv_name=annotations_0_53.csv save_suffix=full-cvnxt-base gt_pkl=g2_0_53.pkl iw=0 nms_thresh=0:0.9:0.1 n_proc=12
<a id="g2_0_15___cvnxt_base_ext_reorg_roi_54_12_6_"></a>
### g2_0_15       @ cvnxt-base/ext_reorg_roi_54_126-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/swi/ipsc_2_class_ext_reorg_roi_g2_54_126-convnext_base_in22k/g2_0_15/csv gt_csv_name=annotations_0_15.csv save_suffix=full-cvnxt-base-g2_0_15 gt_pkl=g2_0_15.pkl iw=0 nms_thresh=0.9 n_proc=12
__iw__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/swi/ipsc_2_class_ext_reorg_roi_g2_54_126-convnext_base_in22k/g2_0_15/csv gt_csv_name=annotations_0_15.csv save_suffix=full-cvnxt-base-g2_0_15 gt_pkl=g2_0_15.pkl iw=0 nms_thresh=0.9 n_proc=1 iw=1

<a id="cvnxt_large___ext_reorg_roi_54_126_"></a>
## cvnxt-large       @ ext_reorg_roi_54_126-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/swi/ipsc_2_class_ext_reorg_roi_g2_54_126-convnext_large_in22k/g2_0_53/csv gt_csv_name=annotations_0_53.csv save_suffix=full-cvnxt-large gt_pkl=g2_0_53.pkl iw=0 nms_thresh=0:0.9:0.1 n_proc=1
<a id="g2_0_15___cvnxt_large_ext_reorg_roi_54_126_"></a>
### g2_0_15       @ cvnxt-large/ext_reorg_roi_54_126-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/swi/ipsc_2_class_ext_reorg_roi_g2_54_126-convnext_large_in22k/g2_0_15/csv gt_csv_name=annotations_0_15.csv save_suffix=full-cvnxt-large-g2_0_15 gt_pkl=g2_0_15.pkl iw=0 nms_thresh=0.9 n_proc=1
__iw__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/swi/ipsc_2_class_ext_reorg_roi_g2_54_126-convnext_large_in22k/g2_0_15/csv gt_csv_name=annotations_0_15.csv save_suffix=full-cvnxt-large-g2_0_15 gt_pkl=g2_0_15.pkl iw=0 nms_thresh=0.9 n_proc=1 iw=1

<a id="idol___ext_reorg_roi_54_126_"></a>
## idol       @ ext_reorg_roi_54_126-->eval_det
<a id="g2_0_53___idol_ext_reorg_roi_54_12_6_"></a>
### g2_0_53       @ idol/ext_reorg_roi_54_126-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/idol-ipsc-ext_reorg_roi_g2_54_126/inference_model_0596999/csv gt_csv_name=annotations_0_53.csv save_suffix=full-idol gt_pkl=g2_0_53.pkl nms_thresh=0:0.9:0.1 n_proc=12 
__-max_length-1-__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/idol-ipsc-ext_reorg_roi_g2_54_126/inference_model_0596999_max_length-1/csv gt_csv_name=annotations_0_53.csv save_suffix=full-idol-max_length-1 gt_pkl=g2_0_53.pkl nms_thresh=0:0.9:0.1 n_proc=12 
__-max_length-2-__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/idol-ipsc-ext_reorg_roi_g2_54_126/inference_model_0596999_max_length-2/csv gt_csv_name=annotations_0_53.csv save_suffix=full-idol-max_length-2 gt_pkl=g2_0_53.pkl nms_thresh=0:0.9:0.1 n_proc=12 
__-max_length-4-__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/idol-ipsc-ext_reorg_roi_g2_54_126/inference_model_0596999_max_length-4/csv gt_csv_name=annotations_0_53.csv save_suffix=full-idol-max_length-4 gt_pkl=g2_0_53.pkl nms_thresh=0:0.9:0.1 n_proc=12 
__-max_length-8-__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/idol-ipsc-ext_reorg_roi_g2_54_126/inference_model_0596999_max_length-8/csv gt_csv_name=annotations_0_53.csv save_suffix=full-idol-max_length-8 gt_pkl=g2_0_53.pkl nms_thresh=0:0.9:0.1 n_proc=12 
__-max_length-19-__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/idol-ipsc-ext_reorg_roi_g2_54_126/inference_model_0596999_max_length-19/csv gt_csv_name=annotations_0_53.csv save_suffix=full-idol-max_length-19 gt_pkl=g2_0_53.pkl nms_thresh=0:0.9:0.1 n_proc=12 

<a id="g2_0_15___idol_ext_reorg_roi_54_12_6_"></a>
### g2_0_15       @ idol/ext_reorg_roi_54_126-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/idol-ipsc-ext_reorg_roi_g2_54_126/inference_model_0596999_g2_0_15/csv gt_csv_name=annotations_0_15.csv save_suffix=full-idol-g2_0_15 gt_pkl=g2_0_15.pkl nms_thresh=0:0.9:0.1 n_proc=1 
__-max_length-1-__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/idol-ipsc-ext_reorg_roi_g2_54_126/inference_model_0596999_g2_0_15-max_length-1/csv gt_csv_name=annotations_0_15.csv save_suffix=full-idol-g2_0_15-max_length-1 gt_pkl=g2_0_15.pkl nms_thresh=0:0.9:0.1 n_proc=1 
__-max_length-2-__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/idol-ipsc-ext_reorg_roi_g2_54_126/inference_model_0596999_g2_0_15-max_length-2/csv gt_csv_name=annotations_0_15.csv save_suffix=full-idol-g2_0_15-max_length-2 gt_pkl=g2_0_15.pkl nms_thresh=0:0.9:0.1 n_proc=1 
__-max_length-4-__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/idol-ipsc-ext_reorg_roi_g2_54_126/inference_model_0596999_g2_0_15-max_length-4/csv gt_csv_name=annotations_0_15.csv save_suffix=full-idol-g2_0_15-max_length-4 gt_pkl=g2_0_15.pkl nms_thresh=0:0.9:0.1 n_proc=1 
__-max_length-8-__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/idol-ipsc-ext_reorg_roi_g2_54_126/inference_model_0596999_g2_0_15-max_length-8/csv gt_csv_name=annotations_0_15.csv save_suffix=full-idol-g2_0_15-max_length-8 gt_pkl=g2_0_15.pkl nms_thresh=0:0.9:0.1 n_proc=1 


<a id="idol_inc___ext_reorg_roi_54_126_"></a>
## idol-inc       @ ext_reorg_roi_54_126-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/idol-ipsc-ext_reorg_roi_g2_54_126/inference_model_0596999_incremental_probs/csv_incremental gt_csv_name=annotations_0_53.csv save_suffix=full-idol-inc-probs gt_pkl=g2_0_53.pkl nms_thresh=0:0.9:0.1 n_proc=12 
<a id="g2_0_15___idol_inc_ext_reorg_roi_54_12_6_"></a>
### g2_0_15       @ idol-inc/ext_reorg_roi_54_126-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/idol-ipsc-ext_reorg_roi_g2_54_126/inference_model_0596999_g2_0_15-incremental_probs/csv_incremental gt_csv_name=annotations_0_15.csv save_suffix=full-idol-inc-probs-g2_0_15 gt_pkl=g2_0_15.pkl nms_thresh=0.1 n_proc=12
__iw__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/idol-ipsc-ext_reorg_roi_g2_54_126/inference_model_0596999_g2_0_15-incremental_probs/csv_incremental gt_csv_name=annotations_0_15.csv save_suffix=full-idol-inc-probs-g2_0_15 gt_pkl=g2_0_15.pkl nms_thresh=0.1 iw=1

<a id="seq___ext_reorg_roi_54_126_"></a>
## seq       @ ext_reorg_roi_54_126-->eval_det
<a id="g2_0_53___seq_ext_reorg_roi_54_126_"></a>
### g2_0_53       @ seq/ext_reorg_roi_54_126-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/seqformer-ipsc-ext_reorg_roi_g2_54_126/inference_model_0495999/csv gt_csv_name=annotations_0_15.csv save_suffix=full-seq gt_pkl=g2_0_53.pkl nms_thresh=0:0.9:0.1 n_proc=1
__-max_length-1-__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/seqformer-ipsc-ext_reorg_roi_g2_54_126/inference_model_0495999_max_length-1/csv gt_csv_name=annotations_0_15.csv save_suffix=full-seq-max_length-1 gt_pkl=g2_0_53.pkl nms_thresh=0:0.9:0.1 n_proc=1
__-max_length-2-__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/seqformer-ipsc-ext_reorg_roi_g2_54_126/inference_model_0495999_max_length-2/csv gt_csv_name=annotations_0_15.csv save_suffix=full-seq-max_length-2 gt_pkl=g2_0_53.pkl nms_thresh=0:0.9:0.1 n_proc=1
__-max_length-4-__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/seqformer-ipsc-ext_reorg_roi_g2_54_126/inference_model_0495999_max_length-4/csv gt_csv_name=annotations_0_15.csv save_suffix=full-seq-max_length-4 gt_pkl=g2_0_53.pkl nms_thresh=0:0.9:0.1 n_proc=1
__-max_length-8-__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/seqformer-ipsc-ext_reorg_roi_g2_54_126/inference_model_0495999_max_length-8/csv gt_csv_name=annotations_0_15.csv save_suffix=full-seq-max_length-8 gt_pkl=g2_0_53.pkl nms_thresh=0:0.9:0.1 n_proc=1
__-max_length-19-__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/seqformer-ipsc-ext_reorg_roi_g2_54_126/inference_model_0495999_max_length-19/csv gt_csv_name=annotations_0_15.csv save_suffix=full-seq-max_length-19 gt_pkl=g2_0_53.pkl nms_thresh=0:0.9:0.1 n_proc=12

<a id="g2_0_15___seq_ext_reorg_roi_54_126_"></a>
### g2_0_15       @ seq/ext_reorg_roi_54_126-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/seqformer-ipsc-ext_reorg_roi_g2_54_126/inference_model_0495999_g2_0_15/csv gt_csv_name=annotations_0_15.csv save_suffix=full-seq-g2_0_15 gt_pkl=g2_0_15.pkl nms_thresh=0:0.9:0.1 n_proc=1
__-max_length-1-__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/seqformer-ipsc-ext_reorg_roi_g2_54_126/inference_model_0495999_g2_0_15-max_length-1/csv gt_csv_name=annotations_0_15.csv save_suffix=full-seq-g2_0_15-max_length-1 gt_pkl=g2_0_15.pkl nms_thresh=0:0.9:0.1 n_proc=1
__-max_length-2-__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/seqformer-ipsc-ext_reorg_roi_g2_54_126/inference_model_0495999_g2_0_15-max_length-2/csv gt_csv_name=annotations_0_15.csv save_suffix=full-seq-g2_0_15-max_length-2 gt_pkl=g2_0_15.pkl nms_thresh=0:0.9:0.1 n_proc=1
__-max_length-4-__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/seqformer-ipsc-ext_reorg_roi_g2_54_126/inference_model_0495999_g2_0_15-max_length-4/csv gt_csv_name=annotations_0_15.csv save_suffix=full-seq-g2_0_15-max_length-4 gt_pkl=g2_0_15.pkl nms_thresh=0:0.9:0.1 n_proc=1
__-max_length-8-__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/seqformer-ipsc-ext_reorg_roi_g2_54_126/inference_model_0495999_g2_0_15-max_length-8/csv gt_csv_name=annotations_0_15.csv save_suffix=full-seq-g2_0_15-max_length-8 gt_pkl=g2_0_15.pkl nms_thresh=0:0.9:0.1 n_proc=12

<a id="seq_inc___ext_reorg_roi_54_126_"></a>
## seq-inc       @ ext_reorg_roi_54_126-->eval_det
<a id="g2_0_15___seq_inc_ext_reorg_roi_54_126_"></a>
### g2_0_15       @ seq-inc/ext_reorg_roi_54_126-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/seqformer-ipsc-ext_reorg_roi_g2_54_126/inference_model_0495999_g2_0_15-incremental_probs/csv_incremental gt_csv_name=annotations_0_15.csv save_suffix=full-seq-inc-g2_0_15 gt_pkl=g2_0_15.pkl nms_thresh=0.9 n_proc=12
__iw__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/seqformer-ipsc-ext_reorg_roi_g2_54_126/inference_model_0495999_g2_0_15-incremental_probs/csv_incremental gt_csv_name=annotations_0_15.csv save_suffix=full-seq-inc-g2_0_15 iw=1 gt_pkl=g2_0_15.pkl nms_thresh=0.9

__-max_length-2-__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vnxt/seqformer-ipsc-ext_reorg_roi_g2_54_126/inference_model_0495999_g2_0_15-max_length-2-incremental_probs/csv_incremental gt_csv_name=annotations_0_15.csv save_suffix=full-seq-inc-g2_0_15-max_length-2 gt_pkl=g2_0_15.pkl nms_thresh=0:0.9:0.1 n_proc=1

<a id="vita___ext_reorg_roi_54_126_"></a>
## vita       @ ext_reorg_roi_54_126-->eval_det
<a id="g2_0_53___vita_ext_reorg_roi_54_12_6_"></a>
### g2_0_53       @ vita/ext_reorg_roi_54_126-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vita/vita-ipsc-ext_reorg_roi_g2_54_126_swin/inference_model_0194999/csv gt_csv_name=annotations_0_53.csv save_suffix=full-vita nms_thresh=0:0.9:0.1 n_proc=12 gt_pkl=g2_0_53.pkl
__-max_length-1-__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vita/vita-ipsc-ext_reorg_roi_g2_54_126_swin/inference_model_0194999_max_length-1/csv gt_csv_name=annotations_0_53.csv save_suffix=full-vita-max_length-1 gt_pkl=g2_0_53.pkl nms_thresh=0:0.9:0.1 n_proc=1
__-max_length-2-__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vita/vita-ipsc-ext_reorg_roi_g2_54_126_swin/inference_model_0194999_max_length-2/csv gt_csv_name=annotations_0_53.csv save_suffix=full-vita-max_length-2 gt_pkl=g2_0_53.pkl nms_thresh=0:0.9:0.1 n_proc=1
__-max_length-4-__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vita/vita-ipsc-ext_reorg_roi_g2_54_126_swin/inference_model_0194999_max_length-4/csv gt_csv_name=annotations_0_53.csv save_suffix=full-vita-max_length-4 gt_pkl=g2_0_53.pkl nms_thresh=0:0.9:0.1 n_proc=1
__-max_length-8-__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vita/vita-ipsc-ext_reorg_roi_g2_54_126_swin/inference_model_0194999_max_length-8/csv gt_csv_name=annotations_0_53.csv save_suffix=full-vita-max_length-8 gt_pkl=g2_0_53.pkl nms_thresh=0:0.9:0.1 n_proc=1
__-max_length-19-__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vita/vita-ipsc-ext_reorg_roi_g2_54_126_swin/inference_model_0194999_max_length-19/csv gt_csv_name=annotations_0_53.csv save_suffix=full-vita-max_length-19 gt_pkl=g2_0_53.pkl nms_thresh=0:0.9:0.1 n_proc=12

<a id="g2_0_15___vita_ext_reorg_roi_54_12_6_"></a>
### g2_0_15       @ vita/ext_reorg_roi_54_126-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vita/vita-ipsc-ext_reorg_roi_g2_54_126_swin/inference_model_0194999_g2_0_15/csv gt_csv_name=annotations_0_15.csv save_suffix=full-vita-g2_0_15 nms_thresh=0 n_proc=1
__-max_length-1-__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vita/vita-ipsc-ext_reorg_roi_g2_54_126_swin/inference_model_0194999_g2_0_15-max_length-1/csv gt_csv_name=annotations_0_15.csv save_suffix=full-vita-g2_0_15-max_length-1 gt_pkl=g2_0_15.pkl nms_thresh=0:0.9:0.1 n_proc=1
__-max_length-2-__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vita/vita-ipsc-ext_reorg_roi_g2_54_126_swin/inference_model_0194999_g2_0_15-max_length-2/csv gt_csv_name=annotations_0_15.csv save_suffix=full-vita-g2_0_15-max_length-2 gt_pkl=g2_0_15.pkl nms_thresh=0:0.9:0.1 n_proc=1
__-max_length-4-__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vita/vita-ipsc-ext_reorg_roi_g2_54_126_swin/inference_model_0194999_g2_0_15-max_length-4/csv gt_csv_name=annotations_0_15.csv save_suffix=full-vita-g2_0_15-max_length-4 gt_pkl=g2_0_15.pkl nms_thresh=0:0.9:0.1 n_proc=1
__-max_length-8-__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vita/vita-ipsc-ext_reorg_roi_g2_54_126_swin/inference_model_0194999_g2_0_15-max_length-8/csv gt_csv_name=annotations_0_15.csv save_suffix=full-vita-g2_0_15-max_length-8 gt_pkl=g2_0_15.pkl nms_thresh=0:0.9:0.1 n_proc=1

<a id="vita_inc___ext_reorg_roi_54_126_"></a>
## vita-inc       @ ext_reorg_roi_54_126-->eval_det
<a id="g2_0_53___vita_inc_ext_reorg_roi_54_12_6_"></a>
### g2_0_53       @ vita-inc/ext_reorg_roi_54_126-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vita/vita-ipsc-ext_reorg_roi_g2_54_126_swin/inference_model_0194999_incremental/csv_incremental gt_csv_name=annotations_0_53.csv save_suffix=full-vita-inc gt_pkl=g2_0_53.pkl nms_thresh=0:0.9:0.1 n_proc=12
__iw__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vita/vita-ipsc-ext_reorg_roi_g2_54_126_swin/inference_model_0194999_incremental/csv_incremental gt_csv_name=annotations_0_53.csv save_suffix=full-vita-inc nms_thresh=0.2 n_proc=1 iw=1
__-max_length-2-__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vita/vita-ipsc-ext_reorg_roi_g2_54_126_swin/inference_model_0194999_max_length-2-incremental/csv_incremental gt_csv_name=annotations_0_53.csv save_suffix=full-vita-inc-max_length-2 gt_pkl=g2_0_53.pkl nms_thresh=0:0.9:0.1 n_proc=12

<a id="g2_0_15___vita_inc_ext_reorg_roi_54_12_6_"></a>
### g2_0_15       @ vita-inc/ext_reorg_roi_54_126-->eval_det
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vita/vita-ipsc-ext_reorg_roi_g2_54_126_swin/inference_model_0194999_g2_0_15-incremental/csv_incremental gt_csv_name=annotations_0_15.csv save_suffix=full-vita-inc-g2_0_15 gt_pkl=g2_0_15.pkl nms_thresh=0.2 n_proc=12
__iw__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vita/vita-ipsc-ext_reorg_roi_g2_54_126_swin/inference_model_0194999_g2_0_15-incremental/csv_incremental gt_csv_name=annotations_0_15.csv save_suffix=full-vita-inc-g2_0_15 gt_pkl=g2_0_15.pkl n_proc=1 iw=1 nms_thresh=0.2
__-max_length-2-__
python3 eval_det.py cfg=ipsc_2_class img_paths=lists/ext_reorg_roi.txt det_paths=log/vita/vita-ipsc-ext_reorg_roi_g2_54_126_swin/inference_model_0194999_g2_0_15-max_length-2-incremental/csv_incremental gt_csv_name=annotations_0_15.csv save_suffix=full-vita-inc-g2_0_15-max_length-2 gt_pkl=g2_0_15.pkl nms_thresh=0:0.9:0.1 n_proc=12

