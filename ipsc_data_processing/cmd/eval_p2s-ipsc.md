<!-- MarkdownTOC -->

- [vit_b-640       @ p2s](#vit_b_640___p2_s_)
    - [16_53-aug-fbb       @ vit_b-640](#16_53_aug_fbb___vit_b_64_0_)
        - [on-0_15       @ 16_53-aug-fbb/vit_b-640](#on_0_15___16_53_aug_fbb_vit_b_64_0_)
    - [54_126-aug-fbb       @ vit_b-640](#54_126_aug_fbb___vit_b_64_0_)
        - [on-0_15       @ 54_126-aug-fbb/vit_b-640](#on_0_15___54_126_aug_fbb_vit_b_640_)
- [resnet-640       @ p2s](#resnet_640___p2_s_)
    - [16_53-buggy       @ resnet-640](#16_53_buggy___resnet_640_)
        - [batch_48       @ 16_53-buggy/resnet-640](#batch_48___16_53_buggy_resnet_640_)
            - [ckpt-1975       @ batch_48/16_53-buggy/resnet-640](#ckpt_1975___batch_48_16_53_buggy_resnet_64_0_)
                - [on-g2_0_15       @ ckpt-1975/batch_48/16_53-buggy/resnet-640](#on_g2_0_15___ckpt_1975_batch_48_16_53_buggy_resnet_64_0_)
                - [on-g2_54_126       @ ckpt-1975/batch_48/16_53-buggy/resnet-640](#on_g2_54_126___ckpt_1975_batch_48_16_53_buggy_resnet_64_0_)
            - [ckpt-12275       @ batch_48/16_53-buggy/resnet-640](#ckpt_12275___batch_48_16_53_buggy_resnet_64_0_)
                - [on-g2_0_15       @ ckpt-12275/batch_48/16_53-buggy/resnet-640](#on_g2_0_15___ckpt_12275_batch_48_16_53_buggy_resnet_640_)
        - [batch_32       @ 16_53-buggy/resnet-640](#batch_32___16_53_buggy_resnet_640_)
            - [on-g2_0_15       @ batch_32/16_53-buggy/resnet-640](#on_g2_0_15___batch_32_16_53_buggy_resnet_64_0_)
        - [batch_6       @ 16_53-buggy/resnet-640](#batch_6___16_53_buggy_resnet_640_)
            - [on-g2_0_15       @ batch_6/16_53-buggy/resnet-640](#on_g2_0_15___batch_6_16_53_buggy_resnet_640_)
            - [on-g2_54_126       @ batch_6/16_53-buggy/resnet-640](#on_g2_54_126___batch_6_16_53_buggy_resnet_640_)
        - [batch_4-scratch       @ 16_53-buggy/resnet-640](#batch_4_scratch___16_53_buggy_resnet_640_)
            - [on-g2_0_15       @ batch_4-scratch/16_53-buggy/resnet-640](#on_g2_0_15___batch_4_scratch_16_53_buggy_resnet_640_)
            - [on-g2_54_126       @ batch_4-scratch/16_53-buggy/resnet-640](#on_g2_54_126___batch_4_scratch_16_53_buggy_resnet_640_)
    - [16_53       @ resnet-640](#16_53___resnet_640_)
    - [16_53-aug       @ resnet-640](#16_53_aug___resnet_640_)
        - [on-0_15       @ 16_53-aug/resnet-640](#on_0_15___16_53_aug_resnet_640_)
        - [on-54_126       @ 16_53-aug/resnet-640](#on_54_126___16_53_aug_resnet_640_)
            - [acc       @ on-54_126/16_53-aug/resnet-640](#acc___on_54_126_16_53_aug_resnet_640_)
    - [16_53-aug-retrain       @ resnet-640](#16_53_aug_retrain___resnet_640_)
        - [on-0_15       @ 16_53-aug-retrain/resnet-640](#on_0_15___16_53_aug_retrain_resnet_640_)
    - [16_53-aug-fbb       @ resnet-640](#16_53_aug_fbb___resnet_640_)
        - [on-0_15       @ 16_53-aug-fbb/resnet-640](#on_0_15___16_53_aug_fbb_resnet_640_)
    - [0_37-gxe       @ resnet-640](#0_37_gxe___resnet_640_)
        - [on-g2_38_53       @ 0_37-gxe/resnet-640](#on_g2_38_53___0_37_gxe_resnet_64_0_)
        - [on-g2_38_53-conf_0       @ 0_37-gxe/resnet-640](#on_g2_38_53_conf_0___0_37_gxe_resnet_64_0_)
    - [0_37       @ resnet-640](#0_37___resnet_640_)
    - [0_37-aug       @ resnet-640](#0_37_aug___resnet_640_)
        - [acc       @ 0_37-aug/resnet-640](#acc___0_37_aug_resnet_64_0_)
    - [54_126-aug       @ resnet-640](#54_126_aug___resnet_640_)
        - [on-0_15       @ 54_126-aug/resnet-640](#on_0_15___54_126_aug_resnet_64_0_)
    - [54_126-aug-fbb       @ resnet-640](#54_126_aug_fbb___resnet_640_)
        - [on-0_15       @ 54_126-aug-fbb/resnet-640](#on_0_15___54_126_aug_fbb_resnet_64_0_)
    - [54_126-aug-fbb-cls_eq       @ resnet-640](#54_126_aug_fbb_cls_eq___resnet_640_)
        - [on-0_15       @ 54_126-aug-fbb-cls_eq/resnet-640](#on_0_15___54_126_aug_fbb_cls_eq_resnet_640_)
    - [pt       @ resnet-640](#pt___resnet_640_)
        - [on-g2_0_1       @ pt/resnet-640](#on_g2_0_1___pt_resnet_64_0_)
        - [on-g2_16_53       @ pt/resnet-640](#on_g2_16_53___pt_resnet_64_0_)
        - [on-g2_0_15       @ pt/resnet-640](#on_g2_0_15___pt_resnet_64_0_)
        - [on-g2_54_126       @ pt/resnet-640](#on_g2_54_126___pt_resnet_64_0_)
- [resnet_1333       @ p2s](#resnet_1333___p2_s_)
    - [g2_16_53       @ resnet_1333](#g2_16_53___resnet_133_3_)
- [resnet-c4-640       @ p2s](#resnet_c4_640___p2_s_)
    - [g2_16_53       @ resnet-c4-640](#g2_16_53___resnet_c4_64_0_)
- [resnet-c4-1333       @ p2s](#resnet_c4_1333___p2_s_)
    - [g2_0_1       @ resnet-c4-1333](#g2_0_1___resnet_c4_1333_)

<!-- /MarkdownTOC -->
<a id="vit_b_640___p2_s_"></a>
# vit_b-640       @ p2s-->eval_det_p2s
<a id="16_53_aug_fbb___vit_b_64_0_"></a>
## 16_53-aug-fbb       @ vit_b-640-->eval_p2s-ipsc
<a id="on_0_15___16_53_aug_fbb_vit_b_64_0_"></a>
### on-0_15       @ 16_53-aug-fbb/vit_b-640-->eval_p2s-ipsc
python eval_det.py cfg=p2s,ipsc:0_15:det-0:gt-1:nms-1:agn:proc-1:_in_-vit_b_640_ext_reorg_roi_g2-16_53-batch_4-jtr-res_1280-fbb/ckpt-__var__-ext_reorg_roi_g2-0_15/csv-batch_2:_out_-p2s-ipsc-16_53-aug-fbb-vit_b

<a id="54_126_aug_fbb___vit_b_64_0_"></a>
## 54_126-aug-fbb       @ vit_b-640-->eval_p2s-ipsc
<a id="on_0_15___54_126_aug_fbb_vit_b_640_"></a>
### on-0_15       @ 54_126-aug-fbb/vit_b-640-->eval_p2s-ipsc
python eval_det.py cfg=p2s,ipsc:0_15:det-0:gt-1:nms-1:agn:proc-1:_in_-vit_b_640_ext_reorg_roi_g2-54_126-batch_4-jtr-res_1280-fbb/ckpt-__var__-ext_reorg_roi_g2-0_15/csv-batch_2:_out_-p2s-ipsc-54_126-aug-fbb-vit_b


<a id="resnet_640___p2_s_"></a>
# resnet-640       @ p2s-->eval_det_p2s
<a id="16_53_buggy___resnet_640_"></a>
## 16_53-buggy       @ resnet-640-->eval_p2s-ipsc
<a id="batch_48___16_53_buggy_resnet_640_"></a>
### batch_48       @ 16_53-buggy/resnet-640-->eval_p2s-ipsc
<a id="ckpt_1975___batch_48_16_53_buggy_resnet_64_0_"></a>
#### ckpt-1975       @ batch_48/16_53-buggy/resnet-640-->eval_p2s-ipsc
<a id="on_g2_0_15___ckpt_1975_batch_48_16_53_buggy_resnet_64_0_"></a>
##### on-g2_0_15       @ ckpt-1975/batch_48/16_53-buggy/resnet-640-->eval_p2s-ipsc
python3 eval_det.py cfg=p2s,ipsc:ext:g2_0_15-agn:_in_-resnet_640_ext_reorg_roi_g2_16_53-batch_48-gxe/ckpt-1975-ext_reorg_roi_g2_0_15/csv-batch_16:_out_-p2s-resnet_640-g2_16_53-g2_0_15-batch_48-1975 show_vis=0 load_gt=0
``cls``
python3 eval_det.py cfg=p2s,ipsc:ext:g2_0_15,p2s:cls:_in_-resnet_640_ext_reorg_roi_g2_16_53-batch_48-gxe/ckpt-1975-ext_reorg_roi_g2_0_15/csv-batch_16:_out_-p2s-resnet_640-g2_16_53-g2_0_15-batch_48-1975-cls show_vis=0 load_gt=0
<a id="on_g2_54_126___ckpt_1975_batch_48_16_53_buggy_resnet_64_0_"></a>
##### on-g2_54_126       @ ckpt-1975/batch_48/16_53-buggy/resnet-640-->eval_p2s-ipsc
python3 eval_det.py cfg=p2s,ipsc:ext:g2_54_126-agn:_in_-resnet_640_ext_reorg_roi_g2_16_53-batch_48-gxe/ckpt-1975-ext_reorg_roi_g2_54_126/csv-batch_32:_out_-p2s-resnet_640-g2_16_53-g2_54_126-batch_48 show_vis=0
``cls``
python3 eval_det.py cfg=p2s,ipsc:ext:g2_54_126,p2s:cls:_in_-resnet_640_ext_reorg_roi_g2_16_53-batch_48-gxe/ckpt-1975-ext_reorg_roi_g2_54_126/csv-batch_32:_out_-p2s-resnet_640-g2_16_53-g2_54_126-batch_48-cls show_vis=0 load_gt=0

<a id="ckpt_12275___batch_48_16_53_buggy_resnet_64_0_"></a>
#### ckpt-12275       @ batch_48/16_53-buggy/resnet-640-->eval_p2s-ipsc
<a id="on_g2_0_15___ckpt_12275_batch_48_16_53_buggy_resnet_640_"></a>
##### on-g2_0_15       @ ckpt-12275/batch_48/16_53-buggy/resnet-640-->eval_p2s-ipsc
python3 eval_det.py cfg=p2s,ipsc:ext:g2_0_15-agn,p2s:fill:_in_-resnet_640_ext_reorg_roi_g2_16_53-batch_48-gxe/ckpt-12275-ext_reorg_roi_g2_0_15/csv-batch_32:_out_-p2s-resnet_640-g2_16_53-g2_0_15-batch_48-12275 show_vis=0 load_gt=0
``cls``
python3 eval_det.py cfg=p2s,ipsc:ext:g2_0_15,p2s:cls:fill:_in_-resnet_640_ext_reorg_roi_g2_16_53-batch_48-gxe/ckpt-12275-ext_reorg_roi_g2_0_15/csv-batch_32:_out_-p2s-resnet_640-g2_16_53-g2_0_15-batch_48-12275-cls show_vis=0 load_gt=0

<a id="batch_32___16_53_buggy_resnet_640_"></a>
### batch_32       @ 16_53-buggy/resnet-640-->eval_p2s-ipsc
<a id="on_g2_0_15___batch_32_16_53_buggy_resnet_64_0_"></a>
#### on-g2_0_15       @ batch_32/16_53-buggy/resnet-640-->eval_p2s-ipsc
python3 eval_det.py cfg=p2s,ipsc:ext:g2_0_15-agn,p2s:fill:_in_-resnet_640_ext_reorg_roi_g2_16_53-batch_32-xe/ckpt-2960-ext_reorg_roi_g2_0_15/csv-batch_64:_out_-p2s-resnet_640-g2_16_53-g2_0_15-batch_32 show_vis=0 load_gt=0
``cls``
python3 eval_det.py cfg=p2s,ipsc:ext:g2_0_15,p2s:cls:fill:_in_-resnet_640_ext_reorg_roi_g2_16_53-batch_32-xe/ckpt-2960-ext_reorg_roi_g2_0_15/csv-batch_64:_out_-p2s-resnet_640-g2_16_53-g2_0_15-batch_32-cls show_vis=0 load_gt=0

<a id="batch_6___16_53_buggy_resnet_640_"></a>
### batch_6       @ 16_53-buggy/resnet-640-->eval_p2s-ipsc
<a id="on_g2_0_15___batch_6_16_53_buggy_resnet_640_"></a>
#### on-g2_0_15       @ batch_6/16_53-buggy/resnet-640-->eval_p2s-ipsc
python3 eval_det.py cfg=p2s,ipsc:ext:g2_0_15-agn:_in_-resnet_640_ext_reorg_roi_g2_16_53-batch_6/ckpt-15876-ext_reorg_roi_g2_0_15/csv-batch_32:_out_-p2s-resnet_640-g2_16_53-g2_0_15-batch_6 show_vis=0 load_gt=0
``cls``
python3 eval_det.py cfg=p2s,ipsc:ext:g2_0_15,p2s:cls:_in_-resnet_640_ext_reorg_roi_g2_16_53-batch_6/ckpt-15876-ext_reorg_roi_g2_0_15/csv-batch_32:_out_-p2s-resnet_640-g2_16_53-g2_0_15-batch_6-cls show_vis=0 load_gt=0
<a id="on_g2_54_126___batch_6_16_53_buggy_resnet_640_"></a>
#### on-g2_54_126       @ batch_6/16_53-buggy/resnet-640-->eval_p2s-ipsc
python3 eval_det.py cfg=p2s,ipsc:ext:g2_54_126-agn:_in_-resnet_640_ext_reorg_roi_g2_16_53-batch_48-gxe/ckpt-15876-ext_reorg_roi_g2_54_126/csv-batch_32:_out_-p2s-resnet_640-g2_16_53-g2_54_126-batch_6 show_vis=0 load_gt=1
``cls``
python3 eval_det.py cfg=p2s,ipsc:ext:g2_54_126,p2s:cls:_in_-resnet_640_ext_reorg_roi_g2_16_53-batch_48-gxe/ckpt-15876-ext_reorg_roi_g2_54_126/csv-batch_32:_out_-p2s-resnet_640-g2_16_53-g2_54_126-batch_6-cls show_vis=0 load_gt=1
<a id="batch_4_scratch___16_53_buggy_resnet_640_"></a>
### batch_4-scratch       @ 16_53-buggy/resnet-640-->eval_p2s-ipsc
<a id="on_g2_0_15___batch_4_scratch_16_53_buggy_resnet_640_"></a>
#### on-g2_0_15       @ batch_4-scratch/16_53-buggy/resnet-640-->eval_p2s-ipsc
python3 eval_det.py cfg=p2s,ipsc:ext:g2_0_15-agn:_in_-resnet_640_ext_reorg_roi_g2_16_53-batch_4-scratch/ckpt-4116-ext_reorg_roi_g2_0_15/csv-batch_32:_out_-p2s-resnet_640-g2_16_53-g2_0_15-batch_4-scratch show_vis=0 load_gt=1
``cls``
python3 eval_det.py cfg=p2s,ipsc:ext:g2_0_15,p2s:cls:_in_-resnet_640_ext_reorg_roi_g2_16_53-batch_4-scratch/ckpt-4116-ext_reorg_roi_g2_0_15/csv-batch_32:_out_-p2s-resnet_640-g2_16_53-g2_0_15-batch_4-scratch-cls show_vis=0 load_gt=0
<a id="on_g2_54_126___batch_4_scratch_16_53_buggy_resnet_640_"></a>
#### on-g2_54_126       @ batch_4-scratch/16_53-buggy/resnet-640-->eval_p2s-ipsc
python3 eval_det.py cfg=p2s,ipsc:ext:g2_54_126-agn:_in_-resnet_640_ext_reorg_roi_g2_16_53-batch_48-gxe/ckpt-4116-ext_reorg_roi_g2_54_126/csv-batch_32:_out_-p2s-resnet_640-g2_16_53-g2_54_126-batch_4-scratch show_vis=0 load_gt=1
``cls``
python3 eval_det.py cfg=p2s,ipsc:ext:g2_54_126,p2s:cls:_in_-resnet_640_ext_reorg_roi_g2_16_53-batch_48-gxe/ckpt-4116-ext_reorg_roi_g2_54_126/csv-batch_32:_out_-p2s-resnet_640-g2_16_53-g2_54_126-batch_4-scratch-cls show_vis=0 load_gt=1

<a id="16_53___resnet_640_"></a>
## 16_53       @ resnet-640-->eval_p2s-ipsc
python3 eval_det.py cfg=p2s,ipsc:16_53:agn:nms:gt-0:_in_-resnet_640_ext_reorg_roi_g2-frame-16_53-batch_18/ckpt-353730-ext_reorg_roi_g2-54_126/csv-batch_36:_out_-p2s-resnet_640-ipsc-640-16_53-54_126-353730


<a id="16_53_aug___resnet_640_"></a>
## 16_53-aug       @ resnet-640-->eval_p2s-ipsc
<a id="on_0_15___16_53_aug_resnet_640_"></a>
### on-0_15       @ 16_53-aug/resnet-640-->eval_p2s-ipsc
python3 eval_det.py cfg=p2s,ipsc:0_15:agn:nms-1:gt-1:show-0:_in_-resnet_640_ext_reorg_roi_g2-16_53-batch_18-jtr-res_1280/ckpt-312000-ext_reorg_roi_g2-0_15/csv-batch_2:_out_-p2s-ipsc-16_53-aug-0_15
<a id="on_54_126___16_53_aug_resnet_640_"></a>
### on-54_126       @ 16_53-aug/resnet-640-->eval_p2s-ipsc
python3 eval_det.py cfg=p2s,ipsc:54_126:agn:nms-1:gt-1:show-0:_in_-resnet_640_ext_reorg_roi_g2-16_53-batch_18-jtr-res_1280/ckpt-312000-ext_reorg_roi_g2-54_126/csv-batch_2:_out_-p2s-ipsc-16_53-aug
<a id="acc___on_54_126_16_53_aug_resnet_640_"></a>
#### acc       @ on-54_126/16_53-aug/resnet-640-->eval_p2s-ipsc
python3 eval_det.py cfg=p2s,ipsc:54_126:agn:nms:gt-0:_in_-resnet_640_ext_reorg_roi_g2-16_53-batch_18-jtr-res_1280/best-val-accuracy_notpad/ckpt-65195-ext_reorg_roi_g2-54_126/csv-batch_16:_out_-p2s-resnet_640-ipsc-640-16_53-54_126-jtr-res_1280-65195-acc

<a id="16_53_aug_retrain___resnet_640_"></a>
## 16_53-aug-retrain       @ resnet-640-->eval_p2s-ipsc
<a id="on_0_15___16_53_aug_retrain_resnet_640_"></a>
### on-0_15       @ 16_53-aug-retrain/resnet-640-->eval_p2s-ipsc
python eval_det.py cfg=p2s,ipsc:0_15:det-0:gt-1:nms-1:agn:proc-1:_in_-resnet_640_ext_reorg_roi_g2-16_53-batch_18-jtr-res_1280-retrain/ckpt-__var__-ext_reorg_roi_g2-0_15/csv-batch_2:_out_-p2s-ipsc-16_53-aug-retrain


<a id="16_53_aug_fbb___resnet_640_"></a>
## 16_53-aug-fbb       @ resnet-640-->eval_p2s-ipsc
<a id="on_0_15___16_53_aug_fbb_resnet_640_"></a>
### on-0_15       @ 16_53-aug-fbb/resnet-640-->eval_p2s-ipsc
python eval_det.py cfg=p2s,ipsc:0_15:det-0:gt-1:nms-1:agn:proc-1:_in_-resnet_640_ext_reorg_roi_g2-16_53-batch_48-jtr-res_1280-fbb/ckpt-__var__-ext_reorg_roi_g2-0_15/csv-batch_2:_out_-p2s-ipsc-16_53-aug-fbb


<a id="0_37_gxe___resnet_640_"></a>
## 0_37-gxe       @ resnet-640-->eval_p2s-ipsc
<a id="on_g2_38_53___0_37_gxe_resnet_64_0_"></a>
### on-g2_38_53       @ 0_37-gxe/resnet-640-->eval_p2s-ipsc
python3 eval_det.py cfg=p2s,ipsc:ext:g2_38_53-agn:_in_-resnet_640_ext_reorg_roi_g2_0_37-batch_48-gxe/ckpt-98175-ext_reorg_roi_g2_38_53/csv-batch_16:_out_-p2s-resnet_640-g2_0_37-g2_38_53-batch_48-98175 show_vis=0 load_gt=0
``cls``
python3 eval_det.py cfg=p2s,ipsc:ext:g2_38_53-agn,p2s:cls:_in_-resnet_640_ext_reorg_roi_g2_0_37-batch_48-gxe/ckpt-98175-ext_reorg_roi_g2_38_53/csv-batch_16:_out_-p2s-resnet_640-g2_0_37-g2_38_53-batch_48-98175-cls show_vis=0 load_gt=0
<a id="on_g2_38_53_conf_0___0_37_gxe_resnet_64_0_"></a>
### on-g2_38_53-conf_0       @ 0_37-gxe/resnet-640-->eval_p2s-ipsc
python3 eval_det.py cfg=p2s,ipsc:ext:g2_38_53-agn:_in_-resnet_640_ext_reorg_roi_g2_0_37-batch_48-gxe/ckpt-98175-ext_reorg_roi_g2_38_53/csv-batch_16-conf_0:_out_-p2s-resnet_640-g2_0_37-g2_38_53-batch_48-98175-conf_0 show_vis=0 load_gt=0
``cls``
python3 eval_det.py cfg=p2s,ipsc:ext:g2_38_53-agn,p2s:cls:_in_-resnet_640_ext_reorg_roi_g2_0_37-batch_48-gxe/ckpt-98175-ext_reorg_roi_g2_38_53/csv-batch_16-conf_0:_out_-p2s-resnet_640-g2_0_37-g2_38_53-batch_48-98175-conf_0-cls show_vis=0 load_gt=0


<a id="0_37___resnet_640_"></a>
## 0_37       @ resnet-640-->eval_p2s-ipsc
python3 eval_det.py cfg=p2s,ipsc:0_37:agn:nms-0:gt-0:_in_-resnet_640_ext_reorg_roi_g2-0_37-batch_18/ckpt-119860-ext_reorg_roi_g2-54_126/csv-batch_36 :_out_-p2s-resnet_640-ipsc-640-0_37-54_126-119860


<a id="0_37_aug___resnet_640_"></a>
## 0_37-aug       @ resnet-640-->eval_p2s-ipsc
python3 eval_det.py cfg=p2s,ipsc:54_126:agn:nms-1:gt-0:_in_-resnet_640_ext_reorg_roi_g2-0_37-batch_18-jtr-res_1280/ckpt-310440-ext_reorg_roi_g2-54_126/csv-batch_2:_out_-p2s-ipsc-0_37-aug
<a id="acc___0_37_aug_resnet_64_0_"></a>
### acc       @ 0_37-aug/resnet-640-->eval_p2s-ipsc
python3 eval_det.py cfg=p2s,ipsc:54_126:agn:nms:gt-0:_in_-resnet_640_ext_reorg_roi_g2-0_37-batch_18-jtr-res_1280/best-val-accuracy_notpad/ckpt-10270-ext_reorg_roi_g2-54_126/csv-batch_16:_out_-p2s-resnet_640-ipsc-640-0_37-54_126-jtr-res_1280-10270-acc


<a id="54_126_aug___resnet_640_"></a>
## 54_126-aug       @ resnet-640-->eval_p2s-ipsc
<a id="on_0_15___54_126_aug_resnet_64_0_"></a>
### on-0_15       @ 54_126-aug/resnet-640-->eval_p2s-ipsc
python eval_det.py cfg=p2s,ipsc:0_15:det-0:gt-1:nms-1:agn:proc-1:_in_-resnet_640_ext_reorg_roi_g2-54_126-batch_18-jtr-res_1280/ckpt-__var__-ext_reorg_roi_g2-0_15/csv-batch_2:_out_-p2s-ipsc-54_126-aug


<a id="54_126_aug_fbb___resnet_640_"></a>
## 54_126-aug-fbb       @ resnet-640-->eval_p2s-ipsc
<a id="on_0_15___54_126_aug_fbb_resnet_64_0_"></a>
### on-0_15       @ 54_126-aug-fbb/resnet-640-->eval_p2s-ipsc
python eval_det.py cfg=p2s,ipsc:0_15:det-0:gt-1:nms-1:agn:proc-1:_in_-resnet_640_ext_reorg_roi_g2-54_126-batch_48-jtr-res_1280-fbb/ckpt-__var__-ext_reorg_roi_g2-0_15/csv-batch_2:_out_-p2s-ipsc-54_126-aug-fbb

<a id="54_126_aug_fbb_cls_eq___resnet_640_"></a>
## 54_126-aug-fbb-cls_eq       @ resnet-640-->eval_p2s-ipsc
<a id="on_0_15___54_126_aug_fbb_cls_eq_resnet_640_"></a>
### on-0_15       @ 54_126-aug-fbb-cls_eq/resnet-640-->eval_p2s-ipsc
python eval_det.py cfg=p2s,ipsc:0_15:det-0:gt-1:nms-1:agn:proc-1:_in_-resnet_640_ext_reorg_roi_g2-54_126-batch_48-jtr-res_1280-fbb-cls_eq/ckpt-__var__-ext_reorg_roi_g2-0_15/csv-batch_4:_out_-p2s-ipsc-54_126-aug-fbb-cls_eq


<a id="pt___resnet_640_"></a>
## pt       @ resnet-640-->eval_p2s-ipsc
<a id="on_g2_0_1___pt_resnet_64_0_"></a>
### on-g2_0_1       @ pt/resnet-640-->eval_p2s-ipsc
<a id="resnet_640___ext_reorg_roi_g2_0_1_p2s_"></a>
<a id="batch_2___resnet_640_ext_reorg_roi_g2_0_1_p2_s_"></a>
``batch-2 ``  
python3 eval_det.py cfg=p2s,ipsc:ext:g2_0_1,p2s:pt-640:_in_-ckpt-74844-ext_reorg_roi_g2_0_1/csv-batch-2:_out_-p2s-resnet_640-0_1-batch-2 show_vis=1
``batch-48``  
python3 eval_det.py cfg=p2s,ipsc:ext:g2_0_1,p2s:pt-640:_in_-ckpt-74844-ext_reorg_roi_g2_0_1/csv-batch-48:_out_-p2s-resnet_640-0_1-batch_48 show_vis=1
<a id="on_g2_16_53___pt_resnet_64_0_"></a>
### on-g2_16_53       @ pt/resnet-640-->eval_p2s-ipsc
python3 eval_det.py cfg=p2s,ipsc:ext:g2_16_53,p2s:pt-640:_in_-ckpt-74844-ext_reorg_roi_g2_16_53/csv-batch_64:_out_-p2s-resnet_640-g2_16_53-batch_64
<a id="on_g2_0_15___pt_resnet_64_0_"></a>
### on-g2_0_15       @ pt/resnet-640-->eval_p2s-ipsc
python3 eval_det.py cfg=p2s,ipsc:ext:g2_0_15,p2s:pt-640:_in_-ckpt-74844-ext_reorg_roi_g2_54_126/csv-batch_32:_out_-p2s-resnet_640-g2_0_15
<a id="on_g2_54_126___pt_resnet_64_0_"></a>
### on-g2_54_126       @ pt/resnet-640-->eval_p2s-ipsc
python3 eval_det.py cfg=p2s,ipsc:ext:g2_54_126,p2s:pt-640:_in_-ckpt-74844-ext_reorg_roi_g2_54_126/csv-batch_32:_out_-p2s-resnet_640-g2_54_126-batch_32


<a id="resnet_1333___p2_s_"></a>
# resnet_1333       @ p2s-->eval_det_p2s
<a id="g2_16_53___resnet_133_3_"></a>
## g2_16_53       @ resnet_1333-->eval_p2s-ipsc
python3 eval_det.py cfg=p2s,ipsc:ext:g2_16_53,p2s:pt-1333:_in_-ckpt-93324-ext_reorg_roi_g2_16_53/csv-batch_24:_out_-p2s-resnet_1333-g2_16_53-batch_24


<a id="resnet_c4_640___p2_s_"></a>
# resnet-c4-640       @ p2s-->eval_det_p2s
<a id="g2_16_53___resnet_c4_64_0_"></a>
## g2_16_53       @ resnet-c4-640-->eval_p2s-ipsc
python3 eval_det.py cfg=p2s,ipsc:ext:g2_16_53,p2s:pt-c4-640:_in_-pretrained/resnet_c4_640/ckpt-56364-ext_reorg_roi_g2_16_53/csv-batch_16:_out_-p2s-resnet_c4_640-g2_16_53-batch_16 class_agnostic=1 enable_mask=0

<a id="resnet_c4_1333___p2_s_"></a>
# resnet-c4-1333       @ p2s-->eval_det_p2s
<a id="g2_0_1___resnet_c4_1333_"></a>
## g2_0_1       @ resnet-c4-1333-->eval_p2s-ipsc
python3 eval_det.py cfg=p2s,ipsc:ext:g2_0_1,p2s,p2s:pt-c4-1333:_in_-pretrained/resnet_c4_1333/ckpt-112728-ext_reorg_roi_g2_0_1/csv-batch_1:_out_-p2s-resnet_c4_1333-g2_0_1-batch_1 class_agnostic=1 enable_mask=0 show_vis=0

