<!-- MarkdownTOC -->

- [resnet-640       @ p2s](#resnet_640___p2_s_)
    - [detrac-0_19       @ resnet-640](#detrac_0_19___resnet_640_)
        - [0_19       @ detrac-0_19/resnet-640](#0_19___detrac_0_19_resnet_640_)
        - [49_68       @ detrac-0_19/resnet-640](#49_68___detrac_0_19_resnet_640_)
    - [mnist-640-5       @ resnet-640](#mnist_640_5___resnet_640_)
        - [test       @ mnist-640-5/resnet-640](#test___mnist_640_5_resnet_640_)
            - [agn       @ test/mnist-640-5/resnet-640](#agn___test_mnist_640_5_resnet_64_0_)
    - [ipsc-16_53       @ resnet-640](#ipsc_16_53___resnet_640_)
    - [ipsc-0_37       @ resnet-640](#ipsc_0_37___resnet_640_)
    - [ipsc-16_53-aug       @ resnet-640](#ipsc_16_53_aug___resnet_640_)
        - [acc       @ ipsc-16_53-aug/resnet-640](#acc___ipsc_16_53_aug_resnet_64_0_)
    - [ipsc-0_37-aug       @ resnet-640](#ipsc_0_37_aug___resnet_640_)
        - [acc       @ ipsc-0_37-aug/resnet-640](#acc___ipsc_0_37_aug_resnet_640_)
    - [ipsc-16_53-buggy       @ resnet-640](#ipsc_16_53_buggy___resnet_640_)
        - [batch_48       @ ipsc-16_53-buggy/resnet-640](#batch_48___ipsc_16_53_buggy_resnet_64_0_)
            - [ckpt-1975       @ batch_48/ipsc-16_53-buggy/resnet-640](#ckpt_1975___batch_48_ipsc_16_53_buggy_resnet_640_)
                - [on-g2_0_15       @ ckpt-1975/batch_48/ipsc-16_53-buggy/resnet-640](#on_g2_0_15___ckpt_1975_batch_48_ipsc_16_53_buggy_resnet_640_)
                - [on-g2_54_126       @ ckpt-1975/batch_48/ipsc-16_53-buggy/resnet-640](#on_g2_54_126___ckpt_1975_batch_48_ipsc_16_53_buggy_resnet_640_)
            - [ckpt-12275       @ batch_48/ipsc-16_53-buggy/resnet-640](#ckpt_12275___batch_48_ipsc_16_53_buggy_resnet_640_)
                - [on-g2_0_15       @ ckpt-12275/batch_48/ipsc-16_53-buggy/resnet-640](#on_g2_0_15___ckpt_12275_batch_48_ipsc_16_53_buggy_resnet_64_0_)
        - [batch_32       @ ipsc-16_53-buggy/resnet-640](#batch_32___ipsc_16_53_buggy_resnet_64_0_)
            - [on-g2_0_15       @ batch_32/ipsc-16_53-buggy/resnet-640](#on_g2_0_15___batch_32_ipsc_16_53_buggy_resnet_640_)
        - [batch_6       @ ipsc-16_53-buggy/resnet-640](#batch_6___ipsc_16_53_buggy_resnet_64_0_)
            - [on-g2_0_15       @ batch_6/ipsc-16_53-buggy/resnet-640](#on_g2_0_15___batch_6_ipsc_16_53_buggy_resnet_64_0_)
            - [on-g2_54_126       @ batch_6/ipsc-16_53-buggy/resnet-640](#on_g2_54_126___batch_6_ipsc_16_53_buggy_resnet_64_0_)
        - [batch_4-scratch       @ ipsc-16_53-buggy/resnet-640](#batch_4_scratch___ipsc_16_53_buggy_resnet_64_0_)
            - [on-g2_0_15       @ batch_4-scratch/ipsc-16_53-buggy/resnet-640](#on_g2_0_15___batch_4_scratch_ipsc_16_53_buggy_resnet_64_0_)
            - [on-g2_54_126       @ batch_4-scratch/ipsc-16_53-buggy/resnet-640](#on_g2_54_126___batch_4_scratch_ipsc_16_53_buggy_resnet_64_0_)
    - [g2_0_37       @ resnet-640](#g2_0_37___resnet_640_)
        - [on-g2_38_53       @ g2_0_37/resnet-640](#on_g2_38_53___g2_0_37_resnet_640_)
        - [on-g2_38_53-conf_0       @ g2_0_37/resnet-640](#on_g2_38_53_conf_0___g2_0_37_resnet_640_)
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
<a id="resnet_640___p2_s_"></a>
# resnet-640       @ p2s-->eval_det_p2s
<a id="detrac_0_19___resnet_640_"></a>
## detrac-0_19       @ resnet-640-->eval_p2s
<a id="49_68___detrac_0_9_resnet_640_vi_d_"></a>
<a id="0_19___detrac_0_19_resnet_640_"></a>
### 0_19       @ detrac-0_19/resnet-640-->eval_p2s
python3 eval_det.py cfg=p2s,detrac:non_empty-0_19:nms:gt-1:show-0:proc-12:_d_-resnet_640_detrac-non_empty-seq-0_19-batch_18/ckpt-242990-detrac-non_empty-seq-0_19/csv-batch_48:_s_-p2s-resnet_640-detrac-0_19-0_19-242990
<a id="49_68___detrac_0_19_resnet_640_"></a>
### 49_68       @ detrac-0_19/resnet-640-->eval_p2s
python3 eval_det.py cfg=p2s,detrac:non_empty:49_68:nms:gt-1:show-0:_d_-resnet_640_detrac-non_empty-seq-0_19-batch_18/ckpt-242990-detrac-non_empty-seq-49_68/csv-batch_48:_s_-p2s-resnet_640-detrac-0_19-49_68-242990

<a id="mnist_640_5___resnet_640_"></a>
## mnist-640-5       @ resnet-640-->eval_p2s
<a id="test___mnist_640_5_resnet_640_"></a>
### test       @ mnist-640-5/resnet-640-->eval_p2s
python3 eval_det.py cfg=mnist:640-5:12_1000:test:nms:gt-1,p2s:_d_-resnet_640_mnist_640_5_12_1000_var-train-batch_18/ckpt-174087-mnist_640_5_12_1000_var-test/csv-batch_96:_s_-p2s-resnet_640-mnist-640-5-12-test-174087
<a id="agn___test_mnist_640_5_resnet_64_0_"></a>
#### agn       @ test/mnist-640-5/resnet-640-->eval_p2s
python3 eval_det.py cfg=mnist:640-5:12_1000:test:nms:det-0:gt-0:agn,p2s:_d_-resnet_640_mnist_640_5_12_1000_var-train-batch_18/ckpt-174087-mnist_640_5_12_1000_var-test/csv-batch_96:_s_-p2s-resnet_640-mnist-640-5-12-test-174087-agn

<a id="ipsc_16_53___resnet_640_"></a>
## ipsc-16_53       @ resnet-640-->eval_p2s
python3 eval_det.py cfg=ipsc:16_53:agn:nms:gt-0,p2s:_d_-resnet_640_ext_reorg_roi_g2-frame-16_53-batch_18/ckpt-353730-ext_reorg_roi_g2-54_126/csv-batch_36:_s_-p2s-resnet_640-ipsc-640-16_53-54_126-353730

<a id="ipsc_0_37___resnet_640_"></a>
## ipsc-0_37       @ resnet-640-->eval_p2s
python3 eval_det.py cfg=ipsc:0_37:agn:nms-0:gt-0,p2s:_d_-resnet_640_ext_reorg_roi_g2-0_37-batch_18/ckpt-119860-ext_reorg_roi_g2-54_126/csv-batch_36 :_s_-p2s-resnet_640-ipsc-640-0_37-54_126-119860

<a id="ipsc_16_53_aug___resnet_640_"></a>
## ipsc-16_53-aug       @ resnet-640-->eval_p2s
python3 eval_det.py cfg=ipsc:54_126:agn:nms:gt-0:show-0,p2s:_d_-resnet_640_ext_reorg_roi_g2-16_53-batch_18-jtr-res_1280/ckpt-312000-ext_reorg_roi_g2-54_126/csv-batch_16:_s_-p2s-resnet_640-ipsc-640-16_53-54_126-jtr-res_1280-312000
<a id="acc___ipsc_16_53_aug_resnet_64_0_"></a>
### acc       @ ipsc-16_53-aug/resnet-640-->eval_p2s
python3 eval_det.py cfg=ipsc:54_126:agn:nms:gt-0,p2s:_d_-resnet_640_ext_reorg_roi_g2-16_53-batch_18-jtr-res_1280/best-val-accuracy_notpad/ckpt-65195-ext_reorg_roi_g2-54_126/csv-batch_16:_s_-p2s-resnet_640-ipsc-640-16_53-54_126-jtr-res_1280-65195-acc

<a id="ipsc_0_37_aug___resnet_640_"></a>
## ipsc-0_37-aug       @ resnet-640-->eval_p2s
python3 eval_det.py cfg=ipsc:54_126:agn:nms:gt-0,p2s:_d_-resnet_640_ext_reorg_roi_g2-0_37-batch_18-jtr-res_1280/ckpt-310440-ext_reorg_roi_g2-54_126/csv-batch_16:_s_-p2s-resnet_640-ipsc-640-0_37-54_126-jtr-res_1280-310440
<a id="acc___ipsc_0_37_aug_resnet_640_"></a>
### acc       @ ipsc-0_37-aug/resnet-640-->eval_p2s
python3 eval_det.py cfg=ipsc:54_126:agn:nms:gt-0,p2s:_d_-resnet_640_ext_reorg_roi_g2-0_37-batch_18-jtr-res_1280/best-val-accuracy_notpad/ckpt-10270-ext_reorg_roi_g2-54_126/csv-batch_16:_s_-p2s-resnet_640-ipsc-640-0_37-54_126-jtr-res_1280-10270-acc

<a id="ipsc_16_53_buggy___resnet_640_"></a>
## ipsc-16_53-buggy       @ resnet-640-->eval_p2s
<a id="batch_48___ipsc_16_53_buggy_resnet_64_0_"></a>
### batch_48       @ ipsc-16_53-buggy/resnet-640-->eval_p2s
<a id="ckpt_1975___batch_48_ipsc_16_53_buggy_resnet_640_"></a>
#### ckpt-1975       @ batch_48/ipsc-16_53-buggy/resnet-640-->eval_p2s
<a id="on_g2_0_15___ckpt_1975_batch_48_ipsc_16_53_buggy_resnet_640_"></a>
##### on-g2_0_15       @ ckpt-1975/batch_48/ipsc-16_53-buggy/resnet-640-->eval_p2s
python3 eval_det.py cfg=ipsc:ext:g2_0_15-agn,p2s:_d_-resnet_640_ext_reorg_roi_g2_16_53-batch_48-gxe/ckpt-1975-ext_reorg_roi_g2_0_15/csv-batch_16:_s_-p2s-resnet_640-g2_16_53-g2_0_15-batch_48-1975 show_vis=0 load_gt=0
``cls``
python3 eval_det.py cfg=ipsc:ext:g2_0_15,p2s:cls:_d_-resnet_640_ext_reorg_roi_g2_16_53-batch_48-gxe/ckpt-1975-ext_reorg_roi_g2_0_15/csv-batch_16:_s_-p2s-resnet_640-g2_16_53-g2_0_15-batch_48-1975-cls show_vis=0 load_gt=0
<a id="on_g2_54_126___ckpt_1975_batch_48_ipsc_16_53_buggy_resnet_640_"></a>
##### on-g2_54_126       @ ckpt-1975/batch_48/ipsc-16_53-buggy/resnet-640-->eval_p2s
python3 eval_det.py cfg=ipsc:ext:g2_54_126-agn,p2s:_d_-resnet_640_ext_reorg_roi_g2_16_53-batch_48-gxe/ckpt-1975-ext_reorg_roi_g2_54_126/csv-batch_32:_s_-p2s-resnet_640-g2_16_53-g2_54_126-batch_48 show_vis=0
``cls``
python3 eval_det.py cfg=ipsc:ext:g2_54_126,p2s:cls:_d_-resnet_640_ext_reorg_roi_g2_16_53-batch_48-gxe/ckpt-1975-ext_reorg_roi_g2_54_126/csv-batch_32:_s_-p2s-resnet_640-g2_16_53-g2_54_126-batch_48-cls show_vis=0 load_gt=0

<a id="ckpt_12275___batch_48_ipsc_16_53_buggy_resnet_640_"></a>
#### ckpt-12275       @ batch_48/ipsc-16_53-buggy/resnet-640-->eval_p2s
<a id="on_g2_0_15___ckpt_12275_batch_48_ipsc_16_53_buggy_resnet_64_0_"></a>
##### on-g2_0_15       @ ckpt-12275/batch_48/ipsc-16_53-buggy/resnet-640-->eval_p2s
python3 eval_det.py cfg=ipsc:ext:g2_0_15-agn,p2s:fill:_d_-resnet_640_ext_reorg_roi_g2_16_53-batch_48-gxe/ckpt-12275-ext_reorg_roi_g2_0_15/csv-batch_32:_s_-p2s-resnet_640-g2_16_53-g2_0_15-batch_48-12275 show_vis=0 load_gt=0
``cls``
python3 eval_det.py cfg=ipsc:ext:g2_0_15,p2s:cls:fill:_d_-resnet_640_ext_reorg_roi_g2_16_53-batch_48-gxe/ckpt-12275-ext_reorg_roi_g2_0_15/csv-batch_32:_s_-p2s-resnet_640-g2_16_53-g2_0_15-batch_48-12275-cls show_vis=0 load_gt=0

<a id="batch_32___ipsc_16_53_buggy_resnet_64_0_"></a>
### batch_32       @ ipsc-16_53-buggy/resnet-640-->eval_p2s
<a id="on_g2_0_15___batch_32_ipsc_16_53_buggy_resnet_640_"></a>
#### on-g2_0_15       @ batch_32/ipsc-16_53-buggy/resnet-640-->eval_p2s
python3 eval_det.py cfg=ipsc:ext:g2_0_15-agn,p2s:fill:_d_-resnet_640_ext_reorg_roi_g2_16_53-batch_32-xe/ckpt-2960-ext_reorg_roi_g2_0_15/csv-batch_64:_s_-p2s-resnet_640-g2_16_53-g2_0_15-batch_32 show_vis=0 load_gt=0
``cls``
python3 eval_det.py cfg=ipsc:ext:g2_0_15,p2s:cls:fill:_d_-resnet_640_ext_reorg_roi_g2_16_53-batch_32-xe/ckpt-2960-ext_reorg_roi_g2_0_15/csv-batch_64:_s_-p2s-resnet_640-g2_16_53-g2_0_15-batch_32-cls show_vis=0 load_gt=0

<a id="batch_6___ipsc_16_53_buggy_resnet_64_0_"></a>
### batch_6       @ ipsc-16_53-buggy/resnet-640-->eval_p2s
<a id="on_g2_0_15___batch_6_ipsc_16_53_buggy_resnet_64_0_"></a>
#### on-g2_0_15       @ batch_6/ipsc-16_53-buggy/resnet-640-->eval_p2s
python3 eval_det.py cfg=ipsc:ext:g2_0_15-agn,p2s:_d_-resnet_640_ext_reorg_roi_g2_16_53-batch_6/ckpt-15876-ext_reorg_roi_g2_0_15/csv-batch_32:_s_-p2s-resnet_640-g2_16_53-g2_0_15-batch_6 show_vis=0 load_gt=0
``cls``
python3 eval_det.py cfg=ipsc:ext:g2_0_15,p2s:cls:_d_-resnet_640_ext_reorg_roi_g2_16_53-batch_6/ckpt-15876-ext_reorg_roi_g2_0_15/csv-batch_32:_s_-p2s-resnet_640-g2_16_53-g2_0_15-batch_6-cls show_vis=0 load_gt=0
<a id="on_g2_54_126___batch_6_ipsc_16_53_buggy_resnet_64_0_"></a>
#### on-g2_54_126       @ batch_6/ipsc-16_53-buggy/resnet-640-->eval_p2s
python3 eval_det.py cfg=ipsc:ext:g2_54_126-agn,p2s:_d_-resnet_640_ext_reorg_roi_g2_16_53-batch_48-gxe/ckpt-15876-ext_reorg_roi_g2_54_126/csv-batch_32:_s_-p2s-resnet_640-g2_16_53-g2_54_126-batch_6 show_vis=0 load_gt=1
``cls``
python3 eval_det.py cfg=ipsc:ext:g2_54_126,p2s:cls:_d_-resnet_640_ext_reorg_roi_g2_16_53-batch_48-gxe/ckpt-15876-ext_reorg_roi_g2_54_126/csv-batch_32:_s_-p2s-resnet_640-g2_16_53-g2_54_126-batch_6-cls show_vis=0 load_gt=1
<a id="batch_4_scratch___ipsc_16_53_buggy_resnet_64_0_"></a>
### batch_4-scratch       @ ipsc-16_53-buggy/resnet-640-->eval_p2s
<a id="on_g2_0_15___batch_4_scratch_ipsc_16_53_buggy_resnet_64_0_"></a>
#### on-g2_0_15       @ batch_4-scratch/ipsc-16_53-buggy/resnet-640-->eval_p2s
python3 eval_det.py cfg=ipsc:ext:g2_0_15-agn,p2s:_d_-resnet_640_ext_reorg_roi_g2_16_53-batch_4-scratch/ckpt-4116-ext_reorg_roi_g2_0_15/csv-batch_32:_s_-p2s-resnet_640-g2_16_53-g2_0_15-batch_4-scratch show_vis=0 load_gt=1
``cls``
python3 eval_det.py cfg=ipsc:ext:g2_0_15,p2s:cls:_d_-resnet_640_ext_reorg_roi_g2_16_53-batch_4-scratch/ckpt-4116-ext_reorg_roi_g2_0_15/csv-batch_32:_s_-p2s-resnet_640-g2_16_53-g2_0_15-batch_4-scratch-cls show_vis=0 load_gt=0
<a id="on_g2_54_126___batch_4_scratch_ipsc_16_53_buggy_resnet_64_0_"></a>
#### on-g2_54_126       @ batch_4-scratch/ipsc-16_53-buggy/resnet-640-->eval_p2s
python3 eval_det.py cfg=ipsc:ext:g2_54_126-agn,p2s:_d_-resnet_640_ext_reorg_roi_g2_16_53-batch_48-gxe/ckpt-4116-ext_reorg_roi_g2_54_126/csv-batch_32:_s_-p2s-resnet_640-g2_16_53-g2_54_126-batch_4-scratch show_vis=0 load_gt=1
``cls``
python3 eval_det.py cfg=ipsc:ext:g2_54_126,p2s:cls:_d_-resnet_640_ext_reorg_roi_g2_16_53-batch_48-gxe/ckpt-4116-ext_reorg_roi_g2_54_126/csv-batch_32:_s_-p2s-resnet_640-g2_16_53-g2_54_126-batch_4-scratch-cls show_vis=0 load_gt=1

<a id="g2_0_37___resnet_640_"></a>
## g2_0_37       @ resnet-640-->eval_p2s
<a id="on_g2_38_53___g2_0_37_resnet_640_"></a>
### on-g2_38_53       @ g2_0_37/resnet-640-->eval_p2s
python3 eval_det.py cfg=ipsc:ext:g2_38_53-agn,p2s:_d_-resnet_640_ext_reorg_roi_g2_0_37-batch_48-gxe/ckpt-98175-ext_reorg_roi_g2_38_53/csv-batch_16:_s_-p2s-resnet_640-g2_0_37-g2_38_53-batch_48-98175 show_vis=0 load_gt=0
``cls``
python3 eval_det.py cfg=ipsc:ext:g2_38_53-agn,p2s:cls:_d_-resnet_640_ext_reorg_roi_g2_0_37-batch_48-gxe/ckpt-98175-ext_reorg_roi_g2_38_53/csv-batch_16:_s_-p2s-resnet_640-g2_0_37-g2_38_53-batch_48-98175-cls show_vis=0 load_gt=0
<a id="on_g2_38_53_conf_0___g2_0_37_resnet_640_"></a>
### on-g2_38_53-conf_0       @ g2_0_37/resnet-640-->eval_p2s
python3 eval_det.py cfg=ipsc:ext:g2_38_53-agn,p2s:_d_-resnet_640_ext_reorg_roi_g2_0_37-batch_48-gxe/ckpt-98175-ext_reorg_roi_g2_38_53/csv-batch_16-conf_0:_s_-p2s-resnet_640-g2_0_37-g2_38_53-batch_48-98175-conf_0 show_vis=0 load_gt=0
``cls``
python3 eval_det.py cfg=ipsc:ext:g2_38_53-agn,p2s:cls:_d_-resnet_640_ext_reorg_roi_g2_0_37-batch_48-gxe/ckpt-98175-ext_reorg_roi_g2_38_53/csv-batch_16-conf_0:_s_-p2s-resnet_640-g2_0_37-g2_38_53-batch_48-98175-conf_0-cls show_vis=0 load_gt=0

<a id="pt___resnet_640_"></a>
## pt       @ resnet-640-->eval_p2s
<a id="on_g2_0_1___pt_resnet_64_0_"></a>
### on-g2_0_1       @ pt/resnet-640-->eval_p2s
<a id="resnet_640___ext_reorg_roi_g2_0_1_p2s_"></a>
<a id="batch_2___resnet_640_ext_reorg_roi_g2_0_1_p2_s_"></a>
``batch-2 ``  
python3 eval_det.py cfg=ipsc:ext:g2_0_1,p2s:pt-640:_d_-ckpt-74844-ext_reorg_roi_g2_0_1/csv-batch-2:_s_-p2s-resnet_640-0_1-batch-2 show_vis=1
``batch-48``  
python3 eval_det.py cfg=ipsc:ext:g2_0_1,p2s:pt-640:_d_-ckpt-74844-ext_reorg_roi_g2_0_1/csv-batch-48:_s_-p2s-resnet_640-0_1-batch_48 show_vis=1
<a id="on_g2_16_53___pt_resnet_64_0_"></a>
### on-g2_16_53       @ pt/resnet-640-->eval_p2s
python3 eval_det.py cfg=ipsc:ext:g2_16_53,p2s:pt-640:_d_-ckpt-74844-ext_reorg_roi_g2_16_53/csv-batch_64:_s_-p2s-resnet_640-g2_16_53-batch_64
<a id="on_g2_0_15___pt_resnet_64_0_"></a>
### on-g2_0_15       @ pt/resnet-640-->eval_p2s
python3 eval_det.py cfg=ipsc:ext:g2_0_15,p2s:pt-640:_d_-ckpt-74844-ext_reorg_roi_g2_54_126/csv-batch_32:_s_-p2s-resnet_640-g2_0_15
<a id="on_g2_54_126___pt_resnet_64_0_"></a>
### on-g2_54_126       @ pt/resnet-640-->eval_p2s
python3 eval_det.py cfg=ipsc:ext:g2_54_126,p2s:pt-640:_d_-ckpt-74844-ext_reorg_roi_g2_54_126/csv-batch_32:_s_-p2s-resnet_640-g2_54_126-batch_32

<a id="resnet_1333___p2_s_"></a>
# resnet_1333       @ p2s-->eval_det_p2s
<a id="g2_16_53___resnet_133_3_"></a>
## g2_16_53       @ resnet_1333-->eval_p2s
python3 eval_det.py cfg=ipsc:ext:g2_16_53,p2s:pt-1333:_d_-ckpt-93324-ext_reorg_roi_g2_16_53/csv-batch_24:_s_-p2s-resnet_1333-g2_16_53-batch_24

<a id="resnet_c4_640___p2_s_"></a>
# resnet-c4-640       @ p2s-->eval_det_p2s
<a id="g2_16_53___resnet_c4_64_0_"></a>
## g2_16_53       @ resnet-c4-640-->eval_p2s
python3 eval_det.py cfg=ipsc:ext:g2_16_53,p2s:pt-c4-640:_d_-pretrained/resnet_c4_640/ckpt-56364-ext_reorg_roi_g2_16_53/csv-batch_16:_s_-p2s-resnet_c4_640-g2_16_53-batch_16 class_agnostic=1 enable_mask=0

<a id="resnet_c4_1333___p2_s_"></a>
# resnet-c4-1333       @ p2s-->eval_det_p2s
<a id="g2_0_1___resnet_c4_1333_"></a>
## g2_0_1       @ resnet-c4-1333-->eval_p2s
python3 eval_det.py cfg=ipsc:ext:g2_0_1,p2s,p2s:pt-c4-1333:_d_-pretrained/resnet_c4_1333/ckpt-112728-ext_reorg_roi_g2_0_1/csv-batch_1:_s_-p2s-resnet_c4_1333-g2_0_1-batch_1 class_agnostic=1 enable_mask=0 show_vis=0

