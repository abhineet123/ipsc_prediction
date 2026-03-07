<!-- MarkdownTOC -->

- [vit_l-640       @ p2s](#vit_l_640___p2_s_)
    - [vid_det_all-aug-fbb       @ vit_l-640](#vid_det_all_aug_fbb___vit_l_64_0_)
        - [train-ratio_1_10_random       @ vid_det_all-aug-fbb/vit_l-640](#train_ratio_1_10_random___vid_det_all_aug_fbb_vit_l_64_0_)
        - [val-16_per_seq_random       @ vid_det_all-aug-fbb/vit_l-640](#val_16_per_seq_random___vid_det_all_aug_fbb_vit_l_64_0_)
- [vit_b-640       @ p2s](#vit_b_640___p2_s_)
    - [vid_det-aug-fbb       @ vit_b-640](#vid_det_aug_fbb___vit_b_64_0_)
        - [train-ratio_1_10_random       @ vid_det-aug-fbb/vit_b-640](#train_ratio_1_10_random___vid_det_aug_fbb_vit_b_64_0_)
        - [val-16_per_seq_random       @ vid_det-aug-fbb/vit_b-640](#val_16_per_seq_random___vid_det_aug_fbb_vit_b_64_0_)
        - [val       @ vid_det-aug-fbb/vit_b-640](#val___vid_det_aug_fbb_vit_b_64_0_)
            - [bnms0       @ val/vid_det-aug-fbb/vit_b-640](#bnms0___val_vid_det_aug_fbb_vit_b_64_0_)
            - [fnms       @ val/vid_det-aug-fbb/vit_b-640](#fnms___val_vid_det_aug_fbb_vit_b_64_0_)
    - [vid-aug-fbb       @ vit_b-640](#vid_aug_fbb___vit_b_64_0_)
        - [train-8_per_seq_random       @ vid-aug-fbb/vit_b-640](#train_8_per_seq_random___vid_aug_fbb_vit_b_64_0_)
        - [val-16_per_seq_random       @ vid-aug-fbb/vit_b-640](#val_16_per_seq_random___vid_aug_fbb_vit_b_64_0_)
        - [val       @ vid-aug-fbb/vit_b-640](#val___vid_aug_fbb_vit_b_64_0_)
            - [batch_32       @ val/vid-aug-fbb/vit_b-640](#batch_32___val_vid_aug_fbb_vit_b_64_0_)
            - [batch_16       @ val/vid-aug-fbb/vit_b-640](#batch_16___val_vid_aug_fbb_vit_b_64_0_)
- [resnet-640       @ p2s](#resnet_640___p2_s_)
    - [vid-aug-fbb       @ resnet-640](#vid_aug_fbb___resnet_640_)
        - [train-8_per_seq_random       @ vid-aug-fbb/resnet-640](#train_8_per_seq_random___vid_aug_fbb_resnet_640_)
        - [val-16_per_seq_random       @ vid-aug-fbb/resnet-640](#val_16_per_seq_random___vid_aug_fbb_resnet_640_)
        - [val       @ vid-aug-fbb/resnet-640](#val___vid_aug_fbb_resnet_640_)
    - [vid_det-aug-fbb       @ resnet-640](#vid_det_aug_fbb___resnet_640_)
        - [train-8_per_seq_random       @ vid_det-aug-fbb/resnet-640](#train_8_per_seq_random___vid_det_aug_fbb_resnet_640_)
        - [train-ratio_1_10_random       @ vid_det-aug-fbb/resnet-640](#train_ratio_1_10_random___vid_det_aug_fbb_resnet_640_)
        - [val-16_per_seq_random       @ vid_det-aug-fbb/resnet-640](#val_16_per_seq_random___vid_det_aug_fbb_resnet_640_)
        - [val       @ vid_det-aug-fbb/resnet-640](#val___vid_det_aug_fbb_resnet_640_)
    - [vid_det-aug       @ resnet-640](#vid_det_aug___resnet_640_)
        - [train-ratio_1_10_random       @ vid_det-aug/resnet-640](#train_ratio_1_10_random___vid_det_aug_resnet_640_)
        - [val-16_per_seq_random       @ vid_det-aug/resnet-640](#val_16_per_seq_random___vid_det_aug_resnet_640_)
    - [vid_det-aug-isc       @ resnet-640](#vid_det_aug_isc___resnet_640_)
        - [train-ratio_1_10_random       @ vid_det-aug-isc/resnet-640](#train_ratio_1_10_random___vid_det_aug_isc_resnet_640_)
        - [val-16_per_seq_random       @ vid_det-aug-isc/resnet-640](#val_16_per_seq_random___vid_det_aug_isc_resnet_640_)
    - [vid_det_all-aug-fbb       @ resnet-640](#vid_det_all_aug_fbb___resnet_640_)
        - [train-ratio_1_10_random       @ vid_det_all-aug-fbb/resnet-640](#train_ratio_1_10_random___vid_det_all_aug_fbb_resnet_640_)
            - [batch_16       @ train-ratio_1_10_random/vid_det_all-aug-fbb/resnet-640](#batch_16___train_ratio_1_10_random_vid_det_all_aug_fbb_resnet_640_)
        - [val-16_per_seq_random       @ vid_det_all-aug-fbb/resnet-640](#val_16_per_seq_random___vid_det_all_aug_fbb_resnet_640_)
            - [batch_16       @ val-16_per_seq_random/vid_det_all-aug-fbb/resnet-640](#batch_16___val_16_per_seq_random_vid_det_all_aug_fbb_resnet_640_)

<!-- /MarkdownTOC -->

<a id="vit_l_640___p2_s_"></a>
# vit_l-640       @ p2s-->eval_det_p2s
<a id="vid_det_all_aug_fbb___vit_l_64_0_"></a>
## vid_det_all-aug-fbb       @ vit_l-640-->eval_p2s-imgn
<a id="train_ratio_1_10_random___vid_det_all_aug_fbb_vit_l_64_0_"></a>
### train-ratio_1_10_random       @ vid_det_all-aug-fbb/vit_l-640-->eval_p2s-imgn
python3 eval_det.py cfg=p2s,imgn:vid_det_all:ratio_1_10_random:nms-s5:gt-0:del-1:agn:proc-1:_in_-vit_b_640_imagenet_vid_det-sampled_eq-batch_40-jtr-res_1440-fbb-self2-0/ckpt-__var__-imagenet_vid_det-ratio_1_10_random/csv-batch_32:_out_-p2s-vit_b-imgn-vid_det-train-ratio_1_10_random-aug-fbb
<a id="val_16_per_seq_random___vid_det_all_aug_fbb_vit_l_64_0_"></a>
### val-16_per_seq_random       @ vid_det_all-aug-fbb/vit_l-640-->eval_p2s-imgn
python3 eval_det.py cfg=p2s,imgn:vid_val:16_per_seq_random:nms-s5:gt-0:del-1:agn:proc-1:_in_-vit_b_640_imagenet_vid_det-sampled_eq-batch_40-jtr-res_1440-fbb-self2-0/ckpt-__var__-imagenet_vid_val-16_per_seq_random/csv-batch_16:_out_-p2s-vit_b-imgn-vid_det-val-16_per_seq_random-aug-fbb

<a id="vit_b_640___p2_s_"></a>
# vit_b-640       @ p2s-->eval_det_p2s
<a id="vid_det_aug_fbb___vit_b_64_0_"></a>
## vid_det-aug-fbb       @ vit_b-640-->eval_p2s-imgn
<a id="train_ratio_1_10_random___vid_det_aug_fbb_vit_b_64_0_"></a>
### train-ratio_1_10_random       @ vid_det-aug-fbb/vit_b-640-->eval_p2s-imgn
python3 eval_det.py cfg=p2s,imgn:vid_det:ratio_1_10_random:nms-s5:gt-0:del-1:agn:proc-1:_in_-vit_b_640_imagenet_vid_det-sampled_eq-batch_40-jtr-res_1440-fbb-self2-0/ckpt-__var__-imagenet_vid_det-ratio_1_10_random/csv-batch_32:_out_-p2s-vit_b-imgn-vid_det-train-ratio_1_10_random-aug-fbb
<a id="val_16_per_seq_random___vid_det_aug_fbb_vit_b_64_0_"></a>
### val-16_per_seq_random       @ vid_det-aug-fbb/vit_b-640-->eval_p2s-imgn
python3 eval_det.py cfg=p2s,imgn:vid_val:16_per_seq_random:nms-s5:gt-0:del-1:agn:proc-1:_in_-vit_b_640_imagenet_vid_det-sampled_eq-batch_40-jtr-res_1440-fbb-self2-0/ckpt-__var__-imagenet_vid_val-16_per_seq_random/csv-batch_16:_out_-p2s-vit_b-imgn-vid_det-val-16_per_seq_random-aug-fbb
<a id="val___vid_det_aug_fbb_vit_b_64_0_"></a>
### val       @ vid_det-aug-fbb/vit_b-640-->eval_p2s-imgn
python3 eval_det.py cfg=p2s,imgn:vid_val:nms-s10:gt-0:del-1:agn:proc-1:_in_-vit_b_640_imagenet_vid_det-sampled_eq-batch_40-jtr-res_1440-fbb-self2-0/ckpt-__var__-imagenet_vid_val/csv-batch_32:_out_-p2s-vit_b-imgn-vid_det-val-aug-fbb
<a id="bnms0___val_vid_det_aug_fbb_vit_b_64_0_"></a>
#### bnms0       @ val/vid_det-aug-fbb/vit_b-640-->eval_p2s-imgn
python3 eval_det.py cfg=p2s,imgn:vid_val:nms-s10:gt-0:del-1:agn:proc-1:_in_-vit_b_640_imagenet_vid_det-sampled_eq-batch_40-jtr-res_1440-fbb-self2-0/ckpt-__var__-imagenet_vid_val/csv-batch_32:_out_-p2s-vit_b-imgn-vid_det-val-aug-fbb-bnms0:bnms0
<a id="fnms___val_vid_det_aug_fbb_vit_b_64_0_"></a>
#### fnms       @ val/vid_det-aug-fbb/vit_b-640-->eval_p2s-imgn
python3 eval_det.py cfg=p2s,imgn:vid_val:nms-s10:gt-0:del-1:agn:proc-1:_in_-vit_b_640_imagenet_vid_det-sampled_eq-batch_40-jtr-res_1440-fbb-self2-0/ckpt-__var__-imagenet_vid_val/csv-batch_32:_out_-p2s-vit_b-imgn-vid_det-val-aug-fbb-fnms:fnms

<a id="vid_aug_fbb___vit_b_64_0_"></a>
## vid-aug-fbb       @ vit_b-640-->eval_p2s-imgn
<a id="train_8_per_seq_random___vid_aug_fbb_vit_b_64_0_"></a>
### train-8_per_seq_random       @ vid-aug-fbb/vit_b-640-->eval_p2s-imgn
python3 eval_det.py cfg=p2s,imgn:vid:8_per_seq_random:nms-s5:gt-0:del-1:agn:proc-1:_in_-vit_b_640_imagenet_vid-batch_40-jtr-res_1440-fbb-self2-0/ckpt-__var__-imagenet_vid-8_per_seq_random/csv-batch_32:_out_-p2s-vit_b-imgn-vid-train-8_per_seq_random-aug-fbb
<a id="val_16_per_seq_random___vid_aug_fbb_vit_b_64_0_"></a>
### val-16_per_seq_random       @ vid-aug-fbb/vit_b-640-->eval_p2s-imgn
python3 eval_det.py cfg=p2s,imgn:vid_val:16_per_seq_random:nms-s5:gt-0:del-1:agn:proc-1:_in_-vit_b_640_imagenet_vid-batch_40-jtr-res_1440-fbb-self2-0/ckpt-__var__-imagenet_vid_val-16_per_seq_random/csv-batch_16:_out_-p2s-vit_b-imgn-vid-val-16_per_seq_random-aug-fbb
<a id="val___vid_aug_fbb_vit_b_64_0_"></a>
### val       @ vid-aug-fbb/vit_b-640-->eval_p2s-imgn
<a id="batch_32___val_vid_aug_fbb_vit_b_64_0_"></a>
#### batch_32       @ val/vid-aug-fbb/vit_b-640-->eval_p2s-imgn
python3 eval_det.py cfg=p2s,imgn:vid_val:nms-s10:gt-1:del-1:agn:proc-1:_in_-vit_b_640_imagenet_vid-batch_40-jtr-res_1440-fbb-self2-0/ckpt-140300-imagenet_vid_val/csv-batch_32:_out_-p2s-vit_b-imgn-vid-val-aug-fbb-b32 allow_missing_dets=1
<a id="batch_16___val_vid_aug_fbb_vit_b_64_0_"></a>
#### batch_16       @ val/vid-aug-fbb/vit_b-640-->eval_p2s-imgn
python3 eval_det.py cfg=p2s,imgn:vid_val:nms-s10:gt-1:del-1:agn:proc-1:_in_-vit_b_640_imagenet_vid-batch_40-jtr-res_1440-fbb-self2-0/ckpt-140300-imagenet_vid_val/csv-batch_16:_out_-p2s-vit_b-imgn-vid-val-aug-fbb allow_missing_dets=1

<a id="resnet_640___p2_s_"></a>
# resnet-640       @ p2s-->eval_det_p2s
<a id="vid_aug_fbb___resnet_640_"></a>
## vid-aug-fbb       @ resnet-640-->eval_p2s-imgn
<a id="train_8_per_seq_random___vid_aug_fbb_resnet_640_"></a>
### train-8_per_seq_random       @ vid-aug-fbb/resnet-640-->eval_p2s-imgn
python3 eval_det.py cfg=p2s,imgn:vid:8_per_seq_random:nms-1:gt-0:agn:proc-1:_in_-resnet_640_imagenet_vid-batch_512-jtr-res_1440-fbb-zexg/ckpt-__var__-imagenet_vid-8_per_seq_random/csv-batch_8:_out_-p2s-imgn-vid-train-8_per_seq_random-aug-fbb:ief
<a id="val_16_per_seq_random___vid_aug_fbb_resnet_640_"></a>
### val-16_per_seq_random       @ vid-aug-fbb/resnet-640-->eval_p2s-imgn
python3 eval_det.py cfg=p2s,imgn:vid_val:16_per_seq_random:nms-1:gt-0:agn:proc-1:_in_-resnet_640_imagenet_vid-batch_512-jtr-res_1440-fbb-zexg/ckpt-__var__-imagenet_vid_val-16_per_seq_random/csv-batch_8:_out_-p2s-imgn-vid_val-16_per_seq_random-aug-fbb:ief
<a id="val___vid_aug_fbb_resnet_640_"></a>
### val       @ vid-aug-fbb/resnet-640-->eval_p2s-imgn
python3 eval_det.py cfg=p2s,imgn:vid_val:nms-s5:gt-1:agn:proc-1:_in_-resnet_640_imagenet_vid-batch_512-jtr-res_1440-fbb-zexg/ckpt-__var__-imagenet_vid_val/csv-batch_8:_out_-p2s-imgn-vid_val-aug-fbb-s5

<a id="vid_det_aug_fbb___resnet_640_"></a>
## vid_det-aug-fbb       @ resnet-640-->eval_p2s-imgn
<a id="train_8_per_seq_random___vid_det_aug_fbb_resnet_640_"></a>
### train-8_per_seq_random       @ vid_det-aug-fbb/resnet-640-->eval_p2s-imgn
python3 eval_det.py cfg=p2s,imgn:vid_det:8_per_seq_random:nms-1:gt-0:agn:proc-1:_in_-resnet_640_imagenet_vid_det-sampled_eq-batch_448-jtr-res_1440-fbb-self2-0/ckpt-__var__-imagenet_vid_det-8_per_seq_random/csv-batch_16:_out_-p2s-imgn-vid_det-train-8_per_seq_random-aug-fbb
<a id="train_ratio_1_10_random___vid_det_aug_fbb_resnet_640_"></a>
### train-ratio_1_10_random       @ vid_det-aug-fbb/resnet-640-->eval_p2s-imgn
python3 eval_det.py cfg=p2s,imgn:vid_det:ratio_1_10_random:nms-1:gt-0:agn:proc-1:_in_-resnet_640_imagenet_vid_det-sampled_eq-batch_448-jtr-res_1440-fbb-self2-0/ckpt-__var__-imagenet_vid_det-ratio_1_10_random/csv-batch_32:_out_-p2s-imgn-vid_det-train-ratio_1_10_random-aug-fbb
<a id="val_16_per_seq_random___vid_det_aug_fbb_resnet_640_"></a>
### val-16_per_seq_random       @ vid_det-aug-fbb/resnet-640-->eval_p2s-imgn
python3 eval_det.py cfg=p2s,imgn:vid_val:16_per_seq_random:nms-1:gt-0:agn:proc-1:_in_-resnet_640_imagenet_vid_det-sampled_eq-batch_448-jtr-res_1440-fbb-self2-0/ckpt-__var__-imagenet_vid_val-16_per_seq_random/csv-batch_16:_out_-p2s-imgn-vid_det-val-16_per_seq_random-aug-fbb
<a id="val___vid_det_aug_fbb_resnet_640_"></a>
### val       @ vid_det-aug-fbb/resnet-640-->eval_p2s-imgn
python3 eval_det.py cfg=p2s,imgn:vid_val:nms-s5:gt-0:agn:proc-1:_in_-resnet_640_imagenet_vid_det-sampled_eq-batch_448-jtr-res_1440-fbb-self2-0/ckpt-__var__-imagenet_vid_val/csv-batch_32:_out_-p2s-imgn-vid_det-val-aug-fbb-s5

<a id="vid_det_aug___resnet_640_"></a>
## vid_det-aug       @ resnet-640-->eval_p2s-imgn
<a id="train_ratio_1_10_random___vid_det_aug_resnet_640_"></a>
### train-ratio_1_10_random       @ vid_det-aug/resnet-640-->eval_p2s-imgn
python3 eval_det.py cfg=p2s,imgn:vid_det:ratio_1_10_random:nms-1:gt-0:agn:proc-1:_in_-resnet_640_imagenet_vid_det-sampled_eq-batch_144-jtr-res_1440-zexg/ckpt-__var__-imagenet_vid_det-ratio_1_10_random/csv-batch_32:_out_-p2s-imgn-vid_det-train-ratio_1_10_random-aug
<a id="val_16_per_seq_random___vid_det_aug_resnet_640_"></a>
### val-16_per_seq_random       @ vid_det-aug/resnet-640-->eval_p2s-imgn
python3 eval_det.py cfg=p2s,imgn:vid_val:16_per_seq_random:nms-1:gt-0:agn:proc-1:_in_-resnet_640_imagenet_vid_det-sampled_eq-batch_144-jtr-res_1440-zexg/ckpt-__var__-imagenet_vid_val-16_per_seq_random/csv-batch_16:_out_-p2s-imgn-vid_det-val-16_per_seq_random-aug

<a id="vid_det_aug_isc___resnet_640_"></a>
## vid_det-aug-isc       @ resnet-640-->eval_p2s-imgn
<a id="train_ratio_1_10_random___vid_det_aug_isc_resnet_640_"></a>
### train-ratio_1_10_random       @ vid_det-aug-isc/resnet-640-->eval_p2s-imgn
python3 eval_det.py cfg=p2s,imgn:vid_det:ratio_1_10_random:nms-1:gt-0:agn:proc-1:_in_-resnet_640_imagenet_vid_det-sampled_eq-batch_144-jtr-res_1440-self2-0/ckpt-__var__-imagenet_vid_det-ratio_1_10_random/csv-batch_32:_out_-p2s-imgn-vid_det-train-ratio_1_10_random-aug
<a id="val_16_per_seq_random___vid_det_aug_isc_resnet_640_"></a>
### val-16_per_seq_random       @ vid_det-aug-isc/resnet-640-->eval_p2s-imgn
python3 eval_det.py cfg=p2s,imgn:vid_val:16_per_seq_random:nms-1:gt-0:agn:proc-1:_in_-resnet_640_imagenet_vid_det-sampled_eq-batch_144-jtr-res_1440-self2-0/ckpt-__var__-imagenet_vid_val-16_per_seq_random/csv-batch_32:_out_-p2s-imgn-vid_det-val-16_per_seq_random-aug


<a id="vid_det_all_aug_fbb___resnet_640_"></a>
## vid_det_all-aug-fbb       @ resnet-640-->eval_p2s-imgn
<a id="train_8_per_seq_random___vid_det_all_aug_fbb_resnet_640_"></a>
<a id="train_ratio_1_10_random___vid_det_all_aug_fbb_resnet_640_"></a>
### train-ratio_1_10_random       @ vid_det_all-aug-fbb/resnet-640-->eval_p2s-imgn
python3 eval_det.py cfg=p2s,imgn:vid_det_all:ratio_1_10_random:nms-1:gt-0:agn:proc-1:_in_-resnet_640_imagenet_vid_det_all-sampled_eq-batch_384-jtr-res_1440-fbb-zeg/ckpt-__var__-imagenet_vid_det_all-ratio_1_10_random/csv-batch_32:_out_-p2s-imgn-vid_det_all-train-ratio_1_10_random-aug-fbb
<a id="batch_16___train_ratio_1_10_random_vid_det_all_aug_fbb_resnet_640_"></a>
#### batch_16       @ train-ratio_1_10_random/vid_det_all-aug-fbb/resnet-640-->eval_p2s-imgn
python3 eval_det.py cfg=p2s,imgn:vid_det_all:ratio_1_10_random:nms-1:gt-0:agn:proc-1:_in_-resnet_640_imagenet_vid_det_all-sampled_eq-batch_384-jtr-res_1440-fbb-zeg/ckpt-__var__-imagenet_vid_det_all-ratio_1_10_random/csv-batch_16:_out_-p2s-imgn-vid_det_all-train-ratio_1_10_random-aug-fbb
<a id="val_16_per_seq_random___vid_det_all_aug_fbb_resnet_640_"></a>
### val-16_per_seq_random       @ vid_det_all-aug-fbb/resnet-640-->eval_p2s-imgn
python3 eval_det.py cfg=p2s,imgn:vid_val:16_per_seq_random:nms-1:del-1:gt-0:agn:proc-1:_in_-resnet_640_imagenet_vid_det_all-sampled_eq-batch_384-jtr-res_1440-fbb-zeg/ckpt-__var__-imagenet_vid_val-16_per_seq_random/csv-batch_16:_out_-p2s-imgn-vid_det_all-val-16_per_seq_random-aug-fbb
<a id="batch_16___val_16_per_seq_random_vid_det_all_aug_fbb_resnet_640_"></a>
#### batch_16       @ val-16_per_seq_random/vid_det_all-aug-fbb/resnet-640-->eval_p2s-imgn
python3 eval_det.py cfg=p2s,imgn:vid_val:16_per_seq_random:nms-1:del-1:gt-0:agn:proc-1:_in_-resnet_640_imagenet_vid_det_all-sampled_eq-batch_384-jtr-res_1440-fbb-zeg/ckpt-__var__-imagenet_vid_val-16_per_seq_random/csv-batch_16:_out_-p2s-imgn-vid_det_all-val-16_per_seq_random-aug-fbb


