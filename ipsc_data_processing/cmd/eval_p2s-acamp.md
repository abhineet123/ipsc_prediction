<!-- MarkdownTOC -->

- [resnet-640       @ p2s](#resnet_640___p2_s_)
    - [1k8_vid_entire_seq       @ resnet-640](#1k8_vid_entire_seq___resnet_640_)
        - [on-inv       @ 1k8_vid_entire_seq/resnet-640](#on_inv___1k8_vid_entire_seq_resnet_64_0_)
    - [10k6_vid_entire_seq       @ resnet-640](#10k6_vid_entire_seq___resnet_640_)
        - [on-inv       @ 10k6_vid_entire_seq/resnet-640](#on_inv___10k6_vid_entire_seq_resnet_640_)
    - [20k6_5_video       @ resnet-640](#20k6_5_video___resnet_640_)
        - [on-inv       @ 20k6_5_video/resnet-640](#on_inv___20k6_5_video_resnet_64_0_)
    - [1k8_vid_entire_seq-aug-fbb       @ resnet-640](#1k8_vid_entire_seq_aug_fbb___resnet_640_)
        - [on-inv-2_per_seq       @ 1k8_vid_entire_seq-aug-fbb/resnet-640](#on_inv_2_per_seq___1k8_vid_entire_seq_aug_fbb_resnet_64_0_)
        - [on-inv       @ 1k8_vid_entire_seq-aug-fbb/resnet-640](#on_inv___1k8_vid_entire_seq_aug_fbb_resnet_64_0_)
    - [10k6_vid_entire_seq-aug-fbb       @ resnet-640](#10k6_vid_entire_seq_aug_fbb___resnet_640_)
        - [on-inv-2_per_seq       @ 10k6_vid_entire_seq-aug-fbb/resnet-640](#on_inv_2_per_seq___10k6_vid_entire_seq_aug_fbb_resnet_640_)
        - [on-inv       @ 10k6_vid_entire_seq-aug-fbb/resnet-640](#on_inv___10k6_vid_entire_seq_aug_fbb_resnet_640_)
    - [20k6_5_video-aug-fbb       @ resnet-640](#20k6_5_video_aug_fbb___resnet_640_)
        - [on-inv-2_per_seq       @ 20k6_5_video-aug-fbb/resnet-640](#on_inv_2_per_seq___20k6_5_video_aug_fbb_resnet_64_0_)
        - [on-inv       @ 20k6_5_video-aug-fbb/resnet-640](#on_inv___20k6_5_video_aug_fbb_resnet_64_0_)
    - [10k6_vid_entire_seq-aug-fbb-b96       @ resnet-640](#10k6_vid_entire_seq_aug_fbb_b96___resnet_640_)
        - [on-inv-2_per_seq       @ 10k6_vid_entire_seq-aug-fbb-b96/resnet-640](#on_inv_2_per_seq___10k6_vid_entire_seq_aug_fbb_b96_resnet_640_)
        - [on-inv       @ 10k6_vid_entire_seq-aug-fbb-b96/resnet-640](#on_inv___10k6_vid_entire_seq_aug_fbb_b96_resnet_640_)
    - [10k6_vid_entire_seq-aug-fbb-cls_eq       @ resnet-640](#10k6_vid_entire_seq_aug_fbb_cls_eq___resnet_640_)
        - [on-inv-2_per_seq       @ 10k6_vid_entire_seq-aug-fbb-cls_eq/resnet-640](#on_inv_2_per_seq___10k6_vid_entire_seq_aug_fbb_cls_eq_resnet_64_0_)
        - [on-inv       @ 10k6_vid_entire_seq-aug-fbb-cls_eq/resnet-640](#on_inv___10k6_vid_entire_seq_aug_fbb_cls_eq_resnet_64_0_)

<!-- /MarkdownTOC -->
<a id="resnet_640___p2_s_"></a>
# resnet-640       @ p2s-->eval_det_p2s
<a id="1k8_vid_entire_seq___resnet_640_"></a>
## 1k8_vid_entire_seq       @ resnet-640-->eval_p2s-acamp
<a id="on_inv___1k8_vid_entire_seq_resnet_64_0_"></a>
### on-inv       @ 1k8_vid_entire_seq/resnet-640-->eval_p2s-acamp
python3 eval_det.py cfg=p2s,acamp:1k8_vid_entire_seq_inv:nms-1:agn:gt-0:proc-12:show-0:_in_-resnet_640_1k8_vid_entire_seq-batch_32/ckpt-119250-1k8_vid_entire_seq_inv/csv-batch_8:_out_-p2s-resnet_640-acamp-1k8_vid_entire_seq
`nms20`
python3 eval_det.py cfg=p2s,acamp:1k8_vid_entire_seq_inv:nms-20:agn:gt-1:proc-2:show-0:_in_-resnet_640_1k8_vid_entire_seq-batch_32/ckpt-119250-1k8_vid_entire_seq_inv/csv-batch_8:_out_-p2s-resnet_640-acamp-1k8_vid_entire_seq-nms20

<a id="10k6_vid_entire_seq___resnet_640_"></a>
## 10k6_vid_entire_seq       @ resnet-640-->eval_p2s-acamp
<a id="on_inv___10k6_vid_entire_seq_resnet_640_"></a>
### on-inv       @ 10k6_vid_entire_seq/resnet-640-->eval_p2s-acamp
python3 eval_det.py cfg=p2s,acamp:10k6_vid_entire_seq_inv:nms-1:agn:gt-0:proc-12:show-0:_in_-resnet_640_10k6_vid_entire_seq-batch_32/ckpt-125625-10k6_vid_entire_seq_inv/csv-batch_8:_out_-p2s-resnet_640-acamp-10k6_vid_entire_seq
`nms20`
python3 eval_det.py cfg=p2s,acamp:10k6_vid_entire_seq_inv:nms-20:agn:gt-1:proc-2:show-0:_in_-resnet_640_10k6_vid_entire_seq-batch_32/ckpt-125625-10k6_vid_entire_seq_inv/csv-batch_8:_out_-p2s-resnet_640-acamp-10k6_vid_entire_seq-nms20

<a id="20k6_5_video___resnet_640_"></a>
## 20k6_5_video       @ resnet-640-->eval_p2s-acamp
<a id="on_inv___20k6_5_video_resnet_64_0_"></a>
### on-inv       @ 20k6_5_video/resnet-640-->eval_p2s-acamp
python3 eval_det.py cfg=p2s,acamp:20k6_5_video_inv:nms-1:agn:gt-1:proc-12:show-0:_in_-resnet_640_20k6_5_video-batch_32/ckpt-124278-20k6_5_video_inv/csv-batch_12:_out_-p2s-resnet_640-acamp-20k6_5_video
`nms20`
python3 eval_det.py cfg=p2s,acamp:20k6_5_video_inv:nms-20:agn:gt-1:proc-2:show-0:_in_-resnet_640_20k6_5_video-batch_32/ckpt-124278-20k6_5_video_inv/csv-batch_12:_out_-p2s-resnet_640-acamp-20k6_5_video-nms20

<a id="1k8_vid_entire_seq_aug_fbb___resnet_640_"></a>
## 1k8_vid_entire_seq-aug-fbb       @ resnet-640-->eval_p2s-acamp
<a id="on_inv_2_per_seq___1k8_vid_entire_seq_aug_fbb_resnet_64_0_"></a>
### on-inv-2_per_seq       @ 1k8_vid_entire_seq-aug-fbb/resnet-640-->eval_p2s-acamp
python3 eval_det.py cfg=p2s,acamp:1k8_vid_entire_seq_inv_2_per_seq:nms-1:agn:gt-1:proc-1:show-0:_in_-resnet_640_1k8_vid_entire_seq-batch_48-jtr-res_1280-fbb/ckpt-__var__-1k8_vid_entire_seq_inv_2_per_seq/csv-batch_2:_out_-p2s-1k8-aug-fbb-2_per_seq
<a id="on_inv___1k8_vid_entire_seq_aug_fbb_resnet_64_0_"></a>
### on-inv       @ 1k8_vid_entire_seq-aug-fbb/resnet-640-->eval_p2s-acamp
python3 eval_det.py cfg=p2s,acamp:1k8_vid_entire_seq_inv:nms-1:agn:gt-1:proc-1:show-0:_in_-resnet_640_1k8_vid_entire_seq-batch_48-jtr-res_1280-fbb/ckpt-__var__-1k8_vid_entire_seq_inv/csv-batch_2:_out_-p2s-1k8-aug-fbb

<a id="10k6_vid_entire_seq_aug_fbb___resnet_640_"></a>
## 10k6_vid_entire_seq-aug-fbb       @ resnet-640-->eval_p2s-acamp
<a id="on_inv_2_per_seq___10k6_vid_entire_seq_aug_fbb_resnet_640_"></a>
### on-inv-2_per_seq       @ 10k6_vid_entire_seq-aug-fbb/resnet-640-->eval_p2s-acamp
python3 eval_det.py cfg=p2s,acamp:10k6_vid_entire_seq_inv_2_per_seq:nms-1:agn:gt-1:proc-1:show-0:_in_-resnet_640_10k6_vid_entire_seq-batch_48-jtr-res_1280-fbb/ckpt-__var__-10k6_vid_entire_seq_inv_2_per_seq/csv-batch_2:_out_-p2s-10k6-aug-fbb-2_per_seq
<a id="on_inv___10k6_vid_entire_seq_aug_fbb_resnet_640_"></a>
### on-inv       @ 10k6_vid_entire_seq-aug-fbb/resnet-640-->eval_p2s-acamp
python3 eval_det.py cfg=p2s,acamp:10k6_vid_entire_seq_inv:nms-1:agn:gt-1:proc-1:show-0:_in_-resnet_640_10k6_vid_entire_seq-batch_48-jtr-res_1280-fbb/ckpt-__var__-10k6_vid_entire_seq_inv/csv-batch_8:_out_-p2s-10k6-aug-fbb

<a id="20k6_5_video_aug_fbb___resnet_640_"></a>
## 20k6_5_video-aug-fbb       @ resnet-640-->eval_p2s-acamp
<a id="on_inv_2_per_seq___20k6_5_video_aug_fbb_resnet_64_0_"></a>
### on-inv-2_per_seq       @ 20k6_5_video-aug-fbb/resnet-640-->eval_p2s-acamp
python3 eval_det.py cfg=p2s,acamp:20k6_5_video_inv_2_per_seq:nms-1:agn:gt-1:proc-1:show-0:_in_-resnet_640_20k6_5_video-batch_48-jtr-res_1280-fbb/ckpt-__var__-20k6_5_video_inv_2_per_seq/csv-batch_2:_out_-p2s-20k6-aug-fbb-2_per_seq
<a id="on_inv___20k6_5_video_aug_fbb_resnet_64_0_"></a>
### on-inv       @ 20k6_5_video-aug-fbb/resnet-640-->eval_p2s-acamp
python3 eval_det.py cfg=p2s,acamp:20k6_5_video_inv:nms-1:agn:gt-1:proc-1:show-0:_in_-resnet_640_20k6_5_video-batch_48-jtr-res_1280-fbb/ckpt-__var__-20k6_5_video_inv/csv-batch_6:_out_-p2s-20k6-aug-fbb

<a id="10k6_vid_entire_seq_aug_fbb_b96___resnet_640_"></a>
## 10k6_vid_entire_seq-aug-fbb-b96       @ resnet-640-->eval_p2s-acamp
<a id="on_inv_2_per_seq___10k6_vid_entire_seq_aug_fbb_b96_resnet_640_"></a>
### on-inv-2_per_seq       @ 10k6_vid_entire_seq-aug-fbb-b96/resnet-640-->eval_p2s-acamp
python3 eval_det.py cfg=p2s,acamp:10k6_vid_entire_seq_inv_2_per_seq:nms-1:agn:gt-1:proc-1:show-0:_in_-resnet_640_10k6_vid_entire_seq-batch_96-jtr-res_1280-fbb/ckpt-__var__-10k6_vid_entire_seq_inv_2_per_seq/csv-batch_4:_out_-p2s-10k6-aug-fbb-b96-2_per_seq
<a id="on_inv___10k6_vid_entire_seq_aug_fbb_b96_resnet_640_"></a>
### on-inv       @ 10k6_vid_entire_seq-aug-fbb-b96/resnet-640-->eval_p2s-acamp
python3 eval_det.py cfg=p2s,acamp:10k6_vid_entire_seq_inv:nms-1:agn:gt-1:proc-1:show-0:_in_-resnet_640_10k6_vid_entire_seq-batch_96-jtr-res_1280-fbb/ckpt-__var__-10k6_vid_entire_seq_inv/csv-batch_8:_out_-p2s-10k6-aug-fbb-b96

<a id="10k6_vid_entire_seq_aug_fbb_cls_eq___resnet_640_"></a>
## 10k6_vid_entire_seq-aug-fbb-cls_eq       @ resnet-640-->eval_p2s-acamp
<a id="on_inv_2_per_seq___10k6_vid_entire_seq_aug_fbb_cls_eq_resnet_64_0_"></a>
### on-inv-2_per_seq       @ 10k6_vid_entire_seq-aug-fbb-cls_eq/resnet-640-->eval_p2s-acamp
python3 eval_det.py cfg=p2s,acamp:10k6_vid_entire_seq_inv_2_per_seq:nms-1:agn:gt-1:proc-1:show-0:_in_-resnet_640_10k6_vid_entire_seq-batch_128-jtr-res_1280-fbb-cls_eq/ckpt-__var__-10k6_vid_entire_seq_inv_2_per_seq/csv-batch_4:_out_-p2s-10k6-aug-fbb-cls_eq-2_per_seq
<a id="on_inv___10k6_vid_entire_seq_aug_fbb_cls_eq_resnet_64_0_"></a>
### on-inv       @ 10k6_vid_entire_seq-aug-fbb-cls_eq/resnet-640-->eval_p2s-acamp
python3 eval_det.py cfg=p2s,acamp:10k6_vid_entire_seq_inv:nms-1:agn:gt-1:proc-1:show-0:_in_-resnet_640_10k6_vid_entire_seq-batch_128-jtr-res_1280-fbb-cls_eq/ckpt-272958-10k6_vid_entire_seq_inv/csv-batch_4:_out_-p2s-10k6-aug-fbb-cls_eq

