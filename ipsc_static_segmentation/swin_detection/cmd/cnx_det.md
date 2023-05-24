<!-- MarkdownTOC -->

- [g2_0_37-no_validate-convnext_base_in22k       @ cvnxt/](#g2_0_37_no_validate_convnext_base_in22k___cvnxt_)
    - [on-g2_38_53       @ g2_0_37-no_validate-convnext_base_in22k](#on_g2_38_53___g2_0_37_no_validate_convnext_base_in22_k_)
- [g2_0_37-convnext_large_in22k       @ cvnxt/](#g2_0_37_convnext_large_in22k___cvnxt_)
    - [coco-pt-no_validate       @ g2_0_37-convnext_large_in22k](#coco_pt_no_validate___g2_0_37_convnext_large_in22k_)
    - [coco-pt       @ g2_0_37-convnext_large_in22k](#coco_pt___g2_0_37_convnext_large_in22k_)
        - [on-g2_38_53       @ coco-pt/g2_0_37-convnext_large_in22k](#on_g2_38_53___coco_pt_g2_0_37_convnext_large_in22k_)
    - [imagenet-pt       @ g2_0_37-convnext_large_in22k](#imagenet_pt___g2_0_37_convnext_large_in22k_)
        - [on-g2_38_53       @ imagenet-pt/g2_0_37-convnext_large_in22k](#on_g2_38_53___imagenet_pt_g2_0_37_convnext_large_in22k_)
- [g2_16_53-convnext_large_in22k](#g2_16_53_convnext_large_in22k_)
    - [on-g2_0_15       @ g2_16_53-convnext_large_in22k](#on_g2_0_15___g2_16_53_convnext_large_in22_k_)
- [g2_54_126-convnext_base_in22k](#g2_54_126_convnext_base_in22k_)
    - [on-g2_0_53       @ g2_54_126-convnext_base_in22k](#on_g2_0_53___g2_54_126_convnext_base_in22_k_)
    - [on-g2_0_15       @ g2_54_126-convnext_base_in22k](#on_g2_0_15___g2_54_126_convnext_base_in22_k_)
- [g2_54_126-convnext_large_in22k](#g2_54_126_convnext_large_in22_k_)
    - [on-g2_0_53       @ g2_54_126-convnext_large_in22k](#on_g2_0_53___g2_54_126_convnext_large_in22k_)
    - [on-g2_0_15       @ g2_54_126-convnext_large_in22k](#on_g2_0_15___g2_54_126_convnext_large_in22k_)
- [g2_54_126-convnext_xlarge_in22k-coco](#g2_54_126_convnext_xlarge_in22k_coc_o_)
- [g2_54_126-convnext_large_in22k-coco](#g2_54_126_convnext_large_in22k_coco_)
- [g2_54_126-convnext_small_in1k-coco](#g2_54_126_convnext_small_in1k_coc_o_)
- [g2_54_126-mask_rcnn_convnext_tiny_in1k-coco](#g2_54_126_mask_rcnn_convnext_tiny_in1k_coco_)

<!-- /MarkdownTOC -->

<a id="g2_0_37_no_validate_convnext_base_in22k___cvnxt_"></a>
# g2_0_37-no_validate-convnext_base_in22k       @ cvnxt/-->cvnxt_det
tools/dist_train.sh configs/convnext/ipsc_2_class_ext_reorg_roi_g2_0_37-no_validate-convnext_base_in22k.py 2 --no-validate --cfg-options model.pretrained=pretrained/cascade_mask_rcnn_convnext_base_22k_3x.pth data.samples_per_gpu=1 --resume-from work_dirs/ipsc_2_class_ext_reorg_roi_g2_0_37-no_validate-convnext_base_in22k/latest.pth

<a id="on_g2_38_53___g2_0_37_no_validate_convnext_base_in22_k_"></a>
## on-g2_38_53       @ g2_0_37-no_validate-convnext_base_in22k-->cnx_det
CUDA_VISIBLE_DEVICES=1 python3 tools/test.py config=configs/convnext/ipsc_2_class_ext_reorg_roi_g2_0_37-no_validate-convnext_base_in22k.py checkpoint=work_dirs/ipsc_2_class_ext_reorg_roi_g2_0_37-no_validate-convnext_base_in22k/epoch_255.pth eval=bbox,segm test_name=g2_38_53

<a id="g2_0_37_convnext_large_in22k___cvnxt_"></a>
# g2_0_37-convnext_large_in22k       @ cvnxt/-->cvnxt_det
<a id="coco_pt_no_validate___g2_0_37_convnext_large_in22k_"></a>
## coco-pt-no_validate       @ g2_0_37-convnext_large_in22k-->cnx_det
```
salloc --nodes=1 --time=0:15:0 --account=def-nilanjan --gpus-per-node=1 --mem=16000M --cpus-per-task=4
salloc --nodes=1 --time=0:15:0 --account=def-nilanjan --gpus-per-node=2 --mem=16000M --cpus-per-task=4

sbatch cmd/ext_reorg_roi-convnext_large_in22k.sh
scancel
```
tools/dist_train.sh configs/convnext/ipsc_2_class_ext_reorg_roi_g2_0_37-no_validate-convnext_large_in22k.py 2 --no-validate --cfg-options model.pretrained=pretrained/cascade_mask_rcnn_convnext_large_22k_3x.pth data.samples_per_gpu=4

<a id="coco_pt___g2_0_37_convnext_large_in22k_"></a>
## coco-pt       @ g2_0_37-convnext_large_in22k-->cnx_det
tools/dist_train.sh configs/convnext/ipsc_2_class_ext_reorg_roi_g2_0_37-convnext_large_in22k.py 2 --cfg-options model.pretrained=pretrained/cascade_mask_rcnn_convnext_large_22k_3x.pth data.samples_per_gpu=1 

<a id="on_g2_38_53___coco_pt_g2_0_37_convnext_large_in22k_"></a>
### on-g2_38_53       @ coco-pt/g2_0_37-convnext_large_in22k-->cnx_det
CUDA_VISIBLE_DEVICES=0 python3 tools/test.py config=configs/convnext/ipsc_2_class_ext_reorg_roi_g2_0_37-convnext_large_in22k.py checkpoint=work_dirs/ipsc_2_class_ext_reorg_roi_g2_0_37-convnext_large_in22k_coco_pretrained/epoch_153.pth eval=bbox,segm test_name=g2_38_53

<a id="imagenet_pt___g2_0_37_convnext_large_in22k_"></a>
## imagenet-pt       @ g2_0_37-convnext_large_in22k-->cnx_det
tools/dist_train.sh configs/convnext/ipsc_2_class_ext_reorg_roi_g2_0_37-convnext_large_in22k.py 2 --cfg-options model.pretrained=pretrained/convnext_large_22k_224.pth data.samples_per_gpu=1

<a id="on_g2_38_53___imagenet_pt_g2_0_37_convnext_large_in22k_"></a>
### on-g2_38_53       @ imagenet-pt/g2_0_37-convnext_large_in22k-->cnx_det
python3 tools/test.py config=configs/convnext/ipsc_2_class_ext_reorg_roi_g2_0_37-convnext_large_in22k.py checkpoint=work_dirs/ipsc_2_class_ext_reorg_roi_g2_0_37-convnext_large_in22k/epoch_977.pth eval=bbox,segm test_name=g2_38_53 

<a id="g2_16_53_convnext_large_in22k_"></a>
# g2_16_53-convnext_large_in22k      
tools/dist_train.sh configs/convnext/ipsc_2_class_ext_reorg_roi_g2_16_53-convnext_large_in22k.py 2 --cfg-options model.pretrained=pretrained/convnext_large_22k_224.pth data.samples_per_gpu=1

<a id="on_g2_0_15___g2_16_53_convnext_large_in22_k_"></a>
## on-g2_0_15       @ g2_16_53-convnext_large_in22k-->cnx_det
python3 tools/test.py config=configs/convnext/ipsc_2_class_ext_reorg_roi_g2_16_53-convnext_large_in22k.py checkpoint=work_dirs/ipsc_2_class_ext_reorg_roi_g2_16_53-convnext_large_in22k/epoch_1014.pth eval=bbox,segm test_name=g2_0_15 

<a id="g2_54_126_convnext_base_in22k_"></a>
# g2_54_126-convnext_base_in22k 
tools/dist_train.sh configs/convnext/ipsc_2_class_ext_reorg_roi_g2_54_126-convnext_base_in22k.py 2 --cfg-options model.pretrained=pretrained/convnext_base_22k_224.pth data.samples_per_gpu=2 --resume-from work_dirs/ipsc_2_class_ext_reorg_roi_g2_54_126-convnext_base_in22k/latest.pth

__coco__
tools/dist_train.sh configs/convnext/ipsc_2_class_ext_reorg_roi_g2_54_126-convnext_base_in22k-coco.py 2 --cfg-options model.pretrained=pretrained/cascade_mask_rcnn_convnext_base_1k_3x.pth data.samples_per_gpu=2

__coco_self__
tools/dist_train.sh configs/convnext/ipsc_2_class_ext_reorg_roi_g2_54_126-convnext_base_in22k-coco.py 2 --cfg-options model.pretrained=pretrained/convnext_base_22k_224.pth  data.samples_per_gpu=2 --resume-from work_dirs/coco17-convnext_base_in22k/latest.pth

<a id="on_g2_0_53___g2_54_126_convnext_base_in22_k_"></a>
## on-g2_0_53       @ g2_54_126-convnext_base_in22k-->cnx_det
CUDA_VISIBLE_DEVICES=1 python3 tools/test.py config=configs/convnext/ipsc_2_class_ext_reorg_roi_g2_54_126-convnext_base_in22k.py checkpoint=work_dirs/ipsc_2_class_ext_reorg_roi_g2_54_126-convnext_base_in22k/epoch_58.pth eval=bbox,segm test_name=g2_0_53
<a id="on_g2_0_15___g2_54_126_convnext_base_in22_k_"></a>
## on-g2_0_15       @ g2_54_126-convnext_base_in22k-->cnx_det
CUDA_VISIBLE_DEVICES=0 python3 tools/test.py config=configs/convnext/ipsc_2_class_ext_reorg_roi_g2_54_126-convnext_base_in22k.py checkpoint=work_dirs/ipsc_2_class_ext_reorg_roi_g2_54_126-convnext_base_in22k/epoch_58.pth eval=bbox,segm test_name=g2_0_15

<a id="g2_54_126_convnext_large_in22_k_"></a>
# g2_54_126-convnext_large_in22k 
tools/dist_train.sh configs/convnext/ipsc_2_class_ext_reorg_roi_g2_54_126-convnext_large_in22k.py 2 --cfg-options model.pretrained=pretrained/convnext_large_22k_224.pth data.samples_per_gpu=1
<a id="on_g2_0_53___g2_54_126_convnext_large_in22k_"></a>
## on-g2_0_53       @ g2_54_126-convnext_large_in22k-->cnx_det
CUDA_VISIBLE_DEVICES=0 python3 tools/test.py config=configs/convnext/ipsc_2_class_ext_reorg_roi_g2_54_126-convnext_large_in22k.py checkpoint=work_dirs/ipsc_2_class_ext_reorg_roi_g2_54_126-convnext_large_in22k/epoch_106.pth eval=bbox,segm test_name=g2_0_53
<a id="on_g2_0_15___g2_54_126_convnext_large_in22k_"></a>
## on-g2_0_15       @ g2_54_126-convnext_large_in22k-->cnx_det
CUDA_VISIBLE_DEVICES=0 python3 tools/test.py config=configs/convnext/ipsc_2_class_ext_reorg_roi_g2_54_126-convnext_large_in22k.py checkpoint=work_dirs/ipsc_2_class_ext_reorg_roi_g2_54_126-convnext_large_in22k/epoch_106.pth eval=bbox,segm test_name=g2_0_15

<a id="g2_54_126_convnext_xlarge_in22k_coc_o_"></a>
# g2_54_126-convnext_xlarge_in22k-coco 
tools/dist_train.sh configs/convnext/ipsc_2_class_ext_reorg_roi_g2_54_126-convnext_large_in22k-coco.py 2 --cfg-options model.pretrained=pretrained/cascade_mask_rcnn_convnext_large_22k_3x.pth data.samples_per_gpu=1

<a id="g2_54_126_convnext_large_in22k_coco_"></a>
# g2_54_126-convnext_large_in22k-coco  
tools/dist_train.sh configs/convnext/ipsc_2_class_ext_reorg_roi_g2_54_126-convnext_large_in22k-coco.py 2 --cfg-options model.pretrained=pretrained/cascade_mask_rcnn_convnext_large_22k_3x.pth data.samples_per_gpu=1

<a id="g2_54_126_convnext_small_in1k_coc_o_"></a>
# g2_54_126-convnext_small_in1k-coco  
tools/dist_train.sh configs/convnext/ipsc_2_class_ext_reorg_roi_g2_54_126-convnext_small_in1k-coco.py 2 --cfg-options model.pretrained=pretrained/cascade_mask_rcnn_convnext_small_1k_3x.pth data.samples_per_gpu=2

<a id="g2_54_126_mask_rcnn_convnext_tiny_in1k_coco_"></a>
# g2_54_126-mask_rcnn_convnext_tiny_in1k-coco
tools/dist_train.sh configs/convnext/ipsc_2_class_ext_reorg_roi_g2_54_126-mask_rcnn_convnext_tiny_in1k.py 2 --cfg-options model.pretrained=pretrained/mask_rcnn_convnext_tiny_1k_3x.pth data.samples_per_gpu=2
