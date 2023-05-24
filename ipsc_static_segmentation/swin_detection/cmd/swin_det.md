<!-- MarkdownTOC -->

- [ext_reorg_roi       @ swin_base_2_class](#ext_reorg_roi___swin_base_2_clas_s_)
    - [g2_0_37       @ ext_reorg_roi](#g2_0_37___ext_reorg_ro_i_)
        - [on-g2_38_53       @ g2_0_37/ext_reorg_roi](#on_g2_38_53___g2_0_37_ext_reorg_ro_i_)
        - [on-g2_38_53       @ g2_0_37/ext_reorg_roi](#on_g2_38_53___g2_0_37_ext_reorg_ro_i__1)
    - [g2_0_37-no_validate       @ ext_reorg_roi](#g2_0_37_no_validate___ext_reorg_ro_i_)
        - [on-g2_38_53       @ g2_0_37-no_validate/ext_reorg_roi](#on_g2_38_53___g2_0_37_no_validate_ext_reorg_ro_i_)
            - [epoch_2751       @ on-g2_38_53/g2_0_37-no_validate/ext_reorg_roi](#epoch_2751___on_g2_38_53_g2_0_37_no_validate_ext_reorg_ro_i_)
            - [epoch_6638       @ on-g2_38_53/g2_0_37-no_validate/ext_reorg_roi](#epoch_6638___on_g2_38_53_g2_0_37_no_validate_ext_reorg_ro_i_)
    - [g2_0_37-rcnn-win7       @ ext_reorg_roi](#g2_0_37_rcnn_win7___ext_reorg_ro_i_)
        - [on-g2_38_53       @ g2_0_37-rcnn-win7/ext_reorg_roi](#on_g2_38_53___g2_0_37_rcnn_win7_ext_reorg_ro_i_)
        - [g2_0_37-no_validate-rcnn       @ g2_0_37-rcnn-win7/ext_reorg_roi](#g2_0_37_no_validate_rcnn___g2_0_37_rcnn_win7_ext_reorg_ro_i_)
        - [on-g2_38_53       @ g2_0_37-rcnn-win7/ext_reorg_roi](#on_g2_38_53___g2_0_37_rcnn_win7_ext_reorg_ro_i__1)
    - [g2_0_37-no_validate-rcnn-win7       @ ext_reorg_roi](#g2_0_37_no_validate_rcnn_win7___ext_reorg_ro_i_)
        - [on-g2_38_53       @ g2_0_37-no_validate-rcnn-win7/ext_reorg_roi](#on_g2_38_53___g2_0_37_no_validate_rcnn_win7_ext_reorg_ro_i_)
    - [g2_15_53-no_validate       @ ext_reorg_roi](#g2_15_53_no_validate___ext_reorg_ro_i_)
        - [on-g2_0_14       @ g2_15_53-no_validate/ext_reorg_roi](#on_g2_0_14___g2_15_53_no_validate_ext_reorg_roi_)
    - [g2_16_53-no_validate       @ ext_reorg_roi](#g2_16_53_no_validate___ext_reorg_ro_i_)
        - [on-g2_0_15       @ g2_16_53-no_validate/ext_reorg_roi](#on_g2_0_15___g2_16_53_no_validate_ext_reorg_roi_)
    - [g2_16_53-no_validate-rcnn       @ ext_reorg_roi](#g2_16_53_no_validate_rcnn___ext_reorg_ro_i_)
        - [on-g2_0_15       @ g2_16_53-no_validate-rcnn/ext_reorg_roi](#on_g2_0_15___g2_16_53_no_validate_rcnn_ext_reorg_ro_i_)
    - [g2_54_126-no_validate-coco       @ ext_reorg_roi](#g2_54_126_no_validate_coco___ext_reorg_ro_i_)
    - [g2_54_126-no_validate       @ ext_reorg_roi](#g2_54_126_no_validate___ext_reorg_ro_i_)
        - [on-g2_0_53       @ g2_54_126-no_validate/ext_reorg_roi](#on_g2_0_53___g2_54_126_no_validate_ext_reorg_ro_i_)
        - [on-g2_0_15       @ g2_54_126-no_validate/ext_reorg_roi](#on_g2_0_15___g2_54_126_no_validate_ext_reorg_ro_i_)
        - [g2_54_126-no_validate-rcnn       @ g2_54_126-no_validate/ext_reorg_roi](#g2_54_126_no_validate_rcnn___g2_54_126_no_validate_ext_reorg_ro_i_)
        - [on-g2_0_53       @ g2_54_126-no_validate/ext_reorg_roi](#on_g2_0_53___g2_54_126_no_validate_ext_reorg_ro_i__1)
        - [on-g2_0_15       @ g2_54_126-no_validate/ext_reorg_roi](#on_g2_0_15___g2_54_126_no_validate_ext_reorg_ro_i__1)

<!-- /MarkdownTOC -->
<a id="ext_reorg_roi___swin_base_2_clas_s_"></a>
# ext_reorg_roi       @ swin_base_2_class-->swin_det
<a id="g2_0_37___ext_reorg_ro_i_"></a>
## g2_0_37       @ ext_reorg_roi-->swin_det
tools/dist_train.sh configs/swin/ipsc_2_class_ext_reorg_roi_g2_0_37.py 2 --cfg-options model.pretrained=pretrained/swin_base_patch4_window12_384.pth model.backbone.use_checkpoint=True data.samples_per_gpu=6 --resume-from work_dirs/ipsc_2_class_ext_reorg_roi_g2_0_37/latest.pth

<a id="on_g2_38_53___g2_0_37_ext_reorg_ro_i_"></a>
### on-g2_38_53       @ g2_0_37/ext_reorg_roi-->swin_det
python3 tools/test.py config=configs/swin/ipsc_2_class_ext_reorg_roi_g2_0_37.py checkpoint=work_dirs/ipsc_2_class_ext_reorg_roi_g2_0_37/epoch_488.pth eval=bbox,segm test_name=g2_38_53

<a id="on_g2_38_53___g2_0_37_ext_reorg_ro_i__1"></a>
### on-g2_38_53       @ g2_0_37/ext_reorg_roi-->swin_det
python3 tools/test.py config=configs/swin/ipsc_2_class_ext_reorg_roi_g2_0_37.py checkpoint=work_dirs/ipsc_2_class_ext_reorg_roi_g2_0_37/epoch_488.pth eval=bbox,segm test_name=g2_38_53

<a id="g2_0_37_no_validate___ext_reorg_ro_i_"></a>
## g2_0_37-no_validate       @ ext_reorg_roi-->swin_det
tools/dist_train.sh configs/swin/ipsc_2_class_ext_reorg_roi_g2_0_37-no_validate.py 2 --cfg-options model.pretrained=pretrained/swin_base_patch4_window12_384.pth model.backbone.use_checkpoint=True data.samples_per_gpu=4 data.workers_per_gpu=6 --resume-from work_dirs/ipsc_2_class_ext_reorg_roi_g2_0_37-no_validate/latest.pth --no-validate

<a id="on_g2_38_53___g2_0_37_no_validate_ext_reorg_ro_i_"></a>
### on-g2_38_53       @ g2_0_37-no_validate/ext_reorg_roi-->swin_det
<a id="epoch_2751___on_g2_38_53_g2_0_37_no_validate_ext_reorg_ro_i_"></a>
#### epoch_2751       @ on-g2_38_53/g2_0_37-no_validate/ext_reorg_roi-->swin_det
python3 tools/test.py config=configs/swin/ipsc_2_class_ext_reorg_roi_g2_0_37-no_validate.py checkpoint=work_dirs/ipsc_2_class_ext_reorg_roi_g2_0_37-no_validate/epoch_2751.pth eval=bbox,segm test_name=g2_38_53
<a id="epoch_6638___on_g2_38_53_g2_0_37_no_validate_ext_reorg_ro_i_"></a>
#### epoch_6638       @ on-g2_38_53/g2_0_37-no_validate/ext_reorg_roi-->swin_det
python3 tools/test.py config=configs/swin/ipsc_2_class_ext_reorg_roi_g2_0_37-no_validate.py checkpoint=work_dirs/ipsc_2_class_ext_reorg_roi_g2_0_37-no_validate/epoch_6638.pth eval=bbox,segm test_name=g2_38_53

<a id="g2_0_37_rcnn_win7___ext_reorg_ro_i_"></a>
## g2_0_37-rcnn-win7       @ ext_reorg_roi-->swin_det
tools/dist_train.sh configs/swin/ipsc_2_class_ext_reorg_roi_g2_0_37-rcnn-win7.py 2 --cfg-options model.pretrained=pretrained/swin_base_patch4_window7_224_22k.pth model.backbone.use_checkpoint=True data.samples_per_gpu=6 data.workers_per_gpu=6
<a id="on_g2_38_53___g2_0_37_rcnn_win7_ext_reorg_ro_i_"></a>
### on-g2_38_53       @ g2_0_37-rcnn-win7/ext_reorg_roi-->swin_det
python3 tools/test.py config=configs/swin/ipsc_2_class_ext_reorg_roi_g2_0_37-rcnn-win7.py checkpoint=work_dirs/ipsc_2_class_ext_reorg_roi_g2_0_37-rcnn-win7/epoch_274.pth eval=bbox test_name=g2_38_53

<a id="g2_0_37_no_validate_rcnn___g2_0_37_rcnn_win7_ext_reorg_ro_i_"></a>
### g2_0_37-no_validate-rcnn       @ g2_0_37-rcnn-win7/ext_reorg_roi-->swin_det
tools/dist_train.sh configs/swin/ipsc_2_class_ext_reorg_roi_g2_0_37-no_validate-rcnn.py 2 --cfg-options model.pretrained=pretrained/swin_base_patch4_window12_384.pth model.backbone.use_checkpoint=True data.samples_per_gpu=6 data.workers_per_gpu=6 --no-validate
<a id="on_g2_38_53___g2_0_37_rcnn_win7_ext_reorg_ro_i__1"></a>
### on-g2_38_53       @ g2_0_37-rcnn-win7/ext_reorg_roi-->swin_det
python3 tools/test.py config=configs/swin/ipsc_2_class_ext_reorg_roi_g2_0_37-no_validate-rcnn.py checkpoint=work_dirs/ipsc_2_class_ext_reorg_roi_g2_0_37-no_validate-rcnn/epoch_144.pth eval=bbox test_name=g2_38_53

<a id="g2_0_37_no_validate_rcnn_win7___ext_reorg_ro_i_"></a>
## g2_0_37-no_validate-rcnn-win7       @ ext_reorg_roi-->swin_det
tools/dist_train.sh configs/swin/ipsc_2_class_ext_reorg_roi_g2_0_37-no_validate-rcnn-win7.py 2 --cfg-options model.pretrained=pretrained/swin_base_patch4_window7_224_22k.pth model.backbone.use_checkpoint=True data.samples_per_gpu=6 data.workers_per_gpu=6 --no-validate
<a id="on_g2_38_53___g2_0_37_no_validate_rcnn_win7_ext_reorg_ro_i_"></a>
### on-g2_38_53       @ g2_0_37-no_validate-rcnn-win7/ext_reorg_roi-->swin_det
python3 tools/test.py config=configs/swin/ipsc_2_class_ext_reorg_roi_g2_0_37-no_validate-rcnn-win7.py checkpoint=work_dirs/ipsc_2_class_ext_reorg_roi_g2_0_37-no_validate-rcnn-win7/epoch_123.pth eval=bbox test_name=g2_38_53

<a id="g2_15_53_no_validate___ext_reorg_ro_i_"></a>
## g2_15_53-no_validate       @ ext_reorg_roi-->swin_det
tools/dist_train.sh configs/swin/ipsc_2_class_ext_reorg_roi_g2_15_53-no_validate.py 2 --cfg-options model.pretrained=pretrained/swin_base_patch4_window12_384.pth model.backbone.use_checkpoint=True data.samples_per_gpu=4 data.workers_per_gpu=4 --no-validate

<a id="on_g2_0_14___g2_15_53_no_validate_ext_reorg_roi_"></a>
### on-g2_0_14       @ g2_15_53-no_validate/ext_reorg_roi-->swin_det
python3 tools/test.py config=configs/swin/ipsc_2_class_ext_reorg_roi_g2_0_37-no_validate.py checkpoint=work_dirs/ipsc_2_class_ext_reorg_roi_g2_0_37-no_validate/epoch_2751.pth eval=bbox,segm test_name=g2_0_14

<a id="g2_16_53_no_validate___ext_reorg_ro_i_"></a>
## g2_16_53-no_validate       @ ext_reorg_roi-->swin_det
tools/dist_train.sh configs/swin/ipsc_2_class_ext_reorg_roi_g2_16_53-no_validate.py 2 --cfg-options model.pretrained=pretrained/swin_base_patch4_window12_384.pth model.backbone.use_checkpoint=True data.samples_per_gpu=4 data.workers_per_gpu=4 --no-validate --resume-from work_dirs/ipsc_2_class_ext_reorg_roi_g2_16_53-no_validate/epoch_2554.pth
<a id="on_g2_0_15___g2_16_53_no_validate_ext_reorg_roi_"></a>
### on-g2_0_15       @ g2_16_53-no_validate/ext_reorg_roi-->swin_det
python3 tools/test.py config=configs/swin/ipsc_2_class_ext_reorg_roi_g2_16_53-no_validate.py checkpoint=work_dirs/ipsc_2_class_ext_reorg_roi_g2_16_53-no_validate/epoch_3273.pth eval=bbox,segm test_name=g2_0_15

<a id="g2_16_53_no_validate_rcnn___ext_reorg_ro_i_"></a>
## g2_16_53-no_validate-rcnn       @ ext_reorg_roi-->swin_det
tools/dist_train.sh configs/swin/ipsc_2_class_ext_reorg_roi_g2_16_53-no_validate-rcnn.py 2 --cfg-options model.pretrained=pretrained/swin_base_patch4_window12_384.pth model.backbone.use_checkpoint=True data.samples_per_gpu=6 data.workers_per_gpu=6 --no-validate
<a id="on_g2_0_15___g2_16_53_no_validate_rcnn_ext_reorg_ro_i_"></a>
### on-g2_0_15       @ g2_16_53-no_validate-rcnn/ext_reorg_roi-->swin_det
python3 tools/test.py config=configs/swin/ipsc_2_class_ext_reorg_roi_g2_16_53-no_validate-rcnn.py checkpoint=work_dirs/ipsc_2_class_ext_reorg_roi_g2_16_53-no_validate-rcnn/latest.pth eval=bbox test_name=g2_0_15

<a id="g2_54_126_no_validate_coco___ext_reorg_ro_i_"></a>
## g2_54_126-no_validate-coco       @ ext_reorg_roi-->swin_det
tools/dist_train.sh configs/swin/ipsc_2_class_ext_reorg_roi_g2_54_126-no_validate-coco.py 2 --cfg-options model.pretrained=pretrained/cascade_mask_rcnn_swin_base_patch4_window7.pth model.backbone.use_checkpoint=True data.samples_per_gpu=2 data.workers_per_gpu=2 --no-validate

<a id="g2_54_126_no_validate___ext_reorg_ro_i_"></a>
## g2_54_126-no_validate       @ ext_reorg_roi-->swin_det
tools/dist_train.sh configs/swin/ipsc_2_class_ext_reorg_roi_g2_54_126-no_validate.py 2 --cfg-options model.pretrained=pretrained/swin_base_patch4_window12_384.pth model.backbone.use_checkpoint=True data.samples_per_gpu=4 data.workers_per_gpu=4 --no-validate --resume-from work_dirs/ipsc_2_class_ext_reorg_roi_g2_54_126-no_validate/latest.pth
<a id="on_g2_0_53___g2_54_126_no_validate_ext_reorg_ro_i_"></a>
### on-g2_0_53       @ g2_54_126-no_validate/ext_reorg_roi-->swin_det
python3 tools/test.py config=configs/swin/ipsc_2_class_ext_reorg_roi_g2_54_126-no_validate.py checkpoint=work_dirs/ipsc_2_class_ext_reorg_roi_g2_54_126-no_validate/epoch_2058.pth eval=bbox,segm test_name=g2_0_53
<a id="on_g2_0_15___g2_54_126_no_validate_ext_reorg_ro_i_"></a>
### on-g2_0_15       @ g2_54_126-no_validate/ext_reorg_roi-->swin_det
python3 tools/test.py config=configs/swin/ipsc_2_class_ext_reorg_roi_g2_54_126-no_validate.py checkpoint=work_dirs/ipsc_2_class_ext_reorg_roi_g2_54_126-no_validate/epoch_2058.pth eval=bbox,segm test_name=g2_0_15

<a id="g2_54_126_no_validate_rcnn___g2_54_126_no_validate_ext_reorg_ro_i_"></a>
### g2_54_126-no_validate-rcnn       @ g2_54_126-no_validate/ext_reorg_roi-->swin_det
tools/dist_train.sh configs/swin/ipsc_2_class_ext_reorg_roi_g2_54_126-no_validate-rcnn.py 2 --cfg-options model.pretrained=pretrained/swin_base_patch4_window12_384.pth model.backbone.use_checkpoint=True data.samples_per_gpu=12 data.workers_per_gpu=6 --no-validate
<a id="on_g2_0_53___g2_54_126_no_validate_ext_reorg_ro_i__1"></a>
### on-g2_0_53       @ g2_54_126-no_validate/ext_reorg_roi-->swin_det
python3 tools/test.py config=configs/swin/ipsc_2_class_ext_reorg_roi_g2_54_126-no_validate-rcnn.py checkpoint=work_dirs/ipsc_2_class_ext_reorg_roi_g2_54_126-no_validate-rcnn/latest.pth eval=bbox test_name=g2_0_53
<a id="on_g2_0_15___g2_54_126_no_validate_ext_reorg_ro_i__1"></a>
### on-g2_0_15       @ g2_54_126-no_validate/ext_reorg_roi-->swin_det
python3 tools/test.py config=configs/swin/ipsc_2_class_ext_reorg_roi_g2_54_126-no_validate-rcnn.py checkpoint=work_dirs/ipsc_2_class_ext_reorg_roi_g2_54_126-no_validate-rcnn/latest.pth eval=bbox test_name=g2_0_15



