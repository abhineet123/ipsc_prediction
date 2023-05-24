<!-- MarkdownTOC -->

- [coco17](#coco1_7_)
    - [cnx-base-in22k       @ coco17](#cnx_base_in22k___coco17_)
    - [cnx-small-in22k       @ coco17](#cnx_small_in22k___coco17_)
    - [swin-small       @ coco17](#swin_small___coco17_)
- [swin_base_2_class](#swin_base_2_class_)
    - [ctc       @ swin_base_2_class](#ctc___swin_base_2_clas_s_)
        - [ctc_all       @ ctc/swin_base_2_class](#ctc_all___ctc_swin_base_2_clas_s_)
        - [ctc_BF_C2DL_HSC       @ ctc/swin_base_2_class](#ctc_bf_c2dl_hsc___ctc_swin_base_2_clas_s_)
        - [ctc_BF_C2DL_MuSC       @ ctc/swin_base_2_class](#ctc_bf_c2dl_musc___ctc_swin_base_2_clas_s_)
        - [ctc_DIC_C2DH_HeLa       @ ctc/swin_base_2_class](#ctc_dic_c2dh_hela___ctc_swin_base_2_clas_s_)
        - [ctc_Fluo_C2DL_MSC       @ ctc/swin_base_2_class](#ctc_fluo_c2dl_msc___ctc_swin_base_2_clas_s_)
        - [ctc_PhC_C2DH_U373       @ ctc/swin_base_2_class](#ctc_phc_c2dh_u373___ctc_swin_base_2_clas_s_)
        - [ctc_PhC_C2DL_PSC       @ ctc/swin_base_2_class](#ctc_phc_c2dl_psc___ctc_swin_base_2_clas_s_)
            - [fixed_lr       @ ctc_PhC_C2DL_PSC/ctc/swin_base_2_class](#fixed_lr___ctc_phc_c2dl_psc_ctc_swin_base_2_class_)
        - [ctc_Fluo_N2DH_GOWT1       @ ctc/swin_base_2_class](#ctc_fluo_n2dh_gowt1___ctc_swin_base_2_clas_s_)
            - [fixed_lr       @ ctc_Fluo_N2DH_GOWT1/ctc/swin_base_2_class](#fixed_lr___ctc_fluo_n2dh_gowt1_ctc_swin_base_2_clas_s_)
        - [ctc_Fluo_N2DH_SIM       @ ctc/swin_base_2_class](#ctc_fluo_n2dh_sim___ctc_swin_base_2_clas_s_)
            - [fixed_lr       @ ctc_Fluo_N2DH_SIM/ctc/swin_base_2_class](#fixed_lr___ctc_fluo_n2dh_sim_ctc_swin_base_2_clas_s_)
        - [ctc_Fluo_N2DL_HeLa       @ ctc/swin_base_2_class](#ctc_fluo_n2dl_hela___ctc_swin_base_2_clas_s_)
            - [fixed_lr       @ ctc_Fluo_N2DL_HeLa/ctc/swin_base_2_class](#fixed_lr___ctc_fluo_n2dl_hela_ctc_swin_base_2_class_)
    - [ctmc       @ swin_base_2_class](#ctmc___swin_base_2_clas_s_)
    - [ext_reorg_roi       @ swin_base_2_class](#ext_reorg_roi___swin_base_2_clas_s_)
        - [g2_0_37       @ ext_reorg_roi/swin_base_2_class](#g2_0_37___ext_reorg_roi_swin_base_2_clas_s_)
            - [on-g2_38_53       @ g2_0_37/ext_reorg_roi/swin_base_2_class](#on_g2_38_53___g2_0_37_ext_reorg_roi_swin_base_2_clas_s_)
            - [on-g2_38_53       @ g2_0_37/ext_reorg_roi/swin_base_2_class](#on_g2_38_53___g2_0_37_ext_reorg_roi_swin_base_2_clas_s__1)
        - [g2_0_37-no_validate       @ ext_reorg_roi/swin_base_2_class](#g2_0_37_no_validate___ext_reorg_roi_swin_base_2_clas_s_)
            - [on-g2_38_53       @ g2_0_37-no_validate/ext_reorg_roi/swin_base_2_class](#on_g2_38_53___g2_0_37_no_validate_ext_reorg_roi_swin_base_2_clas_s_)
                - [epoch_2751       @ on-g2_38_53/g2_0_37-no_validate/ext_reorg_roi/swin_base_2_class](#epoch_2751___on_g2_38_53_g2_0_37_no_validate_ext_reorg_roi_swin_base_2_clas_s_)
                - [epoch_6638       @ on-g2_38_53/g2_0_37-no_validate/ext_reorg_roi/swin_base_2_class](#epoch_6638___on_g2_38_53_g2_0_37_no_validate_ext_reorg_roi_swin_base_2_clas_s_)
        - [rcnn       @ ext_reorg_roi/swin_base_2_class](#rcnn___ext_reorg_roi_swin_base_2_clas_s_)
            - [g2_0_37-rcnn       @ rcnn/ext_reorg_roi/swin_base_2_class](#g2_0_37_rcnn___rcnn_ext_reorg_roi_swin_base_2_class_)
            - [g2_0_37-rcnn-win7       @ rcnn/ext_reorg_roi/swin_base_2_class](#g2_0_37_rcnn_win7___rcnn_ext_reorg_roi_swin_base_2_class_)
            - [g2_0_37-no_validate-rcnn       @ rcnn/ext_reorg_roi/swin_base_2_class](#g2_0_37_no_validate_rcnn___rcnn_ext_reorg_roi_swin_base_2_class_)
                - [on-g2_38_53       @ g2_0_37-no_validate-rcnn/rcnn/ext_reorg_roi/swin_base_2_class](#on_g2_38_53___g2_0_37_no_validate_rcnn_rcnn_ext_reorg_roi_swin_base_2_clas_s_)
            - [g2_0_37-no_validate-rcnn-win7       @ rcnn/ext_reorg_roi/swin_base_2_class](#g2_0_37_no_validate_rcnn_win7___rcnn_ext_reorg_roi_swin_base_2_class_)
                - [on-g2_38_53       @ g2_0_37-no_validate-rcnn-win7/rcnn/ext_reorg_roi/swin_base_2_class](#on_g2_38_53___g2_0_37_no_validate_rcnn_win7_rcnn_ext_reorg_roi_swin_base_2_class_)
        - [g2_15_53-no_validate       @ ext_reorg_roi/swin_base_2_class](#g2_15_53_no_validate___ext_reorg_roi_swin_base_2_clas_s_)
            - [on-g2_0_14       @ g2_15_53-no_validate/ext_reorg_roi/swin_base_2_class](#on_g2_0_14___g2_15_53_no_validate_ext_reorg_roi_swin_base_2_class_)
        - [g2_16_53-no_validate       @ ext_reorg_roi/swin_base_2_class](#g2_16_53_no_validate___ext_reorg_roi_swin_base_2_clas_s_)
            - [on-g2_0_15       @ g2_16_53-no_validate/ext_reorg_roi/swin_base_2_class](#on_g2_0_15___g2_16_53_no_validate_ext_reorg_roi_swin_base_2_class_)
        - [g2_54_126-no_validate-coco       @ ext_reorg_roi/swin_base_2_class](#g2_54_126_no_validate_coco___ext_reorg_roi_swin_base_2_clas_s_)
        - [g2_54_126-no_validate       @ ext_reorg_roi/swin_base_2_class](#g2_54_126_no_validate___ext_reorg_roi_swin_base_2_clas_s_)
            - [on-g2_0_53       @ g2_54_126-no_validate/ext_reorg_roi/swin_base_2_class](#on_g2_0_53___g2_54_126_no_validate_ext_reorg_roi_swin_base_2_clas_s_)
    - [all_frames_roi       @ swin_base_2_class](#all_frames_roi___swin_base_2_clas_s_)
        - [g2_0_37       @ all_frames_roi/swin_base_2_class](#g2_0_37___all_frames_roi_swin_base_2_class_)
            - [no-validate       @ g2_0_37/all_frames_roi/swin_base_2_class](#no_validate___g2_0_37_all_frames_roi_swin_base_2_class_)
            - [g2_38_53       @ g2_0_37/all_frames_roi/swin_base_2_class](#g2_38_53___g2_0_37_all_frames_roi_swin_base_2_class_)
            - [g2_seq_1_39_53       @ g2_0_37/all_frames_roi/swin_base_2_class](#g2_seq_1_39_53___g2_0_37_all_frames_roi_swin_base_2_class_)
    - [g3_4       @ swin_base_2_class](#g3_4___swin_base_2_clas_s_)
    - [g2_4       @ swin_base_2_class](#g2_4___swin_base_2_clas_s_)
        - [reorg_roi       @ g2_4/swin_base_2_class](#reorg_roi___g2_4_swin_base_2_class_)
        - [all_frames_roi       @ g2_4/swin_base_2_class](#all_frames_roi___g2_4_swin_base_2_class_)
        - [all_frames_roi_7777_10249_10111_13349       @ g2_4/swin_base_2_class](#all_frames_roi_7777_10249_10111_13349___g2_4_swin_base_2_class_)
        - [all_frames_roi_8094_13016_11228_15282       @ g2_4/swin_base_2_class](#all_frames_roi_8094_13016_11228_15282___g2_4_swin_base_2_class_)
        - [test       @ g2_4/swin_base_2_class](#test___g2_4_swin_base_2_class_)
        - [nd03       @ g2_4/swin_base_2_class](#nd03___g2_4_swin_base_2_class_)
        - [realtime_test_images       @ g2_4/swin_base_2_class](#realtime_test_images___g2_4_swin_base_2_class_)
    - [g4       @ swin_base_2_class](#g4___swin_base_2_clas_s_)
        - [nd03       @ g4/swin_base_2_class](#nd03___g4_swin_base_2_class_)
    - [g3       @ swin_base_2_class](#g3___swin_base_2_clas_s_)
- [swin_base_5_class](#swin_base_5_class_)
    - [imagenet_pretrained       @ swin_base_5_class](#imagenet_pretrained___swin_base_5_clas_s_)
        - [single_gpu       @ imagenet_pretrained/swin_base_5_class](#single_gpu___imagenet_pretrained_swin_base_5_clas_s_)
            - [g3_4s       @ single_gpu/imagenet_pretrained/swin_base_5_class](#g3_4s___single_gpu_imagenet_pretrained_swin_base_5_class_)
                - [test       @ g3_4s/single_gpu/imagenet_pretrained/swin_base_5_class](#test___g3_4s_single_gpu_imagenet_pretrained_swin_base_5_class_)
                - [on_unlabeled       @ g3_4s/single_gpu/imagenet_pretrained/swin_base_5_class](#on_unlabeled___g3_4s_single_gpu_imagenet_pretrained_swin_base_5_class_)
            - [g3       @ single_gpu/imagenet_pretrained/swin_base_5_class](#g3___single_gpu_imagenet_pretrained_swin_base_5_class_)
                - [test       @ g3/single_gpu/imagenet_pretrained/swin_base_5_class](#test___g3_single_gpu_imagenet_pretrained_swin_base_5_clas_s_)
                - [on_g3       @ g3/single_gpu/imagenet_pretrained/swin_base_5_class](#on_g3___g3_single_gpu_imagenet_pretrained_swin_base_5_clas_s_)
                - [on_g4s       @ g3/single_gpu/imagenet_pretrained/swin_base_5_class](#on_g4s___g3_single_gpu_imagenet_pretrained_swin_base_5_clas_s_)
                - [on_unlabeled       @ g3/single_gpu/imagenet_pretrained/swin_base_5_class](#on_unlabeled___g3_single_gpu_imagenet_pretrained_swin_base_5_clas_s_)
            - [g3_50_50       @ single_gpu/imagenet_pretrained/swin_base_5_class](#g3_50_50___single_gpu_imagenet_pretrained_swin_base_5_class_)
                - [test       @ g3_50_50/single_gpu/imagenet_pretrained/swin_base_5_class](#test___g3_50_50_single_gpu_imagenet_pretrained_swin_base_5_clas_s_)
        - [multi_gpu       @ imagenet_pretrained/swin_base_5_class](#multi_gpu___imagenet_pretrained_swin_base_5_clas_s_)
            - [g3_4s_50_50       @ multi_gpu/imagenet_pretrained/swin_base_5_class](#g3_4s_50_50___multi_gpu_imagenet_pretrained_swin_base_5_clas_s_)
                - [test       @ g3_4s_50_50/multi_gpu/imagenet_pretrained/swin_base_5_class](#test___g3_4s_50_50_multi_gpu_imagenet_pretrained_swin_base_5_clas_s_)
            - [g3_4s       @ multi_gpu/imagenet_pretrained/swin_base_5_class](#g3_4s___multi_gpu_imagenet_pretrained_swin_base_5_clas_s_)
                - [test       @ g3_4s/multi_gpu/imagenet_pretrained/swin_base_5_class](#test___g3_4s_multi_gpu_imagenet_pretrained_swin_base_5_clas_s_)
                - [nd03       @ g3_4s/multi_gpu/imagenet_pretrained/swin_base_5_class](#nd03___g3_4s_multi_gpu_imagenet_pretrained_swin_base_5_clas_s_)
            - [g4s_50_50       @ multi_gpu/imagenet_pretrained/swin_base_5_class](#g4s_50_50___multi_gpu_imagenet_pretrained_swin_base_5_clas_s_)
                - [test       @ g4s_50_50/multi_gpu/imagenet_pretrained/swin_base_5_class](#test___g4s_50_50_multi_gpu_imagenet_pretrained_swin_base_5_clas_s_)
            - [g4s       @ multi_gpu/imagenet_pretrained/swin_base_5_class](#g4s___multi_gpu_imagenet_pretrained_swin_base_5_clas_s_)
                - [test       @ g4s/multi_gpu/imagenet_pretrained/swin_base_5_class](#test___g4s_multi_gpu_imagenet_pretrained_swin_base_5_clas_s_)
                - [realtime_test_images       @ g4s/multi_gpu/imagenet_pretrained/swin_base_5_class](#realtime_test_images___g4s_multi_gpu_imagenet_pretrained_swin_base_5_clas_s_)
                - [on_g4s       @ g4s/multi_gpu/imagenet_pretrained/swin_base_5_class](#on_g4s___g4s_multi_gpu_imagenet_pretrained_swin_base_5_clas_s_)
                - [on_g3       @ g4s/multi_gpu/imagenet_pretrained/swin_base_5_class](#on_g3___g4s_multi_gpu_imagenet_pretrained_swin_base_5_clas_s_)
                - [on_unlabeled       @ g4s/multi_gpu/imagenet_pretrained/swin_base_5_class](#on_unlabeled___g4s_multi_gpu_imagenet_pretrained_swin_base_5_clas_s_)
                - [nd03       @ g4s/multi_gpu/imagenet_pretrained/swin_base_5_class](#nd03___g4s_multi_gpu_imagenet_pretrained_swin_base_5_clas_s_)
                - [test_30       @ g4s/multi_gpu/imagenet_pretrained/swin_base_5_class](#test_30___g4s_multi_gpu_imagenet_pretrained_swin_base_5_clas_s_)
    - [coco_pretrained       @ swin_base_5_class](#coco_pretrained___swin_base_5_clas_s_)
        - [multi_gpu       @ coco_pretrained/swin_base_5_class](#multi_gpu___coco_pretrained_swin_base_5_clas_s_)
- [stvm](#stv_m_)

<!-- /MarkdownTOC -->

<a id="coco1_7_"></a>
# coco17

<a id="cnx_base_in22k___coco17_"></a>
## cnx-base-in22k       @ coco17-->swin_det
tools/dist_train.sh configs/convnext/coco17-convnext_base_in22k.py 2 --init file:///tmp/cnx-base-in22k3 --cfg-options model.pretrained=pretrained/convnext_base_22k_224.pth data.samples_per_gpu=2

<a id="cnx_small_in22k___coco17_"></a>
## cnx-small-in22k       @ coco17-->swin_det
python3 -m  tools.train configs/convnext/cascade_mask_rcnn_convnext_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco_in22k.py --init file:///tmp/cnx-small-in22k  --cfg-options model.pretrained=pretrained/convnext_base_1k_224.pth.pth model.backbone.use_checkpoint=True data.samples_per_gpu=2

<a id="swin_small___coco17_"></a>
## swin-small       @ coco17-->swin_det
python3 -m  tools.train configs/swin/cascade_mask_rcnn_swin_small_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py --cfg-options model.pretrained=pretrained/cascade_mask_rcnn_swin_small_patch4_window7.pth model.backbone.use_checkpoint=True

<a id="swin_base_2_class_"></a>
# swin_base_2_class

<a id="ctc___swin_base_2_clas_s_"></a>
## ctc       @ swin_base_2_class-->swin_det
<a id="ctc_all___ctc_swin_base_2_clas_s_"></a>
### ctc_all       @ ctc/swin_base_2_class-->swin_det
tools/dist_train.sh configs/swin/ctc.py 2

python3 -m tools.train configs/swin/ctc.py --cfg-options data.samples_per_gpu=2

<a id="ctc_bf_c2dl_hsc___ctc_swin_base_2_clas_s_"></a>
### ctc_BF_C2DL_HSC       @ ctc/swin_base_2_class-->swin_det
CUDA_VISIBLE_DEVICES=1 python3 -m tools.train configs/swin/cascade_mask_rcnn_swin_base-ctc_BF_C2DL_HSC.py --init file:///tmp/ctc_BF_C2DL_HSC --cfg-options data.samples_per_gpu=3

<a id="ctc_bf_c2dl_musc___ctc_swin_base_2_clas_s_"></a>
### ctc_BF_C2DL_MuSC       @ ctc/swin_base_2_class-->swin_det
CUDA_VISIBLE_DEVICES=1 python3 -m tools.train configs/swin/cascade_mask_rcnn_swin_base-ctc_BF_C2DL_MuSC.py --init file:///tmp/ctc_BF_C2DL_MuSC --cfg-options data.samples_per_gpu=3

<a id="ctc_dic_c2dh_hela___ctc_swin_base_2_clas_s_"></a>
### ctc_DIC_C2DH_HeLa       @ ctc/swin_base_2_class-->swin_det
CUDA_VISIBLE_DEVICES=0 python3 -m tools.train configs/swin/cascade_mask_rcnn_swin_base-ctc_DIC_C2DH_HeLa.py --init file:///tmp/ctc_DIC_C2DH_HeLa --cfg-options data.samples_per_gpu=3

<a id="ctc_fluo_c2dl_msc___ctc_swin_base_2_clas_s_"></a>
### ctc_Fluo_C2DL_MSC       @ ctc/swin_base_2_class-->swin_det
CUDA_VISIBLE_DEVICES=1 python3 -m tools.train configs/swin/cascade_mask_rcnn_swin_base-ctc_Fluo_C2DL_MSC.py --init file:///tmp/ctc_Fluo_C2DL_MSC --cfg-options data.samples_per_gpu=3

<a id="ctc_phc_c2dh_u373___ctc_swin_base_2_clas_s_"></a>
### ctc_PhC_C2DH_U373       @ ctc/swin_base_2_class-->swin_det
CUDA_VISIBLE_DEVICES=1 python3 -m tools.train configs/swin/cascade_mask_rcnn_swin_base-ctc_PhC_C2DH_U373.py --init file:///tmp/ctc_PhC_C2DH_U373 --cfg-options data.samples_per_gpu=3

<a id="ctc_phc_c2dl_psc___ctc_swin_base_2_clas_s_"></a>
### ctc_PhC_C2DL_PSC       @ ctc/swin_base_2_class-->swin_det
CUDA_VISIBLE_DEVICES=0 python3 -m tools.train configs/swin/cascade_mask_rcnn_swin_base-ctc_PhC_C2DL_PSC.py --init file:///tmp/ctc_PhC_C2DL_PSC --cfg-options data.samples_per_gpu=3

<a id="fixed_lr___ctc_phc_c2dl_psc_ctc_swin_base_2_class_"></a>
#### fixed_lr       @ ctc_PhC_C2DL_PSC/ctc/swin_base_2_class-->swin_det
CUDA_VISIBLE_DEVICES=2 python3 -m tools.train configs/swin/cascade_mask_rcnn_swin_base-ctc_PhC_C2DL_PSC_fixed_lr.py --init file:///tmp/ctc_PhC_C2DL_PSC_fixed_lr --cfg-options data.samples_per_gpu=3

<a id="ctc_fluo_n2dh_gowt1___ctc_swin_base_2_clas_s_"></a>
### ctc_Fluo_N2DH_GOWT1       @ ctc/swin_base_2_class-->swin_det
CUDA_VISIBLE_DEVICES=1 python3 -m tools.train configs/swin/cascade_mask_rcnn_swin_base-ctc_Fluo_N2DH_GOWT1.py --init file:///tmp/ctc_Fluo_N2DH_GOWT1 --cfg-options data.samples_per_gpu=3

<a id="fixed_lr___ctc_fluo_n2dh_gowt1_ctc_swin_base_2_clas_s_"></a>
#### fixed_lr       @ ctc_Fluo_N2DH_GOWT1/ctc/swin_base_2_class-->swin_det
CUDA_VISIBLE_DEVICES=1 python3 -m tools.train configs/swin/cascade_mask_rcnn_swin_base-ctc_Fluo_N2DH_GOWT1_fixed_lr.py --init file:///tmp/ctc_Fluo_N2DH_GOWT1_fixed_lr.py --cfg-options data.samples_per_gpu=3 --resume-from work_dirs/cascade_mask_rcnn_swin_base-ctc_Fluo_N2DH_GOWT1_fixed_lr/latest.pth

<a id="ctc_fluo_n2dh_sim___ctc_swin_base_2_clas_s_"></a>
### ctc_Fluo_N2DH_SIM       @ ctc/swin_base_2_class-->swin_det
CUDA_VISIBLE_DEVICES=0 python3 -m tools.train configs/swin/cascade_mask_rcnn_swin_base-ctc_Fluo_N2DH_SIM.py --init file:///tmp/ctc_Fluo_N2DH_SIM --cfg-options data.samples_per_gpu=3

<a id="fixed_lr___ctc_fluo_n2dh_sim_ctc_swin_base_2_clas_s_"></a>
#### fixed_lr       @ ctc_Fluo_N2DH_SIM/ctc/swin_base_2_class-->swin_det
CUDA_VISIBLE_DEVICES=0 python3 -m tools.train configs/swin/cascade_mask_rcnn_swin_base-ctc_Fluo_N2DH_SIM_fixed_lr.py --init file:///tmp/ctc_Fluo_N2DH_SIM_fixed_lr --cfg-options data.samples_per_gpu=3

<a id="ctc_fluo_n2dl_hela___ctc_swin_base_2_clas_s_"></a>
### ctc_Fluo_N2DL_HeLa       @ ctc/swin_base_2_class-->swin_det
CUDA_VISIBLE_DEVICES=1 python3 -m tools.train configs/swin/cascade_mask_rcnn_swin_base-ctc_Fluo_N2DL_HeLa.py --init file:///tmp/ctc_Fluo_N2DL_HeLa --cfg-options data.samples_per_gpu=3 optimizer.weight_decay=0.001

<a id="fixed_lr___ctc_fluo_n2dl_hela_ctc_swin_base_2_class_"></a>
#### fixed_lr       @ ctc_Fluo_N2DL_HeLa/ctc/swin_base_2_class-->swin_det
CUDA_VISIBLE_DEVICES=1 python3 -m tools.train configs/swin/cascade_mask_rcnn_swin_base-ctc_Fluo_N2DL_HeLa_fixed_lr.py --init file:///tmp/ctc_Fluo_N2DL_HeLa_fixed_lr --cfg-options data.samples_per_gpu=3

<a id="ctmc___swin_base_2_clas_s_"></a>
## ctmc       @ swin_base_2_class-->swin_det
tools/dist_train.sh configs/swin/cascade_rcnn_swin_base_ctmc.py 2
tools/dist_train.sh configs/swin/cascade_rcnn_swin_base_ctmc_small.py 2

CUDA_VISIBLE_DEVICES=0 python3 -m tools.train configs/swin/cascade_rcnn_swin_base_ctmc.py --init file:///tmp/cascade_rcnn_swin_base_ctmc2 --cfg-options data.samples_per_gpu=2 --resume-from work_dirs/cascade_rcnn_swin_base_ctmc/latest.pth

<a id="ext_reorg_roi___swin_base_2_clas_s_"></a>
## ext_reorg_roi       @ swin_base_2_class-->swin_det
<a id="g2_0_37___ext_reorg_roi_swin_base_2_clas_s_"></a>
### g2_0_37       @ ext_reorg_roi/swin_base_2_class-->swin_det
tools/dist_train.sh configs/swin/ipsc_2_class_ext_reorg_roi_g2_0_37.py 2 --cfg-options model.pretrained=pretrained/swin_base_patch4_window12_384.pth model.backbone.use_checkpoint=True data.samples_per_gpu=6 --resume-from work_dirs/ipsc_2_class_ext_reorg_roi_g2_0_37/latest.pth

<a id="on_g2_38_53___g2_0_37_ext_reorg_roi_swin_base_2_clas_s_"></a>
#### on-g2_38_53       @ g2_0_37/ext_reorg_roi/swin_base_2_class-->swin_det
python3 tools/test.py config=configs/swin/ipsc_2_class_ext_reorg_roi_g2_0_37.py checkpoint=work_dirs/ipsc_2_class_ext_reorg_roi_g2_0_37/epoch_488.pth eval=bbox,segm test_name=g2_38_53

<a id="on_g2_38_53___g2_0_37_ext_reorg_roi_swin_base_2_clas_s__1"></a>
#### on-g2_38_53       @ g2_0_37/ext_reorg_roi/swin_base_2_class-->swin_det
python3 tools/test.py config=configs/swin/ipsc_2_class_ext_reorg_roi_g2_0_37.py checkpoint=work_dirs/ipsc_2_class_ext_reorg_roi_g2_0_37/epoch_488.pth eval=bbox,segm test_name=g2_38_53

<a id="g2_0_37_no_validate___ext_reorg_roi_swin_base_2_clas_s_"></a>
### g2_0_37-no_validate       @ ext_reorg_roi/swin_base_2_class-->swin_det
tools/dist_train.sh configs/swin/ipsc_2_class_ext_reorg_roi_g2_0_37-no_validate.py 2 --cfg-options model.pretrained=pretrained/swin_base_patch4_window12_384.pth model.backbone.use_checkpoint=True data.samples_per_gpu=4 data.workers_per_gpu=6 --resume-from work_dirs/ipsc_2_class_ext_reorg_roi_g2_0_37-no_validate/latest.pth --no-validate

<a id="on_g2_38_53___g2_0_37_no_validate_ext_reorg_roi_swin_base_2_clas_s_"></a>
#### on-g2_38_53       @ g2_0_37-no_validate/ext_reorg_roi/swin_base_2_class-->swin_det
<a id="epoch_2751___on_g2_38_53_g2_0_37_no_validate_ext_reorg_roi_swin_base_2_clas_s_"></a>
##### epoch_2751       @ on-g2_38_53/g2_0_37-no_validate/ext_reorg_roi/swin_base_2_class-->swin_det
python3 tools/test.py config=configs/swin/ipsc_2_class_ext_reorg_roi_g2_0_37-no_validate.py checkpoint=work_dirs/ipsc_2_class_ext_reorg_roi_g2_0_37-no_validate/epoch_2751.pth eval=bbox,segm test_name=g2_38_53
<a id="epoch_6638___on_g2_38_53_g2_0_37_no_validate_ext_reorg_roi_swin_base_2_clas_s_"></a>
##### epoch_6638       @ on-g2_38_53/g2_0_37-no_validate/ext_reorg_roi/swin_base_2_class-->swin_det
python3 tools/test.py config=configs/swin/ipsc_2_class_ext_reorg_roi_g2_0_37-no_validate.py checkpoint=work_dirs/ipsc_2_class_ext_reorg_roi_g2_0_37-no_validate/epoch_6638.pth eval=bbox,segm test_name=g2_38_53

<a id="rcnn___ext_reorg_roi_swin_base_2_clas_s_"></a>
### rcnn       @ ext_reorg_roi/swin_base_2_class-->swin_det

<a id="g2_0_37_rcnn___rcnn_ext_reorg_roi_swin_base_2_class_"></a>
#### g2_0_37-rcnn       @ rcnn/ext_reorg_roi/swin_base_2_class-->swin_det
tools/dist_train.sh configs/swin/ipsc_2_class_ext_reorg_roi_g2_0_37-rcnn.py 2 --cfg-options model.pretrained=pretrained/swin_base_patch4_window12_384.pth model.backbone.use_checkpoint=True data.samples_per_gpu=6 data.workers_per_gpu=6

<a id="g2_0_37_rcnn_win7___rcnn_ext_reorg_roi_swin_base_2_class_"></a>
#### g2_0_37-rcnn-win7       @ rcnn/ext_reorg_roi/swin_base_2_class-->swin_det
tools/dist_train.sh configs/swin/ipsc_2_class_ext_reorg_roi_g2_0_37-rcnn-win7.py 2 --cfg-options model.pretrained=pretrained/swin_base_patch4_window7_224_22k.pth model.backbone.use_checkpoint=True data.samples_per_gpu=6 data.workers_per_gpu=6

<a id="g2_0_37_no_validate_rcnn___rcnn_ext_reorg_roi_swin_base_2_class_"></a>
#### g2_0_37-no_validate-rcnn       @ rcnn/ext_reorg_roi/swin_base_2_class-->swin_det
tools/dist_train.sh configs/swin/ipsc_2_class_ext_reorg_roi_g2_0_37-no_validate-rcnn.py 2 --cfg-options model.pretrained=pretrained/swin_base_patch4_window12_384.pth model.backbone.use_checkpoint=True data.samples_per_gpu=6 data.workers_per_gpu=6 --no-validate
<a id="on_g2_38_53___g2_0_37_no_validate_rcnn_rcnn_ext_reorg_roi_swin_base_2_clas_s_"></a>
##### on-g2_38_53       @ g2_0_37-no_validate-rcnn/rcnn/ext_reorg_roi/swin_base_2_class-->swin_det
python3 tools/test.py config=configs/swin/ipsc_2_class_ext_reorg_roi_g2_0_37-no_validate-rcnn.py checkpoint=work_dirs/ipsc_2_class_ext_reorg_roi_g2_0_37-no_validate-rcnn/epoch_144.pth eval=bbox,segm test_name=g2_38_53

<a id="g2_0_37_no_validate_rcnn_win7___rcnn_ext_reorg_roi_swin_base_2_class_"></a>
#### g2_0_37-no_validate-rcnn-win7       @ rcnn/ext_reorg_roi/swin_base_2_class-->swin_det
tools/dist_train.sh configs/swin/ipsc_2_class_ext_reorg_roi_g2_0_37-no_validate-rcnn-win7.py 2 --cfg-options model.pretrained=pretrained/swin_base_patch4_window7_224_22k.pth model.backbone.use_checkpoint=True data.samples_per_gpu=6 data.workers_per_gpu=6 --no-validate
<a id="on_g2_38_53___g2_0_37_no_validate_rcnn_win7_rcnn_ext_reorg_roi_swin_base_2_class_"></a>
##### on-g2_38_53       @ g2_0_37-no_validate-rcnn-win7/rcnn/ext_reorg_roi/swin_base_2_class-->swin_det
python3 tools/test.py config=configs/swin/ipsc_2_class_ext_reorg_roi_g2_0_37-no_validate-rcnn.py checkpoint=work_dirs/ipsc_2_class_ext_reorg_roi_g2_0_37-no_validate-rcnn/epoch_144.pth eval=bbox,segm test_name=g2_38_53

<a id="g2_15_53_no_validate___ext_reorg_roi_swin_base_2_clas_s_"></a>
### g2_15_53-no_validate       @ ext_reorg_roi/swin_base_2_class-->swin_det
tools/dist_train.sh configs/swin/ipsc_2_class_ext_reorg_roi_g2_15_53-no_validate.py 2 --cfg-options model.pretrained=pretrained/swin_base_patch4_window12_384.pth model.backbone.use_checkpoint=True data.samples_per_gpu=4 data.workers_per_gpu=4 --no-validate

<a id="on_g2_0_14___g2_15_53_no_validate_ext_reorg_roi_swin_base_2_class_"></a>
#### on-g2_0_14       @ g2_15_53-no_validate/ext_reorg_roi/swin_base_2_class-->swin_det
python3 tools/test.py config=configs/swin/ipsc_2_class_ext_reorg_roi_g2_0_37-no_validate.py checkpoint=work_dirs/ipsc_2_class_ext_reorg_roi_g2_0_37-no_validate/epoch_2751.pth eval=bbox,segm test_name=g2_0_14

<a id="g2_16_53_no_validate___ext_reorg_roi_swin_base_2_clas_s_"></a>
### g2_16_53-no_validate       @ ext_reorg_roi/swin_base_2_class-->swin_det
tools/dist_train.sh configs/swin/ipsc_2_class_ext_reorg_roi_g2_16_53-no_validate.py 2 --cfg-options model.pretrained=pretrained/swin_base_patch4_window12_384.pth model.backbone.use_checkpoint=True data.samples_per_gpu=4 data.workers_per_gpu=4 --no-validate --resume-from work_dirs/ipsc_2_class_ext_reorg_roi_g2_16_53-no_validate/epoch_2554.pth

<a id="on_g2_0_15___g2_16_53_no_validate_ext_reorg_roi_swin_base_2_class_"></a>
#### on-g2_0_15       @ g2_16_53-no_validate/ext_reorg_roi/swin_base_2_class-->swin_det
python3 tools/test.py config=configs/swin/ipsc_2_class_ext_reorg_roi_g2_16_53-no_validate.py checkpoint=work_dirs/ipsc_2_class_ext_reorg_roi_g2_16_53-no_validate/epoch_3273.pth eval=bbox,segm test_name=g2_0_15

<a id="g2_54_126_no_validate_coco___ext_reorg_roi_swin_base_2_clas_s_"></a>
### g2_54_126-no_validate-coco       @ ext_reorg_roi/swin_base_2_class-->swin_det
tools/dist_train.sh configs/swin/ipsc_2_class_ext_reorg_roi_g2_54_126-no_validate-coco.py 2 --cfg-options model.pretrained=pretrained/cascade_mask_rcnn_swin_base_patch4_window7.pth model.backbone.use_checkpoint=True data.samples_per_gpu=2 data.workers_per_gpu=2 --no-validate

<a id="g2_54_126_no_validate___ext_reorg_roi_swin_base_2_clas_s_"></a>
### g2_54_126-no_validate       @ ext_reorg_roi/swin_base_2_class-->swin_det
tools/dist_train.sh configs/swin/ipsc_2_class_ext_reorg_roi_g2_54_126-no_validate.py 2 --cfg-options model.pretrained=pretrained/swin_base_patch4_window12_384.pth model.backbone.use_checkpoint=True data.samples_per_gpu=4 data.workers_per_gpu=4 --no-validate --resume-from work_dirs/ipsc_2_class_ext_reorg_roi_g2_54_126-no_validate/latest.pth

<a id="on_g2_0_53___g2_54_126_no_validate_ext_reorg_roi_swin_base_2_clas_s_"></a>
#### on-g2_0_53       @ g2_54_126-no_validate/ext_reorg_roi/swin_base_2_class-->swin_det
python3 tools/test.py config=configs/swin/ipsc_2_class_ext_reorg_roi_g2_54_126-no_validate.py checkpoint=work_dirs/ipsc_2_class_ext_reorg_roi_g2_54_126-no_validate/epoch_2058.pth eval=bbox,segm test_name=g2_0_53

<a id="all_frames_roi___swin_base_2_clas_s_"></a>
## all_frames_roi       @ swin_base_2_class-->swin_det
<a id="g2_0_37___all_frames_roi_swin_base_2_class_"></a>
### g2_0_37       @ all_frames_roi/swin_base_2_class-->swin_det
tools/dist_train.sh configs/swin/ipsc_2_class_all_frames_roi_g2_0_37.py 3 --cfg-options model.pretrained=pretrained/swin_base_patch4_window12_384.pth model.backbone.use_checkpoint=True data.samples_per_gpu=1

CUDA_VISIBLE_DEVICES=0 python3 -m tools.train configs/swin/ipsc_2_class_all_frames_roi_g2_0_37.py --init file:///tmp/ipsc_2_class_all_frames_roi_g2_0_37_2 --cfg-options data.samples_per_gpu=1
<a id="no_validate___g2_0_37_all_frames_roi_swin_base_2_class_"></a>
#### no-validate       @ g2_0_37/all_frames_roi/swin_base_2_class-->swin_det
CUDA_VISIBLE_DEVICES=1 python3 -m tools.train configs/swin/ipsc_2_class_all_frames_roi_g2_0_37_backup.py --init file:///tmp/ipsc_2_class_all_frames_roi_g2_0_37_3 --cfg-options data.samples_per_gpu=1 --no-validate

<a id="g2_38_53___g2_0_37_all_frames_roi_swin_base_2_class_"></a>
#### g2_38_53       @ g2_0_37/all_frames_roi/swin_base_2_class-->swin_det
python3 tools/test.py config=configs/swin/ipsc_2_class_all_frames_roi_g2_0_37.py checkpoint=work_dirs/ipsc_2_class_all_frames_roi_g2_0_37/epoch_1000.pth eval=bbox,segm test_name=g2_38_53

<a id="g2_seq_1_39_53___g2_0_37_all_frames_roi_swin_base_2_class_"></a>
#### g2_seq_1_39_53       @ g2_0_37/all_frames_roi/swin_base_2_class-->swin_det
python3 tools/test.py config=configs/swin/ipsc_2_class_all_frames_roi_g2_0_37.py checkpoint=work_dirs/ipsc_2_class_all_frames_roi_g2_0_37/epoch_1000.pth eval=bbox,segm test_name=g2_seq_1_39_53 

<a id="g3_4___swin_base_2_clas_s_"></a>
## g3_4       @ swin_base_2_class-->swin_det
tools/dist_train.sh configs/swin/ipsc_2_class_g3_4.py 2 --cfg-options model.pretrained=pretrained/swin_base_patch4_window12_384.pth model.backbone.use_checkpoint=True --no-validate

<a id="g2_4___swin_base_2_clas_s_"></a>
## g2_4       @ swin_base_2_class-->swin_det
tools/dist_train.sh configs/swin/ipsc_2_class_g2_4.py 2 --cfg-options model.pretrained=pretrained/swin_base_patch4_window12_384.pth model.backbone.use_checkpoint=True --no-validate

<a id="reorg_roi___g2_4_swin_base_2_class_"></a>
### reorg_roi       @ g2_4/swin_base_2_class-->swin_det
python3 tools/test.py config=configs/swin/ipsc_2_class_g2_4.py checkpoint=work_dirs/ipsc_2_class_g2_4/epoch_1000.pth eval=bbox,segm show_dir=work_dirs/ipsc_2_class_g2_4/vis test_name=reorg_roi

<a id="all_frames_roi___g2_4_swin_base_2_class_"></a>
### all_frames_roi       @ g2_4/swin_base_2_class-->swin_det
python3 tools/test.py config=configs/swin/ipsc_2_class_g2_4.py checkpoint=work_dirs/ipsc_2_class_g2_4/epoch_1000.pth eval=bbox,segm show_dir=work_dirs/ipsc_2_class_g2_4/vis test_name=all_frames_roi

<a id="all_frames_roi_7777_10249_10111_13349___g2_4_swin_base_2_class_"></a>
### all_frames_roi_7777_10249_10111_13349       @ g2_4/swin_base_2_class-->swin_det
python3 tools/test.py config=configs/swin/ipsc_2_class_g2_4.py checkpoint=work_dirs/ipsc_2_class_g2_4/epoch_1000.pth eval=bbox,segm show_dir=work_dirs/ipsc_2_class_g2_4/vis test_name=all_frames_roi_7777_10249_10111_13349

<a id="all_frames_roi_8094_13016_11228_15282___g2_4_swin_base_2_class_"></a>
### all_frames_roi_8094_13016_11228_15282       @ g2_4/swin_base_2_class-->swin_det
python3 tools/test.py config=configs/swin/ipsc_2_class_g2_4.py checkpoint=work_dirs/ipsc_2_class_g2_4/epoch_1000.pth eval=bbox,segm show_dir=work_dirs/ipsc_2_class_g2_4/vis test_name=all_frames_roi_8094_13016_11228_15282

<a id="test___g2_4_swin_base_2_class_"></a>
### test       @ g2_4/swin_base_2_class-->swin_det
python3 tools/test.py config=configs/swin/ipsc_2_class_g2_4.py checkpoint=work_dirs/ipsc_2_class_g2_4/epoch_1000.pth eval=bbox,segm show_dir=work_dirs/ipsc_2_class_g2_4/vis test_name=test

<a id="nd03___g2_4_swin_base_2_class_"></a>
### nd03       @ g2_4/swin_base_2_class-->swin_det
python3 tools/test.py config=configs/swin/ipsc_2_class_g2_4.py checkpoint=work_dirs/ipsc_2_class_g2_4/epoch_1000.pth eval=bbox,segm show_dir=work_dirs/ipsc_2_class_g2_4/vis test_name=nd03 show=0 

<a id="realtime_test_images___g2_4_swin_base_2_class_"></a>
### realtime_test_images       @ g2_4/swin_base_2_class-->swin_det
python3 tools/test.py config=configs/swin/ipsc_2_class_g2_4.py checkpoint=work_dirs/ipsc_2_class_g2_4/epoch_1000.pth eval=bbox,segm show_dir=work_dirs/ipsc_2_class_g2_4/vis test_name=realtime_test_images multi_run=1 show=1

<a id="g4___swin_base_2_clas_s_"></a>
## g4       @ swin_base_2_class-->swin_det
tools/dist_train.sh configs/swin/ipsc_2_class_g4.py 2 --cfg-options model.pretrained=pretrained/swin_base_patch4_window12_384.pth model.backbone.use_checkpoint=True --no-validate

<a id="nd03___g4_swin_base_2_class_"></a>
### nd03       @ g4/swin_base_2_class-->swin_det
python3 tools/test.py config=configs/swin/ipsc_2_class_g4.py checkpoint=work_dirs/ipsc_2_class_g4/epoch_500.pth eval=bbox,segm show_dir=work_dirs/ipsc_2_class_g4/vis test_name=nd03 show=0 

<a id="g3___swin_base_2_clas_s_"></a>
## g3       @ swin_base_2_class-->swin_det
tools/dist_train.sh configs/swin/ipsc_2_class_g3.py 2 --cfg-options model.pretrained=pretrained/swin_base_patch4_window12_384.pth model.backbone.use_checkpoint=True --no-validate

<a id="swin_base_5_class_"></a>
# swin_base_5_class

<a id="imagenet_pretrained___swin_base_5_clas_s_"></a>
## imagenet_pretrained       @ swin_base_5_class-->swin_det

<a id="single_gpu___imagenet_pretrained_swin_base_5_clas_s_"></a>
### single_gpu       @ imagenet_pretrained/swin_base_5_class-->swin_det

<a id="g3_4s___single_gpu_imagenet_pretrained_swin_base_5_class_"></a>
#### g3_4s       @ single_gpu/imagenet_pretrained/swin_base_5_class-->swin_det
python3 -m tools.train configs/swin/ipsc_5_class.py --cfg-options model.pretrained=pretrained/swin_base_patch4_window12_384.pth model.backbone.use_checkpoint=True --no-validate

<a id="test___g3_4s_single_gpu_imagenet_pretrained_swin_base_5_class_"></a>
##### test       @ g3_4s/single_gpu/imagenet_pretrained/swin_base_5_class-->swin_det
python3 tools/test.py config=configs/swin/ipsc_5_class.py checkpoint=work_dirs/ipsc_5_class/epoch_28.pth eval=bbox,segm show=1 show_dir=work_dirs/ipsc_5_class/vis

python3 tools/test.py config=configs/swin/ipsc_5_class.py checkpoint=work_dirs/ipsc_5_class/epoch_28.pth eval=bbox,segm 

<a id="on_unlabeled___g3_4s_single_gpu_imagenet_pretrained_swin_base_5_class_"></a>
##### on_unlabeled       @ g3_4s/single_gpu/imagenet_pretrained/swin_base_5_class-->swin_det

python3 tools/test.py config=configs/swin/ipsc_5_class.py checkpoint=work_dirs/ipsc_5_class/epoch_28.pth show=1 show_dir=work_dirs/ipsc_5_class/vis_unlabeled test_name=unlabeled

<a id="g3___single_gpu_imagenet_pretrained_swin_base_5_class_"></a>
#### g3       @ single_gpu/imagenet_pretrained/swin_base_5_class-->swin_det
python3 -m tools.train configs/swin/ipsc_5_class_g3.py --cfg-options model.pretrained=pretrained/swin_base_patch4_window12_384.pth model.backbone.use_checkpoint=True --no-validate --launcher none

<a id="test___g3_single_gpu_imagenet_pretrained_swin_base_5_clas_s_"></a>
##### test       @ g3/single_gpu/imagenet_pretrained/swin_base_5_class-->swin_det
CUDA_VISIBLE_DEVICES=1 python3 tools/test.py config=configs/swin/ipsc_5_class_g3.py checkpoint=work_dirs/ipsc_5_class_g3/epoch_50.pth eval=bbox,segm show=1 show_dir=work_dirs/ipsc_5_class_g3/vis

<a id="on_g3___g3_single_gpu_imagenet_pretrained_swin_base_5_clas_s_"></a>
##### on_g3       @ g3/single_gpu/imagenet_pretrained/swin_base_5_class-->swin_det
CUDA_VISIBLE_DEVICES=0 python3 tools/test.py config=configs/swin/ipsc_5_class_g3.py checkpoint=work_dirs/ipsc_5_class_g3/epoch_50.pth eval=bbox,segm test_name=g3

<a id="on_g4s___g3_single_gpu_imagenet_pretrained_swin_base_5_clas_s_"></a>
##### on_g4s       @ g3/single_gpu/imagenet_pretrained/swin_base_5_class-->swin_det
CUDA_VISIBLE_DEVICES=1 python3 tools/test.py config=configs/swin/ipsc_5_class_g3.py checkpoint=work_dirs/ipsc_5_class_g3/epoch_50.pth eval=bbox,segm test_name=g4s

<a id="on_unlabeled___g3_single_gpu_imagenet_pretrained_swin_base_5_clas_s_"></a>
##### on_unlabeled       @ g3/single_gpu/imagenet_pretrained/swin_base_5_class-->swin_det
python3 tools/test.py config=configs/swin/ipsc_5_class_g3.py checkpoint=work_dirs/ipsc_5_class_g3/epoch_50.pth show=1 show_dir=work_dirs/ipsc_5_class_g3/vis_unlabeled test_name=unlabeled

<a id="g3_50_50___single_gpu_imagenet_pretrained_swin_base_5_class_"></a>
#### g3_50_50       @ single_gpu/imagenet_pretrained/swin_base_5_class-->swin_det
python3 -m tools.train configs/swin/ipsc_5_class_g3_50_50.py --cfg-options model.pretrained=pretrained/swin_base_patch4_window12_384.pth model.backbone.use_checkpoint=True --no-validate --launcher none

<a id="test___g3_50_50_single_gpu_imagenet_pretrained_swin_base_5_clas_s_"></a>
##### test       @ g3_50_50/single_gpu/imagenet_pretrained/swin_base_5_class-->swin_det
CUDA_VISIBLE_DEVICES=1 python3 tools/test.py config=configs/swin/ipsc_5_class_g3_50_50.py checkpoint=work_dirs/ipsc_5_class_g3_50_50/epoch_50.pth eval=bbox,segm show=1 show_dir=work_dirs/ipsc_5_class_g3_50_50/vis

<a id="multi_gpu___imagenet_pretrained_swin_base_5_clas_s_"></a>
### multi_gpu       @ imagenet_pretrained/swin_base_5_class-->swin_det

<a id="g3_4s_50_50___multi_gpu_imagenet_pretrained_swin_base_5_clas_s_"></a>
#### g3_4s_50_50       @ multi_gpu/imagenet_pretrained/swin_base_5_class-->swin_det
tools/dist_train.sh configs/swin/ipsc_5_class_50_50.py 2 --cfg-options model.pretrained=pretrained/swin_base_patch4_window12_384.pth model.backbone.use_checkpoint=True --no-validate

<a id="test___g3_4s_50_50_multi_gpu_imagenet_pretrained_swin_base_5_clas_s_"></a>
##### test       @ g3_4s_50_50/multi_gpu/imagenet_pretrained/swin_base_5_class-->swin_det
python3 tools/test.py config=configs/swin/ipsc_5_class_50_50.py checkpoint=work_dirs/ipsc_5_class_50_50/epoch_100.pth eval=bbox,segm show=1 show_dir=work_dirs/ipsc_5_class_50_50/vis


<a id="g3_4s___multi_gpu_imagenet_pretrained_swin_base_5_clas_s_"></a>
#### g3_4s       @ multi_gpu/imagenet_pretrained/swin_base_5_class-->swin_det
tools/dist_train.sh configs/swin/ipsc_5_class.py 2 --cfg-options model.pretrained=pretrained/swin_base_patch4_window12_384.pth model.backbone.use_checkpoint=True --no-validate

<a id="test___g3_4s_multi_gpu_imagenet_pretrained_swin_base_5_clas_s_"></a>
##### test       @ g3_4s/multi_gpu/imagenet_pretrained/swin_base_5_class-->swin_det
tools/dist_test.sh configs/swin/ipsc_5_class.py work_dirs/ipsc_5_class/epoch_36.pth 2 --eval bbox segm

<a id="nd03___g3_4s_multi_gpu_imagenet_pretrained_swin_base_5_clas_s_"></a>
##### nd03       @ g3_4s/multi_gpu/imagenet_pretrained/swin_base_5_class-->swin_det
CUDA_VISIBLE_DEVICES=1 python3 tools/test.py config=configs/swin/ipsc_5_class.py checkpoint=work_dirs/ipsc_5_class/epoch_100.pth show=1 show_dir=work_dirs/ipsc_5_class/vis_nd03 test_name=nd03

<a id="g4s_50_50___multi_gpu_imagenet_pretrained_swin_base_5_clas_s_"></a>
#### g4s_50_50       @ multi_gpu/imagenet_pretrained/swin_base_5_class-->swin_det
tools/dist_train.sh configs/swin/ipsc_5_class_g4s_50_50.py 2 --cfg-options model.pretrained=pretrained/swin_base_patch4_window12_384.pth model.backbone.use_checkpoint=True --no-validate

<a id="test___g4s_50_50_multi_gpu_imagenet_pretrained_swin_base_5_clas_s_"></a>
##### test       @ g4s_50_50/multi_gpu/imagenet_pretrained/swin_base_5_class-->swin_det
python3 tools/test.py config=configs/swin/ipsc_5_class_g4s_50_50.py checkpoint=work_dirs/ipsc_5_class_g4s_50_50/epoch_100.pth eval=bbox,segm show=1 show_dir=work_dirs/ipsc_5_class_g4s_50_50/vis

<a id="g4s___multi_gpu_imagenet_pretrained_swin_base_5_clas_s_"></a>
#### g4s       @ multi_gpu/imagenet_pretrained/swin_base_5_class-->swin_det
tools/dist_train.sh configs/swin/ipsc_5_class_g4s.py 2 --cfg-options model.pretrained=pretrained/swin_base_patch4_window12_384.pth model.backbone.use_checkpoint=True --no-validate

<a id="test___g4s_multi_gpu_imagenet_pretrained_swin_base_5_clas_s_"></a>
##### test       @ g4s/multi_gpu/imagenet_pretrained/swin_base_5_class-->swin_det
CUDA_VISIBLE_DEVICES=0 python3 tools/test.py config=configs/swin/ipsc_5_class_g4s.py checkpoint=work_dirs/ipsc_5_class_g4s/epoch_100.pth eval=bbox,segm show=1 show_dir=work_dirs/ipsc_5_class_g4s/vis

<a id="realtime_test_images___g4s_multi_gpu_imagenet_pretrained_swin_base_5_clas_s_"></a>
##### realtime_test_images       @ g4s/multi_gpu/imagenet_pretrained/swin_base_5_class-->swin_det
python3 tools/test.py config=configs/swin/ipsc_5_class_g4s.py checkpoint=work_dirs/ipsc_5_class_g4s/epoch_100.pth eval=bbox,segm show=0 show_dir=work_dirs/ipsc_5_class_g4s/vis test_name=realtime_test_images multi_run=1 show=1

<a id="on_g4s___g4s_multi_gpu_imagenet_pretrained_swin_base_5_clas_s_"></a>
##### on_g4s       @ g4s/multi_gpu/imagenet_pretrained/swin_base_5_class-->swin_det
CUDA_VISIBLE_DEVICES=0 python3 tools/test.py config=configs/swin/ipsc_5_class_g4s.py checkpoint=work_dirs/ipsc_5_class_g4s/epoch_100.pth eval=bbox,segm show=0  test_name=g4s

<a id="on_g3___g4s_multi_gpu_imagenet_pretrained_swin_base_5_clas_s_"></a>
##### on_g3       @ g4s/multi_gpu/imagenet_pretrained/swin_base_5_class-->swin_det
CUDA_VISIBLE_DEVICES=1 python3 tools/test.py config=configs/swin/ipsc_5_class_g4s.py checkpoint=work_dirs/ipsc_5_class_g4s/epoch_100.pth eval=bbox,segm show=0 test_name=g3

<a id="on_unlabeled___g4s_multi_gpu_imagenet_pretrained_swin_base_5_clas_s_"></a>
##### on_unlabeled       @ g4s/multi_gpu/imagenet_pretrained/swin_base_5_class-->swin_det
python3 tools/test.py config=configs/swin/ipsc_5_class_g4s.py checkpoint=work_dirs/ipsc_5_class_g4s/epoch_100.pth show=1 show_dir=work_dirs/ipsc_5_class_g4s/vis_unlabeled test_name=unlabeled

<a id="nd03___g4s_multi_gpu_imagenet_pretrained_swin_base_5_clas_s_"></a>
##### nd03       @ g4s/multi_gpu/imagenet_pretrained/swin_base_5_class-->swin_det
python3 tools/test.py config=configs/swin/ipsc_5_class_g4s.py checkpoint=work_dirs/ipsc_5_class_g4s/epoch_100.pth show=1 show_dir=work_dirs/ipsc_5_class_g4s/vis_nd03 test_name=nd03

<a id="test_30___g4s_multi_gpu_imagenet_pretrained_swin_base_5_clas_s_"></a>
##### test_30       @ g4s/multi_gpu/imagenet_pretrained/swin_base_5_class-->swin_det
CUDA_VISIBLE_DEVICES=1 python3 tools/test.py config=configs/swin/ipsc_5_class_g4s.py checkpoint=work_dirs/ipsc_5_class_g4s/epoch_30.pth eval=bbox,segm show=1 show_dir=work_dirs/ipsc_5_class_g4s/vis_30

<a id="coco_pretrained___swin_base_5_clas_s_"></a>
## coco_pretrained       @ swin_base_5_class-->swin_det

python3 -m tools.train configs/swin/ipsc_5_class.py --cfg-options model.pretrained=pretrained/patch4_window7.pth model.backbone.use_checkpoint=True

<a id="multi_gpu___coco_pretrained_swin_base_5_clas_s_"></a>
### multi_gpu       @ coco_pretrained/swin_base_5_class-->swin_det

tools/dist_train.sh configs/swin/patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py 2 --cfg-options model.pretrained=pretrained/patch4_window7.pth

<a id="stv_m_"></a>
# stvm


stvm1 "C:\Datasets\ipsc_5_class\blended_vis,ipsc_5_class/vis"
stvm1 "C:\Datasets\ipsc_5_class\blended_vis,ipsc_5_class_g3/vis"
stvm1 "C:\Datasets\ipsc_5_class\blended_vis,ipsc_5_class_g4s/vis"

stvm "C:\Datasets\ipsc_5_class\blended_vis,ipsc_5_class_50_50/vis"

stvm1 "C:\Datasets\ipsc_5_class\blended_vis,ipsc_5_class_50_50/vis"
stvm1 "C:\Datasets\ipsc_5_class\blended_vis,ipsc_5_class_g3_50_50/vis"
stvm1 "C:\Datasets\ipsc_5_class\blended_vis,ipsc_5_class_g4s_50_50/vis"



