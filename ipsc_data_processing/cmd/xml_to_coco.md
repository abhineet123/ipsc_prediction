<!-- MarkdownTOC -->

- [imgn](#img_n_)
    - [vid       @ imgn](#vid___imgn_)
        - [8_per_seq_random       @ vid/imgn](#8_per_seq_random___vid_imgn_)
    - [vid_val       @ imgn](#vid_val___imgn_)
        - [16_per_seq_random       @ vid_val/imgn](#16_per_seq_random___vid_val_imgn_)
    - [vid_det       @ imgn](#vid_det___imgn_)
        - [sampled_eq       @ vid_det/imgn](#sampled_eq___vid_det_imgn_)
        - [8_per_seq_random       @ vid_det/imgn](#8_per_seq_random___vid_det_imgn_)
        - [ratio_1_10_random       @ vid_det/imgn](#ratio_1_10_random___vid_det_imgn_)
    - [vid_det_all       @ imgn](#vid_det_all___imgn_)
        - [sampled_eq       @ vid_det_all/imgn](#sampled_eq___vid_det_all_imgn_)
        - [ratio_1_10_random       @ vid_det_all/imgn](#ratio_1_10_random___vid_det_all_imgn_)
    - [det_all       @ imgn](#det_all___imgn_)
    - [det_       @ imgn](#det___imgn_)
        - [del       @ det_/imgn](#del___det__img_n_)
    - [det_val       @ imgn](#det_val___imgn_)
        - [del       @ det_val/imgn](#del___det_val_imgn_)
- [detrac](#detra_c_)
    - [0_59       @ detrac](#0_59___detrac_)
        - [100_per_seq_random       @ 0_59/detrac](#100_per_seq_random___0_59_detra_c_)
        - [40_per_seq_random       @ 0_59/detrac](#40_per_seq_random___0_59_detra_c_)
    - [60_99       @ detrac](#60_99___detrac_)
        - [100_per_seq_random       @ 60_99/detrac](#100_per_seq_random___60_99_detrac_)
        - [40_per_seq_random       @ 60_99/detrac](#40_per_seq_random___60_99_detrac_)
- [detrac-non_empty](#detrac_non_empt_y_)
    - [0_19       @ detrac-non_empty](#0_19___detrac_non_empty_)
    - [0_9       @ detrac-non_empty](#0_9___detrac_non_empty_)
    - [0_48       @ detrac-non_empty](#0_48___detrac_non_empty_)
    - [49_68       @ detrac-non_empty](#49_68___detrac_non_empty_)
    - [49_85       @ detrac-non_empty](#49_85___detrac_non_empty_)
        - [100_per_seq_random       @ 49_85/detrac-non_empty](#100_per_seq_random___49_85_detrac_non_empty_)
- [ipsc       @ xml_to_coco](#ipsc___xml_to_coc_o_)
    - [all_frames_roi       @ ipsc](#all_frames_roi___ipsc_)
        - [g2_0_37       @ all_frames_roi/ipsc](#g2_0_37___all_frames_roi_ips_c_)
            - [no_val       @ g2_0_37/all_frames_roi/ipsc](#no_val___g2_0_37_all_frames_roi_ips_c_)
        - [g2_38_53       @ all_frames_roi/ipsc](#g2_38_53___all_frames_roi_ips_c_)
        - [g2_seq_1_39_53       @ all_frames_roi/ipsc](#g2_seq_1_39_53___all_frames_roi_ips_c_)
        - [g3_54_92       @ all_frames_roi/ipsc](#g3_54_92___all_frames_roi_ips_c_)
    - [ext_reorg_roi       @ ipsc](#ext_reorg_roi___ipsc_)
        - [0_0       @ ext_reorg_roi/ipsc](#0_0___ext_reorg_roi_ipsc_)
        - [0_1       @ ext_reorg_roi/ipsc](#0_1___ext_reorg_roi_ipsc_)
        - [0_126       @ ext_reorg_roi/ipsc](#0_126___ext_reorg_roi_ipsc_)
        - [0_37       @ ext_reorg_roi/ipsc](#0_37___ext_reorg_roi_ipsc_)
            - [no_validate       @ 0_37/ext_reorg_roi/ipsc](#no_validate___0_37_ext_reorg_roi_ips_c_)
            - [save_masks       @ 0_37/ext_reorg_roi/ipsc](#save_masks___0_37_ext_reorg_roi_ips_c_)
            - [no_mask       @ 0_37/ext_reorg_roi/ipsc](#no_mask___0_37_ext_reorg_roi_ips_c_)
        - [38_53       @ ext_reorg_roi/ipsc](#38_53___ext_reorg_roi_ipsc_)
            - [save_masks       @ 38_53/ext_reorg_roi/ipsc](#save_masks___38_53_ext_reorg_roi_ipsc_)
        - [15_53       @ ext_reorg_roi/ipsc](#15_53___ext_reorg_roi_ipsc_)
            - [no_validate       @ 15_53/ext_reorg_roi/ipsc](#no_validate___15_53_ext_reorg_roi_ipsc_)
        - [16_53       @ ext_reorg_roi/ipsc](#16_53___ext_reorg_roi_ipsc_)
            - [val-30       @ 16_53/ext_reorg_roi/ipsc](#val_30___16_53_ext_reorg_roi_ipsc_)
            - [no-val       @ 16_53/ext_reorg_roi/ipsc](#no_val___16_53_ext_reorg_roi_ipsc_)
                - [no_mask       @ no-val/16_53/ext_reorg_roi/ipsc](#no_mask___no_val_16_53_ext_reorg_roi_ips_c_)
        - [0_1       @ ext_reorg_roi/ipsc](#0_1___ext_reorg_roi_ipsc__1)
        - [0_15       @ ext_reorg_roi/ipsc](#0_15___ext_reorg_roi_ipsc_)
            - [no_mask       @ 0_15/ext_reorg_roi/ipsc](#no_mask___0_15_ext_reorg_roi_ips_c_)
        - [0_1       @ ext_reorg_roi/ipsc](#0_1___ext_reorg_roi_ipsc__2)
        - [2_3       @ ext_reorg_roi/ipsc](#2_3___ext_reorg_roi_ipsc_)
        - [54_126       @ ext_reorg_roi/ipsc](#54_126___ext_reorg_roi_ipsc_)
            - [strd-5       @ 54_126/ext_reorg_roi/ipsc](#strd_5___54_126_ext_reorg_roi_ips_c_)
            - [strd-8       @ 54_126/ext_reorg_roi/ipsc](#strd_8___54_126_ext_reorg_roi_ips_c_)
            - [list       @ 54_126/ext_reorg_roi/ipsc](#list___54_126_ext_reorg_roi_ips_c_)
            - [no_mask       @ 54_126/ext_reorg_roi/ipsc](#no_mask___54_126_ext_reorg_roi_ips_c_)
        - [g2_0_53       @ ext_reorg_roi/ipsc](#g2_0_53___ext_reorg_roi_ipsc_)
            - [no_mask       @ g2_0_53/ext_reorg_roi/ipsc](#no_mask___g2_0_53_ext_reorg_roi_ipsc_)
    - [ext_reorg_roi-no_annotations       @ ipsc](#ext_reorg_roi_no_annotations___ipsc_)
        - [reorg_roi       @ ext_reorg_roi-no_annotations/ipsc](#reorg_roi___ext_reorg_roi_no_annotations_ips_c_)
        - [all_frames_roi       @ ext_reorg_roi-no_annotations/ipsc](#all_frames_roi___ext_reorg_roi_no_annotations_ips_c_)
            - [all_frames_roi_7777_10249_10111_13349       @ all_frames_roi/ext_reorg_roi-no_annotations/ipsc](#all_frames_roi_7777_10249_10111_13349___all_frames_roi_ext_reorg_roi_no_annotations_ipsc_)
            - [all_frames_roi_8094_13016_11228_15282       @ all_frames_roi/ext_reorg_roi-no_annotations/ipsc](#all_frames_roi_8094_13016_11228_15282___all_frames_roi_ext_reorg_roi_no_annotations_ipsc_)
        - [Test_230710       @ ext_reorg_roi-no_annotations/ipsc](#test_230710___ext_reorg_roi_no_annotations_ips_c_)
        - [Test_230606       @ ext_reorg_roi-no_annotations/ipsc](#test_230606___ext_reorg_roi_no_annotations_ips_c_)
        - [Test_211208       @ ext_reorg_roi-no_annotations/ipsc](#test_211208___ext_reorg_roi_no_annotations_ips_c_)
        - [nd03       @ ext_reorg_roi-no_annotations/ipsc](#nd03___ext_reorg_roi_no_annotations_ips_c_)
    - [g2_4       @ ipsc](#g2_4___ipsc_)
    - [g4       @ ipsc](#g4___ipsc_)
    - [g3       @ ipsc](#g3___ipsc_)
- [ipsc_5_class       @ xml_to_coco](#ipsc_5_class___xml_to_coc_o_)
    - [Test_211208       @ ipsc_5_class](#test_211208___ipsc_5_class_)
    - [k       @ ipsc_5_class](#k___ipsc_5_class_)
    - [nd03       @ ipsc_5_class](#nd03___ipsc_5_class_)
    - [g3_4s       @ ipsc_5_class](#g3_4s___ipsc_5_class_)
        - [50_50       @ g3_4s/ipsc_5_class](#50_50___g3_4s_ipsc_5_class_)
        - [no_val       @ g3_4s/ipsc_5_class](#no_val___g3_4s_ipsc_5_class_)
    - [g3       @ ipsc_5_class](#g3___ipsc_5_class_)
        - [50_50       @ g3/ipsc_5_class](#50_50___g3_ipsc_5_clas_s_)
        - [no_val       @ g3/ipsc_5_class](#no_val___g3_ipsc_5_clas_s_)
    - [g4s       @ ipsc_5_class](#g4s___ipsc_5_class_)
        - [50_50       @ g4s/ipsc_5_class](#50_50___g4s_ipsc_5_class_)
        - [no_val       @ g4s/ipsc_5_class](#no_val___g4s_ipsc_5_class_)
- [ctc       @ xml_to_coco](#ctc___xml_to_coc_o_)
    - [ctc_all       @ ctc](#ctc_all___ct_c_)
    - [ctc_BF_C2DL_HSC       @ ctc](#ctc_bf_c2dl_hsc___ct_c_)
    - [ctc_BF_C2DL_MuSC       @ ctc](#ctc_bf_c2dl_musc___ct_c_)
    - [ctc_DIC_C2DH_HeLa       @ ctc](#ctc_dic_c2dh_hela___ct_c_)
    - [ctc_Fluo_C2DL_Huh7       @ ctc](#ctc_fluo_c2dl_huh7___ct_c_)
    - [ctc_Fluo_C2DL_MSC       @ ctc](#ctc_fluo_c2dl_msc___ct_c_)
    - [ctc_PhC_C2DH_U373       @ ctc](#ctc_phc_c2dh_u373___ct_c_)
    - [ctc_PhC_C2DL_PSC       @ ctc](#ctc_phc_c2dl_psc___ct_c_)
    - [ctc_Fluo_N2DH_GOWT1       @ ctc](#ctc_fluo_n2dh_gowt1___ct_c_)
    - [ctc_Fluo_N2DH_SIM       @ ctc](#ctc_fluo_n2dh_sim___ct_c_)
    - [ctc_Fluo_N2DL_HeLa       @ ctc](#ctc_fluo_n2dl_hela___ct_c_)
- [ctmc_all       @ xml_to_coco](#ctmc_all___xml_to_coc_o_)
- [acamp](#acamp_)
    - [1k8_vid_entire_seq       @ acamp](#1k8_vid_entire_seq___acam_p_)
    - [10k6_vid_entire_seq       @ acamp](#10k6_vid_entire_seq___acam_p_)
    - [20k6_5_video       @ acamp](#20k6_5_video___acam_p_)
    - [1k8_vid_entire_seq_inv       @ acamp](#1k8_vid_entire_seq_inv___acam_p_)
    - [10k6_vid_entire_seq_inv       @ acamp](#10k6_vid_entire_seq_inv___acam_p_)
    - [20k6_5_video_inv       @ acamp](#20k6_5_video_inv___acam_p_)
    - [1k8_vid_entire_seq_inv_2_per_seq       @ acamp](#1k8_vid_entire_seq_inv_2_per_seq___acam_p_)
    - [10k6_vid_entire_seq_inv_2_per_seq       @ acamp](#10k6_vid_entire_seq_inv_2_per_seq___acam_p_)
    - [20k6_5_video_inv_2_per_seq       @ acamp](#20k6_5_video_inv_2_per_seq___acam_p_)

<!-- /MarkdownTOC -->

<a id="img_n_"></a>
# imgn
<a id="vid___imgn_"></a>
## vid       @ imgn-->xml_to_coco
python xml_to_coco.py cfg=imgn:vid
<a id="8_per_seq_random___vid_imgn_"></a>
### 8_per_seq_random       @ vid/imgn-->xml_to_coco
python xml_to_coco.py cfg=imgn:vid:8_per_seq_random

<a id="vid_val___imgn_"></a>
## vid_val       @ imgn-->xml_to_coco
python xml_to_coco.py cfg=imgn:vid_val
<a id="16_per_seq_random___vid_val_imgn_"></a>
### 16_per_seq_random       @ vid_val/imgn-->xml_to_coco
python xml_to_coco.py cfg=imgn:vid_val:16_per_seq_random


<a id="vid_det___imgn_"></a>
## vid_det       @ imgn-->xml_to_coco
python xml_to_coco.py cfg=imgn:vid_det
<a id="sampled_eq___vid_det_imgn_"></a>
### sampled_eq       @ vid_det/imgn-->xml_to_coco
python xml_to_coco.py cfg=imgn:vid_det:sampled_eq:add
<a id="8_per_seq_random___vid_det_imgn_"></a>
### 8_per_seq_random       @ vid_det/imgn-->xml_to_coco
python xml_to_coco.py cfg=imgn:vid_det:8_per_seq_random:add
<a id="ratio_1_10_random___vid_det_imgn_"></a>
### ratio_1_10_random       @ vid_det/imgn-->xml_to_coco
python xml_to_coco.py cfg=imgn:vid_det:ratio_1_10_random:add

<a id="vid_det_all___imgn_"></a>
## vid_det_all       @ imgn-->xml_to_coco
python xml_to_coco.py cfg=imgn:vid_det_all
<a id="sampled_eq___vid_det_all_imgn_"></a>
### sampled_eq       @ vid_det_all/imgn-->xml_to_coco
python xml_to_coco.py cfg=imgn:vid_det_all:sampled_eq:add
<a id="ratio_1_10_random___vid_det_all_imgn_"></a>
### ratio_1_10_random       @ vid_det_all/imgn-->xml_to_coco
python xml_to_coco.py cfg=imgn:vid_det_all:ratio_1_10_random:add

<a id="det_all___imgn_"></a>
## det_all       @ imgn-->xml_to_coco
python xml_to_coco.py cfg=imgn:det_all
<a id="det___imgn_"></a>
## det_       @ imgn-->xml_to_coco
python xml_to_coco.py cfg=imgn:det_only
<a id="del___det__img_n_"></a>
### del       @ det_/imgn-->xml_to_coco
python xml_to_coco.py cfg=imgn:det_only:del-xml

<a id="det_val___imgn_"></a>
## det_val       @ imgn-->xml_to_coco
python xml_to_coco.py cfg=imgn:det_val
<a id="del___det_val_imgn_"></a>
### del       @ det_val/imgn-->xml_to_coco
python xml_to_coco.py cfg=imgn:det_val:del-xml

<a id="detra_c_"></a>
# detrac
<a id="0_59___detrac_"></a>
## 0_59       @ detrac-->xml_to_coco
python xml_to_coco.py cfg=detrac:0_59:zip:ign
<a id="100_per_seq_random___0_59_detra_c_"></a>
### 100_per_seq_random       @ 0_59/detrac-->xml_to_coco
python xml_to_coco.py cfg=detrac:0_59:100_per_seq_random:zip:ign
<a id="40_per_seq_random___0_59_detra_c_"></a>
### 40_per_seq_random       @ 0_59/detrac-->xml_to_coco
python xml_to_coco.py cfg=detrac:0_59:40_per_seq_random:zip:ign

<a id="60_99___detrac_"></a>
## 60_99       @ detrac-->xml_to_coco
python xml_to_coco.py cfg=detrac:60_99:zip:ign
<a id="100_per_seq_random___60_99_detrac_"></a>
### 100_per_seq_random       @ 60_99/detrac-->xml_to_coco
python xml_to_coco.py cfg=detrac:60_99:100_per_seq_random:zip:ign
<a id="40_per_seq_random___60_99_detrac_"></a>
### 40_per_seq_random       @ 60_99/detrac-->xml_to_coco
python xml_to_coco.py cfg=detrac:60_99:40_per_seq_random:zip:ign


<a id="detrac_non_empt_y_"></a>
# detrac-non_empty
<a id="0_19___detrac_non_empty_"></a>
## 0_19       @ detrac-non_empty-->xml_to_coco
python xml_to_coco.py cfg=detrac:non_empty:0_19:zip 
<a id="0_9___detrac_non_empty_"></a>
## 0_9       @ detrac-non_empty-->xml_to_coco
python xml_to_coco.py cfg=detrac:non_empty:0_9:zip  
<a id="0_48___detrac_non_empty_"></a>
## 0_48       @ detrac-non_empty-->xml_to_coco
python xml_to_coco.py cfg=detrac:non_empty:0_48:zip  
<a id="49_68___detrac_non_empty_"></a>
## 49_68       @ detrac-non_empty-->xml_to_coco
python xml_to_coco.py cfg=detrac:non_empty:49_68:zip  
<a id="49_85___detrac_non_empty_"></a>

<a id="49_85___detrac_non_empty_"></a>
## 49_85       @ detrac-non_empty-->xml_to_coco
python xml_to_coco.py cfg=detrac:non_empty:49_85:zip  
<a id="100_per_seq_random___49_85_detrac_non_empty_"></a>
### 100_per_seq_random       @ 49_85/detrac-non_empty-->xml_to_coco
python xml_to_coco.py cfg=detrac:non_empty:49_85:100_per_seq_random:zip


<a id="ipsc___xml_to_coc_o_"></a>
# ipsc       @ xml_to_coco-->coco
<a id="all_frames_roi___ipsc_"></a>
## all_frames_roi       @ ipsc-->xml_to_coco
<a id="g2_0_37___all_frames_roi_ips_c_"></a>
### g2_0_37       @ all_frames_roi/ipsc-->xml_to_coco
python xml_to_coco.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=lists/all_frames_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt output_json=all_frames_roi_g2_0_37.json start_frame_id=0 end_frame_id=37 ignore_invalid_label=1 val_ratio=0.3

<a id="no_val___g2_0_37_all_frames_roi_ips_c_"></a>
#### no_val       @ g2_0_37/all_frames_roi/ipsc-->xml_to_coco
python xml_to_coco.py cfg=ipsc:frame-0_37:zip-0 

<a id="g2_38_53___all_frames_roi_ips_c_"></a>
### g2_38_53       @ all_frames_roi/ipsc-->xml_to_coco
python xml_to_coco.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=lists/all_frames_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt output_json=all_frames_roi_g2_38_53.json start_frame_id=38 end_frame_id=53 ignore_invalid_label=1 val_ratio=0 min_val=0

<a id="g2_seq_1_39_53___all_frames_roi_ips_c_"></a>
### g2_seq_1_39_53       @ all_frames_roi/ipsc-->xml_to_coco
python xml_to_coco.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=lists/all_frames_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt output_json=all_frames_roi_g2_seq_1_39_53.json start_frame_id=39 end_frame_id=53 ignore_invalid_label=1 val_ratio=0 min_val=0 n_seq=1

<a id="g3_54_92___all_frames_roi_ips_c_"></a>
### g3_54_92       @ all_frames_roi/ipsc-->xml_to_coco
python xml_to_coco.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=lists/all_frames_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt output_json=all_frames_roi_g3_54_90.json start_frame_id=54 end_frame_id=92 ignore_invalid_label=1 val_ratio=0.2

<a id="ext_reorg_roi___ipsc_"></a>
## ext_reorg_roi       @ ipsc-->xml_to_coco
<a id="0_0___ext_reorg_roi_ipsc_"></a>
### 0_0       @ ext_reorg_roi/ipsc-->xml_to_coco
python xml_to_coco.py cfg=ipsc:0_0:zip-0 

<a id="0_1___ext_reorg_roi_ipsc_"></a>
### 0_1       @ ext_reorg_roi/ipsc-->xml_to_coco
python xml_to_coco.py cfg=ipsc:0_1:zip-0:mask 

<a id="0_126___ext_reorg_roi_ipsc_"></a>
### 0_126       @ ext_reorg_roi/ipsc-->xml_to_coco
python xml_to_coco.py cfg=ipsc:0_126:zip-0:mask 

<a id="0_37___ext_reorg_roi_ipsc_"></a>
### 0_37       @ ext_reorg_roi/ipsc-->xml_to_coco
python xml_to_coco.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=lists/ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt output_json=ext_reorg_roi_g2_0_37.json start_frame_id=0 end_frame_id=37 ignore_invalid_label=1 val_ratio=0.3

<a id="no_validate___0_37_ext_reorg_roi_ips_c_"></a>
#### no_validate       @ 0_37/ext_reorg_roi/ipsc-->xml_to_coco
python xml_to_coco.py cfg=ipsc:0_37:zip-0 

__only_list__
python xml_to_coco.py root_dir=/data/ipsc/well3/images seq_paths=lists/ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt output_json=ext_reorg_roi_g2_0_37.json start_frame_id=0 end_frame_id=37 ignore_invalid_label=1 val_ratio=0 only_list=1

```
1209 / 1209 valid images :: 7551 objects  ipsc: 1874 diff: 5677
pix_vals_mean: [130.26014205414378, 130.26014205414378, 130.26014205414378]
pix_vals_std: [16.855180446630406, 16.855180446630406, 16.855180446630406]
saving output json for 1209 images to: /data/ipsc/well3/all_frames_roi/ext_reorg_roi_g2_0_37.json
```

<a id="save_masks___0_37_ext_reorg_roi_ips_c_"></a>
#### save_masks       @ 0_37/ext_reorg_roi/ipsc-->xml_to_coco
python xml_to_coco.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=lists/ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt output_json=ext_reorg_roi_g2_0_37.json start_frame_id=0 end_frame_id=37 ignore_invalid_label=1 val_ratio=0 save_masks=1 get_img_stats=0

<a id="no_mask___0_37_ext_reorg_roi_ips_c_"></a>
#### no_mask       @ 0_37/ext_reorg_roi/ipsc-->xml_to_coco
python xml_to_coco.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=lists/ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt output_json=ext_reorg_roi_g2_0_37-no_mask.json start_frame_id=0 end_frame_id=37 ignore_invalid_label=1 val_ratio=0 enable_masks=0
```
1209 / 1209 valid images :: 7557 objects  ipsc: 1877 diff: 5680
pix_vals_mean: [130.2047103166563, 130.2047103166563, 130.2047103166563]
pix_vals_std: [16.979544725358632, 16.979544725358632, 16.979544725358632]
```
<a id="38_53___ext_reorg_roi_ipsc_"></a>
### 38_53       @ ext_reorg_roi/ipsc-->xml_to_coco
python xml_to_coco.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=lists/ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt output_json=ext_reorg_roi_g2_38_53.json start_frame_id=38 end_frame_id=53 ignore_invalid_label=1 val_ratio=0 min_val=0
```
pix_vals_mean: [122.33588318869438, 122.33588318869438, 122.33588318869438]
pix_vals_std: [21.941942680167365, 21.941942680167365, 21.941942680167365]
```
__only_list__
python xml_to_coco.py root_dir=/data/ipsc/well3/images seq_paths=lists/ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt output_json=ext_reorg_roi_g2_38_53.json start_frame_id=38 end_frame_id=53 ignore_invalid_label=1 val_ratio=0 min_val=0 only_list=1

<a id="save_masks___38_53_ext_reorg_roi_ipsc_"></a>
#### save_masks       @ 38_53/ext_reorg_roi/ipsc-->xml_to_coco
python xml_to_coco.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=lists/ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt output_json=ext_reorg_roi_g2_38_53.json start_frame_id=38 end_frame_id=53 ignore_invalid_label=1 val_ratio=0 min_val=0 save_masks=1 get_img_stats=0

<a id="15_53___ext_reorg_roi_ipsc_"></a>
### 15_53       @ ext_reorg_roi/ipsc-->xml_to_coco
python xml_to_coco.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=lists/ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt output_json=ext_reorg_roi_g2_15_53.json start_frame_id=15 end_frame_id=53 ignore_invalid_label=1 val_ratio=0.2
```
pix_vals_mean: [126.62072260746325, 126.62072260746325, 126.62072260746325]
pix_vals_std: [19.203380328648525, 19.203380328648525, 19.203380328648525]
```
<a id="no_validate___15_53_ext_reorg_roi_ipsc_"></a>
#### no_validate       @ 15_53/ext_reorg_roi/ipsc-->xml_to_coco
python xml_to_coco.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=lists/ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt output_json=ext_reorg_roi_g2_15_53.json start_frame_id=15 end_frame_id=53 ignore_invalid_label=1 val_ratio=0

<a id="16_53___ext_reorg_roi_ipsc_"></a>
### 16_53       @ ext_reorg_roi/ipsc-->xml_to_coco
<a id="val_30___16_53_ext_reorg_roi_ipsc_"></a>
#### val-30       @ 16_53/ext_reorg_roi/ipsc-->xml_to_coco
__only_list__
python xml_to_coco.py root_dir=/data/ipsc/well3/images seq_paths=lists/ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt output_json=ext_reorg_roi_g2_16_53.json start_frame_id=16 end_frame_id=53 ignore_invalid_label=1 val_ratio=0.3 only_list=1 shuffle=1
__only_list-seq__
python xml_to_coco.py root_dir=/data/ipsc/well3/images seq_paths=lists/ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt output_json=ext_reorg_roi_g2_16_53-seq.json start_frame_id=16 end_frame_id=53 ignore_invalid_label=1 val_ratio=-0.3 only_list=1 shuffle=0

<a id="no_val___16_53_ext_reorg_roi_ipsc_"></a>
#### no-val       @ 16_53/ext_reorg_roi/ipsc-->xml_to_coco
python xml_to_coco.py cfg=ipsc:frame-16_53:zip-0 

```
pix_vals_mean: [126.21, 126.21, 126.21]
pix_vals_std: [19.33, 19.33, 19.33]
```
__only_list__
python xml_to_coco.py root_dir=/data/ipsc/well3/images seq_paths=lists/ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt output_json=ext_reorg_roi_g2_16_53.json start_frame_id=16 end_frame_id=53 ignore_invalid_label=1 val_ratio=0 only_list=1

<a id="no_mask___no_val_16_53_ext_reorg_roi_ips_c_"></a>
##### no_mask       @ no-val/16_53/ext_reorg_roi/ipsc-->xml_to_coco
python xml_to_coco.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=lists/ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt output_json=ext_reorg_roi_g2_16_53-no_mask.json start_frame_id=16 end_frame_id=53 ignore_invalid_label=1 val_ratio=0 enable_masks=0

<a id="0_1___ext_reorg_roi_ipsc__1"></a>
### 0_1       @ ext_reorg_roi/ipsc-->xml_to_coco
python xml_to_coco.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=lists/ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt output_json=ext_reorg_roi_g2_0_1.json start_frame_id=0 end_frame_id=1 ignore_invalid_label=1 val_ratio=0 get_img_stats=0

<a id="0_15___ext_reorg_roi_ipsc_"></a>
### 0_15       @ ext_reorg_roi/ipsc-->xml_to_coco
python xml_to_coco.py cfg=ipsc:0_15:zip-0 

python xml_to_coco.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=lists/ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt output_json=ext_reorg_roi_g2_0_15.json start_frame_id=0 end_frame_id=15 ignore_invalid_label=1 val_ratio=0 get_img_stats=0
__only_list__
python xml_to_coco.py root_dir=/data/ipsc/well3/images seq_paths=lists/ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt output_json=ext_reorg_roi_g2_0_15.json start_frame_id=0 end_frame_id=15 ignore_invalid_label=1 val_ratio=0 get_img_stats=0 only_list=1

<a id="no_mask___0_15_ext_reorg_roi_ips_c_"></a>
#### no_mask       @ 0_15/ext_reorg_roi/ipsc-->xml_to_coco
python xml_to_coco.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=lists/ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt output_json=ext_reorg_roi_g2_0_15-no_mask.json start_frame_id=0 end_frame_id=15 ignore_invalid_label=1 val_ratio=0 get_img_stats=0 enable_masks=0

<a id="0_1___ext_reorg_roi_ipsc__2"></a>
### 0_1       @ ext_reorg_roi/ipsc-->xml_to_coco
python xml_to_coco.py cfg=ipsc:0_1:zip-0 
<a id="2_3___ext_reorg_roi_ipsc_"></a>
### 2_3       @ ext_reorg_roi/ipsc-->xml_to_coco
python xml_to_coco.py cfg=ipsc:2_3:zip-0 

<a id="54_126___ext_reorg_roi_ipsc_"></a>
### 54_126       @ ext_reorg_roi/ipsc-->xml_to_coco
python xml_to_coco.py cfg=ipsc:54_126:zip-0 
<a id="strd_5___54_126_ext_reorg_roi_ips_c_"></a>
#### strd-5       @ 54_126/ext_reorg_roi/ipsc-->xml_to_coco
python xml_to_coco.py cfg=ipsc:54_126:zip-0:strd-5
<a id="strd_8___54_126_ext_reorg_roi_ips_c_"></a>
#### strd-8       @ 54_126/ext_reorg_roi/ipsc-->xml_to_coco
python xml_to_coco.py cfg=ipsc:54_126:zip-0:strd-8


```
pix_vals_mean: [118.06, 118.06, 118.06]
pix_vals_std: [23.99, 23.99, 23.99]
```
<a id="list___54_126_ext_reorg_roi_ips_c_"></a>
#### list       @ 54_126/ext_reorg_roi/ipsc-->xml_to_coco
python xml_to_coco.py cfg=ipsc:frame-54_126:zip-0:list 

<a id="no_mask___54_126_ext_reorg_roi_ips_c_"></a>
#### no_mask       @ 54_126/ext_reorg_roi/ipsc-->xml_to_coco
python xml_to_coco.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=lists/ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt output_json=ext_reorg_roi_g2_54_126-no_mask.json start_frame_id=54 end_frame_id=126 ignore_invalid_label=1 val_ratio=0 get_img_stats=0 enable_masks=0

<a id="g2_0_53___ext_reorg_roi_ipsc_"></a>
### g2_0_53       @ ext_reorg_roi/ipsc-->xml_to_coco
python xml_to_coco.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=lists/ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt output_json=ext_reorg_roi_g2_0_53.json start_frame_id=0 end_frame_id=53 ignore_invalid_label=1 val_ratio=0 get_img_stats=1
```
pix_vals_mean: [127.94684054255906, 127.94684054255906, 127.94684054255906]
pix_vals_std: [18.35440641227878, 18.35440641227878, 18.35440641227878]
```
python xml_to_coco.py root_dir=/data/ipsc/well3/images seq_paths=lists/ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt output_json=ext_reorg_roi_g2_0_53.json start_frame_id=0 end_frame_id=53 ignore_invalid_label=1 val_ratio=0 get_img_stats=1 only_list=1

<a id="no_mask___g2_0_53_ext_reorg_roi_ipsc_"></a>
#### no_mask       @ g2_0_53/ext_reorg_roi/ipsc-->xml_to_coco
python xml_to_coco.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=lists/ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt output_json=ext_reorg_roi_g2_0_53-no_mask.json start_frame_id=0 end_frame_id=53 ignore_invalid_label=1 val_ratio=0 get_img_stats=0 enable_masks=0

<a id="ext_reorg_roi_no_annotations___ipsc_"></a>
## ext_reorg_roi-no_annotations       @ ipsc-->xml_to_coco
<a id="reorg_roi___ext_reorg_roi_no_annotations_ipsc_2_clas_s_"></a>
python xml_to_coco.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=lists/ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt output_json=ext_reorg_roi.json end_frame_id=-1 no_annotations=1

<a id="reorg_roi___ext_reorg_roi_no_annotations_ips_c_"></a>
### reorg_roi       @ ext_reorg_roi-no_annotations/ipsc-->xml_to_coco
python xml_to_coco.py root_dir=/data/ipsc/well3/reorg_roi seq_paths=lists/reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt output_json=reorg_roi.json end_frame_id=126 no_annotations=1

<a id="all_frames_roi___ext_reorg_roi_no_annotations_ips_c_"></a>
### all_frames_roi       @ ext_reorg_roi-no_annotations/ipsc-->xml_to_coco
python xml_to_coco.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=lists/all_frames_roi_raw.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt output_json=all_frames_roi.json no_annotations=1

<a id="all_frames_roi_7777_10249_10111_13349___all_frames_roi_ext_reorg_roi_no_annotations_ipsc_"></a>
#### all_frames_roi_7777_10249_10111_13349       @ all_frames_roi/ext_reorg_roi-no_annotations/ipsc-->xml_to_coco
python xml_to_coco.py seq_paths=/data/ipsc/well3/all_frames_roi/all_frames_roi_7777_10249_10111_13349 class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt output_json=all_frames_roi_7777_10249_10111_13349.json no_annotations=1

<a id="all_frames_roi_8094_13016_11228_15282___all_frames_roi_ext_reorg_roi_no_annotations_ipsc_"></a>
#### all_frames_roi_8094_13016_11228_15282       @ all_frames_roi/ext_reorg_roi-no_annotations/ipsc-->xml_to_coco
python xml_to_coco.py seq_paths=/data/ipsc/well3/all_frames_roi/all_frames_roi_8094_13016_11228_15282 class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt output_json=all_frames_roi_8094_13016_11228_15282.json no_annotations=1

<a id="test_230710___ext_reorg_roi_no_annotations_ips_c_"></a>
### Test_230710       @ ext_reorg_roi-no_annotations/ipsc-->xml_to_coco
python xml_to_coco.py seq_paths=/data/ipsc/Test_230710 class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt output_json=Test_230710.json no_annotations=1

<a id="test_230606___ext_reorg_roi_no_annotations_ips_c_"></a>
### Test_230606       @ ext_reorg_roi-no_annotations/ipsc-->xml_to_coco
python xml_to_coco.py seq_paths=/data/ipsc/Test_230606 class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt output_json=Test_230606.json no_annotations=1

<a id="test_211208___ext_reorg_roi_no_annotations_ips_c_"></a>
### Test_211208       @ ext_reorg_roi-no_annotations/ipsc-->xml_to_coco
python xml_to_coco.py seq_paths=/data/ipsc/images/Test_211208 class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt output_json=Test_211208_2_class.json no_annotations=1

<a id="nd03___ext_reorg_roi_no_annotations_ips_c_"></a>
### nd03       @ ext_reorg_roi-no_annotations/ipsc-->xml_to_coco
python xml_to_coco.py seq_paths=/data/ipsc/images/nd03 class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt output_json=nd03_2_class.json no_annotations=1

<a id="g2_4___ipsc_"></a>
## g2_4       @ ipsc-->xml_to_coco
python xml_to_coco.py root_dir=/data/ipsc_2_class_raw seq_paths=ipsc_g2_4.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt output_json=ipsc_2_class_g2_4.json 

<a id="g4___ipsc_"></a>
## g4       @ ipsc-->xml_to_coco
python xml_to_coco.py root_dir=/data/ipsc_2_class_raw seq_paths=ipsc_g4.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt output_json=ipsc_2_class_g4.json 

<a id="g3___ipsc_"></a>
## g3       @ ipsc-->xml_to_coco
python xml_to_coco.py root_dir=/data/ipsc_2_class_raw seq_paths=ipsc_g3.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt output_json=ipsc_2_class_g3.json 

<a id="ipsc_5_class___xml_to_coc_o_"></a>
# ipsc_5_class       @ xml_to_coco-->coco

<a id="test_211208___ipsc_5_class_"></a>
## Test_211208       @ ipsc_5_class-->xml_to_coco
python xml_to_coco.py seq_paths=/data/ipsc/images/Test_211208 class_names_path=lists/classes/predefined_classes_ipsc_5_class.txt output_json=Test_211208.json no_annotations=1

<a id="k___ipsc_5_class_"></a>
## k       @ ipsc_5_class-->xml_to_coco
python xml_to_coco.py seq_paths=data/k class_names_path=lists/classes/predefined_classes_person.txt output_json=k.json no_annotations=1

<a id="nd03___ipsc_5_class_"></a>
## nd03       @ ipsc_5_class-->xml_to_coco
python xml_to_coco.py seq_paths=/data/ipsc/images/nd03 class_names_path=lists/classes/predefined_classes_ipsc_5_class.txt output_json=nd03.json no_annotations=1

<a id="g3_4s___ipsc_5_class_"></a>
## g3_4s       @ ipsc_5_class-->xml_to_coco
python xml_to_coco.py root_dir=/data/ipsc_5_class_raw class_names_path=lists/classes/predefined_classes_ipsc_5_class.txt output_json=ipsc_5_class.json 

<a id="50_50___g3_4s_ipsc_5_class_"></a>
### 50_50       @ g3_4s/ipsc_5_class-->xml_to_coco
python xml_to_coco.py root_dir=/data/ipsc_5_class_raw class_names_path=lists/classes/predefined_classes_ipsc_5_class.txt output_json=ipsc_5_class_50_50.json val_ratio=0.5 min_val=1

<a id="no_val___g3_4s_ipsc_5_class_"></a>
### no_val       @ g3_4s/ipsc_5_class-->xml_to_coco
python xml_to_coco.py root_dir=/data/ipsc_5_class_raw class_names_path=lists/classes/predefined_classes_ipsc_5_class.txt output_json=ipsc_5_class.json val_ratio=0 min_val=0

<a id="g3___ipsc_5_class_"></a>
## g3       @ ipsc_5_class-->xml_to_coco
python xml_to_coco.py root_dir=/data/ipsc_5_class_raw seq_paths=ipsc_g3.txt class_names_path=lists/classes/predefined_classes_ipsc_5_class.txt output_json=ipsc_5_class_g3.json

<a id="50_50___g3_ipsc_5_clas_s_"></a>
### 50_50       @ g3/ipsc_5_class-->xml_to_coco
python xml_to_coco.py root_dir=/data/ipsc_5_class_raw seq_paths=ipsc_g3.txt class_names_path=lists/classes/predefined_classes_ipsc_5_class.txt output_json=ipsc_5_class_g3_50_50.json val_ratio=0.5 min_val=1

<a id="no_val___g3_ipsc_5_clas_s_"></a>
### no_val       @ g3/ipsc_5_class-->xml_to_coco
python xml_to_coco.py root_dir=/data/ipsc_5_class_raw seq_paths=ipsc_g3.txt class_names_path=lists/classes/predefined_classes_ipsc_5_class.txt output_json=ipsc_5_class_g3.json val_ratio=0 min_val=0

<a id="g4s___ipsc_5_class_"></a>
## g4s       @ ipsc_5_class-->xml_to_coco
python xml_to_coco.py root_dir=/data/ipsc_5_class_raw seq_paths=ipsc_g4s.txt class_names_path=lists/classes/predefined_classes_ipsc_5_class.txt output_json=ipsc_5_class_g4s.json

<a id="50_50___g4s_ipsc_5_class_"></a>
### 50_50       @ g4s/ipsc_5_class-->xml_to_coco
python xml_to_coco.py root_dir=/data/ipsc_5_class_raw seq_paths=ipsc_g4s.txt class_names_path=lists/classes/predefined_classes_ipsc_5_class.txt output_json=ipsc_5_class_g4s_50_50.json val_ratio=0.5 min_val=0

<a id="no_val___g4s_ipsc_5_class_"></a>
### no_val       @ g4s/ipsc_5_class-->xml_to_coco
python xml_to_coco.py root_dir=/data/ipsc_5_class_raw seq_paths=ipsc_g4s.txt class_names_path=lists/classes/predefined_classes_ipsc_5_class.txt output_json=ipsc_5_class_g4s.json val_ratio=0 min_val=0


<a id="ctc___xml_to_coc_o_"></a>
# ctc       @ xml_to_coco-->coco

<a id="ctc_all___ct_c_"></a>
## ctc_all       @ ctc-->xml_to_coco
python xml_to_coco.py root_dir=/data/CTC/Images seq_paths=ctc_all.txt class_names_path=lists/classes/predefined_classes_cell.txt output_json=ctc_all.json excluded_images_list=missing_seg_images.txt val_ratio=0.3


<a id="ctc_bf_c2dl_hsc___ct_c_"></a>
## ctc_BF_C2DL_HSC       @ ctc-->xml_to_coco
python xml_to_coco.py root_dir=/data/CTC/Images seq_paths=ctc_BF_C2DL_HSC.txt class_names_path=lists/classes/predefined_classes_cell.txt output_json=ctc_BF_C2DL_HSC.json excluded_images_list=missing_seg_images.txt val_ratio=0.3

<a id="ctc_bf_c2dl_musc___ct_c_"></a>
## ctc_BF_C2DL_MuSC       @ ctc-->xml_to_coco
python xml_to_coco.py root_dir=/data/CTC/Images seq_paths=ctc_BF_C2DL_MuSC.txt class_names_path=lists/classes/predefined_classes_cell.txt output_json=ctc_BF_C2DL_MuSC.json excluded_images_list=missing_seg_images.txt val_ratio=0.3
pix_vals_mean: [108.65, 108.65, 108.65]
pix_vals_std: [13.27, 13.27, 13.27]

<a id="ctc_dic_c2dh_hela___ct_c_"></a>
## ctc_DIC_C2DH_HeLa       @ ctc-->xml_to_coco
python xml_to_coco.py root_dir=/data/CTC/Images seq_paths=ctc_DIC_C2DH_HeLa.txt class_names_path=lists/classes/predefined_classes_cell.txt output_json=ctc_DIC_C2DH_HeLa.json excluded_images_list=missing_seg_images.txt val_ratio=0.1
pix_vals_mean: [99.09, 99.09, 99.09]
pix_vals_std: [12.34, 12.34, 12.34]

<a id="ctc_fluo_c2dl_huh7___ct_c_"></a>
## ctc_Fluo_C2DL_Huh7       @ ctc-->xml_to_coco
python xml_to_coco.py root_dir=/data/CTC/Images seq_paths=ctc_Fluo_C2DL_Huh7.txt class_names_path=lists/classes/predefined_classes_cell.txt output_json=ctc_Fluo_C2DL_Huh7.json excluded_images_list=missing_seg_images.txt val_ratio=0.1

<a id="ctc_fluo_c2dl_msc___ct_c_"></a>
## ctc_Fluo_C2DL_MSC       @ ctc-->xml_to_coco
python xml_to_coco.py root_dir=/data/CTC/Images seq_paths=ctc_Fluo_C2DL_MSC.txt class_names_path=lists/classes/predefined_classes_cell.txt output_json=ctc_Fluo_C2DL_MSC.json excluded_images_list=missing_seg_images.txt val_ratio=0.1
pix_vals_mean: [25.06, 25.06, 25.06]
pix_vals_std: [13.85, 13.85, 13.85]

<a id="ctc_phc_c2dh_u373___ct_c_"></a>
## ctc_PhC_C2DH_U373       @ ctc-->xml_to_coco
python xml_to_coco.py root_dir=/data/CTC/Images seq_paths=ctc_PhC_C2DH_U373.txt class_names_path=lists/classes/predefined_classes_cell.txt output_json=ctc_PhC_C2DH_U373.json excluded_images_list=missing_seg_images.txt val_ratio=0.1
pix_vals_mean: [87.98, 87.98, 87.98]
pix_vals_std: [13.71, 13.71, 13.71]

<a id="ctc_phc_c2dl_psc___ct_c_"></a>
## ctc_PhC_C2DL_PSC       @ ctc-->xml_to_coco
python xml_to_coco.py root_dir=/data/CTC/Images seq_paths=ctc_PhC_C2DL_PSC.txt class_names_path=lists/classes/predefined_classes_cell.txt output_json=ctc_PhC_C2DL_PSC.json excluded_images_list=missing_seg_images.txt val_ratio=0.1
pix_vals_mean: [135.42, 135.42, 135.42]
pix_vals_std: [24.22, 24.22, 24.22]

<a id="ctc_fluo_n2dh_gowt1___ct_c_"></a>
## ctc_Fluo_N2DH_GOWT1       @ ctc-->xml_to_coco
python xml_to_coco.py root_dir=/data/CTC/Images seq_paths=ctc_Fluo_N2DH_GOWT1.txt class_names_path=lists/classes/predefined_classes_cell.txt output_json=ctc_Fluo_N2DH_GOWT1.json excluded_images_list=missing_seg_images.txt val_ratio=0.1
pix_vals_mean: [4.61, 4.61, 4.61]
pix_vals_std: [14.15, 14.15, 14.15]

<a id="ctc_fluo_n2dh_sim___ct_c_"></a>
## ctc_Fluo_N2DH_SIM       @ ctc-->xml_to_coco
python xml_to_coco.py root_dir=/data/CTC/Images seq_paths=ctc_Fluo_N2DH_SIM.txt class_names_path=lists/classes/predefined_classes_cell.txt output_json=ctc_Fluo_N2DH_SIM.json excluded_images_list=missing_seg_images.txt val_ratio=0.1
pix_vals_mean: [21.40, 21.40, 21.40]
pix_vals_std: [9.70, 9.70, 9.70]

<a id="ctc_fluo_n2dl_hela___ct_c_"></a>
## ctc_Fluo_N2DL_HeLa       @ ctc-->xml_to_coco
python xml_to_coco.py root_dir=/data/CTC/Images seq_paths=ctc_Fluo_N2DL_HeLa.txt class_names_path=lists/classes/predefined_classes_cell.txt output_json=ctc_Fluo_N2DL_HeLa.json excluded_images_list=missing_seg_images.txt val_ratio=0.1
pix_vals_mean: [9.75, 9.75, 9.75]
pix_vals_std: [15.87, 15.87, 15.87]

<a id="ctmc_all___xml_to_coc_o_"></a>
# ctmc_all       @ xml_to_coco-->coco
python xml_to_coco.py root_dir=/data/CTMC/Images seq_paths=ctmc_train.txt class_names_path=lists/classes/predefined_classes_cell.txt output_json=ctmc_train.json val_ratio=0.3 enable_masks=0

<a id="acamp_"></a>
# acamp
<a id="1k8_vid_entire_seq___acam_p_"></a>
## 1k8_vid_entire_seq       @ acamp-->xml_to_coco
python xml_to_coco.py cfg=acamp:1k8_vid_entire_seq
<a id="10k6_vid_entire_seq___acam_p_"></a>
## 10k6_vid_entire_seq       @ acamp-->xml_to_coco
python xml_to_coco.py cfg=acamp:10k6_vid_entire_seq
<a id="20k6_5_video___acam_p_"></a>
## 20k6_5_video       @ acamp-->xml_to_coco
python xml_to_coco.py cfg=acamp:20k6_5_video

<a id="1k8_vid_entire_seq_inv___acam_p_"></a>
## 1k8_vid_entire_seq_inv       @ acamp-->xml_to_coco
python xml_to_coco.py cfg=acamp:1k8_vid_entire_seq_inv
<a id="10k6_vid_entire_seq_inv___acam_p_"></a>
## 10k6_vid_entire_seq_inv       @ acamp-->xml_to_coco
python xml_to_coco.py cfg=acamp:10k6_vid_entire_seq_inv
<a id="20k6_5_video_inv___acam_p_"></a>
## 20k6_5_video_inv       @ acamp-->xml_to_coco
python xml_to_coco.py cfg=acamp:20k6_5_video_inv

'Z:\UofA\Acamp\acamp_code\tf_api\cmd\csv_to_record.md'

<a id="1k8_vid_entire_seq_inv_2_per_seq___acam_p_"></a>
## 1k8_vid_entire_seq_inv_2_per_seq       @ acamp-->xml_to_coco
python xml_to_coco.py cfg=acamp:1k8_vid_entire_seq_inv_2_per_seq
'Z:\UofA\Acamp\acamp_code\tf_api\cmd\csv_to_record.md'

<a id="10k6_vid_entire_seq_inv_2_per_seq___acam_p_"></a>
## 10k6_vid_entire_seq_inv_2_per_seq       @ acamp-->xml_to_coco
python xml_to_coco.py cfg=acamp:10k6_vid_entire_seq_inv_2_per_seq
'Z:\UofA\Acamp\acamp_code\tf_api\cmd\csv_to_record.md'

<a id="20k6_5_video_inv_2_per_seq___acam_p_"></a>
## 20k6_5_video_inv_2_per_seq       @ acamp-->xml_to_coco
python xml_to_coco.py cfg=acamp:20k6_5_video_inv_2_per_seq
'Z:\UofA\Acamp\acamp_code\tf_api\cmd\csv_to_record.md'

