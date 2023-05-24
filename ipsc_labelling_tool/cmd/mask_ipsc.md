<!-- MarkdownTOC -->

- [labelImg](#labelim_g_)
- [propagate_by_tracking](#propagate_by_tracking_)
    - [all_frames_roi       @ propagate_by_tracking](#all_frames_roi___propagate_by_trackin_g_)
    - [reorg_roi       @ propagate_by_tracking](#reorg_roi___propagate_by_trackin_g_)
        - [bad       @ reorg_roi/propagate_by_tracking](#bad___reorg_roi_propagate_by_trackin_g_)
        - [good       @ reorg_roi/propagate_by_tracking](#good___reorg_roi_propagate_by_trackin_g_)
            - [batch       @ good/reorg_roi/propagate_by_tracking](#batch___good_reorg_roi_propagate_by_tracking_)
            - [individual       @ good/reorg_roi/propagate_by_tracking](#individual___good_reorg_roi_propagate_by_tracking_)
- [manifold_embedding](#manifold_embeddin_g_)
    - [Frame_251__roi_16627_11116_18727_12582       @ manifold_embedding](#frame_251_roi_16627_11116_18727_12582___manifold_embedding_)
    - [g4       @ manifold_embedding](#g4___manifold_embedding_)
        - [0       @ g4/manifold_embedding](#0___g4_manifold_embeddin_g_)
        - [1       @ g4/manifold_embedding](#1___g4_manifold_embeddin_g_)
        - [2       @ g4/manifold_embedding](#2___g4_manifold_embeddin_g_)
        - [3       @ g4/manifold_embedding](#3___g4_manifold_embeddin_g_)
        - [4       @ g4/manifold_embedding](#4___g4_manifold_embeddin_g_)
    - [g3       @ manifold_embedding](#g3___manifold_embedding_)
        - [load_masks       @ g3/manifold_embedding](#load_masks___g3_manifold_embeddin_g_)
        - [0       @ g3/manifold_embedding](#0___g3_manifold_embeddin_g_)
        - [1       @ g3/manifold_embedding](#1___g3_manifold_embeddin_g_)
        - [2       @ g3/manifold_embedding](#2___g3_manifold_embeddin_g_)
        - [3       @ g3/manifold_embedding](#3___g3_manifold_embeddin_g_)
        - [4       @ g3/manifold_embedding](#4___g3_manifold_embeddin_g_)
    - [g2       @ manifold_embedding](#g2___manifold_embedding_)
        - [0       @ g2/manifold_embedding](#0___g2_manifold_embeddin_g_)
        - [1       @ g2/manifold_embedding](#1___g2_manifold_embeddin_g_)
        - [2       @ g2/manifold_embedding](#2___g2_manifold_embeddin_g_)
    - [g1       @ manifold_embedding](#g1___manifold_embedding_)
            - [0       @ g1/manifold_embedding](#0___g1_manifold_embeddin_g_)
            - [1       @ g1/manifold_embedding](#1___g1_manifold_embeddin_g_)
            - [2       @ g1/manifold_embedding](#2___g1_manifold_embeddin_g_)
- [to_mask_seq](#to_mask_seq_)
    - [all       @ to_mask_seq](#all___to_mask_se_q_)
        - [513x513_20       @ all/to_mask_seq](#513x513_20___all_to_mask_se_q_)
        - [Frame_101_150_roi_7777_10249_10111_13349       @ all/to_mask_seq](#frame_101_150_roi_7777_10249_10111_13349___all_to_mask_se_q_)
    - [2_class       @ to_mask_seq](#2_class___to_mask_se_q_)
        - [blended_vis       @ 2_class/to_mask_seq](#blended_vis___2_class_to_mask_se_q_)
        - [Frame_101_150_roi_12660_17981_16026_20081       @ 2_class/to_mask_seq](#frame_101_150_roi_12660_17981_16026_20081___2_class_to_mask_se_q_)
    - [5_class       @ to_mask_seq](#5_class___to_mask_se_q_)
        - [blended_vis       @ 5_class/to_mask_seq](#blended_vis___5_class_to_mask_se_q_)
        - [Frame_201_250_roi_11927_12517_15394_15550       @ 5_class/to_mask_seq](#frame_201_250_roi_11927_12517_15394_15550___5_class_to_mask_se_q_)
    - [5_class_220211       @ to_mask_seq](#5_class_220211___to_mask_se_q_)
        - [blended_vis       @ 5_class_220211/to_mask_seq](#blended_vis___5_class_220211_to_mask_seq_)
        - [Frame_201-250_roi_11927_12517_15394_15550       @ 5_class_220211/to_mask_seq](#frame_201_250_roi_11927_12517_15394_15550___5_class_220211_to_mask_seq_)
    - [all_frames_roi       @ to_mask_seq](#all_frames_roi___to_mask_se_q_)
        - [blended_vis       @ all_frames_roi/to_mask_seq](#blended_vis___all_frames_roi_to_mask_seq_)

<!-- /MarkdownTOC -->

<a id="labelim_g_"></a>
# labelImg
python3 labelImg.py load_prev=0 root_dir=/data/ipsc/well3/reorg_roi file_name=roi_9261_13449_11494_14382

<a id="propagate_by_tracking_"></a>
# propagate_by_tracking

<a id="all_frames_roi___propagate_by_trackin_g_"></a>
## all_frames_roi       @ propagate_by_tracking-->mask_ipsc

python3 propagate_by_tracking.py class_names_path=data/predefined_classes_ipsc_3_class.txt root_dir=/data/ipsc/well3/all_frames_roi seq_paths=all_frames_roi_7777_10249_10111_13349

python3 propagate_by_tracking.py class_names_path=data/predefined_classes_ipsc_3_class.txt root_dir=/data/ipsc/well3/all_frames_roi seq_paths=all_frames_roi_8094_13016_11228_15282

python3 propagate_by_tracking.py class_names_path=data/predefined_classes_ipsc_3_class.txt root_dir=/data/ipsc/well3/all_frames_roi seq_paths=all_frames_roi_9861_9849_12861_11516

python3 propagate_by_tracking.py class_names_path=data/predefined_classes_ipsc_3_class.txt root_dir=/data/ipsc/well3/all_frames_roi seq_paths=all_frames_roi_10127_9782_12527_11782

python3 propagate_by_tracking.py class_names_path=data/predefined_classes_ipsc_3_class.txt root_dir=/data/ipsc/well3/all_frames_roi seq_paths=all_frames_roi_10161_9883_13561_12050

python3 propagate_by_tracking.py class_names_path=data/predefined_classes_ipsc_3_class.txt root_dir=/data/ipsc/well3/all_frames_roi seq_paths=all_frames_roi_11927_12517_15394_15550 save_cell_lines=1 save_track=1

python3 propagate_by_tracking.py class_names_path=data/predefined_classes_ipsc_3_class.txt root_dir=/data/ipsc/well3/all_frames_roi seq_paths=all_frames_roi_12094_17082_16427_20915 save_track=1

python3 propagate_by_tracking.py class_names_path=data/predefined_classes_ipsc_3_class.txt root_dir=/data/ipsc/well3/all_frames_roi seq_paths=all_frames_roi_12527_11015_14493_12615

python3 propagate_by_tracking.py class_names_path=data/predefined_classes_ipsc_3_class.txt root_dir=/data/ipsc/well3/all_frames_roi seq_paths=all_frames_roi_12794_8282_14661_10116

python3 propagate_by_tracking.py class_names_path=data/predefined_classes_ipsc_3_class.txt root_dir=/data/ipsc/well3/all_frames_roi seq_paths=all_frames_roi_12994_10915_15494_12548

python3 propagate_by_tracking.py class_names_path=data/predefined_classes_ipsc_3_class.txt root_dir=/data/ipsc/well3/all_frames_roi seq_paths=all_frames_roi_16627_11116_18727_12582

<a id="reorg_roi___propagate_by_trackin_g_"></a>
## reorg_roi       @ propagate_by_tracking-->mask_ipsc

<a id="bad___reorg_roi_propagate_by_trackin_g_"></a>
### bad       @ reorg_roi/propagate_by_tracking-->mask_ipsc

python3 propagate_by_tracking.py class_names_path=data/predefined_classes_ipsc_3_class.txt root_dir=/data/ipsc/well3/reorg_roi seq_paths=roi_6094_19416_8394_20382

python3 propagate_by_tracking.py class_names_path=data/predefined_classes_ipsc_3_class.txt root_dir=/data/ipsc/well3/reorg_roi seq_paths=roi_7494_15849_9894_17016

python3 propagate_by_tracking.py class_names_path=data/predefined_classes_ipsc_3_class.txt root_dir=/data/ipsc/well3/reorg_roi seq_paths=roi_9426_20150_11527_21482


<a id="good___reorg_roi_propagate_by_trackin_g_"></a>
### good       @ reorg_roi/propagate_by_tracking-->mask_ipsc

<a id="batch___good_reorg_roi_propagate_by_tracking_"></a>
#### batch       @ good/reorg_roi/propagate_by_tracking-->mask_ipsc
python3 propagate_by_tracking.py class_names_path=data/predefined_classes_ipsc_3_class.txt root_dir=/data/ipsc/well3/all_frames_roi seq_paths=all_frames_roi.txt save_track=1 start_id=-1 write_xml=0

python3 propagate_by_tracking.py class_names_path=data/predefined_classes_ipsc_2_class.txt ignore_invalid_class=1 root_dir=/data/ipsc/well3/all_frames_roi seq_paths=ext_reorg_roi.txt save_track=1 start_id=-1 write_xml=0

<a id="individual___good_reorg_roi_propagate_by_tracking_"></a>
#### individual       @ good/reorg_roi/propagate_by_tracking-->mask_ipsc
python3 propagate_by_tracking.py class_names_path=data/predefined_classes_ipsc_3_class.txt root_dir=/data/ipsc/well3/all_frames_roi seq_paths=roi_4961_15682_7127_16949

python3 propagate_by_tracking.py class_names_path=data/predefined_classes_ipsc_3_class.txt root_dir=/data/ipsc/well3/all_frames_roi seq_paths=roi_6661_13749_9061_14816

python3 propagate_by_tracking.py class_names_path=data/predefined_classes_ipsc_3_class.txt root_dir=/data/ipsc/well3/all_frames_roi seq_paths=roi_7594_11916_9927_13149

python3 propagate_by_tracking.py class_names_path=data/predefined_classes_ipsc_3_class.txt root_dir=/data/ipsc/well3/all_frames_roi seq_paths=roi_7694_8682_10194_9682

python3 propagate_by_tracking.py class_names_path=data/predefined_classes_ipsc_3_class.txt root_dir=/data/ipsc/well3/all_frames_roi seq_paths=roi_7727_10749_9961_11749

python3 propagate_by_tracking.py class_names_path=data/predefined_classes_ipsc_3_class.txt root_dir=/data/ipsc/well3/all_frames_roi seq_paths=roi_8461_17782_10194_19016

python3 propagate_by_tracking.py class_names_path=data/predefined_classes_ipsc_3_class.txt root_dir=/data/ipsc/well3/all_frames_roi seq_paths=roi_9261_13449_11494_14382

python3 propagate_by_tracking.py class_names_path=data/predefined_classes_ipsc_3_class.txt root_dir=/data/ipsc/well3/all_frames_roi seq_paths=roi_10228_10182_12394_11915

python3 propagate_by_tracking.py class_names_path=data/predefined_classes_ipsc_3_class.txt root_dir=/data/ipsc/well3/all_frames_roi seq_paths=roi_10494_8849_12494_9849

python3 propagate_by_tracking.py class_names_path=data/predefined_classes_ipsc_3_class.txt root_dir=/data/ipsc/well3/all_frames_roi seq_paths=roi_11661_13082_13594_14849
python3 propagate_by_tracking.py class_names_path=data/predefined_classes_ipsc_2_class.txt root_dir=/data/ipsc/well3/all_frames_roi seq_paths=roi_11661_13082_13594_14849 ignore_invalid_class=1 ui.fmt.size=2 ui.fmt.offset=5,40 start_id=-1 write_xml=0 

python3 propagate_by_tracking.py class_names_path=data/predefined_classes_ipsc_3_class.txt root_dir=/data/ipsc/well3/all_frames_roi seq_paths=roi_12394_17282_14327_20782

python3 propagate_by_tracking.py class_names_path=data/predefined_classes_ipsc_3_class.txt root_dir=/data/ipsc/well3/all_frames_roi seq_paths=roi_12761_10682_14894_11782

python3 propagate_by_tracking.py class_names_path=data/predefined_classes_ipsc_3_class.txt root_dir=/data/ipsc/well3/all_frames_roi seq_paths=roi_12861_8815_15027_10115

python3 propagate_by_tracking.py class_names_path=data/predefined_classes_ipsc_3_class.txt root_dir=/data/ipsc/well3/all_frames_roi seq_paths=roi_12961_11916_14661_12816

python3 propagate_by_tracking.py class_names_path=data/predefined_classes_ipsc_3_class.txt root_dir=/data/ipsc/well3/all_frames_roi seq_paths=roi_13894_13749_16527_15316

python3 propagate_by_tracking.py class_names_path=data/predefined_classes_ipsc_3_class.txt root_dir=/data/ipsc/well3/all_frames_roi seq_paths=roi_14094_17682_15894_19749

python3 propagate_by_tracking.py class_names_path=data/predefined_classes_ipsc_3_class.txt root_dir=/data/ipsc/well3/all_frames_roi seq_paths=roi_15827_11316_17627_12749

python3 propagate_by_tracking.py class_names_path=data/predefined_classes_ipsc_3_class.txt root_dir=/data/ipsc/well3/all_frames_roi seq_paths=roi_15927_17249_17627_19582

python3 propagate_by_tracking.py class_names_path=data/predefined_classes_ipsc_3_class.txt root_dir=/data/ipsc/well3/all_frames_roi seq_paths=roi_17094_13782_19127_16348

python3 propagate_by_tracking.py class_names_path=data/predefined_classes_ipsc_3_class.txt root_dir=/data/ipsc/well3/all_frames_roi seq_paths=roi_17861_11316_19661_12616
python3 propagate_by_tracking.py class_names_path=data/predefined_classes_ipsc_2_class.txt ignore_invalid_class=1 root_dir=/data/ipsc/well3/all_frames_roi seq_paths=roi_17861_11316_19661_12616 save_track=1 start_id=-1 write_xml=0

<a id="manifold_embeddin_g_"></a>
# manifold_embedding

<a id="frame_251_roi_16627_11116_18727_12582___manifold_embedding_"></a>
## Frame_251__roi_16627_11116_18727_12582       @ manifold_embedding-->mask_ipsc
python3 to_mask_seq.py class_names_path=data/predefined_classes_ipsc_dual.txt root_dir=/data/ipsc/201020/masks_0x0_0/images seq_paths=Frame_251__roi_16627_11116_18727_12582 out_mask_size=0x0 out_border=0 show_img=0 extract_features=2

<a id="g4___manifold_embedding_"></a>
## g4       @ manifold_embedding-->mask_ipsc
python3 to_mask_seq.py class_names_path=data/predefined_classes_ipsc_dual.txt root_dir=/data/ipsc/201020/masks_0x0_0/images seq_paths=g4.txt out_mask_size=0x0 out_border=0 show_img=0 extract_features=2

<a id="0___g4_manifold_embeddin_g_"></a>
### 0       @ g4/manifold_embedding-->mask_ipsc
python3 to_mask_seq.py class_names_path=data/predefined_classes_ipsc_dual.txt root_dir=/data/ipsc/201020/masks_0x0_0/images seq_paths=g4.txt out_mask_size=0x0 out_border=0 show_img=0 extract_features=2 single_id=0 crop=0
<a id="1___g4_manifold_embeddin_g_"></a>
### 1       @ g4/manifold_embedding-->mask_ipsc
python3 to_mask_seq.py class_names_path=data/predefined_classes_ipsc_dual.txt root_dir=/data/ipsc/201020/masks_0x0_0/images seq_paths=g4.txt out_mask_size=0x0 out_border=0 show_img=0 extract_features=2 single_id=1 crop=0
<a id="2___g4_manifold_embeddin_g_"></a>
### 2       @ g4/manifold_embedding-->mask_ipsc
python3 to_mask_seq.py class_names_path=data/predefined_classes_ipsc_dual.txt root_dir=/data/ipsc/201020/masks_0x0_0/images seq_paths=g4.txt out_mask_size=0x0 out_border=0 show_img=0 extract_features=2 single_id=2 crop=0
<a id="3___g4_manifold_embeddin_g_"></a>
### 3       @ g4/manifold_embedding-->mask_ipsc
python3 to_mask_seq.py class_names_path=data/predefined_classes_ipsc_dual.txt root_dir=/data/ipsc/201020/masks_0x0_0/images seq_paths=g4.txt out_mask_size=0x0 out_border=0 show_img=0 extract_features=2 single_id=3 crop=0
<a id="4___g4_manifold_embeddin_g_"></a>
### 4       @ g4/manifold_embedding-->mask_ipsc
python3 to_mask_seq.py class_names_path=data/predefined_classes_ipsc_dual.txt root_dir=/data/ipsc/201020/masks_0x0_0/images seq_paths=g4.txt out_mask_size=0x0 out_border=0 show_img=0 extract_features=2 single_id=4 crop=0

<a id="g3___manifold_embedding_"></a>
## g3       @ manifold_embedding-->mask_ipsc
python3 to_mask_seq.py class_names_path=data/predefined_classes_ipsc_dual.txt root_dir=/data/ipsc/201020 seq_paths=g3.txt out_mask_size=0x0 out_border=0 show_img=0 extract_features=1

<a id="load_masks___g3_manifold_embeddin_g_"></a>
### load_masks       @ g3/manifold_embedding-->mask_ipsc
python3 to_mask_seq.py class_names_path=data/predefined_classes_ipsc_dual.txt root_dir=/data/ipsc/201020/masks_0x0_0/images seq_paths=g3.txt extract_features=2

<a id="0___g3_manifold_embeddin_g_"></a>
### 0       @ g3/manifold_embedding-->mask_ipsc
python3 to_mask_seq.py class_names_path=data/predefined_classes_ipsc_dual.txt root_dir=/data/ipsc/201020/masks_0x0_0/images seq_paths=g3.txt extract_features=2 single_id=0 crop=0
<a id="1___g3_manifold_embeddin_g_"></a>
### 1       @ g3/manifold_embedding-->mask_ipsc
python3 to_mask_seq.py class_names_path=data/predefined_classes_ipsc_dual.txt root_dir=/data/ipsc/201020/masks_0x0_0/images seq_paths=g3.txt extract_features=2 single_id=1 crop=0
<a id="2___g3_manifold_embeddin_g_"></a>
### 2       @ g3/manifold_embedding-->mask_ipsc
python3 to_mask_seq.py class_names_path=data/predefined_classes_ipsc_dual.txt root_dir=/data/ipsc/201020/masks_0x0_0/images seq_paths=g3.txt extract_features=2 single_id=2 crop=0
<a id="3___g3_manifold_embeddin_g_"></a>
### 3       @ g3/manifold_embedding-->mask_ipsc
python3 to_mask_seq.py class_names_path=data/predefined_classes_ipsc_dual.txt root_dir=/data/ipsc/201020/masks_0x0_0/images seq_paths=g3.txt extract_features=2 single_id=3 crop=0
<a id="4___g3_manifold_embeddin_g_"></a>
### 4       @ g3/manifold_embedding-->mask_ipsc
python3 to_mask_seq.py class_names_path=data/predefined_classes_ipsc_dual.txt root_dir=/data/ipsc/201020/masks_0x0_0/images seq_paths=g3.txt extract_features=2 single_id=4 crop=0

<a id="g2___manifold_embedding_"></a>
## g2       @ manifold_embedding-->mask_ipsc

python3 to_mask_seq.py class_names_path=data/predefined_classes_ipsc_dual.txt root_dir=/data/ipsc/201020 seq_paths=g2.txt out_mask_size=0x0 out_border=0 show_img=0 extract_features=1

python3 to_mask_seq.py class_names_path=data/predefined_classes_ipsc_dual.txt root_dir=/data/ipsc_patches/images seq_paths=g2.txt out_mask_size=0x0 out_border=0 show_img=0 extract_features=2

<a id="0___g2_manifold_embeddin_g_"></a>
### 0       @ g2/manifold_embedding-->mask_ipsc
python3 to_mask_seq.py class_names_path=data/predefined_classes_ipsc_dual.txt root_dir=/data/ipsc/201020/masks_0x0_0/images seq_paths=g2.txt out_mask_size=0x0 out_border=0 show_img=0 extract_features=2 single_id=0 crop=0
<a id="1___g2_manifold_embeddin_g_"></a>
### 1       @ g2/manifold_embedding-->mask_ipsc
python3 to_mask_seq.py class_names_path=data/predefined_classes_ipsc_dual.txt root_dir=/data/ipsc/201020/masks_0x0_0/images seq_paths=g2.txt out_mask_size=0x0 out_border=0 show_img=0 extract_features=2 single_id=1 crop=0
<a id="2___g2_manifold_embeddin_g_"></a>
### 2       @ g2/manifold_embedding-->mask_ipsc
python3 to_mask_seq.py class_names_path=data/predefined_classes_ipsc_dual.txt root_dir=/data/ipsc/201020/masks_0x0_0/images seq_paths=g2.txt out_mask_size=0x0 out_border=0 show_img=0 extract_features=2 single_id=2 crop=0

<a id="g1___manifold_embedding_"></a>
## g1       @ manifold_embedding-->mask_ipsc
python3 to_mask_seq.py class_names_path=data/predefined_classes_ipsc_dual.txt root_dir=/data/ipsc/201020 seq_paths=g1.txt out_mask_size=0x0 out_border=0 show_img=0 extract_features=1

python3 to_mask_seq.py class_names_path=data/predefined_classes_ipsc_dual.txt root_dir=/data/ipsc_patches/images seq_paths=g1.txt out_mask_size=0x0 out_border=0 show_img=0 extract_features=2

<a id="0___g1_manifold_embeddin_g_"></a>
#### 0       @ g1/manifold_embedding-->mask_ipsc
python3 to_mask_seq.py class_names_path=data/predefined_classes_ipsc_dual.txt root_dir=/data/ipsc/201020/masks_0x0_0/images seq_paths=g1.txt out_mask_size=0x0 out_border=0 show_img=0 extract_features=2 single_id=0 crop=0
<a id="1___g1_manifold_embeddin_g_"></a>
#### 1       @ g1/manifold_embedding-->mask_ipsc
python3 to_mask_seq.py class_names_path=data/predefined_classes_ipsc_dual.txt root_dir=/data/ipsc/201020/masks_0x0_0/images seq_paths=g1.txt out_mask_size=0x0 out_border=0 show_img=0 extract_features=2 single_id=1 crop=0
<a id="2___g1_manifold_embeddin_g_"></a>
#### 2       @ g1/manifold_embedding-->mask_ipsc
python3 to_mask_seq.py class_names_path=data/predefined_classes_ipsc_dual.txt root_dir=/data/ipsc/201020/masks_0x0_0/images seq_paths=g1.txt out_mask_size=0x0 out_border=0 show_img=0 extract_features=2 single_id=2 crop=0


<a id="to_mask_seq_"></a>
# to_mask_seq

<a id="all___to_mask_se_q_"></a>
## all       @ to_mask_seq-->mask_ipsc
python3 to_mask_seq.py class_names_path=data/predefined_classes_ipsc.txt root_dir=/data/ipsc/201020 out_mask_size=0x0 out_border=0 show_img=0 map_to_bbox=0

<a id="513x513_20___all_to_mask_se_q_"></a>
### 513x513_20       @ all/to_mask_seq-->mask_ipsc
python3 to_mask_seq.py class_names_path=data/predefined_classes_ipsc.txt root_dir=/data/ipsc/201020 out_mask_size=513x513 out_border=20 show_img=0 map_to_bbox=0 allow_skipping_images=1

<a id="frame_101_150_roi_7777_10249_10111_13349___all_to_mask_se_q_"></a>
### Frame_101_150_roi_7777_10249_10111_13349       @ all/to_mask_seq-->mask_ipsc
python3 to_mask_seq.py seq_paths=Frame_101_150_roi_7777_10249_10111_13349  class_names_path=data/predefined_classes_ipsc.txt root_dir=/data/ipsc/201020 out_mask_size=0x0 out_border=0 show_img=0 map_to_bbox=0

<a id="2_class___to_mask_se_q_"></a>
## 2_class       @ to_mask_seq-->mask_ipsc
python36 to_mask_seq.py class_names_path=data/predefined_classes_ipsc_dual.txt root_dir=/data/ipsc/201020 out_mask_size=0x0 out_border=0 show_img=0 map_to_bbox=0

<a id="blended_vis___2_class_to_mask_se_q_"></a>
### blended_vis       @ 2_class/to_mask_seq-->mask_ipsc
python36 to_mask_seq.py class_names_path=data/predefined_classes_ipsc_dual.txt root_dir=/data/ipsc/201020 out_mask_size=0x0 out_border=0 show_img=0 map_to_bbox=0 blended_vis=1

<a id="frame_101_150_roi_12660_17981_16026_20081___2_class_to_mask_se_q_"></a>
### Frame_101_150_roi_12660_17981_16026_20081       @ 2_class/to_mask_seq-->mask_ipsc
python36 to_mask_seq.py class_names_path=data/predefined_classes_ipsc_dual.txt root_dir=/data/ipsc/201020 out_mask_size=0x0 out_border=0 show_img=1 map_to_bbox=0 seq_paths=Frame_101_150_roi_12660_17981_16026_20081

<a id="5_class___to_mask_se_q_"></a>
## 5_class       @ to_mask_seq-->mask_ipsc
python36 to_mask_seq.py class_names_path=data/predefined_classes_ipsc_5_class.txt root_dir=/data/ipsc_5_class_raw out_mask_size=0x0 out_border=0 show_img=0 map_to_bbox=0 

<a id="blended_vis___5_class_to_mask_se_q_"></a>
### blended_vis       @ 5_class/to_mask_seq-->mask_ipsc
python36 to_mask_seq.py class_names_path=data/predefined_classes_ipsc_5_class.txt root_dir=/data/ipsc_5_class_raw out_mask_size=0x0 out_border=0 show_img=0 map_to_bbox=0 blended_vis=1

<a id="frame_201_250_roi_11927_12517_15394_15550___5_class_to_mask_se_q_"></a>
### Frame_201_250_roi_11927_12517_15394_15550       @ 5_class/to_mask_seq-->mask_ipsc
python36 to_mask_seq.py class_names_path=data/predefined_classes_ipsc_5_class.txt root_dir=/data/ipsc_5_class_raw out_mask_size=0x0 out_border=0 show_img=0 map_to_bbox=0 seq_paths=Frame_201_250_roi_11927_12517_15394_15550

<a id="5_class_220211___to_mask_se_q_"></a>
## 5_class_220211       @ to_mask_seq-->mask_ipsc
python36 to_mask_seq.py class_names_path=data/predefined_classes_ipsc_5_class.txt root_dir=/data/ipsc_5_class_raw_220211 out_mask_size=0x0 out_border=0 show_img=0 map_to_bbox=0 

<a id="blended_vis___5_class_220211_to_mask_seq_"></a>
### blended_vis       @ 5_class_220211/to_mask_seq-->mask_ipsc
python36 to_mask_seq.py class_names_path=data/predefined_classes_ipsc_5_class.txt root_dir=/data/ipsc_5_class_raw_220211 out_mask_size=0x0 out_border=0 show_img=0 map_to_bbox=0 blended_vis=1

<a id="frame_201_250_roi_11927_12517_15394_15550___5_class_220211_to_mask_seq_"></a>
### Frame_201-250_roi_11927_12517_15394_15550       @ 5_class_220211/to_mask_seq-->mask_ipsc
python36 to_mask_seq.py class_names_path=data/predefined_classes_ipsc_5_class.txt root_dir=/data/ipsc_5_class_raw_220211 out_mask_size=0x0 out_border=0 show_img=0 map_to_bbox=0 seq_paths=Frame_201-250_roi_11927_12517_15394_15550 blended_vis=1

<a id="all_frames_roi___to_mask_se_q_"></a>
## all_frames_roi       @ to_mask_seq-->mask_ipsc
python36 to_mask_seq.py class_names_path=data/predefined_classes_ipsc_3_class.txt root_dir=/data/ipsc_5_class_raw_220211 out_mask_size=0x0 out_border=0 show_img=0 map_to_bbox=0 

<a id="blended_vis___all_frames_roi_to_mask_seq_"></a>
### blended_vis       @ all_frames_roi/to_mask_seq-->mask_ipsc
python36 to_mask_seq.py class_names_path=data/predefined_classes_ipsc_3_class.txt root_dir=/data/ipsc/well3/all_frames_roi out_mask_size=0x0 out_border=0 show_img=0 map_to_bbox=0 blended_vis=1 n_proc=8

