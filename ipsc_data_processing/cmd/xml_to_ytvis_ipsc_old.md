<!-- MarkdownTOC -->

- [gram](#gra_m_)
    - [0_1       @ gram](#0_1___gram_)
- [idot](#ido_t_)
    - [0_1       @ idot](#0_1___idot_)
- [detrac](#detra_c_)
    - [0_0       @ detrac](#0_0___detrac_)
    - [0_19       @ detrac](#0_19___detrac_)
        - [strd-2       @ 0_19/detrac](#strd_2___0_19_detra_c_)
    - [0_9       @ detrac](#0_9___detrac_)
        - [strd-2       @ 0_9/detrac](#strd_2___0_9_detrac_)
    - [49_68       @ detrac](#49_68___detrac_)
        - [strd-2       @ 49_68/detrac](#strd_2___49_68_detrac_)
- [mnist-640-1](#mnist_640_1_)
    - [len-2:strd-1       @ mnist-640-1](#len_2_strd_1___mnist_640_1_)
        - [test       @ len-2:strd-1/mnist-640-1](#test___len_2_strd_1_mnist_640_1_)
            - [strd-2       @ test/len-2:strd-1/mnist-640-1](#strd_2___test_len_2_strd_1_mnist_640_1_)
    - [len-3:strd-1       @ mnist-640-1](#len_3_strd_1___mnist_640_1_)
        - [test       @ len-3:strd-1/mnist-640-1](#test___len_3_strd_1_mnist_640_1_)
            - [strd-3       @ test/len-3:strd-1/mnist-640-1](#strd_3___test_len_3_strd_1_mnist_640_1_)
    - [len-9:strd-1       @ mnist-640-1](#len_9_strd_1___mnist_640_1_)
- [mnist-640-3](#mnist_640_3_)
    - [len-2:strd-1       @ mnist-640-3](#len_2_strd_1___mnist_640_3_)
        - [test       @ len-2:strd-1/mnist-640-3](#test___len_2_strd_1_mnist_640_3_)
- [mnist-640-5](#mnist_640_5_)
    - [len-2:strd-1       @ mnist-640-5](#len_2_strd_1___mnist_640_5_)
    - [len-2:strd-2       @ mnist-640-5](#len_2_strd_2___mnist_640_5_)
    - [len-3:strd-1       @ mnist-640-5](#len_3_strd_1___mnist_640_5_)
    - [len-3:strd-3       @ mnist-640-5](#len_3_strd_3___mnist_640_5_)
    - [len-4:strd-1       @ mnist-640-5](#len_4_strd_1___mnist_640_5_)
    - [len-4:strd-4       @ mnist-640-5](#len_4_strd_4___mnist_640_5_)
    - [len-6:strd-1       @ mnist-640-5](#len_6_strd_1___mnist_640_5_)
    - [len-6:strd-6       @ mnist-640-5](#len_6_strd_6___mnist_640_5_)
    - [len-8:strd-1       @ mnist-640-5](#len_8_strd_1___mnist_640_5_)
    - [len-9:strd-1       @ mnist-640-5](#len_9_strd_1___mnist_640_5_)
        - [test       @ len-9:strd-1/mnist-640-5](#test___len_9_strd_1_mnist_640_5_)
- [ipsc](#ips_c_)
    - [0_4       @ ipsc](#0_4___ipsc_)
        - [12094       @ 0_4/ipsc](#12094___0_4_ipsc_)
    - [5_9       @ ipsc](#5_9___ipsc_)
    - [0_37       @ ipsc](#0_37___ipsc_)
    - [16_53       @ ipsc](#16_53___ipsc_)
        - [len-2       @ 16_53/ipsc](#len_2___16_53_ipsc_)
        - [len-3       @ 16_53/ipsc](#len_3___16_53_ipsc_)
    - [54_126       @ ipsc](#54_126___ipsc_)
- [all_frames_roi_g2_38_53       @ all_frames_roi](#all_frames_roi_g2_38_53___all_frames_roi_)
- [all_frames_roi_g2_seq_1_38_53       @ all_frames_roi](#all_frames_roi_g2_seq_1_38_53___all_frames_roi_)
- [all_frames_roi_g3_53_92       @ all_frames_roi](#all_frames_roi_g3_53_92___all_frames_roi_)
- [ext_reorg_roi_g2_0_37       @ xml_to_ytvis](#ext_reorg_roi_g2_0_37___xml_to_ytvis_)
    - [max_length-20       @ ext_reorg_roi_g2_0_37](#max_length_20___ext_reorg_roi_g2_0_3_7_)
    - [max_length-10       @ ext_reorg_roi_g2_0_37](#max_length_10___ext_reorg_roi_g2_0_3_7_)
- [ext_reorg_roi_g2_38_53       @ xml_to_ytvis](#ext_reorg_roi_g2_38_53___xml_to_ytvis_)
    - [incremental       @ ext_reorg_roi_g2_38_53](#incremental___ext_reorg_roi_g2_38_53_)
- [ext_reorg_roi_g2_16_53       @ xml_to_ytvis](#ext_reorg_roi_g2_16_53___xml_to_ytvis_)
- [ext_reorg_roi_g2_0_1       @ xml_to_ytvis](#ext_reorg_roi_g2_0_1___xml_to_ytvis_)
- [0_15       @ xml_to_ytvis](#0_15___xml_to_ytvis_)
    - [incremental       @ 0_15](#incremental___0_15_)
- [ext_reorg_roi_g2_54_126       @ xml_to_ytvis](#ext_reorg_roi_g2_54_126___xml_to_ytvis_)
    - [subseq       @ ext_reorg_roi_g2_54_126](#subseq___ext_reorg_roi_g2_54_12_6_)
- [ext_reorg_roi_g2_0_53       @ xml_to_ytvis](#ext_reorg_roi_g2_0_53___xml_to_ytvis_)
    - [incremental       @ ext_reorg_roi_g2_0_53](#incremental___ext_reorg_roi_g2_0_5_3_)

<!-- /MarkdownTOC -->

<a id="gra_m_"></a>
# gram
<a id="0_1___gram_"></a>
## 0_1       @ gram-->xml_to_ytvis
python xml_to_ytvis.py cfg=gram:0_1:proc-1:len-9:strd-1:gz:gap-1 
<a id="ido_t_"></a>
# idot
<a id="0_1___idot_"></a>
## 0_1       @ idot-->xml_to_ytvis
python xml_to_ytvis.py cfg=idot:proc-1:len-9:strd-1:gz:gap-1 

<a id="detra_c_"></a>
# detrac
<a id="0_0___detrac_"></a>
## 0_0       @ detrac-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:0_0:proc-1:len-100:strd-100:gz:gap-1:vis 
<a id="0_19___detrac_"></a>
## 0_19       @ detrac-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:0_19:proc-1:len-2:strd-1:gz:gap-1 
python xml_to_ytvis.py cfg=detrac:non_empty:0_19:proc-1:len-3:strd-1:gz:gap-1 
python xml_to_ytvis.py cfg=detrac:non_empty:0_19:proc-1:len-4:strd-1:gz:gap-1 
python xml_to_ytvis.py cfg=detrac:non_empty:0_19:proc-1:len-6:strd-1:gz:gap-1 
python xml_to_ytvis.py cfg=detrac:non_empty:0_19:proc-1:len-8:strd-1:gz:gap-1 
<a id="strd_2___0_19_detra_c_"></a>
### strd-2       @ 0_19/detrac-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:0_19:proc-1:len-2:strd-2:gz:gap-1 
<a id="0_9___detrac_"></a>
## 0_9       @ detrac-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:0_9:proc-1:len-2:strd-1:gz:gap-1:vis 
<a id="strd_2___0_9_detrac_"></a>
### strd-2       @ 0_9/detrac-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:0_9:proc-1:len-2:strd-2:gz:gap-1 
<a id="49_68___detrac_"></a>
## 49_68       @ detrac-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:49_68:proc-12:len-2:strd-1:gz:gap-1
`strd-12 frame-0_360`
python xml_to_ytvis.py cfg=detrac:non_empty:49_68:proc-12:len-3:strd-12:gz:gap-1:frame-0_360
<a id="strd_2___49_68_detrac_"></a>
### strd-2       @ 49_68/detrac-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:49_68:proc-12:len-2:strd-2:gz:gap-1

<a id="mnist_640_1_"></a>
# mnist-640-1
<a id="len_2_strd_1___mnist_640_1_"></a>
## len-2:strd-1       @ mnist-640-1-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=mnist:640-1:12_1000:train:proc-12:len-2:strd-1:gz:gap-1 
<a id="test___len_2_strd_1_mnist_640_1_"></a>
### test       @ len-2:strd-1/mnist-640-1-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=mnist:640-1:12_1000:test:proc-12:len-2:strd-1:gz:gap-1 
<a id="strd_2___test_len_2_strd_1_mnist_640_1_"></a>
#### strd-2       @ test/len-2:strd-1/mnist-640-1-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=mnist:640-1:12_1000:test:proc-1:len-2:strd-2:gz:gap-1 

<a id="len_3_strd_1___mnist_640_1_"></a>
## len-3:strd-1       @ mnist-640-1-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=mnist:640-1:12_1000:train:proc-1:len-3:strd-1:gz:gap-1 
<a id="test___len_3_strd_1_mnist_640_1_"></a>
### test       @ len-3:strd-1/mnist-640-1-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=mnist:640-1:12_1000:test:proc-1:len-3:strd-1:gz:gap-1 
<a id="strd_3___test_len_3_strd_1_mnist_640_1_"></a>
#### strd-3       @ test/len-3:strd-1/mnist-640-1-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=mnist:640-1:12_1000:test:proc-1:len-3:strd-3:gz:gap-1 

<a id="len_9_strd_1___mnist_640_1_"></a>
## len-9:strd-1       @ mnist-640-1-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=mnist:640-1:12_1000:train:proc-12:len-9:strd-1:gz:gap-1 

<a id="mnist_640_3_"></a>
# mnist-640-3
<a id="len_2_strd_1___mnist_640_3_"></a>
## len-2:strd-1       @ mnist-640-3-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=mnist:640-3:12_1000:train:proc-6:len-2:strd-1:gz:gap-1 
<a id="test___len_2_strd_1_mnist_640_3_"></a>
### test       @ len-2:strd-1/mnist-640-3-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=mnist:640-3:12_1000:train:test:proc-6:len-2:strd-1:gz:gap-1 

<a id="mnist_640_5_"></a>
# mnist-640-5
<a id="len_2_strd_1___mnist_640_5_"></a>
## len-2:strd-1       @ mnist-640-5-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=mnist:640-5:12_1000:train:proc-6:len-2:strd-1:gz:gap-1 
python3 xml_to_ytvis.py cfg=mnist:640-5:12_1000:test:proc-6:len-2:strd-1:gz:gap-1 
<a id="len_2_strd_2___mnist_640_5_"></a>
## len-2:strd-2       @ mnist-640-5-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=mnist:640-5:12_1000:test:proc-6:len-2:strd-2:gz:gap-1 

<a id="len_3_strd_1___mnist_640_5_"></a>
## len-3:strd-1       @ mnist-640-5-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=mnist:640-5:12_1000:train:proc-12:len-3:strd-1:gz:gap-1 
python3 xml_to_ytvis.py cfg=mnist:640-5:12_1000:test:proc-12:len-3:strd-1:gz:gap-1 

<a id="len_3_strd_3___mnist_640_5_"></a>
## len-3:strd-3       @ mnist-640-5-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=mnist:640-5:12_1000:test:proc-12:len-3:strd-3:gz:gap-1 

<a id="len_4_strd_1___mnist_640_5_"></a>
## len-4:strd-1       @ mnist-640-5-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=mnist:640-5:12_1000:train:proc-12:len-4:strd-1:gz:gap-1 
python3 xml_to_ytvis.py cfg=mnist:640-5:12_1000:test:proc-12:len-4:strd-1:gz:gap-1 

<a id="len_4_strd_4___mnist_640_5_"></a>
## len-4:strd-4       @ mnist-640-5-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=mnist:640-5:12_1000:test:proc-12:len-4:strd-4:gz:gap-1 

<a id="len_6_strd_1___mnist_640_5_"></a>
## len-6:strd-1       @ mnist-640-5-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=mnist:640-5:12_1000:train:proc-12:len-6:strd-1:gz:gap-1 
python3 xml_to_ytvis.py cfg=mnist:640-5:12_1000:test:proc-12:len-6:strd-1:gz:gap-1 
<a id="len_6_strd_6___mnist_640_5_"></a>
## len-6:strd-6       @ mnist-640-5-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=mnist:640-5:12_1000:test:proc-12:len-6:strd-6:gz:gap-1 

<a id="len_8_strd_1___mnist_640_5_"></a>
## len-8:strd-1       @ mnist-640-5-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=mnist:640-5:12_1000:train:proc-12:len-8:strd-1:gz:gap-1 

<a id="len_9_strd_1___mnist_640_5_"></a>
## len-9:strd-1       @ mnist-640-5-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=mnist:640-5:12_1000:train:proc-6:len-9:strd-1:gz:gap-1 
<a id="test___len_9_strd_1_mnist_640_5_"></a>
### test       @ len-9:strd-1/mnist-640-5-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=mnist:640-5:12_1000:test:proc-6:len-2:strd-1:gz:gap-1 

<a id="ips_c_"></a>
# ipsc
<a id="0_4___ipsc_"></a>
## 0_4       @ ipsc-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=ipsc:2_class:mask-0:frame-0_4:len-2:strd-1:gz-1:gap-1 
python3 xml_to_ytvis.py cfg=ipsc:2_class:mask-0:frame-0_4:len-2:strd-1:gz-1:gap-3 

<a id="12094___0_4_ipsc_"></a>
### 12094       @ 0_4/ipsc-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=ipsc:12094:2_class:mask-0:proc-12:start-0:end-4:len-2:strd-1:gz-1:gap-1 
python3 xml_to_ytvis.py cfg=ipsc:12094_short:2_class:mask-0:proc-12:start-0:end-4:len-2:strd-1:gz-1:gap-1 

<a id="5_9___ipsc_"></a>
## 5_9       @ ipsc-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=ipsc:2_class:mask-0:proc-12:start-5:end-9:len-2:strd-1:gz-1:gap-1 
python3 xml_to_ytvis.py cfg=ipsc:2_class:mask-0:proc-1:start-5:end-9:len-2:strd-1:gz-1:gap-2 
python3 xml_to_ytvis.py cfg=ipsc:2_class:mask-0:proc-1:start-5:end-9:len-2:strd-1:gz-1:gap-3 
python3 xml_to_ytvis.py cfg=ipsc:2_class:mask-0:proc-1:start-5:end-9:len-2:strd-1:gz-1:gap-4 

<a id="0_37___ipsc_"></a>
## 0_37       @ ipsc-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=ipsc:2_class:0_37:len-2:strd-1:gz-1:gap-1:mask-0:proc-1:zip-0 
<a id="16_53___ipsc_"></a>
## 16_53       @ ipsc-->xml_to_ytvis
<a id="len_2___16_53_ipsc_"></a>
### len-2       @ 16_53/ipsc-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=ipsc:2_class:16_53:len-2:strd-1:gz-1:gap-1:mask-0:proc-1:zip-0 
python3 xml_to_ytvis.py cfg=ipsc:2_class:16_53:len-2:strd-1:gz-1:gap-2:mask-0:proc-1:zip-0 
python3 xml_to_ytvis.py cfg=ipsc:2_class:16_53:len-2:strd-1:gz-1:gap-3:mask-0:proc-1:zip-0 
python3 xml_to_ytvis.py cfg=ipsc:2_class:16_53:len-2:strd-1:gz-1:gap-4:mask-0:proc-1:zip-0 
<a id="len_3___16_53_ipsc_"></a>
### len-3       @ 16_53/ipsc-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=ipsc:2_class:16_53:len-3:strd-1:gz-1:gap-1:mask-0:proc-1:zip-0 

<a id="54_126___ipsc_"></a>
## 54_126       @ ipsc-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=ipsc:2_class:54_126:len-2:strd-1:gz:gap-1:mask-0:proc-1:zip-0 
python3 xml_to_ytvis.py cfg=ipsc:2_class:54_126:len-2:strd-2:gz:gap-1:mask-0:proc-1:zip-0 

<a id="all_frames_roi_g2_38_53___all_frames_roi_"></a>
# all_frames_roi_g2_38_53       @ all_frames_roi-->xml_to_ytvis
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=all_frames_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=38 end_frame_id=53 ignore_invalid_label=1 val_ratio=0 min_val=0 allow_missing_images=0 get_img_stats=0 description=all_frames_roi_g2_38_53 save_masks=0 coco_rle=1

<a id="all_frames_roi_g2_seq_1_38_53___all_frames_roi_"></a>
# all_frames_roi_g2_seq_1_38_53       @ all_frames_roi-->xml_to_ytvis
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=all_frames_roi.txt n_seq=1 class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=38 end_frame_id=53 ignore_invalid_label=1 val_ratio=0 min_val=0 allow_missing_images=0 get_img_stats=0 description=all_frames_roi_g2_seq_1_38_53 save_masks=0 coco_rle=1

<a id="all_frames_roi_g3_53_92___all_frames_roi_"></a>
# all_frames_roi_g3_53_92       @ all_frames_roi-->xml_to_ytvis
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=all_frames_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=53 end_frame_id=92 ignore_invalid_label=1 val_ratio=0 min_val=0 allow_missing_images=0 get_img_stats=1 description=all_frames_roi_g3_53_92 save_masks=1

<a id="ext_reorg_roi_g2_0_37___xml_to_ytvis_"></a>
# ext_reorg_roi_g2_0_37       @ xml_to_ytvis-->xml_to_ytvis
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=lists/ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=0 end_frame_id=37 ignore_invalid_label=1 val_ratio=0 allow_missing_images=0 get_img_stats=0 description=ext_reorg_roi_g2_0_37 save_masks=0

<a id="max_length_20___ext_reorg_roi_g2_0_3_7_"></a>
## max_length-20       @ ext_reorg_roi_g2_0_37-->xml_to_ytvis
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=lists/ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=0 end_frame_id=37 ignore_invalid_label=1 val_ratio=0 allow_missing_images=0 get_img_stats=0 description=ext_reorg_roi_g2_0_37 save_masks=0 max_length=20 n_proc=12

<a id="max_length_10___ext_reorg_roi_g2_0_3_7_"></a>
## max_length-10       @ ext_reorg_roi_g2_0_37-->xml_to_ytvis
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=lists/ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=0 end_frame_id=37 ignore_invalid_label=1 val_ratio=0 allow_missing_images=0 get_img_stats=0 description=ext_reorg_roi_g2_0_37 save_masks=0 max_length=10 n_proc=12

<a id="ext_reorg_roi_g2_38_53___xml_to_ytvis_"></a>
# ext_reorg_roi_g2_38_53       @ xml_to_ytvis-->xml_to_ytvis
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=lists/ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=38 end_frame_id=53 ignore_invalid_label=1 val_ratio=0 min_val=0 allow_missing_images=0 get_img_stats=0 description=ext_reorg_roi_g2_38_53 save_masks=0 n_proc=12

<a id="incremental___ext_reorg_roi_g2_38_53_"></a>
## incremental       @ ext_reorg_roi_g2_38_53-->xml_to_ytvis
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=lists/ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=38 end_frame_id=53 ignore_invalid_label=1 val_ratio=0 min_val=0 allow_missing_images=0 get_img_stats=0 description=ext_reorg_roi_g2_38_53 save_masks=0 n_proc=24 incremental=1

<a id="ext_reorg_roi_g2_16_53___xml_to_ytvis_"></a>
# ext_reorg_roi_g2_16_53       @ xml_to_ytvis-->xml_to_ytvis
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=lists/ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=16 end_frame_id=53 ignore_invalid_label=1 val_ratio=0 allow_missing_images=0 get_img_stats=0 description=ext_reorg_roi_g2_16_53 save_masks=0 subseq=1 subseq_split_ids=21 n_proc=12

<a id="ext_reorg_roi_g2_0_1___xml_to_ytvis_"></a>
# ext_reorg_roi_g2_0_1       @ xml_to_ytvis-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=ipsc:2_class start_frame_id=0 end_frame_id=1 description=ext_reorg_roi_g2_0_1 save_masks=0 n_proc=12
length_2-stride_1

<a id="0_15___xml_to_ytvis_"></a>
# 0_15       @ xml_to_ytvis-->xml_to_ytvis
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=lists/ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=0 end_frame_id=15 ignore_invalid_label=1 val_ratio=0 allow_missing_images=0 get_img_stats=0 description=ext_reorg_roi_g2_0_15 save_masks=0 n_proc=12
__-max_length-1__
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=lists/ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=0 end_frame_id=15 ignore_invalid_label=1 val_ratio=0 allow_missing_images=0 get_img_stats=0 description=ext_reorg_roi_g2_0_15 save_masks=0 n_proc=12 max_length=1
__-max_length-2__
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=lists/ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=0 end_frame_id=15 ignore_invalid_label=1 val_ratio=0 allow_missing_images=0 get_img_stats=0 description=ext_reorg_roi_g2_0_15 save_masks=0 n_proc=12 max_length=2
__-max_length-4__
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=lists/ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=0 end_frame_id=15 ignore_invalid_label=1 val_ratio=0 allow_missing_images=0 get_img_stats=0 description=ext_reorg_roi_g2_0_15 save_masks=0 n_proc=12 max_length=4
__-max_length-8-__
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=lists/ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=0 end_frame_id=15 ignore_invalid_label=1 val_ratio=0 allow_missing_images=0 get_img_stats=0 description=ext_reorg_roi_g2_0_15 save_masks=0 n_proc=12 max_length=8

<a id="incremental___0_15_"></a>
## incremental       @ 0_15-->xml_to_ytvis
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=lists/ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=0 end_frame_id=15 ignore_invalid_label=1 val_ratio=0 allow_missing_images=0 get_img_stats=0 description=ext_reorg_roi_g2_0_15 save_masks=0 n_proc=24 incremental=1
__-max_length-2__
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=lists/ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=0 end_frame_id=15 ignore_invalid_label=1 val_ratio=0 allow_missing_images=0 get_img_stats=0 description=ext_reorg_roi_g2_0_15 save_masks=0 n_proc=2 incremental=1 max_length=2
__-max_length-10__
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=lists/ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=0 end_frame_id=15 ignore_invalid_label=1 val_ratio=0 allow_missing_images=0 get_img_stats=0 description=ext_reorg_roi_g2_0_15 save_masks=0 n_proc=2 incremental=1 max_length=10

<a id="ext_reorg_roi_g2_54_126___xml_to_ytvis_"></a>
# ext_reorg_roi_g2_54_126       @ xml_to_ytvis-->xml_to_ytvis
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=lists/ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=54 end_frame_id=126 ignore_invalid_label=1 val_ratio=0 allow_missing_images=0 get_img_stats=0 description=ext_reorg_roi_g2_54_126 save_masks=0 n_proc=12
```
pix_vals_mean: [118.06, 118.06, 118.06]
pix_vals_std: [23.99, 23.99, 23.99]
```
<a id="subseq___ext_reorg_roi_g2_54_12_6_"></a>
## subseq       @ ext_reorg_roi_g2_54_126-->xml_to_ytvis
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=lists/ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=54 end_frame_id=126 ignore_invalid_label=1 val_ratio=0 allow_missing_images=0 get_img_stats=0 description=ext_reorg_roi_g2_54_126 save_masks=0 n_proc=12 subseq=1 subseq_split_ids=15 n_proc=12

<a id="ext_reorg_roi_g2_0_53___xml_to_ytvis_"></a>
# ext_reorg_roi_g2_0_53       @ xml_to_ytvis-->xml_to_ytvis
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=lists/ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=0 end_frame_id=53 ignore_invalid_label=1 val_ratio=0 allow_missing_images=0 get_img_stats=0 description=ext_reorg_roi_g2_0_53 save_masks=0 n_proc=12 subseq=1 subseq_split_ids=37
__-max_length-1-__
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=lists/ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=0 end_frame_id=53 ignore_invalid_label=1 val_ratio=0 allow_missing_images=0 get_img_stats=0 description=ext_reorg_roi_g2_0_53 save_masks=0 n_proc=12 max_length=1
__-max_length-2-__
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=lists/ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=0 end_frame_id=53 ignore_invalid_label=1 val_ratio=0 allow_missing_images=0 get_img_stats=0 description=ext_reorg_roi_g2_0_53 save_masks=0 n_proc=12 max_length=2
__-max_length-4-__
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=lists/ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=0 end_frame_id=53 ignore_invalid_label=1 val_ratio=0 allow_missing_images=0 get_img_stats=0 description=ext_reorg_roi_g2_0_53 save_masks=0 n_proc=12 max_length=4
__-max_length-8-__
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=lists/ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=0 end_frame_id=53 ignore_invalid_label=1 val_ratio=0 allow_missing_images=0 get_img_stats=0 description=ext_reorg_roi_g2_0_53 save_masks=0 n_proc=12 max_length=8
__-max_length-19-__
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=lists/ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=0 end_frame_id=53 ignore_invalid_label=1 val_ratio=0 allow_missing_images=0 get_img_stats=0 description=ext_reorg_roi_g2_0_53 save_masks=0 n_proc=12 max_length=19 subseq=1 subseq_split_ids=37

<a id="incremental___ext_reorg_roi_g2_0_5_3_"></a>
## incremental       @ ext_reorg_roi_g2_0_53-->xml_to_ytvis
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=lists/ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=0 end_frame_id=53 ignore_invalid_label=1 val_ratio=0 allow_missing_images=0 get_img_stats=0 description=ext_reorg_roi_g2_0_53 save_masks=0 n_proc=12 subseq=1 subseq_split_ids=38 incremental=1
__-max_length-2__
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=lists/ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=0 end_frame_id=53 ignore_invalid_label=1 val_ratio=0 allow_missing_images=0 get_img_stats=0 description=ext_reorg_roi_g2_0_53 save_masks=0 n_proc=12 incremental=1 max_length=2
__-max_length-10__
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=lists/ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=0 end_frame_id=53 ignore_invalid_label=1 val_ratio=0 allow_missing_images=0 get_img_stats=0 description=ext_reorg_roi_g2_0_53 save_masks=0 n_proc=12 subseq=1 subseq_split_ids=38 incremental=1 max_length=10
__-max_length-20__
python3 xml_to_ytvis.py root_dir=/data/ipsc/well3/all_frames_roi seq_paths=lists/ext_reorg_roi.txt class_names_path=lists/classes/predefined_classes_ipsc_2_class.txt start_frame_id=0 end_frame_id=53 ignore_invalid_label=1 val_ratio=0 allow_missing_images=0 get_img_stats=0 description=ext_reorg_roi_g2_0_53 save_masks=0 n_proc=12 subseq=1 subseq_split_ids=38 incremental=1 max_length=20
<a id="detrac_len_3_"></a>
