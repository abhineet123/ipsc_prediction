<!-- MarkdownTOC -->

- [imagenet_vid](#imagenet_vi_d_)
    - [train       @ imagenet_vid](#train___imagenet_vid_)
        - [len-2       @ train/imagenet_vid](#len_2___train_imagenet_vid_)
            - [4_per_seq_random_len_2       @ len-2/train/imagenet_vid](#4_per_seq_random_len_2___len_2_train_imagenet_vid_)
    - [val       @ imagenet_vid](#val___imagenet_vid_)
        - [len-2       @ val/imagenet_vid](#len_2___val_imagenet_vid_)
            - [8_per_seq_random_len_2       @ len-2/val/imagenet_vid](#8_per_seq_random_len_2___len_2_val_imagenet_vid_)
- [gram](#gra_m_)
    - [0_1       @ gram](#0_1___gram_)
        - [len-2       @ 0_1/gram](#len_2___0_1_gram_)
        - [len-9       @ 0_1/gram](#len_9___0_1_gram_)
        - [len-14       @ 0_1/gram](#len_14___0_1_gram_)
            - [0_2000       @ len-14/0_1/gram](#0_2000___len_14_0_1_gra_m_)
            - [3000_5000       @ len-14/0_1/gram](#3000_5000___len_14_0_1_gra_m_)
        - [len-16       @ 0_1/gram](#len_16___0_1_gram_)
            - [0_2000       @ len-16/0_1/gram](#0_2000___len_16_0_1_gra_m_)
            - [3000_5000       @ len-16/0_1/gram](#3000_5000___len_16_0_1_gra_m_)
- [idot](#ido_t_)
    - [0_1       @ idot](#0_1___idot_)
    - [8_8       @ idot](#8_8___idot_)
- [detrac](#detra_c_)
    - [0_59       @ detrac](#0_59___detrac_)
        - [len-2       @ 0_59/detrac](#len_2___0_59_detra_c_)
            - [80_per_seq_random_len_2       @ len-2/0_59/detrac](#80_per_seq_random_len_2___len_2_0_59_detra_c_)
        - [len-3       @ 0_59/detrac](#len_3___0_59_detra_c_)
            - [120_per_seq_random_len_3       @ len-3/0_59/detrac](#120_per_seq_random_len_3___len_3_0_59_detra_c_)
        - [len-4       @ 0_59/detrac](#len_4___0_59_detra_c_)
            - [80_per_seq_random_len_4       @ len-4/0_59/detrac](#80_per_seq_random_len_4___len_4_0_59_detra_c_)
        - [len-8       @ 0_59/detrac](#len_8___0_59_detra_c_)
    - [60_99       @ detrac](#60_99___detrac_)
        - [len-2       @ 60_99/detrac](#len_2___60_99_detrac_)
            - [80_per_seq_random_len_2       @ len-2/60_99/detrac](#80_per_seq_random_len_2___len_2_60_99_detrac_)
        - [len-3       @ 60_99/detrac](#len_3___60_99_detrac_)
            - [120_per_seq_random_len_3       @ len-3/60_99/detrac](#120_per_seq_random_len_3___len_3_60_99_detrac_)
        - [len-4       @ 60_99/detrac](#len_4___60_99_detrac_)
            - [80_per_seq_random_len_4       @ len-4/60_99/detrac](#80_per_seq_random_len_4___len_4_60_99_detrac_)
- [detrac-non_empty](#detrac_non_empt_y_)
    - [0_0       @ detrac-non_empty](#0_0___detrac_non_empty_)
    - [0_1       @ detrac-non_empty](#0_1___detrac_non_empty_)
    - [0_19       @ detrac-non_empty](#0_19___detrac_non_empty_)
        - [strd-2       @ 0_19/detrac-non_empty](#strd_2___0_19_detrac_non_empt_y_)
    - [0_9       @ detrac-non_empty](#0_9___detrac_non_empty_)
    - [49_68       @ detrac-non_empty](#49_68___detrac_non_empty_)
        - [len-2       @ 49_68/detrac-non_empty](#len_2___49_68_detrac_non_empty_)
        - [len-6       @ 49_68/detrac-non_empty](#len_6___49_68_detrac_non_empty_)
        - [len-9       @ 49_68/detrac-non_empty](#len_9___49_68_detrac_non_empty_)
        - [len-64       @ 49_68/detrac-non_empty](#len_64___49_68_detrac_non_empty_)
    - [0_48       @ detrac-non_empty](#0_48___detrac_non_empty_)
        - [len-2       @ 0_48/detrac-non_empty](#len_2___0_48_detrac_non_empt_y_)
        - [len-4       @ 0_48/detrac-non_empty](#len_4___0_48_detrac_non_empt_y_)
        - [len-8       @ 0_48/detrac-non_empty](#len_8___0_48_detrac_non_empt_y_)
            - [200_per_seq_random_len_8       @ len-8/0_48/detrac-non_empty](#200_per_seq_random_len_8___len_8_0_48_detrac_non_empt_y_)
        - [len-12       @ 0_48/detrac-non_empty](#len_12___0_48_detrac_non_empt_y_)
        - [len-16       @ 0_48/detrac-non_empty](#len_16___0_48_detrac_non_empt_y_)
        - [len-24       @ 0_48/detrac-non_empty](#len_24___0_48_detrac_non_empt_y_)
        - [len-28       @ 0_48/detrac-non_empty](#len_28___0_48_detrac_non_empt_y_)
        - [len-32       @ 0_48/detrac-non_empty](#len_32___0_48_detrac_non_empt_y_)
        - [len-40       @ 0_48/detrac-non_empty](#len_40___0_48_detrac_non_empt_y_)
        - [len-48       @ 0_48/detrac-non_empty](#len_48___0_48_detrac_non_empt_y_)
        - [len-56       @ 0_48/detrac-non_empty](#len_56___0_48_detrac_non_empt_y_)
        - [len-64       @ 0_48/detrac-non_empty](#len_64___0_48_detrac_non_empt_y_)
    - [0_85       @ detrac-non_empty](#0_85___detrac_non_empty_)
        - [len-2       @ 0_85/detrac-non_empty](#len_2___0_85_detrac_non_empt_y_)
        - [len-32       @ 0_85/detrac-non_empty](#len_32___0_85_detrac_non_empt_y_)
        - [len-40       @ 0_85/detrac-non_empty](#len_40___0_85_detrac_non_empt_y_)
        - [len-48       @ 0_85/detrac-non_empty](#len_48___0_85_detrac_non_empt_y_)
        - [len-56       @ 0_85/detrac-non_empty](#len_56___0_85_detrac_non_empt_y_)
        - [len-64       @ 0_85/detrac-non_empty](#len_64___0_85_detrac_non_empt_y_)
    - [49_85       @ detrac-non_empty](#49_85___detrac_non_empty_)
        - [strd-1       @ 49_85/detrac-non_empty](#strd_1___49_85_detrac_non_empty_)
            - [len-2       @ strd-1/49_85/detrac-non_empty](#len_2___strd_1_49_85_detrac_non_empt_y_)
            - [len-4       @ strd-1/49_85/detrac-non_empty](#len_4___strd_1_49_85_detrac_non_empt_y_)
            - [len-8       @ strd-1/49_85/detrac-non_empty](#len_8___strd_1_49_85_detrac_non_empt_y_)
            - [len-16       @ strd-1/49_85/detrac-non_empty](#len_16___strd_1_49_85_detrac_non_empt_y_)
                - [strds       @ len-16/strd-1/49_85/detrac-non_empty](#strds___len_16_strd_1_49_85_detrac_non_empty_)
            - [len-32       @ strd-1/49_85/detrac-non_empty](#len_32___strd_1_49_85_detrac_non_empt_y_)
                - [strds       @ len-32/strd-1/49_85/detrac-non_empty](#strds___len_32_strd_1_49_85_detrac_non_empty_)
            - [len-40       @ strd-1/49_85/detrac-non_empty](#len_40___strd_1_49_85_detrac_non_empt_y_)
            - [len-48       @ strd-1/49_85/detrac-non_empty](#len_48___strd_1_49_85_detrac_non_empt_y_)
            - [len-56       @ strd-1/49_85/detrac-non_empty](#len_56___strd_1_49_85_detrac_non_empt_y_)
            - [len-64       @ strd-1/49_85/detrac-non_empty](#len_64___strd_1_49_85_detrac_non_empt_y_)
        - [strd-same       @ 49_85/detrac-non_empty](#strd_same___49_85_detrac_non_empty_)
            - [len-2       @ strd-same/49_85/detrac-non_empty](#len_2___strd_same_49_85_detrac_non_empty_)
                - [80_per_seq_random_len_2       @ len-2/strd-same/49_85/detrac-non_empty](#80_per_seq_random_len_2___len_2_strd_same_49_85_detrac_non_empty_)
            - [len-4       @ strd-same/49_85/detrac-non_empty](#len_4___strd_same_49_85_detrac_non_empty_)
                - [80_per_seq_random_len_4       @ len-4/strd-same/49_85/detrac-non_empty](#80_per_seq_random_len_4___len_4_strd_same_49_85_detrac_non_empty_)
            - [len-8       @ strd-same/49_85/detrac-non_empty](#len_8___strd_same_49_85_detrac_non_empty_)
                - [80_per_seq_random_len_8       @ len-8/strd-same/49_85/detrac-non_empty](#80_per_seq_random_len_8___len_8_strd_same_49_85_detrac_non_empty_)
                - [200_per_seq_random_len_8       @ len-8/strd-same/49_85/detrac-non_empty](#200_per_seq_random_len_8___len_8_strd_same_49_85_detrac_non_empty_)
            - [len-12       @ strd-same/49_85/detrac-non_empty](#len_12___strd_same_49_85_detrac_non_empty_)
            - [len-16       @ strd-same/49_85/detrac-non_empty](#len_16___strd_same_49_85_detrac_non_empty_)
                - [256_per_seq_random_len_16       @ len-16/strd-same/49_85/detrac-non_empty](#256_per_seq_random_len_16___len_16_strd_same_49_85_detrac_non_empt_y_)
                - [320_per_seq_random_len_16       @ len-16/strd-same/49_85/detrac-non_empty](#320_per_seq_random_len_16___len_16_strd_same_49_85_detrac_non_empt_y_)
            - [len-32       @ strd-same/49_85/detrac-non_empty](#len_32___strd_same_49_85_detrac_non_empty_)
                - [512_per_seq_random_len_32       @ len-32/strd-same/49_85/detrac-non_empty](#512_per_seq_random_len_32___len_32_strd_same_49_85_detrac_non_empt_y_)
            - [len-40       @ strd-same/49_85/detrac-non_empty](#len_40___strd_same_49_85_detrac_non_empty_)
            - [len-48       @ strd-same/49_85/detrac-non_empty](#len_48___strd_same_49_85_detrac_non_empty_)
            - [len-56       @ strd-same/49_85/detrac-non_empty](#len_56___strd_same_49_85_detrac_non_empty_)
            - [len-64       @ strd-same/49_85/detrac-non_empty](#len_64___strd_same_49_85_detrac_non_empty_)
- [ipsc](#ips_c_)
    - [0_126       @ ipsc](#0_126___ipsc_)
        - [len-2       @ 0_126/ipsc](#len_2___0_126_ipsc_)
    - [0_1       @ ipsc](#0_1___ipsc_)
    - [0_4       @ ipsc](#0_4___ipsc_)
        - [12094       @ 0_4/ipsc](#12094___0_4_ipsc_)
    - [5_9       @ ipsc](#5_9___ipsc_)
    - [0_37       @ ipsc](#0_37___ipsc_)
        - [len-2       @ 0_37/ipsc](#len_2___0_37_ips_c_)
        - [len-3       @ 0_37/ipsc](#len_3___0_37_ips_c_)
    - [16_53       @ ipsc](#16_53___ipsc_)
        - [len-2       @ 16_53/ipsc](#len_2___16_53_ipsc_)
            - [gap       @ len-2/16_53/ipsc](#gap___len_2_16_53_ipsc_)
        - [len-3       @ 16_53/ipsc](#len_3___16_53_ipsc_)
        - [len-4       @ 16_53/ipsc](#len_4___16_53_ipsc_)
        - [len-6       @ 16_53/ipsc](#len_6___16_53_ipsc_)
        - [len-8       @ 16_53/ipsc](#len_8___16_53_ipsc_)
    - [0_15       @ ipsc](#0_15___ipsc_)
        - [len-2       @ 0_15/ipsc](#len_2___0_15_ips_c_)
        - [len-3       @ 0_15/ipsc](#len_3___0_15_ips_c_)
        - [len-4       @ 0_15/ipsc](#len_4___0_15_ips_c_)
        - [len-6       @ 0_15/ipsc](#len_6___0_15_ips_c_)
        - [len-8       @ 0_15/ipsc](#len_8___0_15_ips_c_)
    - [0_53       @ ipsc](#0_53___ipsc_)
        - [len-2       @ 0_53/ipsc](#len_2___0_53_ips_c_)
    - [54_126       @ ipsc](#54_126___ipsc_)
        - [len-2       @ 54_126/ipsc](#len_2___54_126_ips_c_)
            - [sample-8       @ len-2/54_126/ipsc](#sample_8___len_2_54_126_ips_c_)
        - [len-3       @ 54_126/ipsc](#len_3___54_126_ips_c_)
        - [len-4       @ 54_126/ipsc](#len_4___54_126_ips_c_)
        - [len-6       @ 54_126/ipsc](#len_6___54_126_ips_c_)
            - [strd-6       @ len-6/54_126/ipsc](#strd_6___len_6_54_126_ips_c_)
            - [sample-8       @ len-6/54_126/ipsc](#sample_8___len_6_54_126_ips_c_)
            - [sample-4       @ len-6/54_126/ipsc](#sample_4___len_6_54_126_ips_c_)
        - [len-8       @ 54_126/ipsc](#len_8___54_126_ips_c_)
- [acamp](#acamp_)
    - [1k8_vid_entire_seq       @ acamp](#1k8_vid_entire_seq___acam_p_)
        - [inv       @ 1k8_vid_entire_seq/acamp](#inv___1k8_vid_entire_seq_acamp_)
            - [2_per_seq       @ inv/1k8_vid_entire_seq/acamp](#2_per_seq___inv_1k8_vid_entire_seq_acamp_)
            - [6_per_seq       @ inv/1k8_vid_entire_seq/acamp](#6_per_seq___inv_1k8_vid_entire_seq_acamp_)
            - [12_per_seq       @ inv/1k8_vid_entire_seq/acamp](#12_per_seq___inv_1k8_vid_entire_seq_acamp_)
    - [10k6_vid_entire_seq       @ acamp](#10k6_vid_entire_seq___acam_p_)
            - [8_per_seq_random_len_2       @ 10k6_vid_entire_seq/acamp](#8_per_seq_random_len_2___10k6_vid_entire_seq_acam_p_)
    - [10k6_vid_entire_seq-inv       @ acamp](#10k6_vid_entire_seq_inv___acam_p_)
        - [2_per_seq       @ 10k6_vid_entire_seq-inv/acamp](#2_per_seq___10k6_vid_entire_seq_inv_acam_p_)
        - [12_per_seq       @ 10k6_vid_entire_seq-inv/acamp](#12_per_seq___10k6_vid_entire_seq_inv_acam_p_)
        - [8_per_seq_random_len_2       @ 10k6_vid_entire_seq-inv/acamp](#8_per_seq_random_len_2___10k6_vid_entire_seq_inv_acam_p_)
    - [20k6_5_video       @ acamp](#20k6_5_video___acam_p_)
        - [inv       @ 20k6_5_video/acamp](#inv___20k6_5_video_acamp_)
            - [2_per_seq       @ inv/20k6_5_video/acamp](#2_per_seq___inv_20k6_5_video_acamp_)
            - [12_per_seq       @ inv/20k6_5_video/acamp](#12_per_seq___inv_20k6_5_video_acamp_)
    - [2_per_seq_dbg_bear       @ acamp](#2_per_seq_dbg_bear___acam_p_)

<!-- /MarkdownTOC -->

<a id="imagenet_vi_d_"></a>
# imagenet_vid
<a id="train___imagenet_vid_"></a>
## train       @ imagenet_vid-->xml_to_ytvis
<a id="len_2___train_imagenet_vid_"></a>
### len-2       @ train/imagenet_vid-->xml_to_ytvis
python xml_to_ytvis.py cfg=imagenet_vid:proc-1:len-2:strd-1:gap-1:zip-0
<a id="4_per_seq_random_len_2___len_2_train_imagenet_vid_"></a>
#### 4_per_seq_random_len_2       @ len-2/train/imagenet_vid-->xml_to_ytvis
python xml_to_ytvis.py cfg=imagenet_vid:4_per_seq_random_len_2:proc-1:len-2:strd-2:gap-1:zip-0

<a id="val___imagenet_vid_"></a>
## val       @ imagenet_vid-->xml_to_ytvis
<a id="len_2___val_imagenet_vid_"></a>
### len-2       @ val/imagenet_vid-->xml_to_ytvis
python xml_to_ytvis.py cfg=imagenet_vid:val:proc-1:len-2:strd-1:gap-1:zip-0
python xml_to_ytvis.py cfg=imagenet_vid:val:proc-1:len-2:strd-2:gap-1:zip-0
<a id="8_per_seq_random_len_2___len_2_val_imagenet_vid_"></a>
#### 8_per_seq_random_len_2       @ len-2/val/imagenet_vid-->xml_to_ytvis
python xml_to_ytvis.py cfg=imagenet_vid:val:8_per_seq_random_len_2:proc-1:len-2:strd-2:gap-1:zip-0

<a id="gra_m_"></a>
# gram
<a id="0_1___gram_"></a>
## 0_1       @ gram-->xml_to_ytvis
<a id="len_2___0_1_gram_"></a>
### len-2       @ 0_1/gram-->xml_to_ytvis
python xml_to_ytvis.py cfg=gram:0_1:proc-1:len-2:strd-1:gap-1 
python xml_to_ytvis.py cfg=gram:0_1:proc-1:len-2:strd-2:gap-1 

<a id="len_9___0_1_gram_"></a>
### len-9       @ 0_1/gram-->xml_to_ytvis
python xml_to_ytvis.py cfg=gram:0_1:proc-1:len-9:strd-1:gap-1 
python xml_to_ytvis.py cfg=gram:0_1:proc-1:len-9:strd-9:gap-1 

<a id="len_14___0_1_gram_"></a>
### len-14       @ 0_1/gram-->xml_to_ytvis
<a id="0_2000___len_14_0_1_gra_m_"></a>
#### 0_2000       @ len-14/0_1/gram-->xml_to_ytvis
python xml_to_ytvis.py cfg=gram:0_1:frame-0_2000:proc-1:len-14:strd-1:gap-1 
<a id="3000_5000___len_14_0_1_gra_m_"></a>
#### 3000_5000       @ len-14/0_1/gram-->xml_to_ytvis
python xml_to_ytvis.py cfg=gram:0_1:frame-3000_5000:proc-1:len-14:strd-14:gap-1 

<a id="len_16___0_1_gram_"></a>
### len-16       @ 0_1/gram-->xml_to_ytvis
<a id="0_2000___len_16_0_1_gra_m_"></a>
#### 0_2000       @ len-16/0_1/gram-->xml_to_ytvis
python xml_to_ytvis.py cfg=gram:0_1:frame-0_2000:proc-1:len-16:strd-1:gap-1 
<a id="3000_5000___len_16_0_1_gra_m_"></a>
#### 3000_5000       @ len-16/0_1/gram-->xml_to_ytvis
python xml_to_ytvis.py cfg=gram:0_1:frame-3000_5000:proc-1:len-16:strd-16:gap-1 

<a id="ido_t_"></a>
# idot
<a id="0_1___idot_"></a>
## 0_1       @ idot-->xml_to_ytvis
python xml_to_ytvis.py cfg=idot:0_1:proc-1:len-9:strd-1:gap-1 

<a id="8_8___idot_"></a>
## 8_8       @ idot-->xml_to_ytvis
python xml_to_ytvis.py cfg=idot:8_8:proc-1:len-2:strd-10:gap-10 

python xml_to_ytvis.py cfg=idot:8_8:proc-1:len-3:strd-20:gap-10 

python xml_to_ytvis.py cfg=idot:8_8:proc-1:len-6:strd-50:gap-10 

<a id="detra_c_"></a>
# detrac
<a id="0_59___detrac_"></a>
## 0_59       @ detrac-->xml_to_ytvis
<a id="len_2___0_59_detra_c_"></a>
### len-2       @ 0_59/detrac-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:0_59:proc-1:len-2:strd-1:gap-1:ign
`dbg`
python xml_to_ytvis.py cfg=detrac:17_17:proc-1:len-2:strd-1:gap-1:ign
<a id="80_per_seq_random_len_2___len_2_0_59_detra_c_"></a>
#### 80_per_seq_random_len_2       @ len-2/0_59/detrac-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:80_per_seq_random_len_2:0_59:proc-1:len-2:strd-2:gap-1:ign


<a id="len_3___0_59_detra_c_"></a>
### len-3       @ 0_59/detrac-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:0_59:proc-1:len-3:strd-1:gap-1:ign
<a id="120_per_seq_random_len_3___len_3_0_59_detra_c_"></a>
#### 120_per_seq_random_len_3       @ len-3/0_59/detrac-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:120_per_seq_random_len_3:0_59:proc-1:len-3:strd-3:gap-1:ign

<a id="len_4___0_59_detra_c_"></a>
### len-4       @ 0_59/detrac-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:0_59:proc-1:len-4:strd-1:gap-1:ign
`dbg`
python xml_to_ytvis.py cfg=detrac:17_17:proc-1:len-4:strd-1:gap-1:ign
<a id="80_per_seq_random_len_4___len_4_0_59_detra_c_"></a>
#### 80_per_seq_random_len_4       @ len-4/0_59/detrac-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:80_per_seq_random_len_4:0_59:proc-1:len-4:strd-4:gap-1:ign

<a id="len_8___0_59_detra_c_"></a>
### len-8       @ 0_59/detrac-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:0_59:proc-1:len-8:strd-1:gap-1:ign

<a id="60_99___detrac_"></a>
## 60_99       @ detrac-->xml_to_ytvis
<a id="len_2___60_99_detrac_"></a>
### len-2       @ 60_99/detrac-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:60_99:proc-1:len-2:strds:gap-1:ign
python xml_to_ytvis.py cfg=detrac:60_99:proc-1:len-2:strd-2:gap-1:ign
<a id="80_per_seq_random_len_2___len_2_60_99_detrac_"></a>
#### 80_per_seq_random_len_2       @ len-2/60_99/detrac-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:80_per_seq_random_len_2:60_99:proc-1:len-2:strd-2:gap-1:ign

<a id="len_3___60_99_detrac_"></a>
### len-3       @ 60_99/detrac-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:60_99:proc-1:len-3:strds:gap-1:ign
<a id="120_per_seq_random_len_3___len_3_60_99_detrac_"></a>
#### 120_per_seq_random_len_3       @ len-3/60_99/detrac-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:120_per_seq_random_len_3:60_99:proc-1:len-3:strd-3:gap-1:ign

<a id="len_4___60_99_detrac_"></a>
### len-4       @ 60_99/detrac-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:60_99:proc-1:len-4:strds:gap-1:ign
<a id="80_per_seq_random_len_4___len_4_60_99_detrac_"></a>
#### 80_per_seq_random_len_4       @ len-4/60_99/detrac-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:80_per_seq_random_len_4:60_99:proc-1:len-4:strd-4:gap-1:ign

<a id="detrac_non_empt_y_"></a>
# detrac-non_empty
<a id="0_0___detrac_non_empty_"></a>
## 0_0       @ detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:0_0:proc-1:len-100:strd-100:gap-1:vis
python xml_to_ytvis.py cfg=detrac:non_empty:0_0:proc-1:len-2:strd-1:gap-1 
<a id="0_1___detrac_non_empty_"></a>
## 0_1       @ detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:0_1:proc-1:len-2:strd-1:gap-1 
python xml_to_ytvis.py cfg=detrac:non_empty:0_1:proc-1:len-64:strd-1:gap-1 
python xml_to_ytvis.py cfg=detrac:non_empty:0_1:proc-1:len-64:strd-64:gap-1 


<a id="0_19___detrac_non_empty_"></a>
## 0_19       @ detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:0_19:proc-1:len-2:strd-1:gap-1 
python xml_to_ytvis.py cfg=detrac:non_empty:0_19:proc-1:len-3:strd-1:gap-1 
python xml_to_ytvis.py cfg=detrac:non_empty:0_19:proc-1:len-4:strd-1:gap-1 
python xml_to_ytvis.py cfg=detrac:non_empty:0_19:proc-1:len-6:strd-1:gap-1 
python xml_to_ytvis.py cfg=detrac:non_empty:0_19:proc-1:len-8:strd-1:gap-1 
python xml_to_ytvis.py cfg=detrac:non_empty:0_19:proc-1:len-9:strd-1:gap-1 
python xml_to_ytvis.py cfg=detrac:non_empty:0_19:proc-1:len-64:strd-1:gap-1 
<a id="strd_2___0_19_detrac_non_empt_y_"></a>
### strd-2       @ 0_19/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:0_19:proc-1:len-2:strd-2:gap-1 

<a id="0_9___detrac_non_empty_"></a>
## 0_9       @ detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:0_9:proc-1:len-2:strd-1:gap-1:vis 
`strd-2`
python xml_to_ytvis.py cfg=detrac:non_empty:0_9:proc-1:len-2:strd-2:gap-1 

<a id="49_68___detrac_non_empty_"></a>
## 49_68       @ detrac-non_empty-->xml_to_ytvis
<a id="len_2___49_68_detrac_non_empty_"></a>
### len-2       @ 49_68/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:49_68:proc-12:len-2:strds:gap-1
`strd-12 frame-0_360`
python xml_to_ytvis.py cfg=detrac:non_empty:49_68:proc-12:len-3:strd-12:gap-1:frame-0_360
`strd-2`
python xml_to_ytvis.py cfg=detrac:non_empty:49_68:proc-12:len-2:strd-2:gap-1
<a id="len_6___49_68_detrac_non_empty_"></a>
### len-6       @ 49_68/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:49_68:proc-12:len-6:strds:gap-1
<a id="len_9___49_68_detrac_non_empty_"></a>
### len-9       @ 49_68/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:49_68:proc-12:len-9:strds:gap-1
<a id="len_64___49_68_detrac_non_empty_"></a>
### len-64       @ 49_68/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:49_68:proc-1:len-64:strds:gap-1
python xml_to_ytvis.py cfg=detrac:non_empty:49_68:proc-1:len-64:strd-64:gap-1

<a id="0_48___detrac_non_empty_"></a>
## 0_48       @ detrac-non_empty-->xml_to_ytvis
<a id="len_2___0_48_detrac_non_empt_y_"></a>
### len-2       @ 0_48/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:0_48:proc-1:len-2:strd-1:gap-1
<a id="len_4___0_48_detrac_non_empt_y_"></a>
### len-4       @ 0_48/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:0_48:proc-1:len-4:strd-1:gap-1
<a id="len_8___0_48_detrac_non_empt_y_"></a>
### len-8       @ 0_48/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:0_48:proc-1:len-8:strd-1:gap-1
python xml_to_ytvis.py cfg=detrac:non_empty:0_48:proc-1:len-8:strd-8:gap-1
<a id="200_per_seq_random_len_8___len_8_0_48_detrac_non_empt_y_"></a>
#### 200_per_seq_random_len_8       @ len-8/0_48/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:200_per_seq_random_len_8:0_48:proc-1:len-8:strd-8:gap-1
<a id="len_12___0_48_detrac_non_empt_y_"></a>
### len-12       @ 0_48/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:0_48:proc-1:len-12:strd-1:gap-1
<a id="len_16___0_48_detrac_non_empt_y_"></a>
### len-16       @ 0_48/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:0_48:proc-1:len-16:strd-1:gap-1
<a id="len_24___0_48_detrac_non_empt_y_"></a>
### len-24       @ 0_48/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:0_48:proc-1:len-24:strd-1:gap-1
<a id="len_28___0_48_detrac_non_empt_y_"></a>
### len-28       @ 0_48/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:0_48:proc-1:len-28:strd-1:gap-1
<a id="len_32___0_48_detrac_non_empt_y_"></a>
### len-32       @ 0_48/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:0_48:proc-1:len-32:strd-1:gap-1
<a id="len_40___0_48_detrac_non_empt_y_"></a>
### len-40       @ 0_48/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:0_48:proc-1:len-40:strd-1:gap-1
<a id="len_48___0_48_detrac_non_empt_y_"></a>
### len-48       @ 0_48/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:0_48:proc-1:len-48:strd-1:gap-1
<a id="len_56___0_48_detrac_non_empt_y_"></a>
### len-56       @ 0_48/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:0_48:proc-1:len-56:strd-1:gap-1
<a id="len_64___0_48_detrac_non_empt_y_"></a>
### len-64       @ 0_48/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:0_48:proc-1:len-64:strd-1:gap-1

<a id="0_85___detrac_non_empty_"></a>
## 0_85       @ detrac-non_empty-->xml_to_ytvis
<a id="len_2___0_85_detrac_non_empt_y_"></a>
### len-2       @ 0_85/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:0_85:proc-1:len-2:strd-1:gap-1
python xml_to_ytvis.py cfg=detrac:non_empty:0_85:proc-1:len-2:strd-2:gap-1
<a id="len_32___0_85_detrac_non_empt_y_"></a>
### len-32       @ 0_85/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:0_85:proc-1:len-32:strd-1:gap-1
<a id="len_40___0_85_detrac_non_empt_y_"></a>
### len-40       @ 0_85/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:0_85:proc-1:len-40:strd-1:gap-1
<a id="len_48___0_85_detrac_non_empt_y_"></a>
### len-48       @ 0_85/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:0_85:proc-1:len-48:strd-1:gap-1
<a id="len_56___0_85_detrac_non_empt_y_"></a>
### len-56       @ 0_85/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:0_85:proc-1:len-56:strd-1:gap-1
<a id="len_64___0_85_detrac_non_empt_y_"></a>
### len-64       @ 0_85/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:0_85:proc-1:len-64:strd-1:gap-1


<a id="49_85___detrac_non_empty_"></a>
## 49_85       @ detrac-non_empty-->xml_to_ytvis
<a id="strd_1___49_85_detrac_non_empty_"></a>
### strd-1       @ 49_85/detrac-non_empty-->xml_to_ytvis
<a id="len_2___strd_1_49_85_detrac_non_empt_y_"></a>
#### len-2       @ strd-1/49_85/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:49_85:proc-1:len-2:strd-1:gap-1
<a id="len_4___strd_1_49_85_detrac_non_empt_y_"></a>
#### len-4       @ strd-1/49_85/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:49_85:proc-1:len-4:strd-1:gap-1
<a id="len_8___strd_1_49_85_detrac_non_empt_y_"></a>
#### len-8       @ strd-1/49_85/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:49_85:proc-1:len-8:strd-1:gap-1
<a id="len_16___strd_1_49_85_detrac_non_empt_y_"></a>
#### len-16       @ strd-1/49_85/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:49_85:proc-1:len-16:strd-1:gap-1
<a id="strds___len_16_strd_1_49_85_detrac_non_empty_"></a>
##### strds       @ len-16/strd-1/49_85/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:49_85:proc-1:len-16:strds-16:gap-1
<a id="len_32___strd_1_49_85_detrac_non_empt_y_"></a>
#### len-32       @ strd-1/49_85/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:49_85:proc-1:len-32:strd-1:gap-1
<a id="strds___len_32_strd_1_49_85_detrac_non_empty_"></a>
##### strds       @ len-32/strd-1/49_85/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:49_85:proc-1:len-32:strds-32:gap-1
<a id="len_40___strd_1_49_85_detrac_non_empt_y_"></a>
#### len-40       @ strd-1/49_85/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:49_85:proc-1:len-40:strd-1:gap-1
<a id="len_48___strd_1_49_85_detrac_non_empt_y_"></a>
#### len-48       @ strd-1/49_85/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:49_85:proc-1:len-48:strd-1:gap-1
<a id="len_56___strd_1_49_85_detrac_non_empt_y_"></a>
#### len-56       @ strd-1/49_85/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:49_85:proc-1:len-56:strd-1:gap-1
<a id="len_64___strd_1_49_85_detrac_non_empt_y_"></a>
#### len-64       @ strd-1/49_85/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:49_85:proc-1:len-64:strd-1:gap-1

<a id="strd_same___49_85_detrac_non_empty_"></a>
### strd-same       @ 49_85/detrac-non_empty-->xml_to_ytvis
<a id="len_2___strd_same_49_85_detrac_non_empty_"></a>
#### len-2       @ strd-same/49_85/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:49_85:proc-1:len-2:strd-2:gap-1
<a id="80_per_seq_random_len_2___len_2_strd_same_49_85_detrac_non_empty_"></a>
##### 80_per_seq_random_len_2       @ len-2/strd-same/49_85/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:80_per_seq_random_len_2:49_85:proc-1:len-2:strd-2:gap-1


<a id="len_4___strd_same_49_85_detrac_non_empty_"></a>
#### len-4       @ strd-same/49_85/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:49_85:proc-1:len-4:strd-4:gap-1
<a id="80_per_seq_random_len_4___len_4_strd_same_49_85_detrac_non_empty_"></a>
##### 80_per_seq_random_len_4       @ len-4/strd-same/49_85/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:80_per_seq_random_len_4:49_85:proc-1:len-4:strd-4:gap-1

<a id="len_8___strd_same_49_85_detrac_non_empty_"></a>
#### len-8       @ strd-same/49_85/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:49_85:proc-1:len-8:strd-8:gap-1
<a id="80_per_seq_random_len_8___len_8_strd_same_49_85_detrac_non_empty_"></a>
##### 80_per_seq_random_len_8       @ len-8/strd-same/49_85/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:80_per_seq_random_len_8:49_85:proc-1:len-8:strd-8:gap-1
<a id="200_per_seq_random_len_8___len_8_strd_same_49_85_detrac_non_empty_"></a>
##### 200_per_seq_random_len_8       @ len-8/strd-same/49_85/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:200_per_seq_random_len_8:49_85:proc-1:len-8:strd-8:gap-1

<a id="len_12___strd_same_49_85_detrac_non_empty_"></a>
#### len-12       @ strd-same/49_85/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:49_85:proc-1:len-12:strd-12:gap-1

<a id="len_16___strd_same_49_85_detrac_non_empty_"></a>
#### len-16       @ strd-same/49_85/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:49_85:proc-1:len-16:strd-16:gap-1
<a id="256_per_seq_random_len_16___len_16_strd_same_49_85_detrac_non_empt_y_"></a>
##### 256_per_seq_random_len_16       @ len-16/strd-same/49_85/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:256_per_seq_random_len_16:49_85:proc-1:len-16:strd-16:gap-1
`dbg`
python xml_to_ytvis.py cfg=detrac:non_empty:256_per_seq_random_len_16:49_50:proc-1:len-16:strd-16:gap-1
<a id="320_per_seq_random_len_16___len_16_strd_same_49_85_detrac_non_empt_y_"></a>
##### 320_per_seq_random_len_16       @ len-16/strd-same/49_85/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:320_per_seq_random_len_16:49_85:proc-1:len-16:strd-16:gap-1

<a id="len_32___strd_same_49_85_detrac_non_empty_"></a>
#### len-32       @ strd-same/49_85/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:49_85:proc-1:len-32:strd-32:gap-1
<a id="512_per_seq_random_len_32___len_32_strd_same_49_85_detrac_non_empt_y_"></a>
##### 512_per_seq_random_len_32       @ len-32/strd-same/49_85/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:512_per_seq_random_len_32:49_85:proc-1:len-32:strd-32:gap-1


<a id="len_40___strd_same_49_85_detrac_non_empty_"></a>
#### len-40       @ strd-same/49_85/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:49_85:proc-1:len-40:strd-40:gap-1
<a id="len_48___strd_same_49_85_detrac_non_empty_"></a>
#### len-48       @ strd-same/49_85/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:49_85:proc-1:len-48:strd-48:gap-1
<a id="len_56___strd_same_49_85_detrac_non_empty_"></a>
#### len-56       @ strd-same/49_85/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:49_85:proc-1:len-56:strd-56:gap-1
<a id="len_64___strd_same_49_85_detrac_non_empty_"></a>
#### len-64       @ strd-same/49_85/detrac-non_empty-->xml_to_ytvis
python xml_to_ytvis.py cfg=detrac:non_empty:49_85:proc-1:len-64:strd-64:gap-1


<a id="ips_c_"></a>
# ipsc

<a id="0_126___ipsc_"></a>
## 0_126       @ ipsc-->xml_to_ytvis
<a id="len_2___0_126_ipsc_"></a>
### len-2       @ 0_126/ipsc-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=ipsc:0_126:len-2:strds:gap-1:mask-0:proc-1 

<a id="0_1___ipsc_"></a>
## 0_1       @ ipsc-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=ipsc:mask-0:frame-0_1:len-2:gap-1:strds

<a id="0_4___ipsc_"></a>
## 0_4       @ ipsc-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=ipsc:mask-0:frame-0_4:len-2:gap-1:strds 
python3 xml_to_ytvis.py cfg=ipsc:mask-0:frame-0_4:len-2:gap-3:strds 

<a id="12094___0_4_ipsc_"></a>
### 12094       @ 0_4/ipsc-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=ipsc:12094:2_class:mask-0:proc-12:start-0:end-4:len-2:gap-1 
python3 xml_to_ytvis.py cfg=ipsc:12094_short:2_class:mask-0:proc-12:start-0:end-4:len-2:gap-1 

<a id="5_9___ipsc_"></a>
## 5_9       @ ipsc-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=ipsc:mask-0:proc-12:start-5:end-9:len-2:gap-1 
python3 xml_to_ytvis.py cfg=ipsc:mask-0:proc-1:start-5:end-9:len-2:gap-2 
python3 xml_to_ytvis.py cfg=ipsc:mask-0:proc-1:start-5:end-9:len-2:gap-3 
python3 xml_to_ytvis.py cfg=ipsc:mask-0:proc-1:start-5:end-9:len-2:gap-4 

<a id="0_37___ipsc_"></a>
## 0_37       @ ipsc-->xml_to_ytvis
<a id="len_2___0_37_ips_c_"></a>
### len-2       @ 0_37/ipsc-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=ipsc:0_37:len-2:gap-1:mask-0:proc-1:zip-0 
<a id="len_3___0_37_ips_c_"></a>
### len-3       @ 0_37/ipsc-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=ipsc:0_37:len-3:gap-1:mask-0:proc-1:zip-0:strds 

<a id="16_53___ipsc_"></a>
## 16_53       @ ipsc-->xml_to_ytvis
<a id="len_2___16_53_ipsc_"></a>
### len-2       @ 16_53/ipsc-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=ipsc:16_53:len-2:gap-1:mask-0:proc-1:zip-0 
python3 xml_to_ytvis.py cfg=ipsc:16_53:len-2:strd-2:gap-1:mask-0:proc-1:zip-0 
<a id="gap___len_2_16_53_ipsc_"></a>
#### gap       @ len-2/16_53/ipsc-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=ipsc:16_53:len-2:gap-2:mask-0:proc-1:zip-0 
python3 xml_to_ytvis.py cfg=ipsc:16_53:len-2:gap-3:mask-0:proc-1:zip-0 
python3 xml_to_ytvis.py cfg=ipsc:16_53:len-2:gap-4:mask-0:proc-1:zip-0 
<a id="len_3___16_53_ipsc_"></a>
### len-3       @ 16_53/ipsc-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=ipsc:16_53:len-3:gap-1:mask-0:proc-1:zip-0 
python3 xml_to_ytvis.py cfg=ipsc:16_53:len-3:strd-3-1:gap-1:mask-0:proc-1:zip-0 
<a id="len_4___16_53_ipsc_"></a>
### len-4       @ 16_53/ipsc-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=ipsc:16_53:len-4:gap-1:mask-0:proc-1:zip-0:strds
<a id="len_6___16_53_ipsc_"></a>
### len-6       @ 16_53/ipsc-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=ipsc:16_53:len-6:gap-1:mask-0:proc-1:zip-0 
python3 xml_to_ytvis.py cfg=ipsc:16_53:len-6:strds:gap-1:mask-0:proc-1:zip-0 
<a id="len_8___16_53_ipsc_"></a>
### len-8       @ 16_53/ipsc-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=ipsc:16_53:len-8:gap-1:mask-0:proc-1:zip-0 
python3 xml_to_ytvis.py cfg=ipsc:16_53:len-8:strds:gap-1:mask-0:proc-1:zip-0 


<a id="0_15___ipsc_"></a>
## 0_15       @ ipsc-->xml_to_ytvis
<a id="len_2___0_15_ips_c_"></a>
### len-2       @ 0_15/ipsc-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=ipsc:0_15:len-2:strds:gap-1:mask-0:proc-1 
<a id="len_3___0_15_ips_c_"></a>
### len-3       @ 0_15/ipsc-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=ipsc:0_15:len-3:strds:gap-1:mask-0:proc-1 
<a id="len_4___0_15_ips_c_"></a>
### len-4       @ 0_15/ipsc-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=ipsc:0_15:len-4:strds:gap-1:mask-0:proc-1 
<a id="len_6___0_15_ips_c_"></a>
### len-6       @ 0_15/ipsc-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=ipsc:0_15:len-6:strds:gap-1:mask-0:proc-1 
<a id="len_8___0_15_ips_c_"></a>
### len-8       @ 0_15/ipsc-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=ipsc:0_15:len-8:strds:gap-1:mask-0:proc-1 

<a id="0_53___ipsc_"></a>
## 0_53       @ ipsc-->xml_to_ytvis
<a id="len_2___0_53_ips_c_"></a>
### len-2       @ 0_53/ipsc-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=ipsc:0_53:len-2:strds:gap-1:mask-0:proc-1 

<a id="54_126___ipsc_"></a>
## 54_126       @ ipsc-->xml_to_ytvis
<a id="len_2___54_126_ips_c_"></a>
### len-2       @ 54_126/ipsc-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=ipsc:54_126:len-2:strd-1:gap-1:mask-0:proc-1:zip-0 
python3 xml_to_ytvis.py cfg=ipsc:54_126:len-2:strd-2:gap-1:mask-0:proc-1:zip-0 

<a id="sample_8___len_2_54_126_ips_c_"></a>
#### sample-8       @ len-2/54_126/ipsc-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=ipsc:54_126:len-2:strd-2:gap-1:mask-0:proc-1:zip-0:sample-8 

<a id="len_3___54_126_ips_c_"></a>
### len-3       @ 54_126/ipsc-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=ipsc:54_126:len-3:strds:gap-1:mask-0:proc-1:zip-0 

python3 xml_to_ytvis.py cfg=ipsc:54_126:len-3:strd-3:gap-1:mask-0:proc-1:zip-0 

<a id="len_4___54_126_ips_c_"></a>
### len-4       @ 54_126/ipsc-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=ipsc:54_126:len-4:strds:gap-1:mask-0:proc-1:zip-0 

<a id="len_6___54_126_ips_c_"></a>
### len-6       @ 54_126/ipsc-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=ipsc:54_126:len-6:strds:gap-1:mask-0:proc-1:zip-0 
`dbg`
python3 xml_to_ytvis.py cfg=ipsc:frame-54_65:seq-0:len-6:strds:gap-1:mask-0:proc-1:zip-0 
<a id="strd_6___len_6_54_126_ips_c_"></a>
#### strd-6       @ len-6/54_126/ipsc-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=ipsc:54_126:len-6:strds-6:gap-1:mask-0:proc-1:zip-0 
<a id="sample_8___len_6_54_126_ips_c_"></a>
#### sample-8       @ len-6/54_126/ipsc-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=ipsc:54_126:len-6:strd-6:gap-1:mask-0:proc-1:zip-0:sample-8 
<a id="sample_4___len_6_54_126_ips_c_"></a>
#### sample-4       @ len-6/54_126/ipsc-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=ipsc:54_126:len-6:strd-6:gap-1:mask-0:proc-1:zip-0:sample-4 

<a id="len_8___54_126_ips_c_"></a>
### len-8       @ 54_126/ipsc-->xml_to_ytvis
python3 xml_to_ytvis.py cfg=ipsc:54_126:len-8:strds:gap-1:mask-0:proc-1:zip-0 

<a id="acamp_"></a>
# acamp
<a id="1k8_vid_entire_seq___acam_p_"></a>
## 1k8_vid_entire_seq       @ acamp-->xml_to_ytvis
python xml_to_ytvis.py cfg=acamp:1k8_vid_entire_seq:proc-0:len-2:strds:gap-1
<a id="inv___1k8_vid_entire_seq_acamp_"></a>
### inv       @ 1k8_vid_entire_seq/acamp-->xml_to_ytvis
python xml_to_ytvis.py cfg=acamp:1k8_vid_entire_seq_inv:proc-0:len-2:strds:gap-1
<a id="2_per_seq___inv_1k8_vid_entire_seq_acamp_"></a>
#### 2_per_seq       @ inv/1k8_vid_entire_seq/acamp-->xml_to_ytvis
python xml_to_ytvis.py cfg=acamp:1k8_vid_entire_seq_inv_2_per_seq:proc-0:len-2:strds:gap-1
'Z:\UofA\Acamp\acamp_code\tf_api\cmd\csv_to_record.md'

<a id="6_per_seq___inv_1k8_vid_entire_seq_acamp_"></a>
#### 6_per_seq       @ inv/1k8_vid_entire_seq/acamp-->xml_to_ytvis
python xml_to_ytvis.py cfg=acamp:1k8_vid_entire_seq_inv_6_per_seq:proc-0:len-2:strds:gap-1
'Z:\UofA\Acamp\acamp_code\tf_api\cmd\csv_to_record.md'

<a id="12_per_seq___inv_1k8_vid_entire_seq_acamp_"></a>
#### 12_per_seq       @ inv/1k8_vid_entire_seq/acamp-->xml_to_ytvis
python xml_to_ytvis.py cfg=acamp:1k8_vid_entire_seq_inv_12_per_seq:proc-0:len-2:strds:gap-1
'Z:\UofA\Acamp\acamp_code\tf_api\cmd\csv_to_record.md'


<a id="10k6_vid_entire_seq___acam_p_"></a>
## 10k6_vid_entire_seq       @ acamp-->xml_to_ytvis
python xml_to_ytvis.py cfg=acamp:10k6_vid_entire_seq:proc-0:len-2:strds:gap-1
<a id="8_per_seq_random_len_2___10k6_vid_entire_seq_acam_p_"></a>
#### 8_per_seq_random_len_2       @ 10k6_vid_entire_seq/acamp-->xml_to_ytvis
python xml_to_ytvis.py cfg=acamp:10k6_vid_entire_seq_8_per_seq_random_len_2:proc-0:len-2:strds:gap-1


<a id="10k6_vid_entire_seq_inv___acam_p_"></a>
## 10k6_vid_entire_seq-inv       @ acamp-->xml_to_ytvis
python xml_to_ytvis.py cfg=acamp:10k6_vid_entire_seq_inv:proc-0:len-2:strds:gap-1
<a id="2_per_seq___10k6_vid_entire_seq_inv_acam_p_"></a>
### 2_per_seq       @ 10k6_vid_entire_seq-inv/acamp-->xml_to_ytvis
python xml_to_ytvis.py cfg=acamp:10k6_vid_entire_seq_inv_2_per_seq:proc-0:len-2:strds:gap-1
'Z:\UofA\Acamp\acamp_code\tf_api\cmd\csv_to_record.md'

<a id="12_per_seq___10k6_vid_entire_seq_inv_acam_p_"></a>
### 12_per_seq       @ 10k6_vid_entire_seq-inv/acamp-->xml_to_ytvis
python xml_to_ytvis.py cfg=acamp:10k6_vid_entire_seq_inv_12_per_seq:proc-0:len-2:strds:gap-1

<a id="8_per_seq_random_len_2___10k6_vid_entire_seq_inv_acam_p_"></a>
### 8_per_seq_random_len_2       @ 10k6_vid_entire_seq-inv/acamp-->xml_to_ytvis
python xml_to_ytvis.py cfg=acamp:10k6_vid_entire_seq_inv_8_per_seq_random_len_2:proc-0:len-2:strds:gap-1


<a id="20k6_5_video___acam_p_"></a>
## 20k6_5_video       @ acamp-->xml_to_ytvis
python xml_to_ytvis.py cfg=acamp:20k6_5_video:proc-0:len-2:strd-1:gap-1
python xml_to_ytvis.py cfg=acamp:20k6_5_video:proc-0:len-2:strd-2:gap-1
<a id="inv___20k6_5_video_acamp_"></a>
### inv       @ 20k6_5_video/acamp-->xml_to_ytvis
python xml_to_ytvis.py cfg=acamp:20k6_5_video_inv:proc-0:len-2:strds:gap-1
<a id="2_per_seq___inv_20k6_5_video_acamp_"></a>
#### 2_per_seq       @ inv/20k6_5_video/acamp-->xml_to_ytvis
python xml_to_ytvis.py cfg=acamp:20k6_5_video_inv_2_per_seq:proc-0:len-2:strds:gap-1
'Z:\UofA\Acamp\acamp_code\tf_api\cmd\csv_to_record.md'

<a id="12_per_seq___inv_20k6_5_video_acamp_"></a>
#### 12_per_seq       @ inv/20k6_5_video/acamp-->xml_to_ytvis
python xml_to_ytvis.py cfg=acamp:20k6_5_video_inv_12_per_seq:proc-0:len-2:strds:gap-1
'Z:\UofA\Acamp\acamp_code\tf_api\cmd\csv_to_record.md'


<a id="2_per_seq_dbg_bear___acam_p_"></a>
## 2_per_seq_dbg_bear       @ acamp-->xml_to_ytvis
python xml_to_ytvis.py cfg=acamp:2_per_seq_dbg_bear:proc-0:len-2:strd-1:gap-1
