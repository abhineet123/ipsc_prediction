<!-- No Heading Fix -->

This repository contains the code for the paper: [Towards Early Prediction of Human iPSC Reprogramming Success](https://arxiv.org/abs/2305.14575).

<!-- MarkdownTOC -->

- [Code](#cod_e_)
    - [Models](#model_s_)
    - [Annotation](#annotatio_n_)
    - [Data processing and Evaluation](#data_processing_and_evaluation_)
- [Setup and Commands](#setup_and_commands_)
- [Data](#dat_a_)
- [Trained Models](#trained_models_)
    - [classification](#classificatio_n_)
        - [XGB](#xgb_)
        - [SWC](#swc_)
        - [CNC](#cnc_)
    - [static segmentation](#static_segmentation_)
        - [SWD](#swd_)
        - [CND](#cnd_)
    - [video segmentation](#video_segmentation_)
        - [IDOL](#ido_l_)
        - [SEQ](#seq_)
        - [VITA](#vit_a_)
- [Supplementary material](#supplementary_material_)
    - [Visualizations](#visualization_s_)
    - [3D Plots](#3d_plot_s_)

<!-- /MarkdownTOC -->


<a id="cod_e_"></a>
# Code
<a id="model_s_"></a>
## Models
The 8 models reported in the paper are located as follows:    
- classification    
    + **XGB**: [ipsc_data_processing/eval_cls.py](ipsc_data_processing/eval_cls.py)    
    + **SWC**: [ipsc_classification/swin](ipsc_classification/swin)    
    + **CNC**: [ipsc_classification/convnext](ipsc_classification/convnext)    
- static segmentation    
    + **SWD**: [ipsc_static_segmentation/swin_detection](ipsc_static_segmentation/swin_detection)    
    + **CND**: [ipsc_static_segmentation/swin_detection](ipsc_static_segmentation/swin_detection)    
- video segmentation    
    + **IDOL**: [ipsc_video_segmentation/ipsc_vnext](ipsc_video_segmentation/ipsc_vnext)    
    + **SEQ**: [ipsc_video_segmentation/ipsc_vnext](ipsc_video_segmentation/ipsc_vnext)    
    + **VITA**: [ipsc_video_segmentation/ipsc_vita](ipsc_video_segmentation/ipsc_vita)    

The Swin transformer semantic segmentation model not reported in the paper is available in [ipsc_static_segmentation/swin_semantic](ipsc_static_segmentation/swin_semantic).

<a id="annotatio_n_"></a>
## Annotation
-  labeling tool used for stages 1 and 2 of the annotation process can be run using [ipsc_labelling_tool/labelImg.py](ipsc_labelling_tool/labelImg.py)
-  retrospective labelling for stage 3 can be run using [ipsc_labelling_tool/propagate_by_tracking.py](ipsc_labelling_tool/propagate_by_tracking.py)

<a id="data_processing_and_evaluation_"></a>
## Data processing and Evaluation
- scripts for converting annotations between various formats like XML, CSV and JSON are available in [ipsc_data_processing/](ipsc_data_processing/) 
- classifiers can be evaluated using [ipsc_data_processing/eval_cls.py](ipsc_data_processing/eval_cls.py) 
- detectors can be evaluated using [ipsc_data_processing/eval_det.py](ipsc_data_processing/eval_det.py) 
    + this is also the script used for generating the detection failure visualization videos

<a id="setup_and_commands_"></a>
# Setup and Commands
Each of the above folders contains a subfolder named ```cmd``` containing markdown files with hierarchically organized list of commands to reproduce any of the results reported in the paper.
- for example, the commands for **SWD** are in [swin_det.md](ipsc_static_segmentation/swin_detection/cmd/swin_det.md) and [swin_det_setup.md](ipsc_static_segmentation/swin_detection/cmd/swin_det_setup.md) in [ipsc_static_segmentation/swin_detection/cmd](ipsc_static_segmentation/swin_detection/cmd) while those for **IDOL** and **SEQ** are in [vnext.md](ipsc_video_segmentation/ipsc_vnext/cmd/vnext.md) and [vnext_setup.md](ipsc_video_segmentation/ipsc_vnext/cmd/vnext_setup.md) in [ipsc_video_segmentation/swin_detection/cmd](ipsc_video_segmentation/ipsc_vnext/cmd)
- some of the contents of these files might not be easy to understand at present but I am working to make these more user-friendly
- if the commands needed to reproduce any results in the paper or to run any of the models on a new dataset are not clear, please create an [issue](https://github.com/abhineet123/ipsc_prediction/issues) or contact [me](http://webdocs.cs.ualberta.ca/~asingh1/)

<a id="dat_a_"></a>
# Data
Images and annotations can be downloaded from here:  

- [ROI images and labels](https://huggingface.co/abhineet123/ipsc_prediction/blob/main/ipsc_data/roi_images_and_labels.zip)    
- [Raw 714 MP images](https://huggingface.co/abhineet123/ipsc_prediction/blob/main/ipsc_data/raw_images.zip)
- [List TXT files](https://huggingface.co/abhineet123/ipsc_prediction/blob/main/ipsc_data/ipsc_list_txt.zip)
- [Static segmentation JSON files](https://huggingface.co/abhineet123/ipsc_prediction/blob/main/ipsc_data/ipsc_static_json.zip)
- [Video segmentation JSON files](https://huggingface.co/abhineet123/ipsc_prediction/blob/main/ipsc_data/ipsc_video_json.zip)

ROI images and labels archive should be extracted to `/data` while maintaining the folder structure therein, so that the ROI sequences end up at ```/data/ipsc/well3/all_frames_roi/```.

List TXT files and static segmentation JSON files should be extracted to ```/data/ipsc/well3/all_frames_roi/``` while video segmentation JSON files should be extracted to ```/data/ipsc/well3/all_frames_roi/ytvis19/```.

<a id="trained_models_"></a>
# Trained Models
Trained models are available in [this hugging face repo](https://huggingface.co/abhineet123/ipsc_prediction)    
- both [early-stage](https://huggingface.co/abhineet123/ipsc_prediction/tree/main/early_stage) and [late-stage](https://huggingface.co/abhineet123/ipsc_prediction/tree/main/late_stage) trained models are included    
- the zip file for any model should be extracted in its source directory while maintaining the folder structure inside the zip file    
    + for example, the models for IDOL should be extracted inside [    ipsc_video_segmentation/ipsc_vnext](ipsc_video_segmentation/ipsc_vnext)    
    + this would extract the ```.pth``` checkpoint files into a subfolder named ```log/idol-ipsc-ext_reorg_roi_g2_16_53``` and ```log/idol-ipsc-ext_reorg_roi_g2_54_126``` for the [early-stage](https://huggingface.co/abhineet123/ipsc_prediction/blob/main/early_stage/log_idol-ipsc-ext_reorg_roi_g2_16_53_model_0254999_pth.zip) and [late-stage](https://huggingface.co/abhineet123/ipsc_prediction/blob/main/late_stage/log_idol-ipsc-ext_reorg_roi_g2_54_126_model_0596999_pth.zip) models respectively   
<a id="classificatio_n_"></a>
## classification 
<a id="xgb_"></a>
### XGB 
- [early stage training](https://huggingface.co/abhineet123/ipsc_prediction/blob/main/early_stage/log_xgb_ext_reorg_roi_g2_16_53_xgb_trained_model.zip)
- [late stage training](https://huggingface.co/abhineet123/ipsc_prediction/blob/main/late_stage/log_xgb_ext_reorg_roi_g2_54_126_xgb_trained_model.zip) 
   
<a id="swc_"></a>
### SWC 
- [early stage training](https://huggingface.co/abhineet123/ipsc_prediction/blob/main/early_stage/log_ext_reorg_roi_g2_16_53-v1-base-224-1k_ckpt_epoch_00999_pth.zip)
- [late stage training](https://huggingface.co/abhineet123/ipsc_prediction/blob/main/late_stage/log_ext_reorg_roi_g2_54_126-v1-base-224-1k_ckpt_epoch_09999_pth.zip) 

<a id="cnc_"></a>
### CNC
- [early stage training](https://huggingface.co/abhineet123/ipsc_prediction/blob/main/early_stage/log_ext_reorg_roi_g2_16_53-large-384-22k_checkpoint-247_pth.zip)
- [late stage training](https://huggingface.co/abhineet123/ipsc_prediction/blob/main/late_stage/log_ext_reorg_roi_g2_54_126-large-384-22k_checkpoint-116_pth.zip) 

<a id="static_segmentation_"></a>
## static segmentation  
<a id="swd_"></a>
### SWD
- [early stage training](https://huggingface.co/abhineet123/ipsc_prediction/blob/main/early_stage/work_dirs_ipsc_2_class_ext_reorg_roi_g2_16_53-no_validate_epoch_3273_pth.zip)
- [late stage training](https://huggingface.co/abhineet123/ipsc_prediction/blob/main/late_stage/work_dirs_ipsc_2_class_ext_reorg_roi_g2_54_126-no_validate_epoch_2058_pth.zip)   

<a id="cnd_"></a>
### CND
- [early stage training](https://huggingface.co/abhineet123/ipsc_prediction/blob/main/early_stage/work_dirs_ipsc_2_class_ext_reorg_roi_g2_16_53-convnext_large_in22k_epoch_1014_pth.zip)
- [late stage training](https://huggingface.co/abhineet123/ipsc_prediction/blob/main/late_stage/work_dirs_ipsc_2_class_ext_reorg_roi_g2_54_126-convnext_large_in22k_epoch_106_pth.zip)   
  
<a id="video_segmentation_"></a>
## video segmentation   
<a id="ido_l_"></a>
### IDOL
- [early stage training](https://huggingface.co/abhineet123/ipsc_prediction/blob/main/early_stage/log_idol-ipsc-ext_reorg_roi_g2_16_53_model_0254999_pth.zip)
- [late stage training](https://huggingface.co/abhineet123/ipsc_prediction/blob/main/late_stage/log_idol-ipsc-ext_reorg_roi_g2_54_126_model_0596999_pth.zip)   

<a id="seq_"></a>
### SEQ
- [early stage training](https://huggingface.co/abhineet123/ipsc_prediction/blob/main/early_stage/log_seqformer-ipsc-ext_reorg_roi_g2_16_53_model_0241999_pth.zip)
- [late stage training](https://huggingface.co/abhineet123/ipsc_prediction/blob/main/late_stage/log_seqformer-ipsc-ext_reorg_roi_g2_54_126_model_0495999_pth.zip)  

<a id="vit_a_"></a>
### VITA
- [early stage training](https://huggingface.co/abhineet123/ipsc_prediction/blob/main/early_stage/log_vita-ipsc-ext_reorg_roi_g2_16_53_swin_model_0329999_pth.zip)
- [late stage training](https://huggingface.co/abhineet123/ipsc_prediction/blob/main/late_stage/log_vita-ipsc-ext_reorg_roi_g2_54_126_swin_model_0194999_pth.zip)   
     
<a id="supplementary_material_"></a>
# Supplementary material
The supplementary PDF is available [here](docs/ipsc_supplementary.pdf)

<a id="visualization_s_"></a>
## Visualizations
Videos visualizing the annotations along with detection results and failures are available [here](https://huggingface.co/abhineet123/ipsc_prediction/tree/main/supplementary/visualizations).
Detailed description of these videos is in the supplementary PDF.

<a id="3d_plot_s_"></a>
## 3D Plots
Frame-wise partial AUC plots are available [here](https://huggingface.co/abhineet123/ipsc_prediction/tree/main/supplementary/3d_plots) as interactive HTML files.








