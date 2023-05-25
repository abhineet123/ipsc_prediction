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

- [ROI images and labels](https://drive.google.com/file/d/18NCCFAVKFlB7DCfa8Cpo92Sd4v7U6FB7)    
- [Raw 714 MP images](https://drive.google.com/file/d/1WmtyCWeeryxlWP6W8vcF0WmlfSdUroAg)
- [List TXT files](https://drive.google.com/file/d/1a0gVn63TbX2nUWhQJdXvOMzWA2H1Abe1)
- [Static segmentation JSON files](https://drive.google.com/file/d/17bXxZ9Z7Yydt4m2NnXYxS80c6gfUtWyh)
- [Video segmentation JSON files](https://drive.google.com/file/d/1ne2225Rdz0Y75wonmfMlzxv_rhfeRSuu)

ROI images and labels, list TXT files and static segmentation JSON files should be extracted to ```/data/ipsc/well3/all_frames_roi/``` while video segmentation JSON files should be extracted to ```/data/ipsc/well3/all_frames_roi/ytvis19/```.

<a id="trained_models_"></a>
# Trained Models
Trained models can be downloaded from [here](https://drive.google.com/drive/folders/1AHD7I8qHtg9hXqwfEgpNKw0QAG3j_2ae)    
- both early-stage and late-stage trained models are included    
- the zip file for any model should be extracted in its source directory while maintaining the folder structure inside the zip file    
    + for example, the models for IDOL should be extracted inside [    ipsc_video_segmentation/ipsc_vnext](ipsc_video_segmentation/ipsc_vnext)    
    + this would extract the ```.pth``` checkpoint files into a subfolder named ```log/idol-ipsc-ext_reorg_roi_g2_16_53``` and ```log/idol-ipsc-ext_reorg_roi_g2_54_126``` for the early and late-stage models respectively   
     
<a id="supplementary_material_"></a>
# Supplementary material
The supplementary PDF is available [here](https://drive.google.com/file/d/1YNm8N2B-0Cpu5y_MwUEz4iy3QNJAk897)

<a id="visualization_s_"></a>
## Visualizations
Videos visualizing the annotations along with detection results and failures are available [here](https://drive.google.com/drive/folders/1L1NXhSQvLpRSN4WBmbiv9lZYYP3zYgVS).
Detailed description of these videos is in the supplementary PDF.

<a id="3d_plot_s_"></a>
## 3D Plots
Frame-wise partial AUC plots are available [here](https://drive.google.com/drive/folders/1SyEjF9IV8AHnYgxM4kW2ACYvRo1Lp0PE) as interactive HTML files.








