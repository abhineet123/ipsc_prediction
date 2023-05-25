This repository contains the code for the paper: [Towards Early Prediction of Human iPSC Reprogramming Success](https://arxiv.org/abs/2305.14575)

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

The Swin transformer semantic segmentation model not reported in the paper is available in [ipsc_static_segmentation/swin_semantic](ipsc_static_segmentation/swin_semantic)

Each of the above folders have a subfolder named ```cmd``` containing markdown files with hierarchically organized list of commands to reproduce any of the results reported in the paper.
- for example, the commands for **SWD** are in [swin_det.md](ipsc_static_segmentation/swin_detection/cmd/swin_det.md) and [swin_det_setup.md](ipsc_static_segmentation/swin_detection/cmd/swin_det_setup.md) in [ipsc_static_segmentation/swin_detection/cmd](ipsc_static_segmentation/swin_detection/cmd)

Data can be downloaded from here:    
- [ROI images and labels](https://drive.google.com/file/d/18NCCFAVKFlB7DCfa8Cpo92Sd4v7U6FB7/view?usp=sharing)    
- [Raw 714 MP images](https://drive.google.com/file/d/1WmtyCWeeryxlWP6W8vcF0WmlfSdUroAg/view?usp=share_link)

Trained models can be downloaded from [here](https://drive.google.com/drive/folders/1AHD7I8qHtg9hXqwfEgpNKw0QAG3j_2ae?usp=share_link)    
- both early-stage and late-stage trained models are available here    
- the zip file for any model should be extracted in its source directory while maintaining the folder structure inside the zip file    
    + for example, the models for IDOL should be extracted inside [    ipsc_video_segmentation/ipsc_vnext](ipsc_video_segmentation/ipsc_vnext)    
    + this would extract the ```.pth``` checkpoint files into a subfolder named ```log/idol-ipsc-ext_reorg_roi_g2_16_53``` and ```log/idol-ipsc-ext_reorg_roi_g2_54_126``` for the early and late-stage models respectively    

If the commands needed to reproduce any results in the paper or to run any of the models on a new dataset are not clear, please create an [issue](https://github.com/abhineet123/ipsc_prediction/issues) or contact [me](http://webdocs.cs.ualberta.ca/~asingh1/).



