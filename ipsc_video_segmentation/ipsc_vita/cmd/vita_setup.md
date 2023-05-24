<!-- MarkdownTOC -->

- [virtualenv](#virtualen_v_)
    - [grs/python3.6       @ virtualenv](#grs_python3_6___virtualenv_)
        - [3.9       @ grs/python3.6/virtualenv](#3_9___grs_python3_6_virtualenv_)
        - [3.7       @ grs/python3.6/virtualenv](#3_7___grs_python3_6_virtualenv_)
    - [general       @ virtualenv](#general___virtualenv_)
    - [windows       @ virtualenv](#windows___virtualenv_)
    - [cc       @ virtualenv](#cc___virtualenv_)
    - [cuda_version       @ virtualenv](#cuda_version___virtualenv_)
- [install](#install_)
    - [pytorch       @ install](#pytorch___instal_l_)
        - [ubuntu22.04/python_3.10       @ pytorch/install](#ubuntu22_04_python_3_10___pytorch_instal_l_)
    - [detectron2       @ install](#detectron2___instal_l_)
    - [grs/python3.6       @ install](#grs_python3_6___instal_l_)
        - [windows       @ grs/python3.6/install](#windows___grs_python3_6_instal_l_)
    - [opencv       @ install](#opencv___instal_l_)
        - [3.4.11.45       @ opencv/install](#3_4_11_45___opencv_install_)
        - [4.5.4.60       @ opencv/install](#4_5_4_60___opencv_install_)
    - [requirements       @ install](#requirements___instal_l_)
    - [cuda_operators       @ install](#cuda_operators___instal_l_)
    - [cc       @ install](#cc___instal_l_)
- [bugs](#bug_s_)
    - [windows       @ bugs](#windows___bugs_)
- [new_dataset](#new_dataset_)
    - [ytvis19       @ new_dataset](#ytvis19___new_datase_t_)
- [scp](#scp_)
    - [grs       @ scp](#grs___sc_p_)
        - [reorg_roi_annotated_221204_123655.zip       @ grs/scp](#reorg_roi_annotated_221204_123655_zip___grs_sc_p_)
    - [nrw       @ scp](#nrw___sc_p_)
        - [vita-ipsc-ext_reorg_roi_g2_16_53_swin       @ nrw/scp](#vita_ipsc_ext_reorg_roi_g2_16_53_swin___nrw_sc_p_)
    - [from-nrw       @ scp](#from_nrw___sc_p_)
    - [from-grs       @ scp](#from_grs___sc_p_)
        - [vita-ipsc-ext_reorg_roi_g2_16_53_swin_retrain       @ from-grs/scp](#vita_ipsc_ext_reorg_roi_g2_16_53_swin_retrain___from_grs_scp_)
        - [vita-ipsc-ext_reorg_roi_g2_54_126_swin       @ from-grs/scp](#vita_ipsc_ext_reorg_roi_g2_54_126_swin___from_grs_scp_)

<!-- /MarkdownTOC -->

<a id="virtualen_v_"></a>
# virtualenv
<a id="grs_python3_6___virtualenv_"></a>
## grs/python3.6       @ virtualenv-->vita_setup

<a id="3_9___grs_python3_6_virtualenv_"></a>
### 3.9       @ grs/python3.6/virtualenv-->vita_setup
apt-get install python3.9-dev
apt-get install python3.9-tk

wget https://bootstrap.pypa.io/get-pip.py
python3.9 get-pip.py
sudo apt-get install python3.9-distutils

python3.9 -m pip install setuptools==59.5.0
python3.9 -m pip install virtualenv virtualenvwrapper
mkvirtualenv -p python3.9 vita 

<a id="3_7___grs_python3_6_virtualenv_"></a>
### 3.7       @ grs/python3.6/virtualenv-->vita_setup
apt-get install python3.7-dev
apt-get install python3.7-tk

wget https://bootstrap.pypa.io/get-pip.py
python3.7 get-pip.py

python3.7 -m pip install setuptools==59.5.0
python3.7 -m pip install virtualenv virtualenvwrapper
mkvirtualenv -p python3.7 vita 

<a id="general___virtualenv_"></a>
## general       @ virtualenv-->vita_setup
python3 -m pip install virtualenv virtualenvwrapper

nano ~/.bashrc

export PATH=$PATH:/usr/local/cuda/bin

export WORKON_HOME=$HOME/.virtualenvs  
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3  
source /usr/local/bin/virtualenvwrapper.sh  

source ~/.bashrc

mkvirtualenv vita
workon vita

alias vita='workon venv_vita'

<a id="windows___virtualenv_"></a>
## windows       @ virtualenv-->vita_setup
virtualenv venv_vita
venv_vita\Scripts\activate

<a id="cc___virtualenv_"></a>
## cc       @ virtualenv-->vita_setup
module load python/3.8
module load gcc cuda cudnn
virtualenv ~/venv/vita
source ~/venv/vita/bin/activate
deactivate

alias vita='source ~/venv/vita/bin/activate'

diskusage_report

<a id="cuda_version___virtualenv_"></a>
## cuda_version       @ virtualenv-->vita_setup
nvcc --version
/usr/local/cuda/bin/nvcc --version

<a id="install_"></a>
# install
<a id="pytorch___instal_l_"></a>
## pytorch       @ install-->vita_setup
python -m pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio===0.10.2+cu113 -f https://download.pytorch.org/whl/torch_stable.html

<a id="ubuntu22_04_python_3_10___pytorch_instal_l_"></a>
### ubuntu22.04/python_3.10       @ pytorch/install-->vita_setup
python -m pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

<a id="detectron2___instal_l_"></a>
## detectron2       @ install-->vita_setup
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2

<a id="grs_python3_6___instal_l_"></a>
## grs/python3.6       @ install-->vita_setup
nano detectron2/setup.py
`python_requires=">=3.6"`

__does not work__
git clone -b legacy_py3.6 https://github.com/QUVA-Lab/e2cnn.git
cd e2cnn
python setup.py install

<a id="windows___grs_python3_6_instal_l_"></a>
### windows       @ grs/python3.6/install-->vita_setup
change in `detectron2\layers\csrc\nms_rotated\nms_rotated_cuda.cu`
```
/*
#ifdef WITH_CUDA
#include "../box_iou_rotated/box_iou_rotated_utils.h"
#endif
// TODO avoid this when pytorch supports "same directory" hipification
#ifdef WITH_HIP
/\#include "box_iou_rotated/box_iou_rotated_utils.h"
#endif
*/
#include "box_iou_rotated/box_iou_rotated_utils.h"
```
<a id="opencv___instal_l_"></a>
## opencv       @ install-->vita_setup
python -m pip install opencv-python 

<a id="3_4_11_45___opencv_install_"></a>
### 3.4.11.45       @ opencv/install-->vita_setup
python -m pip install opencv-python==3.4.11.45 opencv-contrib-python==3.4.11.45

<a id="4_5_4_60___opencv_install_"></a>
### 4.5.4.60       @ opencv/install-->vita_setup
python36 -m pip install opencv-python==4.5.4.60 opencv-contrib-python==4.5.4.60

<a id="requirements___instal_l_"></a>
## requirements       @ install-->vita_setup
python -m pip install -r requirements.txt

<a id="cuda_operators___instal_l_"></a>
## cuda_operators       @ install-->vita_setup
cd mask2former/modeling/pixel_decoder/ops
python setup.py build install
cd -

<a id="cc___instal_l_"></a>
## cc       @ install-->vita_setup
salloc --nodes=1 --time=0:05:0 --account=def-nilanjan --gpus-per-node=1 --mem=4000M --cpus-per-task=1
module load cuda cudnn gcc python/3.8
source ~/venv/vita/bin/activate
cd mask2former/modeling/pixel_decoder/ops
python3 setup.py build install

<a id="bug_s_"></a>
# bugs
<a id="windows___bugs_"></a>
## windows       @ bugs-->vita_setup
`RuntimeError: Distributed package doesn't have NCCL built in`
change `backend="NCCL"` to `backend="GLOO"` in 
"C:\UofA\PhD\ipsc_cell_tracking\ipsc_vita\detectron2\detectron2\engine\launch.py"

<a id="new_dataset_"></a>
# new_dataset
<a id="ytvis19___new_datase_t_"></a>
## ytvis19       @ new_dataset-->vita_setup
"C:\UofA\PhD\ipsc_cell_tracking\ipsc_vnext\projects\IDOL\idol\data\datasets\builtin.py"
"C:\UofA\PhD\ipsc_cell_tracking\ipsc_vnext\projects\SeqFormer\seqformer\data\datasets\builtin.py"
`_PREDEFINED_SPLITS_YTVIS_2019`
`register_all_ytvis_2019`

<a id="scp_"></a>
# scp

<a id="grs___sc_p_"></a>
## grs       @ scp-->vita_setup
scp -r -P 9738 abhineet@greyshark.cs.ualberta.ca:/home/abhineet/ipsc_vita/pretrained ./
scp -r -P 9738 abhineet@greyshark.cs.ualberta.ca:/home/abhineet/ipsc_vita/pretrained/vita_r101_coco.pth ./

scp -r -P 9738 abhineet@greyshark.cs.ualberta.ca:~/ipsc_vita/pretrained/*.pth ./
scp -r -P 9738 abhineet@greyshark.cs.ualberta.ca:~/ipsc_vita/vita_swin_ytvis2019.pth ./

<a id="reorg_roi_annotated_221204_123655_zip___grs_sc_p_"></a>
### reorg_roi_annotated_221204_123655.zip       @ grs/scp-->vita_setup
scp -r -P 9738 abhineet@greyshark.cs.ualberta.ca:/data/ipsc/well3/all_frames_roi/reorg_roi_annotated_221204_123655.zip ./

scp -r -P 9738 abhineet@greyshark.cs.ualberta.ca:/data/ipsc/well3/all_frames_roi/roi_17861_11316_19661_12616_221205_112359.zip ./

scp -r -P 9738 abhineet@greyshark.cs.ualberta.ca:/data/ipsc/well3/all_frames_roi/ext_reorg_roi_xml_annotations_221205_091419.zip ./

scp -r -P 9738 abhineet@greyshark.cs.ualberta.ca:/data/ipsc/well3/all_frames_roi/ext_reorg_roi_xml_annotations_221205_091419.zip ./

scp -r -P 9738 abhineet@greyshark.cs.ualberta.ca:/data/ipsc/well3/all_frames_roi/ytvis19/ipsc-ext_reorg_roi_g2_0_38.json ./
scp -r -P 9738 abhineet@greyshark.cs.ualberta.ca:/data/ipsc/well3/all_frames_roi/ytvis19/ipsc-ext_reorg_roi_g2_0_38-max_length-10.json ./
scp -r -P 9738 abhineet@greyshark.cs.ualberta.ca:/data/ipsc/well3/all_frames_roi/ytvis19/ipsc-ext_reorg_roi_g2_0_38-max_length-20.json ./
scp -r -P 9738 abhineet@greyshark.cs.ualberta.ca:/data/ipsc/well3/all_frames_roi/ytvis19/ipsc-ext_reorg_roi_g2_39_53.json ./

scp -r -P 9738 abhineet@greyshark.cs.ualberta.ca:/data/ipsc/well3/all_frames_roi/ytvis19/ipsc-ext_reorg_roi_g2_39_53.json ./

scp -r -P 9738 abhineet@greyshark.cs.ualberta.ca:~/ipsc-ext_reorg_roi_g2_0_53_json_grs_230306_212523.zip ./
scp -r -P 9738 abhineet@greyshark.cs.ualberta.ca:~/ipsc-ext_reorg_roi_g2_0_53-incremental_json_grs_230306_212406.zip ./


scp -r -P 9738 abhineet@greyshark.cs.ualberta.ca:~/data_ipsc_well3_all_frames_roi_ytvis19_ipsc-ext_reorg_roi_g2_0_53-max_length-19_json_grs_230420_111810.zip ./
scp -r -P 9738 abhineet@greyshark.cs.ualberta.ca:~/data_ipsc_well3_all_frames_roi_ytvis19_ipsc-ext_reorg_roi_g2_0_53_json_grs_230420_113836.zip ./


<a id="nrw___sc_p_"></a>
## nrw       @ scp-->vita_setup
scp -r asingh1@narval.computecanada.ca:~/scratch/vita_log/vita-ipsc-all_frames_roi_g2_0_38_swin/model_0059999.pth ./

scp -r asingh1@narval.computecanada.ca:~/scratch/vita_log/vita-ipsc-all_frames_roi_g2_0_38_swin/model_0059999.pth ./

<a id="vita_ipsc_ext_reorg_roi_g2_16_53_swin___nrw_sc_p_"></a>
### vita-ipsc-ext_reorg_roi_g2_16_53_swin       @ nrw/scp-->vita_setup
<a id="from_nrw___sc_p_"></a>
## from-nrw       @ scp-->vita_setup
scp -r asingh1@narval.computecanada.ca:/home/asingh1/vita/log/vita-ipsc-ext_reorg_roi_g2_16_53_swin/model_0124999.pth ./

scp -r asingh1@narval.computecanada.ca:/home/asingh1/vita/log/vita-ipsc-ext_reorg_roi_g2_16_53_swin/model_0119999.pth ./

scp -r asingh1@narval.computecanada.ca:/home/asingh1/vita/log/vita-ipsc-ext_reorg_roi_g2_16_53_swin/model_0329999.pth ./

scp -r asingh1@narval.computecanada.ca:/home/asingh1/vita/log/vita-ipsc-ext_reorg_roi_g2_16_53_swin/events.out.** ./

<a id="from_grs___sc_p_"></a>
## from-grs       @ scp-->vita_setup
scp -r -P 9738 abhineet@greyshark.cs.ualberta.ca:~/vita/log/vita-ipsc-ext_reorg_roi_g2_16_53_swin/*.pth ./

<a id="vita_ipsc_ext_reorg_roi_g2_16_53_swin_retrain___from_grs_scp_"></a>
### vita-ipsc-ext_reorg_roi_g2_16_53_swin_retrain       @ from-grs/scp-->vita_setup
scp -r asingh1@narval.computecanada.ca:/home/asingh1/vita/log/vita-ipsc-ext_reorg_roi_g2_16_53_swin_retrain/model_0104999.pth ./

scp -r asingh1@narval.computecanada.ca:/home/asingh1/vita/log/vita-ipsc-ext_reorg_roi_g2_16_53_swin_retrain/model_0079999.pth ./

scp -r asingh1@narval.computecanada.ca:/home/asingh1/vita/log/vita-ipsc-ext_reorg_roi_g2_16_53_swin_retrain/model_0004999.pth ./

scp -r asingh1@narval.computecanada.ca:/home/asingh1/vita/log/vita-ipsc-ext_reorg_roi_g2_16_53_swin_retrain/events.out.** ./


<a id="vita_ipsc_ext_reorg_roi_g2_54_126_swin___from_grs_scp_"></a>
### vita-ipsc-ext_reorg_roi_g2_54_126_swin       @ from-grs/scp-->vita_setup
scp -r asingh1@narval.computecanada.ca:/home/asingh1/vita/log/vita-ipsc-ext_reorg_roi_g2_54_126_swin/model_0194999.pth ./

scp -r asingh1@narval.computecanada.ca:/home/asingh1/vita/log/vita-ipsc-ext_reorg_roi_g2_54_126_swin/events.out.** ./




















