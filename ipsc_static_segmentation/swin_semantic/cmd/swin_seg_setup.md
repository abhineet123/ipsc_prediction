
<!-- MarkdownTOC -->

- [virtualenv](#virtualen_v_)
    - [ubuntu22.04/py3.10       @ virtualenv](#ubuntu22_04_py3_10___virtualenv_)
        - [python3.8       @ ubuntu22.04/py3.10/virtualenv](#python3_8___ubuntu22_04_py3_10_virtualen_v_)
        - [python3.7       @ ubuntu22.04/py3.10/virtualenv](#python3_7___ubuntu22_04_py3_10_virtualen_v_)
    - [windows       @ virtualenv](#windows___virtualenv_)
    - [cuda_version       @ virtualenv](#cuda_version___virtualenv_)
- [install](#install_)
    - [pytorch       @ install](#pytorch___instal_l_)
        - [latest       @ pytorch/install](#latest___pytorch_instal_l_)
        - [py3.10       @ pytorch/install](#py3_10___pytorch_instal_l_)
        - [from_readme/1.8.0       @ pytorch/install](#from_readme_1_8_0___pytorch_instal_l_)
            - [compat_list       @ from_readme/1.8.0/pytorch/install](#compat_list___from_readme_1_8_0_pytorch_instal_l_)
            - [windows       @ from_readme/1.8.0/pytorch/install](#windows___from_readme_1_8_0_pytorch_instal_l_)
    - [mmcv       @ install](#mmcv___instal_l_)
        - [py3.10       @ mmcv/install](#py3_10___mmcv_install_)
        - [latest       @ mmcv/install](#latest___mmcv_install_)
        - [1.8.0       @ mmcv/install](#1_8_0___mmcv_install_)
            - [manual       @ 1.8.0/mmcv/install](#manual___1_8_0_mmcv_install_)
            - [windows       @ 1.8.0/mmcv/install](#windows___1_8_0_mmcv_install_)
            - [from_readme       @ 1.8.0/mmcv/install](#from_readme___1_8_0_mmcv_install_)
    - [mmsegmentation       @ install](#mmsegmentation___instal_l_)
    - [rest       @ install](#rest___instal_l_)
- [bugs](#bug_s_)
    - [enable_tensorboard       @ bugs](#enable_tensorboard___bugs_)
- [convert_datasets](#convert_dataset_s_)
    - [CHASEDB1       @ convert_datasets](#chasedb1___convert_datasets_)
    - [pascal_context       @ convert_datasets](#pascal_context___convert_datasets_)
- [from_mojow_rocks       @ convert_datasets](#from_mojow_rocks___convert_datasets_)
    - [db3-part1       @ from_mojow_rocks](#db3_part1___from_mojow_rocks_)
    - [db3-part10       @ from_mojow_rocks](#db3_part10___from_mojow_rocks_)
    - [syn-part4_on_part5_on_september_5_2020_2K_100       @ from_mojow_rocks](#syn_part4_on_part5_on_september_5_2020_2k_100___from_mojow_rocks_)
    - [syn-part2_on_part8       @ from_mojow_rocks](#syn_part2_on_part8___from_mojow_rocks_)
- [scp](#scp_)

<!-- /MarkdownTOC -->

__only works with python 3.8__

<a id="virtualen_v_"></a>
# virtualenv
sudo pip3 install virtualenv virtualenvwrapper
nano ~/.bashrc

export PATH=$PATH:/usr/local/cuda/bin

export WORKON_HOME=$HOME/.virtualenvs  
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python  
source /usr/local/bin/virtualenvwrapper.sh  

source ~/.bashrc

mkvirtualenv swin_s
workon swin_s

alias sws='workon swin_s'

<a id="ubuntu22_04_py3_10___virtualenv_"></a>
## ubuntu22.04/py3.10       @ virtualenv-->swin_seg_setup
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

<a id="python3_8___ubuntu22_04_py3_10_virtualen_v_"></a>
### python3.8       @ ubuntu22.04/py3.10/virtualenv-->swin_seg_setup
sudo apt install python3.8
sudo apt-get install python3.8-dev
sudo apt-get install python3.8-distutils
sudo apt-get install python3.8-apt

wget https://bootstrap.pypa.io/get-pip.py
python3.8 get-pip.py
python3.8 -m pip install virtualenv virtualenvwrapper
mkvirtualenv -p python3.8 swin_s
pip install --upgrade pip

<a id="python3_7___ubuntu22_04_py3_10_virtualen_v_"></a>
### python3.7       @ ubuntu22.04/py3.10/virtualenv-->swin_seg_setup
sudo apt install python3.7
sudo apt-get install python3.7-distutils
sudo apt-get install python3.7-dev

wget https://bootstrap.pypa.io/get-pip.py
python3.7 get-pip.py
python3.7 -m pip install virtualenv virtualenvwrapper
mkvirtualenv -p python3.7 swin_s
pip install --upgrade pip

<a id="windows___virtualenv_"></a>
## windows       @ virtualenv-->swin_seg_setup
virtualenv swin_s
cd swin_s/Scripts
activate

<a id="cuda_version___virtualenv_"></a>
## cuda_version       @ virtualenv-->swin_seg_setup
nvcc --version
/usr/local/cuda/bin/nvcc --version

<a id="install_"></a>
# install

<a id="pytorch___instal_l_"></a>
## pytorch       @ install-->swin_seg_setup
__only 1.8.0 works__

python -m pip uninstall -y torch torchvision torchaudio 

<a id="latest___pytorch_instal_l_"></a>
### latest       @ pytorch/install-->swin_seg_setup
python -m pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio===0.10.2+cu113 -f https://download.pytorch.org/whl/torch_stable.html

<a id="py3_10___pytorch_instal_l_"></a>
### py3.10       @ pytorch/install-->swin_seg_setup
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html

<a id="from_readme_1_8_0___pytorch_instal_l_"></a>
### from_readme/1.8.0       @ pytorch/install-->swin_seg_setup
python -m pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

<a id="compat_list___from_readme_1_8_0_pytorch_instal_l_"></a>
#### compat_list       @ from_readme/1.8.0/pytorch/install-->swin_seg_setup
https://pypi.org/project/torchvision/
https://github.com/pytorch/audio

<a id="windows___from_readme_1_8_0_pytorch_instal_l_"></a>
#### windows       @ from_readme/1.8.0/pytorch/install-->swin_seg_setup
python -m pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

<a id="mmcv___instal_l_"></a>
## mmcv       @ install-->swin_seg_setup
python -m pip uninstall -y mmcv-full

<a id="py3_10___mmcv_install_"></a>
### py3.10       @ mmcv/install-->swin_seg_setup
python -m pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html

<a id="latest___mmcv_install_"></a>
### latest       @ mmcv/install-->swin_seg_setup
python -m pip install mmcv-full==1.4.2 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html

<a id="1_8_0___mmcv_install_"></a>
### 1.8.0       @ mmcv/install-->swin_seg_setup
<a id="manual___1_8_0_mmcv_install_"></a>
#### manual       @ 1.8.0/mmcv/install-->swin_seg_setup
wget https://download.openmmlab.com/mmcv/dist/1.3.5/torch1.8.0/cu111/mmcv_full-latest%2Btorch1.8.0%2Bcu111-cp38-cp38-manylinux1_x86_64.whl
python -m pip install mmcv_full-latest+torch1.8.0+cu111-cp38-cp38-manylinux1_x86_64.whl

<a id="windows___1_8_0_mmcv_install_"></a>
#### windows       @ 1.8.0/mmcv/install-->swin_seg_setup
https://download.openmmlab.com/mmcv/dist/1.1.5/torch1.6.0/cu102/mmcv_full-1.1.5%2Btorch1.6.0%2Bcu102-cp37-cp37m-win_amd64.whl
python -m pip install mmcv_full-1.1.5+torch1.6.0+cu102-cp37-cp37m-win_amd64.whl

<a id="from_readme___1_8_0_mmcv_install_"></a>
#### from_readme       @ 1.8.0/mmcv/install-->swin_seg_setup
__does not work___
python -m pip install mmcv-full==latest+torch1.8.0+cu111 -f https://download.openmmlab.com/mmcv/dist/index.html

<a id="mmsegmentation___instal_l_"></a>
## mmsegmentation       @ install-->swin_seg_setup
__do not install mmsegmentation__

nano ~/.bashrc
export PYTHONPATH=$PYTHONPATH:/home/abhineet/ipsc_segmentation/swin_semantic
. ~/.bashrc

python -m pip install mmsegmentation
python -m pip uninstall mmsegmentation

<a id="rest___instal_l_"></a>
## rest       @ install-->swin_seg_setup
python -m pip install matplotlib numpy terminaltables timm
python -m pip install tensorflow tensorboard
python -m pip install pycocotools
python -m pip install imagesize pandas
python -m pip install setuptools==59.5.0


<a id="bug_s_"></a>
# bugs
`RuntimeError: Default process group has not been initialized, please make sure to call init_process_group.`

add
```
parser.add_argument('--init',
                    default='file:///tmp/file',
                    type=str,
                    help='the dir to save logs and models')
```
and
``` 
import torch.distributed as dist
dist.init_process_group('gloo',
                        init_method=args.init,
                        rank=0, world_size=1
                        )
```
in train.py

`AttributeError: module 'distutils' has no attribute 'version'`
python -m pip install setuptools==59.5.0

`ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 512, 1, 1])`
use batch size > 1

`TypeError: EncoderDecoder: SwinTransformer: __init__() got an unexpected keyword argument 'embed_dim'`

__do not install mmsegmentation__
add existing mmseg to pythonpath instead

embed_dim --> embed_dims in config
configs/_base_/models/upernet_swin.py
https://github.com/open-mmlab/mmsegmentation/issues/752

`runtimeerror: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.cuda.floattensor [2, 512, 30, 30]], which is output 0 of relubackward0, is at version 1; expected version 0 instead. hint: enable anomaly detection to find the operation that failed to compute its gradient, with torch.autograd.set_detect_anomaly(true).`
use pytorch 1.8.0 with corresponding mmcv


<a id="enable_tensorboard___bugs_"></a>
## enable_tensorboard       @ bugs-->swin_seg_setup
`configs/_base_/default_runtime.py`
comment out:
`dict(type='TensorboardLoggerHook')`


<a id="convert_dataset_s_"></a>
# convert_datasets

<a id="chasedb1___convert_datasets_"></a>
## CHASEDB1       @ convert_datasets-->swin_seg_setup
python tools/convert_datasets/chase_db1.py /data/CHASEDB1.zip

<a id="pascal_context___convert_datasets_"></a>
## pascal_context       @ convert_datasets-->swin_seg_setup
cd ~
git clone https://github.com/zhanghang1989/detail-api
cd detail-api/PythonAPI
sws
python -m pip install cython
make
make install

python -m pip install scikit-image pandas paramparse imagesize

python tools/convert_datasets/pascal_context.py /data/VOCdevkit /data/VOCdevkit/VOC2010/trainval_merged.json

<a id="from_mojow_rocks___convert_datasets_"></a>
# from_mojow_rocks       @ convert_datasets-->swin_seg

<a id="db3_part1___from_mojow_rocks_"></a>
## db3-part1       @ from_mojow_rocks-->swin_seg_setup
python tools/convert_datasets/from_mojow_rocks.py root_dir=/data/mojow_rock/rock_dataset3 csv_paths=part1 sizes=large,huge img_list_name=part1.txt  write_masks=0 show=0 class_info=lists/classes_rock.txt

<a id="db3_part10___from_mojow_rocks_"></a>
## db3-part10       @ from_mojow_rocks-->swin_seg_setup
python tools/convert_datasets/from_mojow_rocks.py root_dir=/data/mojow_rock/rock_dataset3 csv_paths=part10 sizes=large,huge img_list_name=part10.txt  write_masks=1 show=0 class_info=lists/classes_rock.txt

<a id="syn_part4_on_part5_on_september_5_2020_2k_100___from_mojow_rocks_"></a>
## syn-part4_on_part5_on_september_5_2020_2K_100       @ from_mojow_rocks-->swin_seg_setup
python tools/convert_datasets/from_mojow_rocks.py root_dir=/data/mojow_rock/rock_dataset3 csv_paths=syn/part4_on_part5_on_september_5_2020_2K_100 sizes=large,huge class_info=lists/classes_rock_with_syn.txt

<a id="syn_part2_on_part8___from_mojow_rocks_"></a>
## syn-part2_on_part8       @ from_mojow_rocks-->swin_seg_setup
python tools/convert_datasets/from_mojow_rocks.py root_dir=/data/mojow_rock/rock_dataset3 csv_paths=syn/part2_on_part8 sizes=large,huge class_info=lists/classes_rock_with_syn.txt

<a id="scp_"></a>
# scp      
 scp -r -P 9738 ~/pretrained_upernet_swin_base_patch4_window7_512x512_pth_mj_221206_235737.zip abhineet@greyshark.cs.ualberta.ca:~/





















