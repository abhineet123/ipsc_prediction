<!-- MarkdownTOC -->

- [virtualenv](#virtualen_v_)
    - [cc       @ virtualenv](#cc___virtualenv_)
    - [windows       @ virtualenv](#windows___virtualenv_)
    - [cuda_version       @ virtualenv](#cuda_version___virtualenv_)
- [install](#install_)
    - [pytorch       @ install](#pytorch___instal_l_)
        - [python_3.10       @ pytorch/install](#python_3_10___pytorch_instal_l_)
    - [requirements       @ install](#requirements___instal_l_)
    - [detectron2       @ install](#detectron2___instal_l_)
        - [windows       @ detectron2/install](#windows___detectron2_install_)
    - [misc       @ install](#misc___instal_l_)
        - [geos_c       @ misc/install](#geos_c___misc_install_)
    - [cocoapi       @ install](#cocoapi___instal_l_)
        - [windows       @ cocoapi/install](#windows___cocoapi_instal_l_)
    - [cuda_operators       @ install](#cuda_operators___instal_l_)
        - [cc       @ cuda_operators/install](#cc___cuda_operators_install_)
- [bugs](#bug_s_)
    - [cocoapi_RLE_encoding       @ bugs](#cocoapi_rle_encoding___bugs_)
- [new_dataset](#new_dataset_)
    - [ytvis19       @ new_dataset](#ytvis19___new_datase_t_)
    - [ipsc       @ new_dataset](#ipsc___new_datase_t_)
        - [all_frames_roi_g2_0_37_swinL-ytvis       @ ipsc/new_dataset](#all_frames_roi_g2_0_37_swinl_ytvis___ipsc_new_dataset_)
            - [on-all_frames_roi_g2_39_53       @ all_frames_roi_g2_0_37_swinL-ytvis/ipsc/new_dataset](#on_all_frames_roi_g2_39_53___all_frames_roi_g2_0_37_swinl_ytvis_ipsc_new_datase_t_)
    - [mj_rocks       @ new_dataset](#mj_rocks___new_datase_t_)
- [scp       @ cc/virtualenv](#scp___cc_virtualen_v_)
    - [from_grs       @ scp](#from_grs___sc_p_)
        - [ext_reorg_roi_g2_16_53       @ from_grs/scp](#ext_reorg_roi_g2_16_53___from_grs_scp_)
        - [all_frames_roi_g2_0_37       @ from_grs/scp](#all_frames_roi_g2_0_37___from_grs_scp_)
    - [from_nrw       @ scp](#from_nrw___sc_p_)
        - [idol-all_frames_roi_g2_0_37       @ from_nrw/scp](#idol_all_frames_roi_g2_0_37___from_nrw_scp_)
        - [seqformer-all_frames_roi_g2_0_37       @ from_nrw/scp](#seqformer_all_frames_roi_g2_0_37___from_nrw_scp_)
        - [idol-ext_reorg_roi_g2_0_37       @ from_nrw/scp](#idol_ext_reorg_roi_g2_0_37___from_nrw_scp_)
        - [idol-ext_reorg_roi_g2_0_37-max_length-20       @ from_nrw/scp](#idol_ext_reorg_roi_g2_0_37_max_length_20___from_nrw_scp_)
        - [idol-ipsc-ext_reorg_roi_g2_16_53       @ from_nrw/scp](#idol_ipsc_ext_reorg_roi_g2_16_53___from_nrw_scp_)
        - [idol-ipsc-ext_reorg_roi_g2_54_126       @ from_nrw/scp](#idol_ipsc_ext_reorg_roi_g2_54_126___from_nrw_scp_)
        - [seqformer-ipsc-ext_reorg_roi_g2_0_37       @ from_nrw/scp](#seqformer_ipsc_ext_reorg_roi_g2_0_37___from_nrw_scp_)
        - [seqformer-ipsc-ext_reorg_roi_g2_16_53       @ from_nrw/scp](#seqformer_ipsc_ext_reorg_roi_g2_16_53___from_nrw_scp_)
        - [seqformer-ipsc-ext_reorg_roi_g2_54_126       @ from_nrw/scp](#seqformer_ipsc_ext_reorg_roi_g2_54_126___from_nrw_scp_)

<!-- /MarkdownTOC -->

<a id="virtualen_v_"></a>
# virtualenv
python3 -m pip install virtualenv virtualenvwrapper

nano ~/.bashrc

export PATH=$PATH:/usr/local/cuda/bin

export WORKON_HOME=$HOME/.virtualenvs  
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3  
source /usr/local/bin/virtualenvwrapper.sh  

source ~/.bashrc

mkvirtualenv vnext
workon vnext

alias vnxt='workon vnext'

<a id="cc___virtualenv_"></a>
## cc       @ virtualenv-->vnext_setup
module load python/3.8
module load gcc cuda cudnn
virtualenv ~/venv/vnext
source ~/venv/vnext/bin/activate
deactivate

alias vnxt='source ~/venv/vnext/bin/activate'

diskusage_report

<a id="windows___virtualenv_"></a>
## windows       @ virtualenv-->vnext_setup
virtualenv vnext
vnext\Scripts\activate

<a id="cuda_version___virtualenv_"></a>
## cuda_version       @ virtualenv-->vnext_setup
nvcc --version
/usr/local/cuda/bin/nvcc --version

<a id="install_"></a>
# install
<a id="pytorch___instal_l_"></a>
## pytorch       @ install-->vnext_setup
python -m pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio===0.10.2+cu113 -f https://download.pytorch.org/whl/torch_stable.html

python -m pip uninstall -y torch torchvision torchaudio
python -m pip install --no-index torch torchvision torchaudio

<a id="python_3_10___pytorch_instal_l_"></a>
### python_3.10       @ pytorch/install-->vnext_setup
python -m pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

<a id="requirements___instal_l_"></a>
## requirements       @ install-->vnext_setup
python -m pip install -r requirements.txt

<a id="detectron2___instal_l_"></a>
## detectron2       @ install-->vnext_setup
python -m pip install -e .

<a id="windows___detectron2_install_"></a>
### windows       @ detectron2/install-->vnext_setup
change in `setup.py`
```
PROJECTS = {
    "detectron2.projects.idol": "projects/IDOL/idol",
     "detectron2.projects.seqformer": "projects/SeqFormer/seqformer",

}
```
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

<a id="misc___instal_l_"></a>
## misc       @ install-->vnext_setup
python -m pip install imagesize shapely==1.7.1

<a id="geos_c___misc_install_"></a>
### geos_c       @ misc/install-->vnext_setup
sudo apt-get install libgeos-dev

<a id="cocoapi___instal_l_"></a>
## cocoapi       @ install-->vnext_setup
python -m pip install pycocotools
python -m pip uninstall pycocotools
git clone https://github.com/youtubevos/cocoapi
cd cocoapi/PythonAPI
python setup.py build_ext install
cd -

<a id="windows___cocoapi_instal_l_"></a>
### windows       @ cocoapi/install-->vnext_setup
change line 12 from:
```
extra_compile_args=['-Wno-cpp', '-Wno-unused-function', '-std=c99'],
```
to
```
extra_compile_args={'gcc': ['/Qstd=c99']},
```
__buggy__
pip install git+https://github.com/youtubevos/cocoapi.git#"egg=pycocotools&subdirectory=PythonAPI"
python -m pip install pycocotools

<a id="cuda_operators___instal_l_"></a>
## cuda_operators       @ install-->vnext_setup
cd projects/IDOL/idol/models/ops/
python setup.py build install
cd -

<a id="cc___cuda_operators_install_"></a>
### cc       @ cuda_operators/install-->vnext_setup
salloc --nodes=1 --time=0:15:0 --account=def-nilanjan --gpus-per-node=1 --mem=4000M --cpus-per-task=4

module load cuda cudnn gcc python/3.8
source ~/venv/vnext/bin/activate

cp -r projects/IDOL/idol/models/ops ~/
cd ~/ops/
python3 setup.py build install

<a id="bug_s_"></a>
# bugs
`AttributeError: module 'distutils' has no attribute 'version'`
python3 -m pip install setuptools==59.5.0

`RuntimeError: received 0 items of ancdata`
https://github.com/pytorch/pytorch/issues/973
torch.multiprocessing.set_sharing_strategy('file_system')
https://github.com/pytorch/pytorch/issues/973#issuecomment-459398189

<a id="cocoapi_rle_encoding___bugs_"></a>
## cocoapi_RLE_encoding       @ bugs-->vnext_setup
https://github.com/cocodataset/cocoapi/issues/492
https://github.com/cocodataset/cocoapi/issues/386

uncompressed / plain text list RLE to compressed / binary RLE
line 322 of projects\IDOL\idol\data\datasets\ytvis.py
```
segm = mask_util.frPyObjects(segm, *segm["size"])
```
input json loading works with both uncompressed and compressed RLE

`findContours TypeError: Expected Ptr<cv::UMat> for argument 'image'`
copy image returned by mask_util.decode


<a id="new_dataset_"></a>
# new_dataset
<a id="ytvis19___new_datase_t_"></a>
## ytvis19       @ new_dataset-->vnext_setup
"C:\UofA\PhD\ipsc_cell_tracking\ipsc_vnext\projects\IDOL\idol\data\datasets\builtin.py"
"C:\UofA\PhD\ipsc_cell_tracking\ipsc_vnext\projects\SeqFormer\seqformer\data\datasets\builtin.py"
`_PREDEFINED_SPLITS_YTVIS_2019`
`register_all_ytvis_2019`

<a id="ipsc___new_datase_t_"></a>
## ipsc       @ new_dataset-->vnext_setup
scp -r -P 9738 abhineet@greyshark.cs.ualberta.ca:~/all_frames_roi_grs_221007.zip ./
scp -r -P 9738 abhineet@greyshark.cs.ualberta.ca:pretrained_cocopretrain_SWINL_pth_grs_221007_224219.zip ./


ln -s /data/ipsc ./datasets/ipsc
ln -s /data/ipsc/well3/all_frames_roi /data/ipsc/well3/all_frames_roi
__cc__
ln -s ~/projects/def-nilanjan/asingh1/data/ipsc ./datasets/ipsc
ln -s  ~/projects/def-nilanjan/asingh1/data/ipsc/well3/all_frames_roi  ~/projects/def-nilanjan/asingh1/data/ipsc/well3/all_frames_roi
Running scp -r -i ~/.ssh/id_rsa -P 22 asingh1@narval.computecanada.ca:"/home/asingh1/ipsc-all_frames_roi_g2_0_37_ytvis_swinL_10278087.out" "/home/Tommy"

<a id="all_frames_roi_g2_0_37_swinl_ytvis___ipsc_new_dataset_"></a>
### all_frames_roi_g2_0_37_swinL-ytvis       @ ipsc/new_dataset-->vnext_setup
mv ipsc/well3/all_frames_roi/ytvis19/all_frames_roi_g2_0_37-train.json ipsc/well3/all_frames_roi/ytvis19/ipsc-all_frames_roi_g2_0_37-train.json

mv ipsc/well3/all_frames_roi/ytvis19/all_frames_roi_g2_0_37-val.json ipsc/well3/all_frames_roi/ytvis19/ipsc-all_frames_roi_g2_0_37-val.json

<a id="on_all_frames_roi_g2_39_53___all_frames_roi_g2_0_37_swinl_ytvis_ipsc_new_datase_t_"></a>
#### on-all_frames_roi_g2_39_53       @ all_frames_roi_g2_0_37_swinL-ytvis/ipsc/new_dataset-->vnext_setup
mv ipsc-all_frames_roi_g2_39_53-train.json ipsc-all_frames_roi_g2_39_53-test.json

<a id="mj_rocks___new_datase_t_"></a>
## mj_rocks       @ new_dataset-->vnext_setup
ln -s ~/data/mojow_rock ./datasets/mojow_rock
ln -s ~/data/mojow_rock/rock_dataset3 ~/data/mojow_rock/rock_dataset3/ytvis19/JPEGImages


<a id="scp___cc_virtualen_v_"></a>
# scp       @ cc/virtualenv-->vnext_setup
<a id="from_grs___sc_p_"></a>
## from_grs       @ scp-->vnext_setup
scp -r -P 9738 abhineet@greyshark.cs.ualberta.ca:~/all_frames_roi_grs_221007.zip ./
scp -r -P 9738 abhineet@greyshark.cs.ualberta.ca:~/scripts ~/
scp -r -P 9738 abhineet@greyshark.cs.ualberta.ca:~/ipsc_vnext/pretrained ./

scp -r -P 9738 abhineet@greyshark.cs.ualberta.ca:/data/ipsc/well3 ./
scp -r -P 9738 abhineet@greyshark.cs.ualberta.ca:/data/ipsc/well3/all_frames_roi/ytvis19 ./

<a id="ext_reorg_roi_g2_16_53___from_grs_scp_"></a>
### ext_reorg_roi_g2_16_53       @ from_grs/scp-->vnext_setup
scp -r -P 9738 abhineet@greyshark.cs.ualberta.ca:/data/ipsc/well3/all_frames_roi/ytvis19/ipsc-ext_reorg_roi_g2_16_53.json ./
scp -r -P 9738 abhineet@greyshark.cs.ualberta.ca:/data/ipsc/well3/all_frames_roi/ytvis19/ipsc-ext_reorg_roi_g2_0_15.json ./

<a id="all_frames_roi_g2_0_37___from_grs_scp_"></a>
### all_frames_roi_g2_0_37       @ from_grs/scp-->vnext_setup
scp -r -P 9738 abhineet@greyshark.cs.ualberta.ca:/data/vnext_log/idol-ipsc-all_frames_roi_g2_0_37/model_0056999.pth ./
scp -r -P 9738 abhineet@greyshark.cs.ualberta.ca:/data/vnext_log/idol-ipsc-all_frames_roi_g2_0_37 ./
 
scp -r -P 9738 abhineet@greyshark.cs.ualberta.ca:/data/vnext_log/seqformer-ipsc-all_frames_roi_g2_0_37/model_0049999.pth ./


<a id="from_nrw___sc_p_"></a>
## from_nrw       @ scp-->vnext_setup
mkdir /data/vnxt_log
cd /data/vnxt_log

<a id="idol_all_frames_roi_g2_0_37___from_nrw_scp_"></a>
### idol-all_frames_roi_g2_0_37       @ from_nrw/scp-->vnext_setup
mkdir idol-ipsc-all_frames_roi_g2_0_37
scp -r asingh1@narval.computecanada.ca:~/scratch/ipsc_vnext_log/idol-ipsc-all_frames_roi_g2_0_37/model_0099999.pth ./

ln -s /data/vnxt_log/idol-ipsc-all_frames_roi_g2_0_37 ./

scp -r asingh1@narval.computecanada.ca:~/scratch/vnext_log/idol-ipsc-all_frames_roi_g2_0_37/model_0056999.pth ./

<a id="seqformer_all_frames_roi_g2_0_37___from_nrw_scp_"></a>
### seqformer-all_frames_roi_g2_0_37       @ from_nrw/scp-->vnext_setup
scp -r asingh1@narval.computecanada.ca:~/scratch/vnext_log/seqformer-ipsc-all_frames_roi_g2_0_37/model_0049999.pth ./

<a id="idol_ext_reorg_roi_g2_0_37___from_nrw_scp_"></a>
### idol-ext_reorg_roi_g2_0_37       @ from_nrw/scp-->vnext_setup
scp -r asingh1@narval.computecanada.ca:/home/asingh1/vnext/log/idol-ipsc-ext_reorg_roi_g2_0_37/model_0099999.pth ./
scp -r asingh1@narval.computecanada.ca:/home/asingh1/vnext/log/idol-ipsc-ext_reorg_roi_g2_0_37/events.out.** ./

<a id="idol_ext_reorg_roi_g2_0_37_max_length_20___from_nrw_scp_"></a>
### idol-ext_reorg_roi_g2_0_37-max_length-20       @ from_nrw/scp-->vnext_setup
scp -r asingh1@narval.computecanada.ca:/home/asingh1/vnext/log/idol-ipsc-ext_reorg_roi_g2_0_37-max_length-20/model_0099999.pth ./
scp -r asingh1@narval.computecanada.ca:/home/asingh1/vnext/log/idol-ipsc-ext_reorg_roi_g2_0_37/events.out.** ./


<a id="idol_ipsc_ext_reorg_roi_g2_16_53___from_nrw_scp_"></a>
### idol-ipsc-ext_reorg_roi_g2_16_53       @ from_nrw/scp-->vnext_setup
scp -r asingh1@narval.computecanada.ca:/home/asingh1/vnext/log/idol-ipsc-ext_reorg_roi_g2_16_53/model_0099999.pth ./
scp -r asingh1@narval.computecanada.ca:/home/asingh1/vnext/log/idol-ipsc-ext_reorg_roi_g2_16_53/model_0254999.pth ./
scp -r asingh1@narval.computecanada.ca:/home/asingh1/vnext/log/idol-ipsc-ext_reorg_roi_g2_16_53/model_0253999.pth ./
scp -r asingh1@narval.computecanada.ca:/home/asingh1/vnext/log/idol-ipsc-ext_reorg_roi_g2_16_53/model_0252999.pth ./

scp -r asingh1@narval.computecanada.ca:/home/asingh1/vnext/log/idol-ipsc-ext_reorg_roi_g2_16_53/events.out.** ./

<a id="idol_ipsc_ext_reorg_roi_g2_54_126___from_nrw_scp_"></a>
### idol-ipsc-ext_reorg_roi_g2_54_126       @ from_nrw/scp-->vnext_setup
scp -r asingh1@narval.computecanada.ca:/home/asingh1/vnext/log/idol-ipsc-ext_reorg_roi_g2_54_126/model_0293999.pth ./

scp -r asingh1@narval.computecanada.ca:/home/asingh1/vnext/log/idol-ipsc-ext_reorg_roi_g2_54_126/model_0596999.pth ./

scp -r asingh1@narval.computecanada.ca:/home/asingh1/vnext/log/idol-ipsc-ext_reorg_roi_g2_54_126/events.out.** ./

<a id="seqformer_ipsc_ext_reorg_roi_g2_0_37___from_nrw_scp_"></a>
### seqformer-ipsc-ext_reorg_roi_g2_0_37       @ from_nrw/scp-->vnext_setup
scp -r asingh1@narval.computecanada.ca:/home/asingh1/vnext/log/seqformer-ipsc-ext_reorg_roi_g2_0_37/model_0099999.pth ./

scp -r asingh1@narval.computecanada.ca:/home/asingh1/vnext/log/seqformer-ipsc-ext_reorg_roi_g2_0_37/model_0098999.pth ./

scp -r asingh1@narval.computecanada.ca:/home/asingh1/vnext/log/seqformer-ipsc-ext_reorg_roi_g2_0_37 ./

scp -r asingh1@narval.computecanada.ca:/home/asingh1/vnext/log/seqformer-ipsc-ext_reorg_roi_g2_0_37/events.out.** ./

scp -r asingh1@narval.computecanada.ca:/home/asingh1/vnext/log/seqformer-ipsc-ext_reorg_roi_g2_0_37 ./

<a id="seqformer_ipsc_ext_reorg_roi_g2_16_53___from_nrw_scp_"></a>
### seqformer-ipsc-ext_reorg_roi_g2_16_53       @ from_nrw/scp-->vnext_setup
scp -r asingh1@narval.computecanada.ca:/home/asingh1/vnext/log/seqformer-ipsc-ext_reorg_roi_g2_16_53/model_0099999.pth ./

scp -r asingh1@narval.computecanada.ca:/home/asingh1/vnext/log/seqformer-ipsc-ext_reorg_roi_g2_16_53/model_0241999.pth ./

scp -r asingh1@narval.computecanada.ca:/home/asingh1/vnext/log/seqformer-ipsc-ext_reorg_roi_g2_16_53/model_0240999.pth ./

scp -r asingh1@narval.computecanada.ca:/home/asingh1/vnext/log/seqformer-ipsc-ext_reorg_roi_g2_16_53/events.out.** ./

<a id="seqformer_ipsc_ext_reorg_roi_g2_54_126___from_nrw_scp_"></a>
### seqformer-ipsc-ext_reorg_roi_g2_54_126       @ from_nrw/scp-->vnext_setup
scp -r asingh1@narval.computecanada.ca:/home/asingh1/vnext/log/seqformer-ipsc-ext_reorg_roi_g2_54_126/model_0495999.pth ./

scp -r asingh1@narval.computecanada.ca:/home/asingh1/vnext/log/seqformer-ipsc-ext_reorg_roi_g2_54_126/model_0040999.pth ./

scp -r asingh1@narval.computecanada.ca:/home/asingh1/vnext/log/seqformer-ipsc-ext_reorg_roi_g2_54_126/model_0239999.pth ./

scp -r asingh1@narval.computecanada.ca:/home/asingh1/vnext/log/seqformer-ipsc-ext_reorg_roi_g2_54_126/model_0495999.pth ./

scp -r asingh1@narval.computecanada.ca:/home/asingh1/vnext/log/seqformer-ipsc-ext_reorg_roi_g2_54_126/events.out.** ./

scp -r asingh1@narval.computecanada.ca:/home/asingh1/__log_seqformer-ipsc-ext_reorg_roi_g2_54_126_inference_model_0495999_max_length-19_results_json_nrw_230420_134050.zip ./











