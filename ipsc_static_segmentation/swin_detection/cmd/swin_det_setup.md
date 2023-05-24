<!-- MarkdownTOC -->

- [virtualenv](#virtualen_v_)
    - [cc       @ virtualenv](#cc___virtualenv_)
    - [windows       @ virtualenv](#windows___virtualenv_)
- [install](#install_)
    - [torch       @ install](#torch___instal_l_)
        - [cc       @ torch/install](#cc___torch_instal_l_)
        - [cu111       @ torch/install](#cu111___torch_instal_l_)
        - [cu102       @ torch/install](#cu102___torch_instal_l_)
        - [cu113       @ torch/install](#cu113___torch_instal_l_)
            - [py3.10       @ cu113/torch/install](#py3_10___cu113_torch_instal_l_)
    - [mmcv       @ install](#mmcv___instal_l_)
        - [cc       @ mmcv/install](#cc___mmcv_install_)
            - [from_source       @ cc/mmcv/install](#from_source___cc_mmcv_instal_l_)
        - [cu111       @ mmcv/install](#cu111___mmcv_install_)
        - [cu102       @ mmcv/install](#cu102___mmcv_install_)
        - [cu113       @ mmcv/install](#cu113___mmcv_install_)
            - [torch1.10.0       @ cu113/mmcv/install](#torch1_10_0___cu113_mmcv_install_)
            - [torch1.10.2       @ cu113/mmcv/install](#torch1_10_2___cu113_mmcv_install_)
        - [py3.10       @ mmcv/install](#py3_10___mmcv_install_)
            - [torch1.12       @ py3.10/mmcv/install](#torch1_12___py3_10_mmcv_instal_l_)
            - [torch1.11       @ py3.10/mmcv/install](#torch1_11___py3_10_mmcv_instal_l_)
        - [uninstall       @ mmcv/install](#uninstall___mmcv_install_)
        - [misc       @ mmcv/install](#misc___mmcv_install_)
    - [openmim       @ install](#openmim___instal_l_)
    - [mmdet       @ install](#mmdet___instal_l_)
    - [misc       @ install](#misc___instal_l_)
        - [apex       @ misc/install](#apex___misc_install_)
    - [bugs       @ install](#bugs___instal_l_)
- [file locations](#file_location_s_)
    - [optimizer_config       @ file_locations](#optimizer_config___file_locations_)
        - [lr_updater       @ optimizer_config/file_locations](#lr_updater___optimizer_config_file_location_s_)
    - [checkpoint_saving       @ file_locations](#checkpoint_saving___file_locations_)
    - [epochs       @ file_locations](#epochs___file_locations_)
    - [data_pipelines       @ file_locations](#data_pipelines___file_locations_)
- [train_on_new_dataset](#train_on_new_datase_t_)
    - [only_bboxes       @ train_on_new_dataset](#only_bboxes___train_on_new_dataset_)
    - [enable_tensorboard       @ train_on_new_dataset](#enable_tensorboard___train_on_new_dataset_)
    - [multiple_non_distributed_on_separate_gpus       @ train_on_new_dataset](#multiple_non_distributed_on_separate_gpus___train_on_new_dataset_)
    - [issues       @ train_on_new_dataset](#issues___train_on_new_dataset_)
    - [visualize       @ train_on_new_dataset](#visualize___train_on_new_dataset_)
- [test_on_new_unlabeled_images](#test_on_new_unlabeled_image_s_)
    - [k       @ test_on_new_unlabeled_images](#k___test_on_new_unlabeled_images_)
    - [output_to_annotations       @ test_on_new_unlabeled_images](#output_to_annotations___test_on_new_unlabeled_images_)
- [create_roi](#create_ro_i_)
    - [reorg_rois       @ create_roi](#reorg_rois___create_roi_)
- [scp](#scp_)
- [mask-rle](#mask_rl_e_)

<!-- /MarkdownTOC -->

<a id="virtualen_v_"></a>
# virtualenv
sudo pip3 install virtualenv virtualenvwrapper
python3 -m pip install virtualenv virtualenvwrapper

export WORKON_HOME=$HOME/.virtualenvs  
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python  
source /usr/local/bin/virtualenvwrapper.sh  

nano ~/.bashrc
alias swi='workon swin_i'
source ~/.bashrc

mkvirtualenv swin_i
workon swin_i

<a id="cc___virtualenv_"></a>
## cc       @ virtualenv-->swin_det_setup
module load python/3.7
module load gcc cuda cudnn
virtualenv ~/venv/swin_i
source ~/venv/swin_i/bin/activate
deactivate

pip install --upgrade pip setuptools wheel

alias swi='source ~/venv/swin_i/bin/activate'


<a id="windows___virtualenv_"></a>
## windows       @ virtualenv-->swin_det_setup
virtualenv swin_i
swin_i\Scripts\activate.bat

<a id="install_"></a>
# install

<a id="torch___instal_l_"></a>
## torch       @ install-->swin_det_setup

<a id="cc___torch_instal_l_"></a>
### cc       @ torch/install-->swin_det_setup
python -m pip install torch torchvision

<a id="cu111___torch_instal_l_"></a>
### cu111       @ torch/install-->swin_det_setup
__this is the one for ubuntu 20.04__
python -m pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

<a id="cu102___torch_instal_l_"></a>
### cu102       @ torch/install-->swin_det_setup
python -m pip install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

<a id="cu113___torch_instal_l_"></a>
### cu113       @ torch/install-->swin_det_setup
python -m pip install -v torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio===0.10.2+cu113 -f https://download.pytorch.org/whl/torch_stable.html

<a id="py3_10___cu113_torch_instal_l_"></a>
#### py3.10       @ cu113/torch/install-->swin_det_setup
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

__buggy errors in resuming training__
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116

<a id="mmcv___instal_l_"></a>
## mmcv       @ install-->swin_det_setup

<a id="cc___mmcv_install_"></a>
### cc       @ mmcv/install-->swin_det_setup
https://mmcv.readthedocs.io/en/latest/get_started/installation.html

python -c 'import torch;print(torch.__version__);print(torch.version.cuda)'
pip install -v mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html

<a id="from_source___cc_mmcv_instal_l_"></a>
#### from_source       @ cc/mmcv/install-->swin_det_setup
CUDA_HOME=/cvmfs/soft.computecanada.ca/easybuild/software/2020/Core/cudacore/11.4.2 MMCV_WITH_OPS=1 pip install -e . -v
MMCV_WITH_OPS=1 pip install -e . -v

<a id="cu111___mmcv_install_"></a>
### cu111       @ mmcv/install-->swin_det_setup
python -m pip install mmcv-full==1.3.9 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

<a id="cu102___mmcv_install_"></a>
### cu102       @ mmcv/install-->swin_det_setup
__won't work with RTX 3000___
python -m pip install mmcv-full==1.3.9 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.9.0/index.html

python -m pip install mmcv-full==1.4.2 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.9.0/index.html

<a id="cu113___mmcv_install_"></a>
### cu113       @ mmcv/install-->swin_det_setup
<a id="torch1_10_0___cu113_mmcv_install_"></a>
#### torch1.10.0       @ cu113/mmcv/install-->swin_det_setup
python -m pip install mmcv-full==1.4.2 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html -v
<a id="torch1_10_2___cu113_mmcv_install_"></a>
#### torch1.10.2       @ cu113/mmcv/install-->swin_det_setup
__might take a while__
python -m pip install -v mmcv-full==1.4.2 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.2/index.html

<a id="py3_10___mmcv_install_"></a>
### py3.10       @ mmcv/install-->swin_det_setup

<a id="torch1_12___py3_10_mmcv_instal_l_"></a>
#### torch1.12       @ py3.10/mmcv/install-->swin_det_setup
python -m pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12.0/index.html

<a id="torch1_11___py3_10_mmcv_instal_l_"></a>
#### torch1.11       @ py3.10/mmcv/install-->swin_det_setup
python -m pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html

<a id="uninstall___mmcv_install_"></a>
### uninstall       @ mmcv/install-->swin_det_setup
python -m pip uninstall mmcv-full
python -m pip install -v mmcv-full

python -m pip uninstall torch torchvision
python -m pip install -v torch torchvision

<a id="misc___mmcv_install_"></a>
### misc       @ mmcv/install-->swin_det_setup
'utils/spconv/spconv/geometry.h'

python -m pip install libboost-filesystem-dev
python -m pip install libboost-dev

<a id="openmim___instal_l_"></a>
## openmim       @ install-->swin_det_setup
python -m pip install -v openmim
python -m pip install -v --upgrade timm

<a id="mmdet___instal_l_"></a>
## mmdet       @ install-->swin_det_setup
__buggy__
python -m pip install mmdet

run
```
python -m pip uninstall mmdet
```
and add to test / train script to use custom mmdet:
```
swin_det_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f'swin_det_dir: {swin_det_dir}')
sys.path.append(swin_det_dir)
```

<a id="misc___instal_l_"></a>
## misc       @ install-->swin_det_setup
python -m pip install -v setuptools==59.5.0
python -m pip install -v tqdm
python -m pip install -v terminaltables
python -m pip install -v lxml
python -m pip install -v fiftyone
python -m pip install -v instaboostfast
python -m pip install -v paramparse

python -m pip install -v git+https://github.com/cocodataset/panopticapi.git
python -m pip install -v git+https://github.com/lvis-dataset/lvis-api.git

python -m pip install -v albumentations>=0.3.2 --no-binary imgaug,albumentations

python -m pip uninstall pycocotools
python -m pip uninstall mmpycocotools
python -m pip install mmpycocotools

python -m pip install tensorflow tensorboard

<a id="apex___misc_install_"></a>
### apex       @ misc/install-->swin_det_setup
__best_not_installed__
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

<a id="bugs___instal_l_"></a>
## bugs       @ install-->swin_det_setup
__AttributeError: partially initialized module 'cv2' has no attribute '_registerMatType' (most likely due to a circular import)__
https://github.com/opencv/opencv-python/issues/591
uninstall opencv

python -m pip uninstall opencv-python
python -m pip uninstall opencv-python-headles

python -m pip install -v opencv-python


python -m  pip install -v "opencv-python-headless<4.3
python -m pip install -v opencv-python==4.1.2.30+computecanada

__AttributeError: module 'numpy' has no attribute 'float'__
python -m pip install "numpy<1.24"

__AttributeError: module 'torch.distributed' has no attribute '_reduce_scatter_base'__
uninstall apex

__progress bar for training__

~/.virtualenvs/swin_i/lib/python3.10/site-packages/mmcv/runner/epoch_based_runner.py
or
~/.virtualenvs/swin_i/lib/python3.8/site-packages/mmcv/runner/epoch_based_runner.py

add tqdm in line 47
```
        from tqdm import tqdm
        for i, data_batch in enumerate(tqdm(self.data_loader, total=len(self.data_loader), ncols=100)):
```
__training stuck at start__
https://github.com/open-mmlab/mmdetection/issues/166
https://github.com/pytorch/pytorch/issues/33296

At the beginning of the program ( in train.py ), first import cv2, before import torch. Seem like pytorch internal problem.
__annoying issues in terminal due to duplicate mmdet__

embed_dim --> embed_dims
https://github.com/open-mmlab/mmsegmentation/issues/752#issuecomment-893258968
ape --> use_abs_pos_embed
use_checkpoint --> with_cp

add swin_detection to pythonpath to fix it

__AssertionError: If capturable=False, state_steps should not be CUDA tensors.__
on resuming from checkpoint

https://github.com/thu-ml/tianshou/issues/681
https://github.com/pytorch/pytorch/issues/80809
caused by annoying new adam param in 1.12
revert to 1.11 or older

__AttributeError: 'MMDistributedDataParallel' object has no attribute '_sync_params'__
https://github.com/open-mmlab/mmcv/pull/1816
pytorch 1.11 - mmcv incompatibility

__system restart on starting training__
https://discuss.pytorch.org/t/system-reboot-when-training/108317/4
sudo nvidia-smi -i 0,1 -pl 250
sudo nvidia-smi -i 0,1 -pl 300

<a id="file_location_s_"></a>
# file locations
<a id="optimizer_config___file_locations_"></a>
## optimizer_config       @ file_locations-->swin_det_setup
configs/_base_/schedules/schedule_1x.py
<a id="lr_updater___optimizer_config_file_location_s_"></a>
### lr_updater       @ optimizer_config/file_locations-->swin_det_setup
C:/python7/Lib/site-packages/mmcv/runner/hooks/lr_updater.py
<a id="checkpoint_saving___file_locations_"></a>
## checkpoint_saving       @ file_locations-->swin_det_setup
mmcv_custom/runner/checkpoint.py
<a id="epochs___file_locations_"></a>
## epochs       @ file_locations-->swin_det_setup
mmcv_custom/runner/epoch_based_runner.py

<a id="data_pipelines___file_locations_"></a>
## data_pipelines       @ file_locations-->swin_det_setup
Resize, RandomFlip, Normalize, Pad: swin_detection\mmdet\datasets\pipelines\transforms.py

<a id="train_on_new_datase_t_"></a>
# train_on_new_dataset
0. create xml annotations if needed (using `mot_to_xml` for instance)
1. create json for the images using `xml_to_coco` as in `nd03` or `k` after creating a new class file in `acamp_code/labelling_tool/data` if needed
2. create a dataset file in `configs/_base_/datasets`, for example, by copying `coco_instance_person.py`
3. if needed, create a new base class in `mmdet/datasets`, e.g by copying `coco_person.py` and update second line in the file created in the last step
4. if needed, add name of the new base dataset in `mmdet/datasets/__init__.py`
5. if needed, create a new base model for the dataset in `configs/_base_/models`
6. create config file in `configs/swin`, for example, by copying `cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco_person.py`
7. add a dict entry for the images in `data` in this file
8. copy over the new class file, if needed, to `data/`

<a id="only_bboxes___train_on_new_dataset_"></a>
## only_bboxes       @ train_on_new_dataset-->swin_det_setup
- `configs\_base_\models\<name>.py`
    - comment out:
        # mask_roi_extractor=dict(
        #     type='SingleRoIExtractor',
        #     roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
        #     out_channels=256,
        #     featmap_strides=[4, 8, 16, 32]),
        # mask_head=dict(
        #     type='FCNMaskHead',
        #     num_convs=4,
        #     in_channels=256,
        #     conv_out_channels=256,
        #     num_classes=1,
        #     loss_mask=dict(
        #         type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)) 
- `configs\swin\<name>.py`
    - change:
    `dict(type='LoadAnnotations', with_bbox=True, with_mask=False)`
- `configs/_base_/datasets/<name>.py`:
    - delete `gt_masks`: 
    `dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])`
    - delete `segm`:
    `evaluation = dict(metric=['bbox'])`

<a id="enable_tensorboard___train_on_new_dataset_"></a>
## enable_tensorboard       @ train_on_new_dataset-->swin_det_setup
`configs/_base_/default_runtime.py`
comment out:
`dict(type='TensorboardLoggerHook')`

<a id="multiple_non_distributed_on_separate_gpus___train_on_new_dataset_"></a>
## multiple_non_distributed_on_separate_gpus       @ train_on_new_dataset-->swin_det_setup
`dist.init_process_group` with different file passed to `init_method` in line 124 of `tools.train`

<a id="issues___train_on_new_dataset_"></a>
## issues       @ train_on_new_dataset-->swin_det_setup
__apex IndexError: tuple index out of range__
comment out  ```optimizer_config = dict``` part completely in the `configs\swin\<name>.py` file


<a id="visualize___train_on_new_dataset_"></a>
## visualize       @ train_on_new_dataset-->swin_det_setup
python tools/misc/browse_dataset.py configs/swin/cascade_mask_rcnn_swin_base_ipsc_2_class_all_frames_roi_g2_0_37.py

<a id="test_on_new_unlabeled_image_s_"></a>
# test_on_new_unlabeled_images
1. create dummy json for the images using `xml_to_coco` as in `nd03` or `k` after creating a new class list file in `acamp_code/labelling_tool/data` if needed
2. create a dataset file in `configs/_base_/datasets`, for example, by copying `coco_instance_person.py`
3. if needed create a new base class in `mmdet/datasets`, e.g by copying `coco_person.py` and update second line in the file created in the last step
    3.1. if needed, add name of the new base dataset in `mmdet/datasets/__init__.py`
4. create config file in `configs/swin`, for example, by copying `cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco_person.py`
5. add a dict entry for the images in `data` in this file
6. copy over the new class file if needed to `data/`

<a id="k___test_on_new_unlabeled_images_"></a>
## k       @ test_on_new_unlabeled_images-->swin_det_setup
python tools/test.py config=configs/swin/cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco_person.py checkpoint=work_dirs/cascade_mask_rcnn_swin_base_patch4_window7.pth show=1 show_dir=work_dirs/cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco_person/vis test_name=k class_info=data/classes_person.txt filter_objects=1

<a id="output_to_annotations___test_on_new_unlabeled_images_"></a>
## output_to_annotations       @ test_on_new_unlabeled_images-->swin_det_setup
1. replace `<name>diff</name>` and `<name>ipsc</name>` with `<name>cell</name>`
2. if xml files not in annotations folder, first find and isolate all xml files and then use mftf: `mftf ann 1000 roi_6094_19416_8394_20382`
3. change ann1 to annotations: `rrepfr ann1 annotations`

<a id="create_ro_i_"></a>
# create_roi 
<a id="reorg_rois___create_roi_"></a>
## reorg_rois       @ create_roi-->swin_det_setup
python create_ROIs.py root_dir=/data/ipsc/well3 rois_path=reorg_rois.txt src_path=all_frames

<a id="scp_"></a>
# scp
scp -r -P 9738 abhineet@greyshark.cs.ualberta.ca:~/ipsc_segmentation/swin_detection/work_dirs/cascade_mask_rcnn_swin_base_ipsc_2_class_all_frames_roi_g2_0_37/epoch_1000.pth ./

scp -r -P 9738 abhineet@greyshark.cs.ualberta.ca:~/ipsc_segmentation/swin_detection/pretrained/* ./
scp -r -P 9738 abhineet@greyshark.cs.ualberta.ca:/data/ipsc/well3/all_frames_roi/ext_reorg_roi_g2_0_37.json ./

<a id="mask_rl_e_"></a>
# mask-rle
line 304 of
 "C:\UofA\PhD\ipsc_cell_tracking\ipsc_segmentation\swin_detection\ipsc_seg\Lib\site-packages\pycocotools\coco.py"

tools/dist_train.sh configs/convnext/cascade_mask_rcnn_convnext_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco_in1k.py 1 --cfg-options model.pretrained=pretrained/convnext_tiny_1k_224.pth
