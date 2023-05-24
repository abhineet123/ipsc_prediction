<!-- MarkdownTOC -->

- [virtualenv](#virtualen_v_)
    - [windows       @ virtualenv](#windows___virtualenv_)
- [install](#install_)
    - [torch       @ install](#torch___instal_l_)
- [ext_reorg_roi_g2_0_37](#ext_reorg_roi_g2_0_37_)
    - [v1-base-224-1k       @ ext_reorg_roi_g2_0_37](#v1_base_224_1k___ext_reorg_roi_g2_0_3_7_)
        - [eval       @ v1-base-224-1k/ext_reorg_roi_g2_0_37](#eval___v1_base_224_1k_ext_reorg_roi_g2_0_37_)
    - [v1-large-224-22k       @ ext_reorg_roi_g2_0_37](#v1_large_224_22k___ext_reorg_roi_g2_0_3_7_)
    - [v1-large-384-22k       @ ext_reorg_roi_g2_0_37](#v1_large_384_22k___ext_reorg_roi_g2_0_3_7_)
        - [eval       @ v1-large-384-22k/ext_reorg_roi_g2_0_37](#eval___v1_large_384_22k_ext_reorg_roi_g2_0_37_)
    - [v2-base-256-1k       @ ext_reorg_roi_g2_0_37](#v2_base_256_1k___ext_reorg_roi_g2_0_3_7_)
        - [eval       @ v2-base-256-1k/ext_reorg_roi_g2_0_37](#eval___v2_base_256_1k_ext_reorg_roi_g2_0_37_)
    - [v2-large-192-22k       @ ext_reorg_roi_g2_0_37](#v2_large_192_22k___ext_reorg_roi_g2_0_3_7_)
- [ext_reorg_roi_g2_0_37_masked](#ext_reorg_roi_g2_0_37_maske_d_)
    - [v1-base-224-1k       @ ext_reorg_roi_g2_0_37_masked](#v1_base_224_1k___ext_reorg_roi_g2_0_37_masked_)
        - [eval       @ v1-base-224-1k/ext_reorg_roi_g2_0_37_masked](#eval___v1_base_224_1k_ext_reorg_roi_g2_0_37_maske_d_)
- [ext_reorg_roi_g2_16_53](#ext_reorg_roi_g2_16_5_3_)
    - [v1-base-224-1k       @ ext_reorg_roi_g2_16_53](#v1_base_224_1k___ext_reorg_roi_g2_16_53_)
        - [eval       @ v1-base-224-1k/ext_reorg_roi_g2_16_53](#eval___v1_base_224_1k_ext_reorg_roi_g2_16_5_3_)
- [ext_reorg_roi_g2_16_53_masked](#ext_reorg_roi_g2_16_53_masked_)
    - [v1-base-224-1k       @ ext_reorg_roi_g2_16_53_masked](#v1_base_224_1k___ext_reorg_roi_g2_16_53_maske_d_)
        - [eval       @ v1-base-224-1k/ext_reorg_roi_g2_16_53_masked](#eval___v1_base_224_1k_ext_reorg_roi_g2_16_53_masked_)
- [ext_reorg_roi_g2_54_126](#ext_reorg_roi_g2_54_126_)
    - [v1-base-224-1k       @ ext_reorg_roi_g2_54_126](#v1_base_224_1k___ext_reorg_roi_g2_54_12_6_)
        - [eval       @ v1-base-224-1k/ext_reorg_roi_g2_54_126](#eval___v1_base_224_1k_ext_reorg_roi_g2_54_126_)
        - [on-g2_0_15       @ v1-base-224-1k/ext_reorg_roi_g2_54_126](#on_g2_0_15___v1_base_224_1k_ext_reorg_roi_g2_54_126_)

<!-- /MarkdownTOC -->

<a id="virtualen_v_"></a>
# virtualenv
python3 -m pip install virtualenv virtualenvwrapper

mkvirtualenv swin_c
workon swin_c

nano ~/.bashrc
alias swc='workon swin_c'
source ~/.bashrc

<a id="windows___virtualenv_"></a>
## windows       @ virtualenv-->swin_cls
virtualenv swin_c
swin_c\Scripts\activate.bat

<a id="install_"></a>
# install

<a id="torch___instal_l_"></a>
## torch       @ install-->swin_cls
python -m pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

python -m pip install tensorboardX
python -m pip install tqdm
python -m pip install timm==0.4.12
python -m pip install opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8 pyyaml scipy

cd kernels/window_process
python setup.py install

python -m pip install tensorflow tensorboard

<a id="ext_reorg_roi_g2_0_37_"></a>
# ext_reorg_roi_g2_0_37
<a id="v1_base_224_1k___ext_reorg_roi_g2_0_3_7_"></a>
## v1-base-224-1k       @ ext_reorg_roi_g2_0_37-->swin_cls
python -m torch.distributed.launch --nproc_per_node 3 --master_port 12345  main.py --cfg configs/swin/swin_base_patch4_window7_224_ipsc2class.yaml --data-path /data/ipsc/well3/all_frames_roi/swc/ext_reorg_roi_g2_0_37 --batch-size 128 --output log/ext_reorg_roi_g2_0_37-v1-base-224-1k  --pretrained pretrained/swin_base_patch4_window7_224_22k.pth --use-checkpoint

<a id="eval___v1_base_224_1k_ext_reorg_roi_g2_0_37_"></a>
### eval       @ v1-base-224-1k/ext_reorg_roi_g2_0_37-->swin_cls
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py --eval --cfg configs/swin/swin_base_patch4_window7_224_ipsc2class.yaml --data-path /data/ipsc/well3/all_frames_roi/swc/ext_reorg_roi_g2_0_37 --batch-size 24 --output log/ext_reorg_roi_g2_0_37-v1-base-224-1k --resume log/ext_reorg_roi_g2_0_37-v1-base-224-1k/ckpt_epoch_796.pth

<a id="v1_large_224_22k___ext_reorg_roi_g2_0_3_7_"></a>
## v1-large-224-22k       @ ext_reorg_roi_g2_0_37-->swin_cls
python -m torch.distributed.launch --nproc_per_node 3 --master_port 12345  main.py --cfg configs/swin/swin_large_patch4_window7_224_22k_ipsc2class.yaml --data-path /data/ipsc/well3/all_frames_roi/swc/ext_reorg_roi_g2_0_37 --batch-size 64 --output log/ext_reorg_roi_g2_0_37-v1-large-224-22k  --pretrained pretrained/swin_large_patch4_window7_224_22k.pth --use-checkpoint

<a id="v1_large_384_22k___ext_reorg_roi_g2_0_3_7_"></a>
## v1-large-384-22k       @ ext_reorg_roi_g2_0_37-->swin_cls
python -m torch.distributed.launch --nproc_per_node 3 --master_port 12345  main.py --cfg configs/swin/swin_large_patch4_window12_384_22kto1k_finetune_ipsc2class.yaml --data-path /data/ipsc/well3/all_frames_roi/swc/ext_reorg_roi_g2_0_37 --batch-size 12 --output log/ext_reorg_roi_g2_0_37-v1-large-384-22k  --pretrained pretrained/swin_large_patch4_window12_384_22k.pth --use-checkpoint

<a id="eval___v1_large_384_22k_ext_reorg_roi_g2_0_37_"></a>
### eval       @ v1-large-384-22k/ext_reorg_roi_g2_0_37-->swin_cls
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py --eval --cfg configs/swin/swin_large_patch4_window12_384_22kto1k_finetune_ipsc2class.yaml --data-path /data/ipsc/well3/all_frames_roi/swc/ext_reorg_roi_g2_0_37 --batch-size 24 --output log/ext_reorg_roi_g2_0_37-v1-large-384-22k --resume log/ext_reorg_roi_g2_0_37-v1-large-384-22k/ckpt_epoch_00999.pth

<a id="v2_base_256_1k___ext_reorg_roi_g2_0_3_7_"></a>
## v2-base-256-1k       @ ext_reorg_roi_g2_0_37-->swin_cls
python -m torch.distributed.launch --nproc_per_node 3 --master_port 12345  main.py --cfg configs/swinv2/swinv2_base_patch4_window16_256_ipsc2class.yaml --data-path /data/ipsc/well3/all_frames_roi/swc/ext_reorg_roi_g2_0_37 --batch-size 48 --output log/ext_reorg_roi_g2_0_37-v2-base-256-1k --pretrained pretrained/swinv2_base_patch4_window16_256.pth --use-checkpoint

<a id="eval___v2_base_256_1k_ext_reorg_roi_g2_0_37_"></a>
### eval       @ v2-base-256-1k/ext_reorg_roi_g2_0_37-->swin_cls
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py --eval --cfg configs/swinv2/swinv2_base_patch4_window16_256_ipsc2class.yaml --data-path /data/ipsc/well3/all_frames_roi/swc/ext_reorg_roi_g2_0_37 --batch-size 24 --output log/ext_reorg_roi_g2_0_37-v2-base-256-1k --resume log/ext_reorg_roi_g2_0_37-v2-base-256-1k/ckpt_epoch_00999.pth

<a id="v2_large_192_22k___ext_reorg_roi_g2_0_3_7_"></a>
## v2-large-192-22k       @ ext_reorg_roi_g2_0_37-->swin_cls
python -m torch.distributed.launch --nproc_per_node 3 --master_port 12345  main.py --cfg configs/swinv2/swinv2_large_patch4_window12_192_22k_ipsc2class.yaml --data-path /data/ipsc/well3/all_frames_roi/swc/ext_reorg_roi_g2_0_37 --batch-size 48 --output log/ext_reorg_roi_g2_0_37-v2-large-192-22k --pretrained pretrained/swinv2_large_patch4_window12_192_22k.pth --use-checkpoint

<a id="ext_reorg_roi_g2_0_37_maske_d_"></a>
# ext_reorg_roi_g2_0_37_masked
<a id="v1_base_224_1k___ext_reorg_roi_g2_0_37_masked_"></a>
## v1-base-224-1k       @ ext_reorg_roi_g2_0_37_masked-->swin_cls
python -m torch.distributed.launch --nproc_per_node 3 --master_port 12345  main.py --cfg configs/swin/swin_base_patch4_window7_224_ipsc2class.yaml --data-path /data/ipsc/well3/all_frames_roi/swc/ext_reorg_roi_g2_0_37_masked --batch-size 128 --output log/ext_reorg_roi_g2_0_37_masked-v1-base-224-1k  --pretrained pretrained/swin_base_patch4_window7_224_22k.pth --use-checkpoint

<a id="eval___v1_base_224_1k_ext_reorg_roi_g2_0_37_maske_d_"></a>
### eval       @ v1-base-224-1k/ext_reorg_roi_g2_0_37_masked-->swin_cls
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py --eval --cfg configs/swin/swin_base_patch4_window7_224_ipsc2class.yaml --data-path /data/ipsc/well3/all_frames_roi/swc/ext_reorg_roi_g2_0_37_masked --batch-size 24 --output log/ext_reorg_roi_g2_0_37_masked-v1-base-224-1k --resume log/ext_reorg_roi_g2_0_37_masked-v1-base-224-1k/ckpt_epoch_00999.pth

<a id="ext_reorg_roi_g2_16_5_3_"></a>
# ext_reorg_roi_g2_16_53
<a id="v1_base_224_1k___ext_reorg_roi_g2_16_53_"></a>
## v1-base-224-1k       @ ext_reorg_roi_g2_16_53-->swin_cls
python -m torch.distributed.launch --nproc_per_node 3 --master_port 12345  main.py --cfg configs/swin/swin_base_patch4_window7_224_ipsc2class.yaml --data-path /data/ipsc/well3/all_frames_roi/swc/ext_reorg_roi_g2_16_53 --batch-size 128 --output log/ext_reorg_roi_g2_16_53-v1-base-224-1k  --pretrained pretrained/swin_base_patch4_window7_224_22k.pth --use-checkpoint

<a id="eval___v1_base_224_1k_ext_reorg_roi_g2_16_5_3_"></a>
### eval       @ v1-base-224-1k/ext_reorg_roi_g2_16_53-->swin_cls
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py --eval --cfg configs/swin/swin_base_patch4_window7_224_ipsc2class.yaml --data-path /data/ipsc/well3/all_frames_roi/swc/ext_reorg_roi_g2_16_53 --batch-size 24 --output log/ext_reorg_roi_g2_16_53-v1-base-224-1k --resume log/ext_reorg_roi_g2_16_53-v1-base-224-1k/ckpt_epoch_00999.pth

<a id="ext_reorg_roi_g2_16_53_masked_"></a>
# ext_reorg_roi_g2_16_53_masked
<a id="v1_base_224_1k___ext_reorg_roi_g2_16_53_maske_d_"></a>
## v1-base-224-1k       @ ext_reorg_roi_g2_16_53_masked-->swin_cls
python -m torch.distributed.launch --nproc_per_node 3 --master_port 12345  main.py --cfg configs/swin/swin_base_patch4_window7_224_ipsc2class.yaml --data-path /data/ipsc/well3/all_frames_roi/swc/ext_reorg_roi_g2_16_53_masked --batch-size 128 --output log/ext_reorg_roi_g2_16_53_masked-v1-base-224-1k  --pretrained pretrained/swin_base_patch4_window7_224_22k.pth --use-checkpoint

<a id="eval___v1_base_224_1k_ext_reorg_roi_g2_16_53_masked_"></a>
### eval       @ v1-base-224-1k/ext_reorg_roi_g2_16_53_masked-->swin_cls
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py --eval --cfg configs/swin/swin_base_patch4_window7_224_ipsc2class.yaml --data-path /data/ipsc/well3/all_frames_roi/swc/ext_reorg_roi_g2_16_53_masked --batch-size 24 --output log/ext_reorg_roi_g2_16_53_masked-v1-base-224-1k --resume log/ext_reorg_roi_g2_16_53_masked-v1-base-224-1k/ckpt_epoch_00999.pth

<a id="ext_reorg_roi_g2_54_126_"></a>
# ext_reorg_roi_g2_54_126
<a id="v1_base_224_1k___ext_reorg_roi_g2_54_12_6_"></a>
## v1-base-224-1k       @ ext_reorg_roi_g2_54_126-->swin_cls
python -m torch.distributed.launch --nproc_per_node 3 --master_port 12345  main.py --cfg configs/swin/swin_base_patch4_window7_224_ipsc2class.yaml --data-path /data/ipsc/well3/all_frames_roi/swc/ext_reorg_roi_g2_54_126 --batch-size 128 --output log/ext_reorg_roi_g2_54_126-v1-base-224-1k  --pretrained pretrained/swin_base_patch4_window7_224_22k.pth --use-checkpoint --resume log/ext_reorg_roi_g2_54_126-v1-base-224-1k/ckpt_epoch_09999.pth

<a id="eval___v1_base_224_1k_ext_reorg_roi_g2_54_126_"></a>
### eval       @ v1-base-224-1k/ext_reorg_roi_g2_54_126-->swin_cls
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py --eval --cfg configs/swin/swin_base_patch4_window7_224_ipsc2class.yaml --data-path /data/ipsc/well3/all_frames_roi/swc/ext_reorg_roi_g2_54_126 --batch-size 24 --output log/ext_reorg_roi_g2_54_126-v1-base-224-1k --resume log/ext_reorg_roi_g2_54_126-v1-base-224-1k/ckpt_epoch_09999.pth
<a id="on_g2_0_15___v1_base_224_1k_ext_reorg_roi_g2_54_126_"></a>
### on-g2_0_15       @ v1-base-224-1k/ext_reorg_roi_g2_54_126-->swin_cls
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py --eval --cfg configs/swin/swin_base_patch4_window7_224_ipsc2class.yaml --data-path /data/ipsc/well3/all_frames_roi/swc/ext_reorg_roi_g2_16_53 --batch-size 24 --output log/ext_reorg_roi_g2_54_126-v1-base-224-1k --resume log/ext_reorg_roi_g2_54_126-v1-base-224-1k/ckpt_epoch_09999.pth

log/ext_reorg_roi_g2_54_126-v1-base-224-1k/ckpt_epoch_09999.pth
log/ext_reorg_roi_g2_54_126-v1-base-224-1k/ckpt_epoch_09999.pth
log/ext_reorg_roi_g2_54_126-v1-base-224-1k/ckpt_epoch_09999.pth

