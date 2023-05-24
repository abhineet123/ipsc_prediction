<!-- MarkdownTOC -->

- [virtualenv](#virtualen_v_)
    - [windows       @ virtualenv](#windows___virtualenv_)
- [install](#install_)
- [base-384-22k](#base_384_22_k_)
    - [ext_reorg_roi_g2_0_37       @ base-384-22k](#ext_reorg_roi_g2_0_37___base_384_22k_)
        - [eval       @ ext_reorg_roi_g2_0_37/base-384-22k](#eval___ext_reorg_roi_g2_0_37_base_384_22k_)
            - [855       @ eval/ext_reorg_roi_g2_0_37/base-384-22k](#855___eval_ext_reorg_roi_g2_0_37_base_384_22_k_)
            - [856       @ eval/ext_reorg_roi_g2_0_37/base-384-22k](#856___eval_ext_reorg_roi_g2_0_37_base_384_22_k_)
            - [857       @ eval/ext_reorg_roi_g2_0_37/base-384-22k](#857___eval_ext_reorg_roi_g2_0_37_base_384_22_k_)
    - [ext_reorg_roi_g2_16_53       @ base-384-22k](#ext_reorg_roi_g2_16_53___base_384_22k_)
        - [eval       @ ext_reorg_roi_g2_16_53/base-384-22k](#eval___ext_reorg_roi_g2_16_53_base_384_22_k_)
    - [ext_reorg_roi_g2_16_53_masked       @ base-384-22k](#ext_reorg_roi_g2_16_53_masked___base_384_22k_)
        - [eval       @ ext_reorg_roi_g2_16_53_masked/base-384-22k](#eval___ext_reorg_roi_g2_16_53_masked_base_384_22k_)
    - [ext_reorg_roi_g2_54_126       @ base-384-22k](#ext_reorg_roi_g2_54_126___base_384_22k_)
        - [eval       @ ext_reorg_roi_g2_54_126/base-384-22k](#eval___ext_reorg_roi_g2_54_126_base_384_22k_)
        - [on_g2_0_15       @ ext_reorg_roi_g2_54_126/base-384-22k](#on_g2_0_15___ext_reorg_roi_g2_54_126_base_384_22k_)
- [large-384-22k](#large_384_22k_)
    - [ext_reorg_roi_g2_0_37       @ large-384-22k](#ext_reorg_roi_g2_0_37___large_384_22_k_)
        - [eval       @ ext_reorg_roi_g2_0_37/large-384-22k](#eval___ext_reorg_roi_g2_0_37_large_384_22_k_)
    - [ext_reorg_roi_g2_0_37_masked       @ large-384-22k](#ext_reorg_roi_g2_0_37_masked___large_384_22_k_)
        - [eval       @ ext_reorg_roi_g2_0_37_masked/large-384-22k](#eval___ext_reorg_roi_g2_0_37_masked_large_384_22k_)
    - [ext_reorg_roi_g2_16_53       @ large-384-22k](#ext_reorg_roi_g2_16_53___large_384_22_k_)
        - [eval       @ ext_reorg_roi_g2_16_53/large-384-22k](#eval___ext_reorg_roi_g2_16_53_large_384_22k_)
    - [ext_reorg_roi_g2_16_53_masked       @ large-384-22k](#ext_reorg_roi_g2_16_53_masked___large_384_22_k_)
        - [eval       @ ext_reorg_roi_g2_16_53_masked/large-384-22k](#eval___ext_reorg_roi_g2_16_53_masked_large_384_22_k_)
    - [ext_reorg_roi_g2_54_126       @ large-384-22k](#ext_reorg_roi_g2_54_126___large_384_22_k_)
        - [eval       @ ext_reorg_roi_g2_54_126/large-384-22k](#eval___ext_reorg_roi_g2_54_126_large_384_22_k_)
            - [on_g2_0_15       @ eval/ext_reorg_roi_g2_54_126/large-384-22k](#on_g2_0_15___eval_ext_reorg_roi_g2_54_126_large_384_22k_)
- [bugs](#bug_s_)

<!-- /MarkdownTOC -->

<a id="virtualen_v_"></a>
# virtualenv
sudo pip3 install virtualenv virtualenvwrapper
python3 -m pip install virtualenv virtualenvwrapper

export WORKON_HOME=$HOME/.virtualenvs  
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python  
source /usr/local/bin/virtualenvwrapper.sh  

nano ~/.bashrc
alias cnx='workon cnxt'
source ~/.bashrc

mkvirtualenv cnxt
workon cnxt

<a id="windows___virtualenv_"></a>
## windows       @ virtualenv-->cnx_cls
virtualenv cnxt
cnxt\Scripts\activate.bat

<a id="install_"></a>
# install
python -m pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio===0.10.2+cu113 -f https://download.pytorch.org/whl/torch_stable.html

python -m pip install timm==0.4.12
python -m pip install tensorboardX six tqdm


<a id="base_384_22_k_"></a>
# base-384-22k
<a id="ext_reorg_roi_g2_0_37___base_384_22k_"></a>
## ext_reorg_roi_g2_0_37       @ base-384-22k-->cnx_cls
python -m torch.distributed.launch --nproc_per_node=3 main.py --model convnext_base --drop_path 0.2 --input_size 384 --batch_size 4 --lr 5e-5 --update_freq 2 --warmup_epochs 0 --epochs 10000 --weight_decay 1e-8 --layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 --finetune  pretrained/convnext_base_22k_1k_384.pth --data_path /data/ipsc/well3/all_frames_roi/swc/ext_reorg_roi_g2_0_37 --output_dir log/ext_reorg_roi_g2_0_37-base-384-22k --nb_classes 2

<a id="eval___ext_reorg_roi_g2_0_37_base_384_22k_"></a>
### eval       @ ext_reorg_roi_g2_0_37/base-384-22k-->cnx_cls
<a id="855___eval_ext_reorg_roi_g2_0_37_base_384_22_k_"></a>
#### 855       @ eval/ext_reorg_roi_g2_0_37/base-384-22k-->cnx_cls
python -m torch.distributed.launch --nproc_per_node=1 main.py --model convnext_base --drop_path 0.2 --input_size 384 --batch_size 10 --lr 5e-5 --update_freq 2 --warmup_epochs 0 --epochs 10000 --weight_decay 1e-8 --layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 --data_path /data/ipsc/well3/all_frames_roi/swc/ext_reorg_roi_g2_0_37 --output_dir log/ext_reorg_roi_g2_0_37-base-384-22k --nb_classes 2 --eval true --resume log/ext_reorg_roi_g2_0_37-base-384-22k/checkpoint-855.pth
<a id="856___eval_ext_reorg_roi_g2_0_37_base_384_22_k_"></a>
#### 856       @ eval/ext_reorg_roi_g2_0_37/base-384-22k-->cnx_cls
python -m torch.distributed.launch --nproc_per_node=1 main.py --model convnext_base --drop_path 0.2 --input_size 384 --batch_size 10 --lr 5e-5 --update_freq 2 --warmup_epochs 0 --epochs 10000 --weight_decay 1e-8 --layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 --data_path /data/ipsc/well3/all_frames_roi/swc/ext_reorg_roi_g2_0_37 --output_dir log/ext_reorg_roi_g2_0_37-base-384-22k --nb_classes 2 --eval true --resume log/ext_reorg_roi_g2_0_37-base-384-22k/checkpoint-856.pth
<a id="857___eval_ext_reorg_roi_g2_0_37_base_384_22_k_"></a>
#### 857       @ eval/ext_reorg_roi_g2_0_37/base-384-22k-->cnx_cls
python -m torch.distributed.launch --nproc_per_node=1 main.py --model convnext_base --drop_path 0.2 --input_size 384 --batch_size 10 --lr 5e-5 --update_freq 2 --warmup_epochs 0 --epochs 10000 --weight_decay 1e-8 --layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 --data_path /data/ipsc/well3/all_frames_roi/swc/ext_reorg_roi_g2_0_37 --output_dir log/ext_reorg_roi_g2_0_37-base-384-22k --nb_classes 2 --eval true --resume log/ext_reorg_roi_g2_0_37-base-384-22k/checkpoint-857.pth

<a id="ext_reorg_roi_g2_16_53___base_384_22k_"></a>
## ext_reorg_roi_g2_16_53       @ base-384-22k-->cnx_cls
python -m torch.distributed.launch --nproc_per_node=3 main.py --model convnext_base --drop_path 0.2 --input_size 384 --batch_size 4 --lr 5e-5 --update_freq 2 --warmup_epochs 0 --epochs 10000 --weight_decay 1e-8 --layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 --finetune  pretrained/convnext_base_22k_1k_384.pth --data_path /data/ipsc/well3/all_frames_roi/swc/ext_reorg_roi_g2_16_53 --output_dir log/ext_reorg_roi_g2_16_53-base-384-22k --nb_classes 2
<a id="eval___ext_reorg_roi_g2_16_53_base_384_22_k_"></a>
### eval       @ ext_reorg_roi_g2_16_53/base-384-22k-->cnx_cls
python -m torch.distributed.launch --nproc_per_node=1 main.py --model convnext_base --drop_path 0.2 --input_size 384 --batch_size 10 --lr 5e-5 --update_freq 2 --warmup_epochs 0 --epochs 10000 --weight_decay 1e-8 --layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 --data_path /data/ipsc/well3/all_frames_roi/swc/ext_reorg_roi_g2_16_53 --output_dir log/ext_reorg_roi_g2_16_53-base-384-22k --nb_classes 2 --eval true --resume log/ext_reorg_roi_g2_16_53-base-384-22k/checkpoint-1596.pth

<a id="ext_reorg_roi_g2_16_53_masked___base_384_22k_"></a>
## ext_reorg_roi_g2_16_53_masked       @ base-384-22k-->cnx_cls
python -m torch.distributed.launch --nproc_per_node=3 main.py --model convnext_base --drop_path 0.2 --input_size 384 --batch_size 4 --lr 5e-5 --update_freq 2 --warmup_epochs 0 --epochs 10000 --weight_decay 1e-8 --layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 --finetune  pretrained/convnext_base_22k_1k_384.pth --data_path /data/ipsc/well3/all_frames_roi/swc/ext_reorg_roi_g2_16_53_masked --output_dir log/ext_reorg_roi_g2_16_53_masked-base-384-22k --nb_classes 2
<a id="eval___ext_reorg_roi_g2_16_53_masked_base_384_22k_"></a>
### eval       @ ext_reorg_roi_g2_16_53_masked/base-384-22k-->cnx_cls
python -m torch.distributed.launch --nproc_per_node=1 main.py --model convnext_base --drop_path 0.2 --input_size 384 --batch_size 10 --lr 5e-5 --update_freq 2 --warmup_epochs 0 --epochs 10000 --weight_decay 1e-8 --layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 --data_path /data/ipsc/well3/all_frames_roi/swc/ext_reorg_roi_g2_16_53_masked --output_dir log/ext_reorg_roi_g2_16_53_masked-base-384-22k --nb_classes 2 --eval true --resume log/ext_reorg_roi_g2_16_53_masked-base-384-22k/checkpoint-1596.pth

<a id="ext_reorg_roi_g2_54_126___base_384_22k_"></a>
## ext_reorg_roi_g2_54_126       @ base-384-22k-->cnx_cls
python -m torch.distributed.launch --nproc_per_node=3 main.py --model convnext_base --drop_path 0.2 --input_size 384 --batch_size 4 --lr 5e-5 --update_freq 2 --warmup_epochs 0 --epochs 10000 --weight_decay 1e-8 --layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 --finetune  pretrained/convnext_base_22k_1k_384.pth --data_path /data/ipsc/well3/all_frames_roi/swc/ext_reorg_roi_g2_54_126 --output_dir log/ext_reorg_roi_g2_54_126-base-384-22k --nb_classes 2
<a id="eval___ext_reorg_roi_g2_54_126_base_384_22k_"></a>
### eval       @ ext_reorg_roi_g2_54_126/base-384-22k-->cnx_cls
python -m torch.distributed.launch --nproc_per_node=1 main.py --model convnext_base --drop_path 0.2 --input_size 384 --batch_size 10 --lr 5e-5 --update_freq 2 --warmup_epochs 0 --epochs 10000 --weight_decay 1e-8 --layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 --data_path /data/ipsc/well3/all_frames_roi/swc/ext_reorg_roi_g2_54_126 --output_dir log/ext_reorg_roi_g2_54_126-base-384-22k --nb_classes 2 --eval true --resume log/ext_reorg_roi_g2_54_126-base-384-22k/checkpoint-1065.pth
<a id="on_g2_0_15___ext_reorg_roi_g2_54_126_base_384_22k_"></a>
### on_g2_0_15       @ ext_reorg_roi_g2_54_126/base-384-22k-->cnx_cls
python -m torch.distributed.launch --nproc_per_node=1 main.py --model convnext_base --drop_path 0.2 --input_size 384 --batch_size 10 --lr 5e-5 --update_freq 2 --warmup_epochs 0 --epochs 10000 --weight_decay 1e-8 --layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 --data_path /data/ipsc/well3/all_frames_roi/swc/ext_reorg_roi_g2_16_53 --output_dir log/ext_reorg_roi_g2_54_126-base-384-22k/on_g2_0_15 --nb_classes 2 --eval true --resume log/ext_reorg_roi_g2_54_126-base-384-22k/checkpoint-1065.pth

<a id="large_384_22k_"></a>
# large-384-22k
<a id="ext_reorg_roi_g2_0_37___large_384_22_k_"></a>
## ext_reorg_roi_g2_0_37       @ large-384-22k-->cnx_cls
python -m torch.distributed.launch --nproc_per_node=3 main.py --model convnext_large --drop_path 0.2 --input_size 384 --batch_size 2 --lr 5e-5 --update_freq 2 --warmup_epochs 0 --epochs 10000 --weight_decay 1e-8 --layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 --finetune  pretrained/convnext_large_22k_1k_384.pth --data_path /data/ipsc/well3/all_frames_roi/swc/ext_reorg_roi_g2_0_37 --output_dir log/ext_reorg_roi_g2_0_37-large-384-22k --nb_classes 2

<a id="eval___ext_reorg_roi_g2_0_37_large_384_22_k_"></a>
### eval       @ ext_reorg_roi_g2_0_37/large-384-22k-->cnx_cls
python -m torch.distributed.launch --nproc_per_node=1 main.py --model convnext_large --drop_path 0.2 --input_size 384 --batch_size 3 --lr 5e-5 --update_freq 2 --warmup_epochs 0 --epochs 10000 --weight_decay 1e-8 --layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 --data_path /data/ipsc/well3/all_frames_roi/swc/ext_reorg_roi_g2_0_37 --output_dir log/ext_reorg_roi_g2_0_37-large-384-22k --nb_classes 2 --eval true --resume log/ext_reorg_roi_g2_0_37-large-384-22k/checkpoint-242.pth

<a id="ext_reorg_roi_g2_0_37_masked___large_384_22_k_"></a>
## ext_reorg_roi_g2_0_37_masked       @ large-384-22k-->cnx_cls
python -m torch.distributed.launch --nproc_per_node=3 main.py --model convnext_large --drop_path 0.2 --input_size 384 --batch_size 2 --lr 5e-5 --update_freq 2 --warmup_epochs 0 --epochs 10000 --weight_decay 1e-8 --layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 --finetune  pretrained/convnext_large_22k_1k_384.pth --data_path /data/ipsc/well3/all_frames_roi/swc/ext_reorg_roi_g2_0_37_masked --output_dir log/ext_reorg_roi_g2_0_37_masked-large-384-22k --nb_classes 2

<a id="eval___ext_reorg_roi_g2_0_37_masked_large_384_22k_"></a>
### eval       @ ext_reorg_roi_g2_0_37_masked/large-384-22k-->cnx_cls
python -m torch.distributed.launch --nproc_per_node=1 main.py --model convnext_large --drop_path 0.2 --input_size 384 --batch_size 3 --lr 5e-5 --update_freq 2 --warmup_epochs 0 --epochs 10000 --weight_decay 1e-8 --layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 --data_path /data/ipsc/well3/all_frames_roi/swc/ext_reorg_roi_g2_0_37_masked --output_dir log/ext_reorg_roi_g2_0_37_masked-large-384-22k --nb_classes 2 --eval true --resume log/ext_reorg_roi_g2_0_37_masked-large-384-22k/checkpoint-best.pth

<a id="ext_reorg_roi_g2_16_53___large_384_22_k_"></a>
## ext_reorg_roi_g2_16_53       @ large-384-22k-->cnx_cls
python -m torch.distributed.launch --nproc_per_node=3 main.py --model convnext_large --drop_path 0.2 --input_size 384 --batch_size 2 --lr 5e-5 --update_freq 2 --warmup_epochs 0 --epochs 10000 --weight_decay 1e-8 --layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 --finetune  pretrained/convnext_large_22k_1k_384.pth --data_path /data/ipsc/well3/all_frames_roi/swc/ext_reorg_roi_g2_16_53 --output_dir log/ext_reorg_roi_g2_16_53-large-384-22k --nb_classes 2

<a id="eval___ext_reorg_roi_g2_16_53_large_384_22k_"></a>
### eval       @ ext_reorg_roi_g2_16_53/large-384-22k-->cnx_cls
python -m torch.distributed.launch --nproc_per_node=1 main.py --model convnext_large --drop_path 0.2 --input_size 384 --batch_size 3 --lr 5e-5 --update_freq 2 --warmup_epochs 0 --epochs 10000 --weight_decay 1e-8 --layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 --data_path /data/ipsc/well3/all_frames_roi/swc/ext_reorg_roi_g2_16_53 --output_dir log/ext_reorg_roi_g2_16_53-large-384-22k --nb_classes 2 --eval true --resume log/ext_reorg_roi_g2_16_53-large-384-22k/checkpoint-247.pth

<a id="ext_reorg_roi_g2_16_53_masked___large_384_22_k_"></a>
## ext_reorg_roi_g2_16_53_masked       @ large-384-22k-->cnx_cls
python -m torch.distributed.launch --nproc_per_node=3 main.py --model convnext_large --drop_path 0.2 --input_size 384 --batch_size 2 --lr 5e-5 --update_freq 2 --warmup_epochs 0 --epochs 10000 --weight_decay 1e-8 --layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 --finetune  pretrained/convnext_large_22k_1k_384.pth --data_path /data/ipsc/well3/all_frames_roi/swc/ext_reorg_roi_g2_16_53_masked --output_dir log/ext_reorg_roi_g2_16_53_masked-large-384-22k --nb_classes 2

<a id="eval___ext_reorg_roi_g2_16_53_masked_large_384_22_k_"></a>
### eval       @ ext_reorg_roi_g2_16_53_masked/large-384-22k-->cnx_cls
python -m torch.distributed.launch --nproc_per_node=1 main.py --model convnext_large --drop_path 0.2 --input_size 384 --batch_size 3 --lr 5e-5 --update_freq 2 --warmup_epochs 0 --epochs 10000 --weight_decay 1e-8 --layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 --data_path /data/ipsc/well3/all_frames_roi/swc/ext_reorg_roi_g2_16_53_masked --output_dir log/ext_reorg_roi_g2_16_53_masked-large-384-22k --nb_classes 2 --eval true --resume log/ext_reorg_roi_g2_16_53_masked-large-384-22k/checkpoint-242.pth

<a id="ext_reorg_roi_g2_54_126___large_384_22_k_"></a>
## ext_reorg_roi_g2_54_126       @ large-384-22k-->cnx_cls
python -m torch.distributed.launch --nproc_per_node=3 main.py --model convnext_large --drop_path 0.2 --input_size 384 --batch_size 2 --lr 5e-5 --update_freq 2 --warmup_epochs 0 --epochs 10000 --weight_decay 1e-8 --layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 --finetune  pretrained/convnext_large_22k_1k_384.pth --data_path /data/ipsc/well3/all_frames_roi/swc/ext_reorg_roi_g2_54_126 --output_dir log/ext_reorg_roi_g2_54_126-large-384-22k --nb_classes 2

<a id="eval___ext_reorg_roi_g2_54_126_large_384_22_k_"></a>
### eval       @ ext_reorg_roi_g2_54_126/large-384-22k-->cnx_cls
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 main.py --model convnext_large --drop_path 0.2 --input_size 384 --batch_size 3 --lr 5e-5 --update_freq 2 --warmup_epochs 0 --epochs 10000 --weight_decay 1e-8 --layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 --data_path /data/ipsc/well3/all_frames_roi/swc/ext_reorg_roi_g2_54_126 --output_dir log/ext_reorg_roi_g2_54_126-large-384-22k --nb_classes 2 --eval true --resume log/ext_reorg_roi_g2_54_126-large-384-22k/checkpoint-116.pth
<a id="on_g2_0_15___eval_ext_reorg_roi_g2_54_126_large_384_22k_"></a>
#### on_g2_0_15       @ eval/ext_reorg_roi_g2_54_126/large-384-22k-->cnx_cls
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 main.py --model convnext_large --drop_path 0.2 --input_size 384 --batch_size 3 --lr 5e-5 --update_freq 2 --warmup_epochs 0 --epochs 10000 --weight_decay 1e-8 --layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 --data_path /data/ipsc/well3/all_frames_roi/swc/ext_reorg_roi_g2_16_53 --output_dir log/ext_reorg_roi_g2_54_126-large-384-22k/on_g2_0_15 --nb_classes 2 --eval true --resume log/ext_reorg_roi_g2_54_126-large-384-22k/checkpoint-116.pth


<a id="bug_s_"></a>
# bugs
ImportError: cannot import name 'container_abcs' from 'torch._six'
https://stackoverflow.com/a/70457724/10101014







