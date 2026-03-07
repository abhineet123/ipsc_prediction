<!-- MarkdownTOC -->

- [mnist_mot-r50](#mnist_mot_r50_)
    - [n-1       @ mnist_mot-r50](#n_1___mnist_mot_r5_0_)
    - [n-1-no_fpn       @ mnist_mot-r50](#n_1_no_fpn___mnist_mot_r5_0_)
        - [test_1_10       @ n-1-no_fpn/mnist_mot-r50](#test_1_10___n_1_no_fpn_mnist_mot_r50_)
    - [n-3       @ mnist_mot-r50](#n_3___mnist_mot_r5_0_)
- [mnist_mot](#mnist_mot_)
    - [n-1       @ mnist_mot](#n_1___mnist_mo_t_)
    - [n-1-no_fpn       @ mnist_mot](#n_1_no_fpn___mnist_mo_t_)
        - [test       @ n-1-no_fpn/mnist_mot](#test___n_1_no_fpn_mnist_mot_)
            - [pool-0       @ test/n-1-no_fpn/mnist_mot](#pool_0___test_n_1_no_fpn_mnist_mo_t_)
            - [pool-2       @ test/n-1-no_fpn/mnist_mot](#pool_2___test_n_1_no_fpn_mnist_mo_t_)
            - [pool-4       @ test/n-1-no_fpn/mnist_mot](#pool_4___test_n_1_no_fpn_mnist_mo_t_)
            - [pool-8       @ test/n-1-no_fpn/mnist_mot](#pool_8___test_n_1_no_fpn_mnist_mo_t_)
            - [pool-16       @ test/n-1-no_fpn/mnist_mot](#pool_16___test_n_1_no_fpn_mnist_mo_t_)
            - [set_zero       @ test/n-1-no_fpn/mnist_mot](#set_zero___test_n_1_no_fpn_mnist_mo_t_)
                - [0,1,2       @ set_zero/test/n-1-no_fpn/mnist_mot](#0_1_2___set_zero_test_n_1_no_fpn_mnist_mot_)
                - [0,1       @ set_zero/test/n-1-no_fpn/mnist_mot](#0_1___set_zero_test_n_1_no_fpn_mnist_mot_)
                - [0       @ set_zero/test/n-1-no_fpn/mnist_mot](#0___set_zero_test_n_1_no_fpn_mnist_mot_)
                - [3       @ set_zero/test/n-1-no_fpn/mnist_mot](#3___set_zero_test_n_1_no_fpn_mnist_mot_)
                - [1,2,3       @ set_zero/test/n-1-no_fpn/mnist_mot](#1_2_3___set_zero_test_n_1_no_fpn_mnist_mot_)
                - [2,3       @ set_zero/test/n-1-no_fpn/mnist_mot](#2_3___set_zero_test_n_1_no_fpn_mnist_mot_)
        - [extract_features       @ n-1-no_fpn/mnist_mot](#extract_features___n_1_no_fpn_mnist_mot_)
            - [train       @ extract_features/n-1-no_fpn/mnist_mot](#train___extract_features_n_1_no_fpn_mnist_mo_t_)
                - [f0_max_16       @ train/extract_features/n-1-no_fpn/mnist_mot](#f0_max_16___train_extract_features_n_1_no_fpn_mnist_mo_t_)
                - [f0_max_4       @ train/extract_features/n-1-no_fpn/mnist_mot](#f0_max_4___train_extract_features_n_1_no_fpn_mnist_mo_t_)
            - [100_960       @ extract_features/n-1-no_fpn/mnist_mot](#100_960___extract_features_n_1_no_fpn_mnist_mo_t_)
                - [f0_max_16       @ 100_960/extract_features/n-1-no_fpn/mnist_mot](#f0_max_16___100_960_extract_features_n_1_no_fpn_mnist_mo_t_)
                - [f0_max_4       @ 100_960/extract_features/n-1-no_fpn/mnist_mot](#f0_max_4___100_960_extract_features_n_1_no_fpn_mnist_mo_t_)
    - [n-3       @ mnist_mot](#n_3___mnist_mo_t_)

<!-- /MarkdownTOC -->
<a id="mnist_mot_r50_"></a>
# mnist_mot-r50
<a id="n_1___mnist_mot_r5_0_"></a>
## n-1       @ mnist_mot-r50-->swin_det_mnist
CUDA_VISIBLE_DEVICES=0 tools/dist_train.sh configs/faster_rcnn/faster_rcnn_r50_fpn_1x_mnist_mot_rgb_512_1k_9600_1_var-rcnn.py 1 --cfg-options model.pretrained=pretrained/resnet50-19c8e357.pth model.backbone.use_ckpt=True data.samples_per_gpu=48 data.workers_per_gpu=6
<a id="n_1_no_fpn___mnist_mot_r5_0_"></a>
## n-1-no_fpn       @ mnist_mot-r50-->swin_det_mnist
PORT=29502 CUDA_VISIBLE_DEVICES=1 tools/dist_train.sh configs/faster_rcnn/faster_rcnn_r50_fpn_1x_mnist_mot_rgb_512_1k_9600_1_var-rcnn_no_fpn.py 1 --cfg-options model.pretrained=pretrained/resnet50-19c8e357.pth model.backbone.use_ckpt=True data.samples_per_gpu=48 data.workers_per_gpu=6

python3 tools/extract_features.py config=configs/faster_rcnn/faster_rcnn_r50_fpn_1x_mnist_mot_rgb_512_1k_9600_1_var-rcnn_no_fpn.py ckpt_name=best_bbox_mAP.pth set=MNIST_MOT_RGB_512x512_1_1000_9600_var seq=0
<a id="test_1_10___n_1_no_fpn_mnist_mot_r50_"></a>
### test_1_10       @ n-1-no_fpn/mnist_mot-r50-->swin_det_mnist
CUDA_VISIBLE_DEVICES=0 python3 tools/test.py config=configs/faster_rcnn/faster_rcnn_r50_fpn_1x_mnist_mot_rgb_512_1k_9600_1_var-rcnn_no_fpn.py ckpt_name=best_bbox_mAP.pth eval=bbox test_name=test_1_10 batch_size=8

[pool](batch/pool___test_n_1_no_fpn_mnist_mot_r5_0_.bsh)    

[set_zero](batch/1_2_3___set_zero_test_n_1_no_fpn_mnist_mot_r50_.bsh)

<a id="n_3___mnist_mot_r5_0_"></a>
## n-3       @ mnist_mot-r50-->swin_det_mnist
PORT=29501 CUDA_VISIBLE_DEVICES=1 tools/dist_train.sh configs/faster_rcnn/faster_rcnn_r50_fpn_1x_mnist_mot_rgb_512_1k_9600_3_var-rcnn.py 1 --init file:///tmp/faster_rcnn_r50_fpn_1x_mnist_mot_rgb_512_1k_9600_3_var-00656565 --cfg-options model.pretrained=pretrained/resnet50-19c8e357.pth model.backbone.use_ckpt=True data.samples_per_gpu=48 data.workers_per_gpu=6

```
--nproc_per_node=1 --master_port=29501 tools/train.py configs/faster_rcnn/faster_rcnn_r50_fpn_1x_mnist_mot_rgb_512_1k_9600_3_var-rcnn.py --cfg-options model.pretrained=pretrained/resnet50-19c8e357.pth model.backbone.use_ckpt=True data.samples_per_gpu=48 data.workers_per_gpu=6 --resume
```
<a id="mnist_mot_"></a>
# mnist_mot
<a id="n_1___mnist_mo_t_"></a>
## n-1       @ mnist_mot-->swin_det_mnist
tools/dist_train.sh configs/swin/mnist_mot_rgb_512_1k_9600_1_var-rcnn.py 2 --cfg-options model.pretrained=pretrained/swin_base_patch4_window12_384.pth model.backbone.use_ckpt=True data.samples_per_gpu=6 data.workers_per_gpu=6 --resume

python3 tools/test.py config=configs/swin/mnist_mot_rgb_512_1k_9600_1_var-rcnn.py ckpt=work_dirs/mnist_mot_rgb_512_1k_9600_1_var-rcnn/best_bbox_mAP.pth eval=bbox test_name=val

python3 tools/extract_features.py config=configs/swin/mnist_mot_rgb_512_1k_9600_1_var-rcnn.py ckpt_name=best_bbox_mAP.pth set=MNIST_MOT_RGB_512x512_1_1000_9600_var seq=0

<a id="n_1_no_fpn___mnist_mo_t_"></a>
## n-1-no_fpn       @ mnist_mot-->swin_det_mnist
CUDA_VISIBLE_DEVICES=0,1 tools/dist_train.sh configs/swin/mnist_mot_rgb_512_1k_9600_1_var-rcnn_no_fpn.py 2 --cfg-options model.pretrained=pretrained/swin_base_patch4_window12_384.pth model.backbone.use_ckpt=True data.samples_per_gpu=16 data.workers_per_gpu=6
```
--nproc_per_node=2 --master_port=29500 tools/train.py configs/swin/mnist_mot_rgb_512_1k_9600_1_var-rcnn_no_fpn.py --cfg-options model.pretrained=pretrained/swin_base_patch4_window12_384.pth model.backbone.use_ckpt=True data.samples_per_gpu=6 data.workers_per_gpu=6
```
<a id="test___n_1_no_fpn_mnist_mot_"></a>
### test       @ n-1-no_fpn/mnist_mot-->swin_det_mnist
<a id="pool_0___test_n_1_no_fpn_mnist_mo_t_"></a>
#### pool-0       @ test/n-1-no_fpn/mnist_mot-->swin_det_mnist
CUDA_VISIBLE_DEVICES=0 python3 tools/test.py config=configs/swin/mnist_mot_rgb_512_1k_9600_1_var-rcnn_no_fpn.py ckpt_name=best_bbox_mAP.pth eval=bbox test_name=test batch_size=8 pool=0
<a id="pool_2___test_n_1_no_fpn_mnist_mo_t_"></a>
#### pool-2       @ test/n-1-no_fpn/mnist_mot-->swin_det_mnist
CUDA_VISIBLE_DEVICES=0 python3 tools/test.py config=configs/swin/mnist_mot_rgb_512_1k_9600_1_var-rcnn_no_fpn.py ckpt_name=best_bbox_mAP.pth eval=bbox test_name=test batch_size=8 pool=2
<a id="pool_4___test_n_1_no_fpn_mnist_mo_t_"></a>
#### pool-4       @ test/n-1-no_fpn/mnist_mot-->swin_det_mnist
CUDA_VISIBLE_DEVICES=0 python3 tools/test.py config=configs/swin/mnist_mot_rgb_512_1k_9600_1_var-rcnn_no_fpn.py ckpt_name=best_bbox_mAP.pth eval=bbox test_name=test batch_size=8 pool=4
<a id="pool_8___test_n_1_no_fpn_mnist_mo_t_"></a>
#### pool-8       @ test/n-1-no_fpn/mnist_mot-->swin_det_mnist
CUDA_VISIBLE_DEVICES=1 python3 tools/test.py config=configs/swin/mnist_mot_rgb_512_1k_9600_1_var-rcnn_no_fpn.py ckpt_name=best_bbox_mAP.pth eval=bbox test_name=test batch_size=8 pool=8
<a id="pool_16___test_n_1_no_fpn_mnist_mo_t_"></a>
#### pool-16       @ test/n-1-no_fpn/mnist_mot-->swin_det_mnist
CUDA_VISIBLE_DEVICES=1 python3 tools/test.py config=configs/swin/mnist_mot_rgb_512_1k_9600_1_var-rcnn_no_fpn.py ckpt_name=best_bbox_mAP.pth eval=bbox test_name=test batch_size=8 pool=16

<a id="set_zero___test_n_1_no_fpn_mnist_mo_t_"></a>
#### set_zero       @ test/n-1-no_fpn/mnist_mot-->swin_det_mnist
<a id="0_1_2___set_zero_test_n_1_no_fpn_mnist_mot_"></a>
##### 0,1,2       @ set_zero/test/n-1-no_fpn/mnist_mot-->swin_det_mnist
CUDA_VISIBLE_DEVICES=1 python3 tools/test.py config=configs/swin/mnist_mot_rgb_512_1k_9600_1_var-rcnn_no_fpn.py ckpt_name=best_bbox_mAP.pth eval=bbox test_name=test_1_10 batch_size=8 set_zero=0,1,2
<a id="0_1___set_zero_test_n_1_no_fpn_mnist_mot_"></a>
##### 0,1       @ set_zero/test/n-1-no_fpn/mnist_mot-->swin_det_mnist
CUDA_VISIBLE_DEVICES=1 python3 tools/test.py config=configs/swin/mnist_mot_rgb_512_1k_9600_1_var-rcnn_no_fpn.py ckpt_name=best_bbox_mAP.pth eval=bbox test_name=test_1_10 batch_size=8 set_zero=0,1
<a id="0___set_zero_test_n_1_no_fpn_mnist_mot_"></a>
##### 0       @ set_zero/test/n-1-no_fpn/mnist_mot-->swin_det_mnist
CUDA_VISIBLE_DEVICES=0 python3 tools/test.py config=configs/swin/mnist_mot_rgb_512_1k_9600_1_var-rcnn_no_fpn.py ckpt_name=best_bbox_mAP.pth eval=bbox test_name=test_1_10 batch_size=8 set_zero=0
<a id="3___set_zero_test_n_1_no_fpn_mnist_mot_"></a>
##### 3       @ set_zero/test/n-1-no_fpn/mnist_mot-->swin_det_mnist
CUDA_VISIBLE_DEVICES=1 python3 tools/test.py config=configs/swin/mnist_mot_rgb_512_1k_9600_1_var-rcnn_no_fpn.py ckpt_name=best_bbox_mAP.pth eval=bbox test_name=test_1_10 batch_size=8 set_zero=3
<a id="1_2_3___set_zero_test_n_1_no_fpn_mnist_mot_"></a>
##### 1,2,3       @ set_zero/test/n-1-no_fpn/mnist_mot-->swin_det_mnist
CUDA_VISIBLE_DEVICES=0 python3 tools/test.py config=configs/swin/mnist_mot_rgb_512_1k_9600_1_var-rcnn_no_fpn.py ckpt_name=best_bbox_mAP.pth eval=bbox test_name=test_1_10 batch_size=8 set_zero=1,2,3

CUDA_VISIBLE_DEVICES=0 python3 tools/test.py config=configs/swin/mnist_mot_rgb_512_1k_9600_1_var-rcnn_no_fpn.py ckpt_name=best_bbox_mAP.pth eval=bbox test_name=test_1_10 batch_size=8 set_zero=1,2,3 pool=16

CUDA_VISIBLE_DEVICES=0 python3 tools/test.py config=configs/swin/mnist_mot_rgb_512_1k_9600_1_var-rcnn_no_fpn.py ckpt_name=best_bbox_mAP.pth eval=bbox test_name=test_1_10 batch_size=8 set_zero=1,2,3 pool=8

CUDA_VISIBLE_DEVICES=1 python3 tools/test.py config=configs/swin/mnist_mot_rgb_512_1k_9600_1_var-rcnn_no_fpn.py ckpt_name=best_bbox_mAP.pth eval=bbox test_name=test_1_10 batch_size=8 set_zero=1,2,3 pool=4

CUDA_VISIBLE_DEVICES=1 python3 tools/test.py config=configs/swin/mnist_mot_rgb_512_1k_9600_1_var-rcnn_no_fpn.py ckpt_name=best_bbox_mAP.pth eval=bbox test_name=test_1_10 batch_size=8 set_zero=1,2,3 pool=2

<a id="2_3___set_zero_test_n_1_no_fpn_mnist_mot_"></a>
##### 2,3       @ set_zero/test/n-1-no_fpn/mnist_mot-->swin_det_mnist
CUDA_VISIBLE_DEVICES=0 python3 tools/test.py config=configs/swin/mnist_mot_rgb_512_1k_9600_1_var-rcnn_no_fpn.py ckpt_name=best_bbox_mAP.pth eval=bbox test_name=test_1_10 batch_size=8 set_zero=2,3

<a id="extract_features___n_1_no_fpn_mnist_mot_"></a>
### extract_features       @ n-1-no_fpn/mnist_mot-->swin_det_mnist
<a id="train___extract_features_n_1_no_fpn_mnist_mo_t_"></a>
#### train       @ extract_features/n-1-no_fpn/mnist_mot-->swin_det_mnist
CUDA_VISIBLE_DEVICES=0 python3 tools/extract_features.py config=configs/swin/mnist_mot_rgb_512_1k_9600_1_var-rcnn_no_fpn.py ckpt_name=best_bbox_mAP.pth set=MNIST_MOT_RGB_512x512_1_1000_9600_var start_seq=0,1000 end_seq=99,1099 batch_size=32 test_name=train_480_2 @slide size=480 num=2 

<a id="f0_max_16___train_extract_features_n_1_no_fpn_mnist_mo_t_"></a>
##### f0_max_16       @ train/extract_features/n-1-no_fpn/mnist_mot-->swin_det_mnist
CUDA_VISIBLE_DEVICES=1 python3 tools/extract_features.py config=configs/swin/mnist_mot_rgb_512_1k_9600_1_var-rcnn_no_fpn.py ckpt_name=best_bbox_mAP.pth set=MNIST_MOT_RGB_512x512_1_1000_9600_var start_seq=0,1000 end_seq=99,1099 batch_size=24 test_name=train_480_2 reduce=f0_max_16 @slide size=480 num=2 
<a id="f0_max_4___train_extract_features_n_1_no_fpn_mnist_mo_t_"></a>
##### f0_max_4       @ train/extract_features/n-1-no_fpn/mnist_mot-->swin_det_mnist
CUDA_VISIBLE_DEVICES=0 python3 tools/extract_features.py config=configs/swin/mnist_mot_rgb_512_1k_9600_1_var-rcnn_no_fpn.py ckpt_name=best_bbox_mAP.pth set=MNIST_MOT_RGB_512x512_1_1000_9600_var start_seq=0,1000 end_seq=99,1099 batch_size=24 test_name=train_480_2 reduce=f0_max_4 @slide size=480 num=2 
__dbg__
CUDA_VISIBLE_DEVICES=0 python3 tools/extract_features.py config=configs/swin/mnist_mot_rgb_512_1k_9600_1_var-rcnn_no_fpn.py ckpt_name=best_bbox_mAP.pth set=MNIST_MOT_RGB_512x512_1_1000_9600_var start_seq=1000 end_seq=1000 batch_size=1 test_name=train_480_2 raw=1 pool=16 vis=0 @slide size=480 num=1 

<a id="100_960___extract_features_n_1_no_fpn_mnist_mo_t_"></a>
#### 100_960       @ extract_features/n-1-no_fpn/mnist_mot-->swin_det_mnist
<a id="f0_max_16___100_960_extract_features_n_1_no_fpn_mnist_mo_t_"></a>
##### f0_max_16       @ 100_960/extract_features/n-1-no_fpn/mnist_mot-->swin_det_mnist
CUDA_VISIBLE_DEVICES=1 python3 tools/extract_features.py config=configs/swin/mnist_mot_rgb_512_1k_9600_1_var-rcnn_no_fpn.py ckpt_name=best_bbox_mAP.pth set=MNIST_MOT_RGB_512x512_1_100_960_var batch_size=48 test_name=test_100_960_480 reduce=f0_max_16 @slide size=480 
<a id="f0_max_4___100_960_extract_features_n_1_no_fpn_mnist_mo_t_"></a>
##### f0_max_4       @ 100_960/extract_features/n-1-no_fpn/mnist_mot-->swin_det_mnist
CUDA_VISIBLE_DEVICES=0 python3 tools/extract_features.py config=configs/swin/mnist_mot_rgb_512_1k_9600_1_var-rcnn_no_fpn.py ckpt_name=best_bbox_mAP.pth set=MNIST_MOT_RGB_512x512_1_100_960_var batch_size=48 test_name=test_100_960_480 reduce=f0_max_4 out_dir=/data_ssd/MNIST_MOT_RGB_512x512_1_100_960_var @slide size=480

<a id="n_3___mnist_mo_t_"></a>
## n-3       @ mnist_mot-->swin_det_mnist
tools/dist_train.sh configs/swin/mnist_mot_rgb_512_1k_9600_3_var-rcnn.py 2 --cfg-options model.pretrained=pretrained/swin_base_patch4_window12_384.pth model.backbone.use_ckpt=True data.samples_per_gpu=6 data.workers_per_gpu=6
```
--nproc_per_node=2 --master_port=29500 tools/train.py configs/swin/mnist_mot_rgb_512_1k_9600_3_var-rcnn.py --cfg-options model.pretrained=pretrained/swin_base_patch4_window12_384.pth model.backbone.use_ckpt=True data.samples_per_gpu=12 data.workers_per_gpu=6 --no-validate
```



