<!-- MarkdownTOC -->

- [db3_2_to_17_except_6](#db3_2_to_17_except_6_)
    - [r50       @ db3_2_to_17_except_6](#r50___db3_2_to_17_except_6_)
        - [coco       @ r50/db3_2_to_17_except_6](#coco___r50_db3_2_to_17_except_6_)
        - [on-september_5_2020       @ r50/db3_2_to_17_except_6](#on_september_5_2020___r50_db3_2_to_17_except_6_)
        - [on-part1       @ r50/db3_2_to_17_except_6](#on_part1___r50_db3_2_to_17_except_6_)
    - [r101       @ db3_2_to_17_except_6](#r101___db3_2_to_17_except_6_)
        - [coco       @ r101/db3_2_to_17_except_6](#coco___r101_db3_2_to_17_except_6_)
        - [on-september_5_2020       @ r101/db3_2_to_17_except_6](#on_september_5_2020___r101_db3_2_to_17_except_6_)

<!-- /MarkdownTOC -->

<a id="db3_2_to_17_except_6_"></a>
# db3_2_to_17_except_6
<a id="r50___db3_2_to_17_except_6_"></a>
## r50       @ db3_2_to_17_except_6-->vita_mj
<a id="coco___r50_db3_2_to_17_except_6_"></a>
### coco       @ r50/db3_2_to_17_except_6-->vita_mj
python train_net_vita.py --num-gpus 2 --config-file configs/ytvis19/vita-db3_2_to_17_except_6-R50_bs2.yaml MODEL.WEIGHTS pretrained/R50_coco.pth SOLVER.IMS_PER_BATCH 2

<a id="on_september_5_2020___r50_db3_2_to_17_except_6_"></a>
### on-september_5_2020       @ r50/db3_2_to_17_except_6-->vita_mj
PYTORCH_NO_CUDA_MEMORY_CACHING=1 CUDA_VISIBLE_DEVICES=0 python train_net_vita.py --config-file configs/ytvis19/vita-db3_2_to_17_except_6-R50_bs2.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/db3_2_to_17_except_6-R50/model_0139999.pth

<a id="on_part1___r50_db3_2_to_17_except_6_"></a>
### on-part1       @ r50/db3_2_to_17_except_6-->vita_mj
PYTORCH_NO_CUDA_MEMORY_CACHING=1 CUDA_VISIBLE_DEVICES=0 python train_net_vita.py --config-file configs/ytvis19/vita-db3_2_to_17_except_6-R50_bs2.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/db3_2_to_17_except_6-R50/model_0139999.pth


<a id="r101___db3_2_to_17_except_6_"></a>
## r101       @ db3_2_to_17_except_6-->vita_mj
<a id="coco___r101_db3_2_to_17_except_6_"></a>
### coco       @ r101/db3_2_to_17_except_6-->vita_mj
python train_net_vita.py --num-gpus 2 --config-file configs/ytvis19/vita-db3_2_to_17_except_6-R101_bs2.yaml MODEL.WEIGHTS pretrained/R101_coco.pth SOLVER.IMS_PER_BATCH 2

<a id="on_september_5_2020___r101_db3_2_to_17_except_6_"></a>
### on-september_5_2020       @ r101/db3_2_to_17_except_6-->vita_mj
PYTORCH_NO_CUDA_MEMORY_CACHING=1 CUDA_VISIBLE_DEVICES=1 python train_net_vita.py --config-file configs/ytvis19/vita-db3_2_to_17_except_6-R101_bs2.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/db3_2_to_17_except_6-R101/model_0134999.pth

