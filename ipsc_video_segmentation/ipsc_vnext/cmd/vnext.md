<!-- MarkdownTOC -->

- [idol](#ido_l_)
    - [all_frames_roi_g2_0_37_swinL       @ idol](#all_frames_roi_g2_0_37_swinl___idol_)
    - [all_frames_roi_g2_0_37_swinL-ytvis       @ idol](#all_frames_roi_g2_0_37_swinl_ytvis___idol_)
        - [on-all_frames_roi_g2_38_53       @ all_frames_roi_g2_0_37_swinL-ytvis/idol](#on_all_frames_roi_g2_38_53___all_frames_roi_g2_0_37_swinl_ytvis_ido_l_)
        - [on-all_frames_roi_g2_seq_1_38_53       @ all_frames_roi_g2_0_37_swinL-ytvis/idol](#on_all_frames_roi_g2_seq_1_38_53___all_frames_roi_g2_0_37_swinl_ytvis_ido_l_)
    - [ext_reorg_roi_g2_0_37       @ idol](#ext_reorg_roi_g2_0_37___idol_)
        - [on-g2_38_53       @ ext_reorg_roi_g2_0_37/idol](#on_g2_38_53___ext_reorg_roi_g2_0_37_idol_)
        - [on-g2_0_53       @ ext_reorg_roi_g2_0_37/idol](#on_g2_0_53___ext_reorg_roi_g2_0_37_idol_)
            - [probs       @ on-g2_0_53/ext_reorg_roi_g2_0_37/idol](#probs___on_g2_0_53_ext_reorg_roi_g2_0_37_ido_l_)
    - [ext_reorg_roi_g2_0_37-max_length-10       @ idol](#ext_reorg_roi_g2_0_37_max_length_10___idol_)
    - [ext_reorg_roi_g2_0_37-max_length-20       @ idol](#ext_reorg_roi_g2_0_37_max_length_20___idol_)
        - [on-g2_38_53       @ ext_reorg_roi_g2_0_37-max_length-20/idol](#on_g2_38_53___ext_reorg_roi_g2_0_37_max_length_20_idol_)
    - [ext_reorg_roi_g2_16_53       @ idol](#ext_reorg_roi_g2_16_53___idol_)
        - [on-g2_0_15       @ ext_reorg_roi_g2_16_53/idol](#on_g2_0_15___ext_reorg_roi_g2_16_53_ido_l_)
            - [sigmoid       @ on-g2_0_15/ext_reorg_roi_g2_16_53/idol](#sigmoid___on_g2_0_15_ext_reorg_roi_g2_16_53_idol_)
            - [incremental       @ on-g2_0_15/ext_reorg_roi_g2_16_53/idol](#incremental___on_g2_0_15_ext_reorg_roi_g2_16_53_idol_)
            - [probs       @ on-g2_0_15/ext_reorg_roi_g2_16_53/idol](#probs___on_g2_0_15_ext_reorg_roi_g2_16_53_idol_)
            - [incremental       @ on-g2_0_15/ext_reorg_roi_g2_16_53/idol](#incremental___on_g2_0_15_ext_reorg_roi_g2_16_53_idol__1)
                - [g2_0_53       @ incremental/on-g2_0_15/ext_reorg_roi_g2_16_53/idol](#g2_0_53___incremental_on_g2_0_15_ext_reorg_roi_g2_16_53_idol_)
    - [ext_reorg_roi_g2_54_126       @ idol](#ext_reorg_roi_g2_54_126___idol_)
        - [on-g2_0_53       @ ext_reorg_roi_g2_54_126/idol](#on_g2_0_53___ext_reorg_roi_g2_54_126_idol_)
            - [incremental       @ on-g2_0_53/ext_reorg_roi_g2_54_126/idol](#incremental___on_g2_0_53_ext_reorg_roi_g2_54_126_ido_l_)
        - [on-g2_0_15       @ ext_reorg_roi_g2_54_126/idol](#on_g2_0_15___ext_reorg_roi_g2_54_126_idol_)
            - [incremental       @ on-g2_0_15/ext_reorg_roi_g2_54_126/idol](#incremental___on_g2_0_15_ext_reorg_roi_g2_54_126_ido_l_)
- [seqformer       @ seqformer](#seqformer___seqforme_r_)
    - [all_frames_roi_g2_0_37_swinL-ytvis       @ seqformer](#all_frames_roi_g2_0_37_swinl_ytvis___seqforme_r_)
        - [on-all_frames_roi_g2_38_53       @ all_frames_roi_g2_0_37_swinL-ytvis/seqformer](#on_all_frames_roi_g2_38_53___all_frames_roi_g2_0_37_swinl_ytvis_seqformer_)
            - [model_0049999       @ on-all_frames_roi_g2_38_53/all_frames_roi_g2_0_37_swinL-ytvis/seqformer](#model_0049999___on_all_frames_roi_g2_38_53_all_frames_roi_g2_0_37_swinl_ytvis_seqforme_r_)
    - [ext_reorg_roi_g2_0_37       @ seqformer](#ext_reorg_roi_g2_0_37___seqforme_r_)
            - [on-g2_38_53       @ ext_reorg_roi_g2_0_37/seqformer](#on_g2_38_53___ext_reorg_roi_g2_0_37_seqforme_r_)
                - [probs       @ on-g2_38_53/ext_reorg_roi_g2_0_37/seqformer](#probs___on_g2_38_53_ext_reorg_roi_g2_0_37_seqforme_r_)
                - [topk-100       @ on-g2_38_53/ext_reorg_roi_g2_0_37/seqformer](#topk_100___on_g2_38_53_ext_reorg_roi_g2_0_37_seqforme_r_)
    - [ext_reorg_roi_g2_16_53       @ seqformer](#ext_reorg_roi_g2_16_53___seqforme_r_)
            - [on-g2_0_15       @ ext_reorg_roi_g2_16_53/seqformer](#on_g2_0_15___ext_reorg_roi_g2_16_53_seqformer_)
                - [probs       @ on-g2_0_15/ext_reorg_roi_g2_16_53/seqformer](#probs___on_g2_0_15_ext_reorg_roi_g2_16_53_seqforme_r_)
    - [ext_reorg_roi_g2_54_126       @ seqformer](#ext_reorg_roi_g2_54_126___seqforme_r_)
            - [on-g2_0_53       @ ext_reorg_roi_g2_54_126/seqformer](#on_g2_0_53___ext_reorg_roi_g2_54_126_seqforme_r_)
                - [incremental       @ on-g2_0_53/ext_reorg_roi_g2_54_126/seqformer](#incremental___on_g2_0_53_ext_reorg_roi_g2_54_126_seqformer_)
            - [on-g2_0_15       @ ext_reorg_roi_g2_54_126/seqformer](#on_g2_0_15___ext_reorg_roi_g2_54_126_seqforme_r_)
                - [incremental       @ on-g2_0_15/ext_reorg_roi_g2_54_126/seqformer](#incremental___on_g2_0_15_ext_reorg_roi_g2_54_126_seqformer_)

<!-- /MarkdownTOC -->
<a id="ido_l_"></a>
# idol

<a id="all_frames_roi_g2_0_37_swinl___idol_"></a>
## all_frames_roi_g2_0_37_swinL       @ idol-->vnext
python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-ipsc-all_frames_roi_g2_0_37_swinL.yaml --num-gpus 2 

<a id="all_frames_roi_g2_0_37_swinl_ytvis___idol_"></a>
## all_frames_roi_g2_0_37_swinL-ytvis       @ idol-->vnext
__cc__
```
salloc --nodes=1 --time=0:15:0 --account=def-nilanjan --gpus-per-node=1 --mem=16000M --cpus-per-task=4
salloc --nodes=1 --time=0:15:0 --account=def-nilanjan --gpus-per-node=2 --mem=16000M --cpus-per-task=4
exit

ln -s ~/scratch/ipsc_vnext_log/idol-ipsc-all_frames_roi_g2_0_37/ .

sbatch cmd/ipsc-all_frames_roi_g2_0_37_ytvis_swinL.sh

squeue -u asingh1
scancel

MAX_JOBS=1
```
python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-ipsc-all_frames_roi_g2_0_37_ytvis_swinL.yaml --num-gpus 1 --resume

<a id="on_all_frames_roi_g2_38_53___all_frames_roi_g2_0_37_swinl_ytvis_ido_l_"></a>
### on-all_frames_roi_g2_38_53       @ all_frames_roi_g2_0_37_swinL-ytvis/idol-->vnext
python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-ipsc-all_frames_roi_g2_0_37_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/idol-ipsc-all_frames_roi_g2_0_37/model_0056999.pth

<a id="on_all_frames_roi_g2_seq_1_38_53___all_frames_roi_g2_0_37_swinl_ytvis_ido_l_"></a>
### on-all_frames_roi_g2_seq_1_38_53       @ all_frames_roi_g2_0_37_swinL-ytvis/idol-->vnext
python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-ipsc-ext_reorg_roi_g2_0_37_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/idol-ipsc-all_frames_roi_g2_0_37/model_0056999.pth DATASETS.TEST ('ytvis-ipsc-all_frames_roi_g2_seq_1_38_53-test',)

<a id="ext_reorg_roi_g2_0_37___idol_"></a>
## ext_reorg_roi_g2_0_37       @ idol-->vnext
python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-ipsc-ext_reorg_roi_g2_0_37_ytvis_swinL.yaml --num-gpus 2 
```
sbatch cmd/ipsc-ext_reorg_roi_g2_0_37_ytvis_swinL.sh
```
<a id="on_g2_38_53___ext_reorg_roi_g2_0_37_idol_"></a>
### on-g2_38_53       @ ext_reorg_roi_g2_0_37/idol-->vnext
python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-ipsc-ext_reorg_roi_g2_0_37_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/idol-ipsc-ext_reorg_roi_g2_0_37/model_0098999.pth
__incremental__
CUDA_VISIBLE_DEVICES=0 python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-ipsc-ext_reorg_roi_g2_0_37_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/idol-ipsc-ext_reorg_roi_g2_0_37/model_0098999.pth TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_38_53-incremental OUT_SUFFIX incremental

<a id="on_g2_0_53___ext_reorg_roi_g2_0_37_idol_"></a>
### on-g2_0_53       @ ext_reorg_roi_g2_0_37/idol-->vnext
CUDA_VISIBLE_DEVICES=0 python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-ipsc-ext_reorg_roi_g2_0_37_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/idol-ipsc-ext_reorg_roi_g2_0_37/model_0098999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_53-incremental OUT_SUFFIX 0_53-incremental-probs

<a id="probs___on_g2_0_53_ext_reorg_roi_g2_0_37_ido_l_"></a>
#### probs       @ on-g2_0_53/ext_reorg_roi_g2_0_37/idol-->vnext
python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-ipsc-ext_reorg_roi_g2_0_37_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/idol-ipsc-ext_reorg_roi_g2_0_37/model_0098999.pth USE_PROBS 1 OUT_SUFFIX probs

<a id="ext_reorg_roi_g2_0_37_max_length_10___idol_"></a>
## ext_reorg_roi_g2_0_37-max_length-10       @ idol-->vnext
python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-ipsc-ext_reorg_roi_g2_0_37-max_length-10_ytvis_swinL.yaml --num-gpus 2 
```
sbatch cmd/ipsc-ext_reorg_roi_g2_0_37-max_length-10_ytvis_swinL.sh
```
<a id="ext_reorg_roi_g2_0_37_max_length_20___idol_"></a>
## ext_reorg_roi_g2_0_37-max_length-20       @ idol-->vnext
python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-ipsc-ext_reorg_roi_g2_0_37-max_length-20_ytvis_swinL.yaml --num-gpus 2 
```
sbatch cmd/ipsc-ext_reorg_roi_g2_0_37-max_length-20_ytvis_swinL.sh
```
<a id="on_g2_38_53___ext_reorg_roi_g2_0_37_max_length_20_idol_"></a>
### on-g2_38_53       @ ext_reorg_roi_g2_0_37-max_length-20/idol-->vnext
python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-ipsc-ext_reorg_roi_g2_0_37-max_length-20_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/idol-ipsc-ext_reorg_roi_g2_0_37-max_length-20/model_0098999.pth

<a id="ext_reorg_roi_g2_16_53___idol_"></a>
## ext_reorg_roi_g2_16_53       @ idol-->vnext
```
sbatch cmd/ipsc-ext_reorg_roi_g2_16_53_ytvis_swinL.sh
```

python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-ipsc-ext_reorg_roi_g2_16_53_ytvis_swinL.yaml --num-gpus 2 --resume

<a id="on_g2_0_15___ext_reorg_roi_g2_16_53_ido_l_"></a>
### on-g2_0_15       @ ext_reorg_roi_g2_16_53/idol-->vnext
<a id="sigmoid___on_g2_0_15_ext_reorg_roi_g2_16_53_idol_"></a>
#### sigmoid       @ on-g2_0_15/ext_reorg_roi_g2_16_53/idol-->vnext
python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-ipsc-ext_reorg_roi_g2_16_53_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/idol-ipsc-ext_reorg_roi_g2_16_53/model_0254999.pth
<a id="incremental___on_g2_0_15_ext_reorg_roi_g2_16_53_idol_"></a>
#### incremental       @ on-g2_0_15/ext_reorg_roi_g2_16_53/idol-->vnext
python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-ipsc-ext_reorg_roi_g2_16_53_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/idol-ipsc-ext_reorg_roi_g2_16_53/model_0254999.pth TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_15-incremental OUT_SUFFIX incremental

<a id="probs___on_g2_0_15_ext_reorg_roi_g2_16_53_idol_"></a>
#### probs       @ on-g2_0_15/ext_reorg_roi_g2_16_53/idol-->vnext
CUDA_VISIBLE_DEVICES=1 python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-ipsc-ext_reorg_roi_g2_16_53_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/idol-ipsc-ext_reorg_roi_g2_16_53/model_0254999.pth USE_PROBS 1 OUT_SUFFIX probs

CUDA_VISIBLE_DEVICES=1 python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-ipsc-ext_reorg_roi_g2_16_53_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/idol-ipsc-ext_reorg_roi_g2_16_53/model_0254999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_15
__-max_length-1-__
CUDA_VISIBLE_DEVICES=1 python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-ipsc-ext_reorg_roi_g2_16_53_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/idol-ipsc-ext_reorg_roi_g2_16_53/model_0254999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_15-max_length-1 OUT_SUFFIX max_length-1
__-max_length-2-__
CUDA_VISIBLE_DEVICES=1 python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-ipsc-ext_reorg_roi_g2_16_53_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/idol-ipsc-ext_reorg_roi_g2_16_53/model_0254999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_15-max_length-2 OUT_SUFFIX max_length-2
__-max_length-4-__
CUDA_VISIBLE_DEVICES=1 python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-ipsc-ext_reorg_roi_g2_16_53_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/idol-ipsc-ext_reorg_roi_g2_16_53/model_0254999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_15-max_length-4 OUT_SUFFIX max_length-4
__-max_length-8-__
CUDA_VISIBLE_DEVICES=1 python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-ipsc-ext_reorg_roi_g2_16_53_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/idol-ipsc-ext_reorg_roi_g2_16_53/model_0254999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_15-max_length-8 OUT_SUFFIX max_length-8

<a id="incremental___on_g2_0_15_ext_reorg_roi_g2_16_53_idol__1"></a>
#### incremental       @ on-g2_0_15/ext_reorg_roi_g2_16_53/idol-->vnext
CUDA_VISIBLE_DEVICES=1 python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-ipsc-ext_reorg_roi_g2_16_53_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/idol-ipsc-ext_reorg_roi_g2_16_53/model_0254999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_15-incremental OUT_SUFFIX incremental_probs

<a id="g2_0_53___incremental_on_g2_0_15_ext_reorg_roi_g2_16_53_idol_"></a>
##### g2_0_53       @ incremental/on-g2_0_15/ext_reorg_roi_g2_16_53/idol-->vnext
CUDA_VISIBLE_DEVICES=1 python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-ipsc-ext_reorg_roi_g2_16_53_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/idol-ipsc-ext_reorg_roi_g2_16_53/model_0254999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_53-incremental OUT_SUFFIX g2_0_53-incremental_probs

<a id="ext_reorg_roi_g2_54_126___idol_"></a>
## ext_reorg_roi_g2_54_126       @ idol-->vnext
```
sbatch cmd/ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL.sh
```

python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL.yaml --num-gpus 2 --resume

<a id="on_g2_0_53___ext_reorg_roi_g2_54_126_idol_"></a>
### on-g2_0_53       @ ext_reorg_roi_g2_54_126/idol-->vnext
CUDA_VISIBLE_DEVICES=0 python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/idol-ipsc-ext_reorg_roi_g2_54_126/model_0596999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_53 
__-max_length-1__
CUDA_VISIBLE_DEVICES=0 python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/idol-ipsc-ext_reorg_roi_g2_54_126/model_0596999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_53-max_length-1 OUT_SUFFIX max_length-1
__-max_length-2__
CUDA_VISIBLE_DEVICES=0 python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/idol-ipsc-ext_reorg_roi_g2_54_126/model_0596999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_53-max_length-2 OUT_SUFFIX max_length-2
__-max_length-4__
CUDA_VISIBLE_DEVICES=1 python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/idol-ipsc-ext_reorg_roi_g2_54_126/model_0596999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_53-max_length-4 OUT_SUFFIX max_length-4
__-max_length-8__
CUDA_VISIBLE_DEVICES=1 python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/idol-ipsc-ext_reorg_roi_g2_54_126/model_0596999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_53-max_length-8 OUT_SUFFIX max_length-8
__-max_length-19-__
CUDA_VISIBLE_DEVICES=1 python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/idol-ipsc-ext_reorg_roi_g2_54_126/model_0596999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_53-max_length-19 OUT_SUFFIX max_length-19

<a id="incremental___on_g2_0_53_ext_reorg_roi_g2_54_126_ido_l_"></a>
#### incremental       @ on-g2_0_53/ext_reorg_roi_g2_54_126/idol-->vnext
```
sbatch cmd/ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL_on_0_53_inc.sh
```
CUDA_VISIBLE_DEVICES=0 python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/idol-ipsc-ext_reorg_roi_g2_54_126/model_0596999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_53-incremental OUT_SUFFIX incremental_probs

```
bash cmd/batch/idol-54_126_on_0_53__0_15_max_length_inc.sh
```
__-max_length-2__
CUDA_VISIBLE_DEVICES=0 python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/idol-ipsc-ext_reorg_roi_g2_54_126/model_0596999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_53-max_length-2-incremental OUT_SUFFIX max_length-2-incremental_probs
__-max_length-10__
CUDA_VISIBLE_DEVICES=0 python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/idol-ipsc-ext_reorg_roi_g2_54_126/model_0596999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_53-max_length-10-incremental OUT_SUFFIX max_length-10-incremental_probs
__-max_length-20__
CUDA_VISIBLE_DEVICES=0 python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/idol-ipsc-ext_reorg_roi_g2_54_126/model_0596999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_53-max_length-20-incremental OUT_SUFFIX max_length-20-incremental_probs

<a id="on_g2_0_15___ext_reorg_roi_g2_54_126_idol_"></a>
### on-g2_0_15       @ ext_reorg_roi_g2_54_126/idol-->vnext
CUDA_VISIBLE_DEVICES=0 python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/idol-ipsc-ext_reorg_roi_g2_54_126/model_0596999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_15 OUT_SUFFIX g2_0_15
__-max_length-1__
CUDA_VISIBLE_DEVICES=1 python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/idol-ipsc-ext_reorg_roi_g2_54_126/model_0596999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_15-max_length-1 OUT_SUFFIX g2_0_15-max_length-1
__-max_length-2__
CUDA_VISIBLE_DEVICES=1 python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/idol-ipsc-ext_reorg_roi_g2_54_126/model_0596999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_15-max_length-2 OUT_SUFFIX g2_0_15-max_length-2
__-max_length-4__
CUDA_VISIBLE_DEVICES=0 python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/idol-ipsc-ext_reorg_roi_g2_54_126/model_0596999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_15-max_length-4 OUT_SUFFIX g2_0_15-max_length-4
__-max_length-8__
CUDA_VISIBLE_DEVICES=1 python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/idol-ipsc-ext_reorg_roi_g2_54_126/model_0596999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_15-max_length-8 OUT_SUFFIX g2_0_15-max_length-8

<a id="incremental___on_g2_0_15_ext_reorg_roi_g2_54_126_ido_l_"></a>
#### incremental       @ on-g2_0_15/ext_reorg_roi_g2_54_126/idol-->vnext
CUDA_VISIBLE_DEVICES=0 python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/idol-ipsc-ext_reorg_roi_g2_54_126/model_0596999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_15-incremental OUT_SUFFIX g2_0_15-incremental_probs
__-max_length-2__
CUDA_VISIBLE_DEVICES=0 python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/idol-ipsc-ext_reorg_roi_g2_54_126/model_0596999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_15-max_length-2-incremental OUT_SUFFIX g2_0_15-max_length-2-incremental_probs

```
mv inference_model_0596999_incremental_probs-max_length-2 inference_model_0596999_max_length-2-incremental_probs

mv inference_model_0596999_incremental_probs-max_length-10 inference_model_0596999_max_length-10-incremental_probs

mv inference_model_0596999_incremental_probs-max_length-20 inference_model_0596999_max_length-20-incremental_probs

mv inference_model_0596999_g2_0_15-incremental_probs-max_length-2 inference_model_0596999_g2_0_15_max_length-2-incremental_probs
```

<a id="seqformer___seqforme_r_"></a>
# seqformer       @ seqformer-->vnext
<a id="all_frames_roi_g2_0_37_swinl_ytvis___seqforme_r_"></a>
## all_frames_roi_g2_0_37_swinL-ytvis       @ seqformer-->vnext
sbatch cmd/seq-ipsc-all_frames_roi_g2_0_37_ytvis_swinL.sh

python3 projects/SeqFormer/train_net.py --config-file projects/SeqFormer/configs/seqformer-ipsc-all_frames_roi_g2_0_37_ytvis_swinL.yaml --num-gpus 2

<a id="on_all_frames_roi_g2_38_53___all_frames_roi_g2_0_37_swinl_ytvis_seqformer_"></a>
### on-all_frames_roi_g2_38_53       @ all_frames_roi_g2_0_37_swinL-ytvis/seqformer-->vnext
python3 projects/SeqFormer/train_net.py --config-file projects/SeqFormer/configs/seqformer-ipsc-all_frames_roi_g2_0_37_ytvis_swinL.yaml --num-gpus 1 --eval-only 

<a id="model_0049999___on_all_frames_roi_g2_38_53_all_frames_roi_g2_0_37_swinl_ytvis_seqforme_r_"></a>
#### model_0049999       @ on-all_frames_roi_g2_38_53/all_frames_roi_g2_0_37_swinL-ytvis/seqformer-->vnext
python projects/SeqFormer/train_net.py --config-file projects/SeqFormer/configs/seqformer-ipsc-all_frames_roi_g2_0_37_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/seqformer-ipsc-all_frames_roi_g2_0_37/model_0049999.pth

<a id="ext_reorg_roi_g2_0_37___seqforme_r_"></a>
## ext_reorg_roi_g2_0_37       @ seqformer-->vnext
```
sbatch cmd/seq-ipsc-ext_reorg_roi_g2_0_37_ytvis_swinL.sh
```
python3 projects/SeqFormer/train_net.py --config-file projects/SeqFormer/configs/seqformer-ipsc-ext_reorg_roi_g2_0_37_ytvis_swinL.yaml --num-gpus 2

<a id="on_g2_38_53___ext_reorg_roi_g2_0_37_seqforme_r_"></a>
#### on-g2_38_53       @ ext_reorg_roi_g2_0_37/seqformer-->vnext
CUDA_VISIBLE_DEVICES=1 python projects/SeqFormer/train_net.py --config-file projects/SeqFormer/configs/seqformer-ipsc-ext_reorg_roi_g2_0_37_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/seqformer-ipsc-ext_reorg_roi_g2_0_37/model_final.pth 
__incremental__
CUDA_VISIBLE_DEVICES=1 python projects/SeqFormer/train_net.py --config-file projects/SeqFormer/configs/seqformer-ipsc-ext_reorg_roi_g2_0_37_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/seqformer-ipsc-ext_reorg_roi_g2_0_37/model_final.pth TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_38_53-incremental OUT_SUFFIX incremental

<a id="probs___on_g2_38_53_ext_reorg_roi_g2_0_37_seqforme_r_"></a>
##### probs       @ on-g2_38_53/ext_reorg_roi_g2_0_37/seqformer-->vnext
python projects/SeqFormer/train_net.py --config-file projects/SeqFormer/configs/seqformer-ipsc-ext_reorg_roi_g2_0_37_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/seqformer-ipsc-ext_reorg_roi_g2_0_37/model_final.pth USE_PROBS 1 OUT_SUFFIX probs 
<a id="topk_100___on_g2_38_53_ext_reorg_roi_g2_0_37_seqforme_r_"></a>
##### topk-100       @ on-g2_38_53/ext_reorg_roi_g2_0_37/seqformer-->vnext
CUDA_VISIBLE_DEVICES=1 python projects/SeqFormer/train_net.py --config-file projects/SeqFormer/configs/seqformer-ipsc-ext_reorg_roi_g2_0_37_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/seqformer-ipsc-ext_reorg_roi_g2_0_37/model_final.pth OUT_SUFFIX topk-100 MODEL.SeqFormer.N_TOPK 100


<a id="ext_reorg_roi_g2_16_53___seqforme_r_"></a>
## ext_reorg_roi_g2_16_53       @ seqformer-->vnext
```
sbatch cmd/seq-ipsc-ext_reorg_roi_g2_16_53_ytvis_swinL.sh
```
python3 projects/SeqFormer/train_net.py --config-file projects/SeqFormer/configs/seqformer-ipsc-ext_reorg_roi_g2_16_53_ytvis_swinL.yaml --num-gpus 2 --resume

<a id="on_g2_0_15___ext_reorg_roi_g2_16_53_seqformer_"></a>
#### on-g2_0_15       @ ext_reorg_roi_g2_16_53/seqformer-->vnext
CUDA_VISIBLE_DEVICES=0 python projects/SeqFormer/train_net.py --config-file projects/SeqFormer/configs/seqformer-ipsc-ext_reorg_roi_g2_16_53_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/seqformer-ipsc-ext_reorg_roi_g2_16_53/model_0241999.pth 
__incremental__
CUDA_VISIBLE_DEVICES=1 python projects/SeqFormer/train_net.py --config-file projects/SeqFormer/configs/seqformer-ipsc-ext_reorg_roi_g2_16_53_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/seqformer-ipsc-ext_reorg_roi_g2_16_53/model_0241999.pth TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_15-incremental OUT_SUFFIX incremental

<a id="probs___on_g2_0_15_ext_reorg_roi_g2_16_53_seqforme_r_"></a>
##### probs       @ on-g2_0_15/ext_reorg_roi_g2_16_53/seqformer-->vnext
CUDA_VISIBLE_DEVICES=1 python projects/SeqFormer/train_net.py --config-file projects/SeqFormer/configs/seqformer-ipsc-ext_reorg_roi_g2_16_53_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/seqformer-ipsc-ext_reorg_roi_g2_16_53/model_0241999.pth USE_PROBS 1 OUT_SUFFIX probs 

CUDA_VISIBLE_DEVICES=1 python projects/SeqFormer/train_net.py --config-file projects/SeqFormer/configs/seqformer-ipsc-ext_reorg_roi_g2_16_53_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/seqformer-ipsc-ext_reorg_roi_g2_16_53/model_0241999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_15  
__-max_length-1-__
CUDA_VISIBLE_DEVICES=1 python projects/SeqFormer/train_net.py --config-file projects/SeqFormer/configs/seqformer-ipsc-ext_reorg_roi_g2_16_53_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/seqformer-ipsc-ext_reorg_roi_g2_16_53/model_0241999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_15-max_length-1 OUT_SUFFIX max_length-1 
__-max_length-2-__
CUDA_VISIBLE_DEVICES=1 python projects/SeqFormer/train_net.py --config-file projects/SeqFormer/configs/seqformer-ipsc-ext_reorg_roi_g2_16_53_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/seqformer-ipsc-ext_reorg_roi_g2_16_53/model_0241999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_15-max_length-2 OUT_SUFFIX max_length-2 
__-max_length-4-__
CUDA_VISIBLE_DEVICES=1 python projects/SeqFormer/train_net.py --config-file projects/SeqFormer/configs/seqformer-ipsc-ext_reorg_roi_g2_16_53_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/seqformer-ipsc-ext_reorg_roi_g2_16_53/model_0241999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_15-max_length-4 OUT_SUFFIX max_length-4 
__-max_length-8-__
CUDA_VISIBLE_DEVICES=1 python projects/SeqFormer/train_net.py --config-file projects/SeqFormer/configs/seqformer-ipsc-ext_reorg_roi_g2_16_53_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/seqformer-ipsc-ext_reorg_roi_g2_16_53/model_0241999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_15-max_length-8 OUT_SUFFIX max_length-8 

__incremental__
CUDA_VISIBLE_DEVICES=0 python projects/SeqFormer/train_net.py --config-file projects/SeqFormer/configs/seqformer-ipsc-ext_reorg_roi_g2_16_53_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/seqformer-ipsc-ext_reorg_roi_g2_16_53/model_0241999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_15-incremental OUT_SUFFIX incremental_probs

<a id="ext_reorg_roi_g2_54_126___seqforme_r_"></a>
## ext_reorg_roi_g2_54_126       @ seqformer-->vnext
```
sbatch cmd/seq-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL.sh
```
python3 projects/SeqFormer/train_net.py --config-file projects/SeqFormer/configs/seqformer-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL.yaml --num-gpus 2 --resume

<a id="on_g2_0_53___ext_reorg_roi_g2_54_126_seqforme_r_"></a>
#### on-g2_0_53       @ ext_reorg_roi_g2_54_126/seqformer-->vnext
```
sbatch cmd/seq-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL_on_0_53.sh
```
CUDA_VISIBLE_DEVICES=1 python projects/SeqFormer/train_net.py --config-file projects/SeqFormer/configs/seqformer-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/seqformer-ipsc-ext_reorg_roi_g2_54_126/model_0495999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_53  
__-max_length-1-__
CUDA_VISIBLE_DEVICES=0 python projects/SeqFormer/train_net.py --config-file projects/SeqFormer/configs/seqformer-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/seqformer-ipsc-ext_reorg_roi_g2_54_126/model_0495999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_53-max_length-1 OUT_SUFFIX max_length-1
__-max_length-2-__
CUDA_VISIBLE_DEVICES=0 python projects/SeqFormer/train_net.py --config-file projects/SeqFormer/configs/seqformer-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/seqformer-ipsc-ext_reorg_roi_g2_54_126/model_0495999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_53-max_length-2 OUT_SUFFIX max_length-2
__-max_length-4-__
CUDA_VISIBLE_DEVICES=0 python projects/SeqFormer/train_net.py --config-file projects/SeqFormer/configs/seqformer-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/seqformer-ipsc-ext_reorg_roi_g2_54_126/model_0495999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_53-max_length-4 OUT_SUFFIX max_length-4
__-max_length-8-__
CUDA_VISIBLE_DEVICES=0 python projects/SeqFormer/train_net.py --config-file projects/SeqFormer/configs/seqformer-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/seqformer-ipsc-ext_reorg_roi_g2_54_126/model_0495999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_53-max_length-8 OUT_SUFFIX max_length-8
__-max_length-19-__
```
sbatch cmd/seq-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL_on_0_53_max_length_19.sh
squeue -u asingh1
scancel


```
CUDA_VISIBLE_DEVICES=0 python projects/SeqFormer/train_net.py --config-file projects/SeqFormer/configs/seqformer-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/seqformer-ipsc-ext_reorg_roi_g2_54_126/model_0495999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_53-max_length-19 OUT_SUFFIX max_length-19

<a id="incremental___on_g2_0_53_ext_reorg_roi_g2_54_126_seqformer_"></a>
##### incremental       @ on-g2_0_53/ext_reorg_roi_g2_54_126/seqformer-->vnext
```
sbatch cmd/seq-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL_on_0_53_inc.sh
```
CUDA_VISIBLE_DEVICES=1 python projects/SeqFormer/train_net.py --config-file projects/SeqFormer/configs/seqformer-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/seqformer-ipsc-ext_reorg_roi_g2_54_126/model_0495999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_53-incremental OUT_SUFFIX incremental_probs 

```
bash cmd/batch/seq-54_126_on_0_53__0_15_max_length_inc.sh
```
__-max_length-2-__
CUDA_VISIBLE_DEVICES=1 python projects/SeqFormer/train_net.py --config-file projects/SeqFormer/configs/seqformer-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/seqformer-ipsc-ext_reorg_roi_g2_54_126/model_0495999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_53-max_length-2-incremental OUT_SUFFIX max_length-2-incremental_probs
__-max_length-10-__
CUDA_VISIBLE_DEVICES=1 python projects/SeqFormer/train_net.py --config-file projects/SeqFormer/configs/seqformer-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/seqformer-ipsc-ext_reorg_roi_g2_54_126/model_0495999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_53-max_length-10-incremental OUT_SUFFIX max_length-10-incremental_probs 
__-max_length-20-__
CUDA_VISIBLE_DEVICES=1 python projects/SeqFormer/train_net.py --config-file projects/SeqFormer/configs/seqformer-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/seqformer-ipsc-ext_reorg_roi_g2_54_126/model_0495999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_53-max_length-20-incremental OUT_SUFFIX max_length-20-incremental_probs

<a id="on_g2_0_15___ext_reorg_roi_g2_54_126_seqforme_r_"></a>
#### on-g2_0_15       @ ext_reorg_roi_g2_54_126/seqformer-->vnext
CUDA_VISIBLE_DEVICES=0 python projects/SeqFormer/train_net.py --config-file projects/SeqFormer/configs/seqformer-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/seqformer-ipsc-ext_reorg_roi_g2_54_126/model_0495999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_15 OUT_SUFFIX g2_0_15
__-max_length-1-__
CUDA_VISIBLE_DEVICES=1 python projects/SeqFormer/train_net.py --config-file projects/SeqFormer/configs/seqformer-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/seqformer-ipsc-ext_reorg_roi_g2_54_126/model_0495999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_15-max_length-1 OUT_SUFFIX g2_0_15-max_length-1
__-max_length-2-__
CUDA_VISIBLE_DEVICES=1 python projects/SeqFormer/train_net.py --config-file projects/SeqFormer/configs/seqformer-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/seqformer-ipsc-ext_reorg_roi_g2_54_126/model_0495999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_15-max_length-2 OUT_SUFFIX g2_0_15-max_length-2
__-max_length-4-__
CUDA_VISIBLE_DEVICES=1 python projects/SeqFormer/train_net.py --config-file projects/SeqFormer/configs/seqformer-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/seqformer-ipsc-ext_reorg_roi_g2_54_126/model_0495999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_15-max_length-4 OUT_SUFFIX g2_0_15-max_length-4
__-max_length-8-__
CUDA_VISIBLE_DEVICES=1 python projects/SeqFormer/train_net.py --config-file projects/SeqFormer/configs/seqformer-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/seqformer-ipsc-ext_reorg_roi_g2_54_126/model_0495999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_15-max_length-2 OUT_SUFFIX g2_0_15-max_length-8

<a id="incremental___on_g2_0_15_ext_reorg_roi_g2_54_126_seqformer_"></a>
##### incremental       @ on-g2_0_15/ext_reorg_roi_g2_54_126/seqformer-->vnext
CUDA_VISIBLE_DEVICES=1 python projects/SeqFormer/train_net.py --config-file projects/SeqFormer/configs/seqformer-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/seqformer-ipsc-ext_reorg_roi_g2_54_126/model_0495999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_15-incremental OUT_SUFFIX g2_0_15-incremental_probs
__-max_length-2-__
CUDA_VISIBLE_DEVICES=1 python projects/SeqFormer/train_net.py --config-file projects/SeqFormer/configs/seqformer-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/seqformer-ipsc-ext_reorg_roi_g2_54_126/model_0495999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_15-max_length-2-incremental OUT_SUFFIX g2_0_15-max_length-2-incremental_probs
__-max_length-4-__
CUDA_VISIBLE_DEVICES=0 python projects/SeqFormer/train_net.py --config-file projects/SeqFormer/configs/seqformer-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/seqformer-ipsc-ext_reorg_roi_g2_54_126/model_0495999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_15-max_length-4-incremental OUT_SUFFIX g2_0_15-max_length-4-incremental_probs
__-max_length-10-__
CUDA_VISIBLE_DEVICES=1 python projects/SeqFormer/train_net.py --config-file projects/SeqFormer/configs/seqformer-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/seqformer-ipsc-ext_reorg_roi_g2_54_126/model_0495999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_15-max_length-10-incremental OUT_SUFFIX g2_0_15-max_length-10-incremental_probs

```
mv inference_model_0495999_incremental_probs-max_length-2 inference_model_0495999_max_length-2-incremental_probs
mv inference_model_0495999_incremental_probs-max_length-10 inference_model_0495999_max_length-10-incremental_probs
mv inference_model_0495999_incremental_probs-max_length-20 inference_model_0495999_max_length-20-incremental_probs

mv inference_model_0495999_g2_0_15-incremental_probs-max_length-2 inference_model_0495999_g2_0_15-max_length-2-incremental_probs

mv inference_model_0495999_g2_0_15-max_length-20-incremental_probs inference_model_0495999_g2_0_15-max_length-2-incremental_probs

inference_model_0495999_g2_0_15-max_length-2-incremental_probs
```











