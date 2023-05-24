<!-- MarkdownTOC -->

- [idol       @ idol](#idol___idol_)
    - [db3_part12_ytvis_swinL       @ idol](#db3_part12_ytvis_swinl___idol_)
    - [db3_2_to_17_except_6_with_syn_ytvis_swinL       @ idol](#db3_2_to_17_except_6_with_syn_ytvis_swinl___idol_)
    - [db3_2_to_17_except_6_ytvis_swinL       @ idol](#db3_2_to_17_except_6_ytvis_swinl___idol_)
        - [on-september_5_2020       @ db3_2_to_17_except_6_ytvis_swinL/idol](#on_september_5_2020___db3_2_to_17_except_6_ytvis_swinl_ido_l_)
    - [db3_2_to_17_except_6_large_huge_ytvis_swinL       @ idol](#db3_2_to_17_except_6_large_huge_ytvis_swinl___idol_)
        - [on-september_5_2020       @ db3_2_to_17_except_6_large_huge_ytvis_swinL/idol](#on_september_5_2020___db3_2_to_17_except_6_large_huge_ytvis_swinl_idol_)
- [seqformer       @ seqformer-ipsc](#seqformer___seqformer_ipsc_)
        - [db3_2_to_17_except_6_ytvis_swinL       @ seqformer/](#db3_2_to_17_except_6_ytvis_swinl___seqformer_)

<!-- /MarkdownTOC -->

<a id="idol___idol_"></a>
# idol       @ idol-->vnext

<a id="db3_part12_ytvis_swinl___idol_"></a>
## db3_part12_ytvis_swinL       @ idol-->vnext_mj
python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-db3_part12_ytvis_swinL.yaml --num-gpus 2

CUDA_VISIBLE_DEVICES=1 python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-db3_part12_ytvis_swinL.yaml --num-gpus 1 

<a id="db3_2_to_17_except_6_with_syn_ytvis_swinl___idol_"></a>
## db3_2_to_17_except_6_with_syn_ytvis_swinL       @ idol-->vnext_mj
python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-db3_2_to_17_except_6_with_syn_ytvis_swinL.yaml --num-gpus 2

<a id="db3_2_to_17_except_6_ytvis_swinl___idol_"></a>
## db3_2_to_17_except_6_ytvis_swinL       @ idol-->vnext_mj
python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-db3_2_to_17_except_6_ytvis_swinL.yaml --num-gpus 2

<a id="on_september_5_2020___db3_2_to_17_except_6_ytvis_swinl_ido_l_"></a>
### on-september_5_2020       @ db3_2_to_17_except_6_ytvis_swinL/idol-->vnext_mj
PYTORCH_NO_CUDA_MEMORY_CACHING=1 python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-db3_2_to_17_except_6_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/idol-ytvis-mj_rock-db3_2_to_17_except_6/model_0151999.pth SOLVER.IMS_PER_BATCH 1

<a id="db3_2_to_17_except_6_large_huge_ytvis_swinl___idol_"></a>
## db3_2_to_17_except_6_large_huge_ytvis_swinL       @ idol-->vnext_mj
python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-db3_2_to_17_except_6_large_huge_ytvis_swinL.yaml --num-gpus 2

<a id="on_september_5_2020___db3_2_to_17_except_6_large_huge_ytvis_swinl_idol_"></a>
### on-september_5_2020       @ db3_2_to_17_except_6_large_huge_ytvis_swinL/idol-->vnext_mj
python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-db3_2_to_17_except_6_large_huge_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/idol-ytvis-mj_rock-db3_2_to_17_except_6_large_huge/model_0209999.pth SOLVER.IMS_PER_BATCH 1


<a id="seqformer___seqformer_ipsc_"></a>
# seqformer       @ seqformer-ipsc-->vnext
<a id="db3_2_to_17_except_6_ytvis_swinl___seqformer_"></a>
### db3_2_to_17_except_6_ytvis_swinL       @ seqformer/-->vnext_mj
python3 projects/SeqFormer/train_net.py --config-file projects/SeqFormer/configs/seqformer-ipsc-db3_2_to_17_except_6_ytvis_swinL.yaml --num-gpus 2

