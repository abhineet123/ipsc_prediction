#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/idol-ipsc-ext_reorg_roi_g2_54_126/model_0596999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_53-max_length-2-incremental OUT_SUFFIX max_length-2-incremental_probs

CUDA_VISIBLE_DEVICES=0 python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/idol-ipsc-ext_reorg_roi_g2_54_126/model_0596999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_53-max_length-10-incremental OUT_SUFFIX max_length-10-incremental_probs

CUDA_VISIBLE_DEVICES=0 python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/idol-ipsc-ext_reorg_roi_g2_54_126/model_0596999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_53-max_length-20-incremental OUT_SUFFIX max_length-20-incremental_probs

CUDA_VISIBLE_DEVICES=0 python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/idol-ipsc-ext_reorg_roi_g2_54_126/model_0596999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_15-max_length-2-incremental OUT_SUFFIX g2_0_15-max_length-2-incremental_probs