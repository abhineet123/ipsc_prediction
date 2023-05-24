#!/bin/bash
#SBATCH --account=def-nilanjan
#SBATCH --nodes=1
#SBATCH --mem=32000M
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4

#SBATCH --job-name=seqformer-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL
#SBATCH --output=%x_%j.out

#SBATCH --time=0-00:30           # time (DD-HH:MM)

#SBATCH --mail-user=asingh1@ualberta.ca
#SBATCH --mail-type=ALL

module load cuda cudnn gcc python/3.8

source ~/venv/vnext/bin/activate

nvidia-smi

CUDA_VISIBLE_DEVICES=0 python projects/SeqFormer/train_net.py --config-file projects/SeqFormer/configs/seqformer-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/seqformer-ipsc-ext_reorg_roi_g2_54_126/model_0495999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_53-max_length-19 OUT_SUFFIX max_length-19


