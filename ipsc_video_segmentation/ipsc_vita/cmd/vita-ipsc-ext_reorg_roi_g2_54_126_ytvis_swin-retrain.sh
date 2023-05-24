#!/bin/bash
#SBATCH --account=def-nilanjan
#SBATCH --nodes=1
#SBATCH --mem=24000M
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4

#SBATCH --job-name=vita-ipsc-ext_reorg_roi_g2_54_126-swin-retrain
#SBATCH --output=%x_%j.out

#SBATCH --time=0-96:00            # time (DD-HH:MM)

#SBATCH --mail-user=asingh1@ualberta.ca
#SBATCH --mail-type=ALL

module load cuda cudnn gcc python/3.8

source ~/venv/vita/bin/activate

nvidia-smi

python train_net_vita.py --num-gpus 2 --config-file configs/ytvis19/vita-ipsc-ext_reorg_roi_g2_54_126-vita_SWIN_bs8-retrain.yaml MODEL.WEIGHTS pretrained/vita_swin_coco.pth SOLVER.IMS_PER_BATCH 2



