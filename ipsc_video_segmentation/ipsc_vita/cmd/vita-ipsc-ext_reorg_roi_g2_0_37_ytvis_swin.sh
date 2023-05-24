#!/bin/bash
#SBATCH --account=def-nilanjan
#SBATCH --nodes=1
#SBATCH --mem=16000M
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4

#SBATCH --job-name=vita-ipsc-ext_reorg_roi_g2_0_37-swin
#SBATCH --output=%x_%j.out

#SBATCH --time=0-24:00            # time (DD-HH:MM)

#SBATCH --mail-user=asingh1@ualberta.ca
#SBATCH --mail-type=ALL

module load cuda cudnn gcc python/3.8

source ~/venv/vita/bin/activate

nvidia-smi

python train_net_vita.py --resume --num-gpus 2 --config-file configs/youtubevis_2019/vita-ipsc-ext_reorg_roi_g2_0_37-vita_SWIN_bs8.yaml MODEL.WEIGHTS pretrained/vita_swin_coco.pth SOLVER.IMS_PER_BATCH 2



