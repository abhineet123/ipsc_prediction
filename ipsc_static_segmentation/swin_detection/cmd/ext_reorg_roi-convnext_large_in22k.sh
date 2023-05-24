#!/bin/bash
#SBATCH --account=def-nilanjan
#SBATCH --nodes=1
#SBATCH --mem=16000M
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4

#SBATCH --job-name=ipsc_2_class_ext_reorg_roi_g2_0_38-no_validate-convnext_large_in22k
#SBATCH --output=%x_%j.out

#SBATCH --time=0-48:00            # time (DD-HH:MM)

#SBATCH --mail-user=asingh1@ualberta.ca
#SBATCH --mail-type=ALL

module load cuda cudnn gcc python/3.7

source ~/venv/swin_i/bin/activate

nvidia-smi

tools/dist_train.sh configs/convnext/ipsc_2_class_ext_reorg_roi_g2_0_38-no_validate-convnext_large_in22k.py 2 --no-validate --cfg-options model.pretrained=pretrained/cascade_mask_rcnn_convnext_large_22k_3x.pth data.samples_per_gpu=4

