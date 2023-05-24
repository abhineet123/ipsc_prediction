#!/bin/bash
#SBATCH --account=def-nilanjan
#SBATCH --nodes=1
#SBATCH --mem=32000M
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4

#SBATCH --job-name=idol-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL
#SBATCH --output=%x_%j.out

#SBATCH --time=0-6:00            # time (DD-HH:MM)

#SBATCH --mail-user=asingh1@ualberta.ca
#SBATCH --mail-type=ALL

module load cuda cudnn gcc python/3.8

source ~/venv/vnext/bin/activate

nvidia-smi

python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/idol-ipsc-ext_reorg_roi_g2_54_126_ytvis_swinL.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS log/idol-ipsc-ext_reorg_roi_g2_54_126/model_0596999.pth USE_PROBS 1 TEST_NAME ytvis-ipsc-ext_reorg_roi_g2_0_53-incremental OUT_SUFFIX incremental_probs

