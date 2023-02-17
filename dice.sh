#!/usr/bin/env bash
#PBS -N dice_att
#PBS -l ncpus=1
#PBS -l mem=64GB
#PBS -l ngpus=1
#PBS -l gputype=A100
#PBS -l walltime=22:00:00
#PBS -o dice_test.out
#PBS -e dice_test_err.out

module load cuda/11.3.1
export CUDA_VISIBLE_DEVICES=0
source /home/nazib/miniconda3/etc/profile.d/conda.sh
conda activate medical
cd ~/Medical/MIDL_code
python train.py --model_name Dice_attn_raw --loss Dice --isprob yes 