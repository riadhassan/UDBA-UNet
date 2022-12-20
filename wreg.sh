#!/usr/bin/env bash
#PBS -N Wreg_test
#PBS -l ncpus=1
#PBS -l mem=64GB
#PBS -l ngpus=1
#PBS -l gputype=T4
#PBS -l walltime=10:00:00
#PBS -o Wreg_test.out
#PBS -e Wreg_test_err.out

module load cuda/11.3.1
export CUDA_VISIBLE_DEVICES=0
source /home/nazib/miniconda3/etc/profile.d/conda.sh
conda activate medical
cd ~/Medical/code
python train.py --model_name Wreg_with_aug_ep_100 --loss Wreg