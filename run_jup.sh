#!/bin/bash

#PBS -q normal
#PBS -j oe
#PBS -l select=1:ncpus=16:ngpus=1
#PBS -l walltime=6:00:00
#PBS -N jup
#PBS -P personal-elim078

cd scratch/code-r1
module load miniforge3
conda activate rl_cre

export CUDA_VISIBLE_DEVICES=0



# python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"

python setup.py