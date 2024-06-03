#!/bin/bash

#SBATCH --job-name=yf
#SBATCH --dependency=singleton
#SBATCH --mail-user=shesterg@ttic.edu
#SBATCH --mail-type=ALL
#SBATCH --output=slurm-logs/slurm_%j.out


cd /share/data/2pals/shester/dinov2
source /share/data/2pals/shester/mc3/bin/activate dinov22

export TORCH_HOME=/share/data/2pals/shester/hf_cache


torchrun --nproc_per_node=8 dinov2/train/train.py --config-file dinov2/configs/yasl_faces.yaml --output-dir /share/data/2pals/shester/dinov2/output_yasl_faces

