#!/bin/bash

for i in {0..125}; do
    cat >tempD_slurm_script_$i.sh <<EOL
#!/bin/bash
#SBATCH --job-name=hf$i
#SBATCH --dependency=singleton
#SBATCH --partition=gpu
#SBATCH -G1
#SBATCH --mail-user=shesterg@ttic.edu
#SBATCH --mail-type=FAIL
#SBATCH --output=slurm-logs/slurm_%j.out

env

cd /path/to/feature_extraction
source /path/mc3/bin/activate dinov2_env

export TORCH_HOME=/path/to/dino_original_model/hf_cache


python fe.py --index "$i"

EOL

    # submit the temporary slurm script  python _extract_frames.py --index "$i"
    sbatch tempD_slurm_script_$i.sh

    # remove the temporary slurm script
    rm tempD_slurm_script_$i.sh
done