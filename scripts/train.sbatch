#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=128:00:00
#SBATCH --mem=128GB 
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=vitae
#SBATCH --output=vitae_%j.out

singularity exec --nv \
	    --overlay /scratch/$USER/containers/overlay5.sqf \
		--overlay /scratch/work/public/imagenet/imagenet21k_resized.sqf \
	    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
	    /bin/bash -c "source /ext3/env.sh; cd /scratch/$USER/TorchFundations; \
		python -m torch_fundations autoencoder load-config configs/autoencoder.yml "