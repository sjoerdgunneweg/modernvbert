#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=contrastive_vbert_m2
#SBATCH --time=01:00:00
#SBATCH --output=output/%x_%j.out
#SBATCH --error=output/%x_%j.err

set -e

mkdir -p output

module purge
module load 2025
module load Anaconda3/2025.06-1
module load CUDA/12.8.0

cd $HOME/ir2/modernvbert
source activate modernvbert

export HF_HOME=$SCRATCH/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME

export WANDB_PROJECT=vbert_m2
export WANDB_RUN_GROUP=colmodernvbert-m2


srun python colpali/scripts/train/train_colbert.py \
  colpali/scripts/configs/modernvbert/train_colmodernvbert_m2.yaml

echo "Training completed at $(date)"