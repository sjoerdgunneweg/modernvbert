#!/bin/bash
#SBATCH --job-name=colmv_m2_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/colmv_m2_%j.out
#SBATCH --error=logs/colmv_m2_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Snellius-specific module setup
module purge
module load 2025
module load Anaconda3/2025.06-1
module load CUDA/12.8.0

# Activate conda environment
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate modernvbert

# Set cache directories to avoid repeated HF model downloads
export HF_HOME=/scratch/$USER/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
mkdir -p $HF_HOME

# Disable W&B to avoid login issues; set to "online" if you have wandb configured
export WANDB_MODE=offline

# Navigate to project root (adjust path if needed)
cd $HOME/ir2/modernvbert

# Number of GPUs available on this node
NUM_GPUS=4
MAIN_PORT=29502

echo "Starting training on $(hostname)"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"
echo "Using $NUM_GPUS GPUs"
echo "Config: colpali/scripts/configs/modernvbert/train_colmodernvbert_m2.yaml"
echo "Time: $(date)"

# Run training using the colpali training script with configue
python colpali/scripts/train/train_colbert.py colpali/scripts/configs/modernvbert/train_colmodernvbert_m2.yaml

echo "Training completed at $(date)"
