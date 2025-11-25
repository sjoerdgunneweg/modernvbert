# modernvbert

### Setup

Create and activate the Conda environment:

```bash
# Snellius specific part--------------
# Allows for conda use and activates CUDA 12.8
module purge
module load 2025
module load Anaconda3/2025.06-1
module load CUDA/12.8.0
#-------------------------------------

conda create -n modernvbert python==3.12.0
conda activate modernvbert
# source activate modernvbert # for snellius

# installing torch:
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# cd into this repo and install the copali-engine in the following way:
cd modernvbert
pip install -e colpali

# install flash attention:
pip install flash-attn --no-build-isolation
```

### Eval

Evaluation uses the vbert branch from mteb. Use the following command to install mteb:

```bash
git clone git@github.com:paultltc/mteb-vlm.git
cd mteb-vlm
git switch vbert
pip install -e .  #-e is important as otherwise you get a can't find package error due to a missing __init__.py file in a subdirectory -_-
pip install dacite
```
