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
cd modernvbert/colpali
pip install -e .

# install flash attention:
pip install flash-attn --no-build-isolation

pip install wandb
```

### Eval

Evaluation uses the vbert branch from mteb. Use the following command to install mteb:

```bash
git clone git@github.com:paultltc/mteb-vlm.git
cd mteb-vlm
git switch a5102b8f0479eab07defe8376b1d758e7cf2b5cd --detach
pip install -e .  #-e is important as otherwise you get a can't find package error due to a missing __init__.py file in a subdirectory -_-
pip install dacite
pip install timm
```

Then run the aggregate script to get the aggregated results:

```bash
cd modernvbert
python aggregate_ndcg_at_5.py -f <PATH TO RESULTS> -b <BENCHMARK, CHOOSE FROM {ViDoRe_V1,ViDoRe_V2,ViDoRe_V3}>
```

Example:
```bash
python aggregate_ndcg_at_5.py -f results/ModernVBERT__colmodernvbert/no_revision_available -b ViDoRe_V1
```


### ViDoRe Leaderboard Repdruction Results

| Model | Late Interaction | Model Size (B) | ViDoRe (v1) | ViDoRe (v2, eng) | Average | Latency (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$\ge$ 1 B Parameters** | | | | | | |
| MoCa-3B (Chen et al., 2025) | | 3.75 | - | - | - | X |
| GME-Qwen2 (Zhang et al., 2025) | | 3.75 | - | - | - | X |
| VLM2Vec (Jiang et al., 2025) | | 4.15 | - | - | - | X |
| E-V (Jiang et al., 2024) | | 8.36 | - | - | - | X |
| ColPali (Faysse et al., 2025) | $\checkmark$ | 2.92 | 70.3 | 45.4 | 57.9 | - |
| ColQwen2.5 (Faysse et al., 2025) | $\checkmark$ | 3.75 | 89.6 | 61.9 | 75,8 | - |
| Jina-v4 (Günther et al., 2025) | $\checkmark$ | 3.75 | - | - | - | X |
| NemoRetriever-3B (Xu et al., 2025) | $\checkmark$ | 4.40 | - | - | - | - |
| **< 1 B Parameters** | | | | | | |
| Jina CLIP (Koukounas et al., 2024) | | 0.22 | - | - | - | - |
| BGE Visualized M3 (Zhou et al., 2024) | | 0.87 | - | - | - | - |
| SigLIP2-L-512/16 (Tschannen et al., 2025) | | 0.88 | - | - | - | - |
| ColFlor (Masry & Hoque, 2024) | $\checkmark$ | 0.17 | - | - | - | - |
| BiModernVBERT | | 0.25 | - | - | - | - |
| ColModernVBERT (Our Contrastive Training) | $\checkmark$ | 0.25 | 3.3 | 3.8 | 3.6 | - |
| ColModernVBERT (Theirs) | $\checkmark$ | 0.25 | 3.4 | 4.0 | 3.7 | - |


# repo information:
The starting points of these repo are:
- (modernvbert repo at specific commithash) https://github.com/illuin-tech/modernvbert/tree/d4727bde0525991585cc13ccdedecf32c69ed25c
- (colpali repo at vbert branch and specifc commithash) https://github.com/illuin-tech/colpali/tree/eb561da6dd665c42e90572c8488ad3c6e114a409
