# modernvbert

### Setup

Create and activate the Conda environment:

```bash
cd modernvbert
conda create -n modernvbert python==3.11
source activate modernvbert
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e . # installing copali-engine
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
