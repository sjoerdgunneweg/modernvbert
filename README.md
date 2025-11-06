# modernvbert

### Setup

Create and activate the Conda environment:

```bash
cd modernvbert
conda create -n modernvbert python==3.11
conda activate modernvbert
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e .
pip install flash-attn --no-build-isolation
```
