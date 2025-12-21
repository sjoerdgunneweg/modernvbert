# SparseModernVBERT: A Sparse Visual Document Retriever 


<!-- [![ViDoRe](https://img.shields.io/badge/MTEB_ModernVBERT-100000?style=for-the-badge\&logo=github\&logoColor=white)](https://github.com/sjoerdgunneweg/mteb_modernvbert) -->

<!-- [![Code](https://img.shields.io/badge/Code-Reproducibility-blue?style=for-the-badge\&logo=github\&logoColor=white)](https://github.com/illuin-tech/modernvbert) -->

[[Source Paper]](https://arxiv.org/abs/2510.01149)
[[Source Codebase]](https://github.com/illuin-tech/modernvbert)
[[ViDoRe Leaderboard]](https://huggingface.co/spaces/vidore/vidore-leaderboard)

## Associated Paper

This repository contains the **reproducibility study and sparse extension** accompanying the paper:

> **SparseModernVBERT: A Sparse Visual Document Retriever**
> Ruben Figge, Sjoerd Gunneweg, Ken Maradiaga Rosales

The paper builds upon **ModernVBERT** and introduces a learned sparse retrieval formulation for **visual document retrieval (VDR)**. In addition, this repository provides a *faithful reproduction* of previously reported ModernVBERT and ColModernVBERT results on the ViDoRe benchmarks.

---

## Introduction

Visual Document Retrieval (VDR) requires jointly reasoning over **textual content**, **layout**, and **visual cues**. Recent approaches such as ColPali and ModernVBERT have demonstrated that vision-language models can effectively retrieve document pages directly from pixels, without relying on OCR pipelines.

In this work, we introduce **SparseModernVBERT**, a sparse visual retriever built on top of ModernVBERT. Our model maps multimodal representations into a **vocabulary-aligned sparse space**, enabling efficient inverted-index retrieval while retaining strong retrieval performance.

Key contributions:

* A **reproducibility study** of ModernVBERT and ColModernVBERT on ViDoRe v1 and v2
* Identification and correction of **evaluation and precision inconsistencies**
* An extension of the evaluation to **ViDoRe v3**
* A new **learned sparse retrieval (LSR)** formulation for visual document retrieval
* Extensive evaluation across dense, late-interaction, and sparse settings

# TODO zet v3 hier nog tussen

---

## Models

| Model                        | Late Interaction | Sparse | Params (B) | ViDoRe v1 | ViDoRe v2 (EN) | ViDoRe v3 |
| ---------------------------- | :--------------: | :----: | :--------: | :-------: | :------------: | :-------: |
| ColPali                      |         ✓        |   –    |    2.92    |   70.3    |      45.4      |   48.9    |
| ColQwen2.5                   |         ✓        |   –    |    3.75    |   89.6    |      61.9      |   59.6    |
| BiModernVBERT                |         –        |   –    |    0.25    |   63.6    |      35.7      |   30.1    |
| ColModernVBERT               |         ✓        |   –    |    0.25    |   80.4    |      55.3      |   44.1    |
| **SparseModernVBERT (ours)** |         ✓        |   ✓    |    0.25    |   TBD     |      TBD       |   TBD     |

---

## Setup

We used **Python 3.12** and **PyTorch 2.8**. The codebase is compatible with CUDA 12.8.

```bash
# Snellius-specific setup
module purge
module load 2025
module load Anaconda3/2025.06-1
module load CUDA/12.8.0

conda create -n modernvbert python=3.12.0
conda activate modernvbert

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128

cd modernvbert/colpali
pip install -e .

pip install flash-attn --no-build-isolation
pip install wandb
```

---

## Evaluation

### ViDoRe v1 & ViDoRe v2 Evaluation

Evaluation follows the **ModernVBERT ViDoRe protocol** using their fork of MTEB: [[mteb-vlm]](https://github.com/paultltc/mteb-vlm)

```bash
git clone git@github.com:paultltc/mteb-vlm.git
cd mteb-vlm
git switch a5102b8f0479eab07defe8376b1d758e7cf2b5cd --detach
pip install -e .
pip install dacite timm
```

Run the ViDoRe v1/v2 evaluation:
```bash
python modernvbert/src/modernvbert/contrastive_training/evaluate.py --config configs/evaluation/<CONFIG FILE> --<ViDoRe_V1|ViDoRe_V2>
```


### ViDoRe v3 Evaluation

MTEB-VLM does not yet support ViDoRe v3. Therefore, we use a more recent fork of MTEB: [[mteb_modernvbert]](https://github.com/sjoerdgunneweg/mteb_modernvbert)

First, uninstall `mteb-vlm`:
```bash
pip uninstall mteb
```
The install the updated fork:
```bash
git clone git@github.com:sjoerdgunneweg/mteb_modernvbert.git
cd mteb_modernvbert
git switch vbert
pip install -e .
```

Run the ViDoRe v3 evaluation (note: no `ViDoRe_V3` flag required):
```bash
python modernvbert/src/modernvbert/contrastive_training/evaluate_vidore_v3.py --config configs/evaluation/eval_colmodernvbert.yaml
```

### Aggregate the results:

```bash
cd modernvbert
python aggregate_ndcg_at_5.py -f <PATH_TO_RESULTS> -b <ViDoRe_V1|ViDoRe_V2|ViDoRe_V3>
```



---

## Reproducibility Notes ⚠️

During reproduction, we identified several issues in the original ModernVBERT evaluation pipeline:

* **Precision mismatch**: released checkpoints are stored in float32, while evaluation scripts cast to bfloat16
* **FlashAttention incompatibility** with released checkpoints
* Hardcoded paths and missing English-only evaluation support

We provide a **corrected evaluation pipeline** and report both uncorrected and corrected results in the paper.

---

## Citation

If you use this repository, please cite the original works:

```bibtex
@misc{teiletche2025modernvbert,
  title={ModernVBERT: Towards Smaller Visual Document Retrievers},
  author={Teiletche et al.},
  year={2025}
}

@misc{faysse2024colpali,
  title={ColPali: Efficient Document Retrieval with Vision Language Models},
  author={Faysse et al.},
  year={2024}
}
```

---

## Acknowledgements

This work builds directly on **ModernVBERT** and **ColPali**. We thank the original authors for releasing their code and for their assistance with reproduction questions.
