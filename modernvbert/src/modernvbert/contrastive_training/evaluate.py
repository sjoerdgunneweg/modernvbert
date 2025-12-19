#!/usr/bin/env python
import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import mteb
from mteb.benchmarks import Benchmark
from mteb.overview import get_tasks
from mteb.models import coleurovbert_models, colmodernvbert_models, colvllama_models, colqwen_models, colpali_models, jina_models, jina_clip, colflor_models
from mteb.model_meta import ModelMeta

from mteb.benchmarks.benchmark import VidoreBenchmark
#--------------------------------------------
# from mteb.benchmarks import VIDORE, VIDORE_V2 # NOTE: commented out since these are not in mteb-vlm package
#--------------------------------------------

from config import load_config

MODELS_MODULES = [coleurovbert_models, colmodernvbert_models, colvllama_models, colqwen_models, colpali_models, jina_models, jina_clip, colflor_models]
smolmieb = Benchmark(
    name="MIEB(smol)",
    tasks=get_tasks(
        tasks=[
        # Document retrieval
        "VidoreArxivQARetrieval",
        "VidoreDocVQARetrieval",
        "VidoreInfoVQARetrieval",
        "VidoreTabfquadRetrieval",
        "VidoreTatdqaRetrieval",
        "VidoreShiftProjectRetrieval",
        "VidoreSyntheticDocQAAIRetrieval",
        "VidoreSyntheticDocQAEnergyRetrieval",
        "VidoreSyntheticDocQAGovernmentReportsRetrieval",
        "VidoreSyntheticDocQAHealthcareIndustryRetrieval",
        "Vidore2ESGReportsRetrieval",
        "Vidore2EconomicsReportsRetrieval",
        "Vidore2BioMedicalLecturesRetrieval",
        "Vidore2ESGReportsHLRetrieval",
        # Caption retrieval.
        "MSCOCOT2IRetrieval",
        "MSCOCOI2TRetrieval",
        "Flickr30kT2IRetrieval",
        "Flickr30kI2TRetrieval",
        # Image class.
        "Caltech101",
        "Caltech101ZeroShot",
        "DTD",
        "DTDZeroShot",
        "FER2013",
        "FER2013ZeroShot",
        "EuroSAT",
        "EuroSATZeroShot",
        "OxfordFlowersClassification",
        "OxfordPets",
        "StanfordCars",
        "Food101Classification",
        ],
    ),
    description="""MIEB(lite) is a comprehensive image embeddings benchmark, spanning 10 task types, covering 51 tasks.
    This is a lite version of MIEB(Multilingual), designed to be run at a fraction of the cost while maintaining
    relative rank of models.""",
    reference="https://arxiv.org/abs/2504.10471",
    contacts=["gowitheflow-1998", "isaac-chung"],
    citation=r"""
@article{xiao2025mieb,
  author = {Chenghao Xiao and Isaac Chung and Imene Kerboua and Jamie Stirling and Xin Zhang and Márton Kardos and Roman Solomatin and Noura Al Moubayed and Kenneth Enevoldsen and Niklas Muennighoff},
  doi = {10.48550/ARXIV.2504.10471},
  journal = {arXiv preprint arXiv:2504.10471},
  publisher = {arXiv},
  title = {MIEB: Massive Image Embedding Benchmark},
  url = {https://arxiv.org/abs/2504.10471},
  year = {2025},
}
""",
)

# ---------------------------
# CLI
# ---------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MTEB evaluation with a config file.")
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to a YAML or JSON config file.",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--ViDoRe_V1", "-v1",
        action="store_true",
        help="Use ViDoRe V1 benchmark instead of ViDoRe V2.",
    )
    group.add_argument(
        "--ViDoRe_V2", "-v2",
        action="store_true",
        help="Use ViDoRe V2 benchmark instead of ViDoRe V1.",
    )

    return parser.parse_args()

def str_to_torch_dtype(dtype_str: str):
    # e.g. "float16" -> torch.float16
    try:
        return getattr(torch, dtype_str)
    except AttributeError as e:
        raise ValueError(f"Unknown torch dtype: {dtype_str}") from e

def str_to_model_class(model_cls_str: str):
    for module in MODELS_MODULES:
        if hasattr(module, model_cls_str):
            return getattr(module, model_cls_str)
    raise ValueError(f"Unknown model class: {model_cls_str}. Available classes: {[m for mod in MODELS_MODULES for m in dir(mod) if not m.startswith('_')]}.")
# ---------------------------
# Main
# ---------------------------
def main(cfg, args) -> None:
    # --- Logging ---
    logging.getLogger("mteb").setLevel(logging.INFO)

    # --- Model metadata / loading ---
    model_class = str_to_model_class(cfg.eval_config.wrapper_cls)
    if cfg.eval_config.model_name_or_path:
        model_name_or_path = cfg.eval_config.model_name_or_path
    else:
        model_name_or_path = cfg.tr_args.output_dir + "/final"

    # if len(model_name_or_path.split("/")) > 2: # NOTE: removed this since path of locally saved model is longer than 2
    #     name = f"SmolVEncoder/{model_name_or_path.split('/')[-2]}"
    # else:
        # name = model_name_or_path.split("/")[-1]

    name = model_name_or_path

    print(f"Model class: {model_class}")
    print(f"Model name:  {name}")

    custom_model_meta = ModelMeta(
        loader=model_class,
        name=name,
        modalities=["image", "text"],
        framework=cfg.eval_config.framework,
        similarity_fn_name=cfg.eval_config.similarity_fn_name,
        use_instructions=True,
        revision=None,
        release_date=None,
        languages=None,
        n_parameters=None,
        memory_usage_mb=None,
        max_tokens=None,
        embed_dim=128,
        license="apache-2.0",
        open_weights=True,
        public_training_code=None,
        public_training_data=None,
        training_datasets=None,
        )

    custom_model = custom_model_meta.load_model(
        model_name=name,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

#-------------------------------------------------------------------------
    print("[INFO] Model loaded successfully:", custom_model)
#---------------------------------------------------------------------------

    if name == "ModernVBERT/colmodernvbert" or name == "ModernVBERT/bimodernvbert" or "colmodernvbert_reproduction" in name:
        print("[INFO] Adjusting image processor settings for ColModernVBert model...")
        custom_model.processor.image_processor.size["longest_edge"] = cfg.eval_config.encode_kwargs.pop("max_image_size", 2048)
        custom_model.processor.image_processor.do_resize = cfg.eval_config.encode_kwargs.pop("do_resize", True)

    # --- Load tasks ---
    # tasks = mteb.get_tasks(tasks=cfg.eval_config.tasks) # NOTE: this is unnused, came from original evaluate.py
    # tasks = smolmieb

    #-----------------------------------------------------------------------------------------

    VIDORE= Benchmark(
        name="ViDoRe(v1)",
        tasks=get_tasks(
            tasks=[
                "VidoreArxivQARetrieval",
                "VidoreDocVQARetrieval",
                "VidoreInfoVQARetrieval",
                "VidoreTabfquadRetrieval",
                "VidoreTatdqaRetrieval",
                "VidoreShiftProjectRetrieval",
                "VidoreSyntheticDocQAAIRetrieval",
                "VidoreSyntheticDocQAEnergyRetrieval",
                "VidoreSyntheticDocQAGovernmentReportsRetrieval",
                "VidoreSyntheticDocQAHealthcareIndustryRetrieval",
            ],
            languages=["eng"], # NOTE: set to english only as mentioned in the paper
        ),
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/abs/2407.01449",
        citation=r"""
    @article{faysse2024colpali,
    author = {Faysse, Manuel and Sibille, Hugues and Wu, Tony and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
    journal = {arXiv preprint arXiv:2407.01449},
    title = {ColPali: Efficient Document Retrieval with Vision Language Models},
    year = {2024},

    }
    """,
    )

    VIDORE_V2 = Benchmark(
        name="ViDoRe(v2)",
        tasks=get_tasks(
            tasks=[
                "Vidore2ESGReportsRetrieval",
                "Vidore2EconomicsReportsRetrieval",
                "Vidore2BioMedicalLecturesRetrieval",
                "Vidore2ESGReportsHLRetrieval",
            ],
            languages=["eng"], # NOTE: set to english only as mentioned in the paper
        ),
        description="Retrieve associated pages according to questions.",
        reference="https://arxiv.org/abs/2407.01449",
        citation=r"""
    @article{mace2025vidorev2,
    author = {Macé, Quentin and Loison António and Faysse, Manuel},
    journal = {arXiv preprint arXiv:2505.17166},
    title = {ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval},
    year = {2025},
    }
    """,
    )
    
    VIDORE_V3 = VidoreBenchmark(
        name="ViDoRe(v3)",
        display_name="ViDoRe V3",
        language_view=[
            "deu-Latn",
            "eng-Latn",
            "fra-Latn",
            "ita-Latn",
            "por-Latn",
            "spa-Latn",
        ],
        icon="https://cdn-uploads.huggingface.co/production/uploads/66e16a677c2eb2da5109fb5c/x99xqw__fl2UaPbiIdC_f.png",
        tasks=get_tasks(
            tasks=[
                "Vidore3FinanceEnRetrieval",
                "Vidore3IndustrialRetrieval",
                "Vidore3ComputerScienceRetrieval",
                "Vidore3PharmaceuticalsRetrieval",
                "Vidore3HrRetrieval",
                "Vidore3FinanceFrRetrieval",
                "Vidore3PhysicsRetrieval",
                "Vidore3EnergyRetrieval",
                "Vidore3TelecomRetrieval",
                "Vidore3NuclearRetrieval",
            ],
            languages=["eng"], # NOTE: set to english only as mentioned in the paper
        ),
        description="ViDoRe V3 sets a new industry gold standard for multi-modal, enterprise document visual retrieval evaluation. It addresses a critical challenge in production RAG systems: retrieving accurate information from complex, visually-rich documents. The benchmark includes both open and closed datasets: to submit results on private tasks, please [open an issue](https://github.com/embeddings-benchmark/mteb/issues?template=eval_request.yaml).",
        reference="https://huggingface.co/blog/QuentinJG/introducing-vidore-v3",
        citation=r"""
    @misc{mace2025vidorev3,
    author = {Macé, Quentin and Loison, Antonio and EDY, Antoine and Xing, Victor and Viaud, Gautier},
    day = {5},
    howpublished = {\url{https://huggingface.co/blog/QuentinJG/introducing-vidore-v3}},
    journal = {Hugging Face Blog},
    month = {November},
    publisher = {Hugging Face},
    title = {ViDoRe V3: a comprehensive evaluation of retrieval for enterprise use-cases},
    year = {2025},
    }
    """,
    )

#-----------------------------------------------------------------------------------------
    if args.ViDoRe_V1:
        tasks = VIDORE
    elif args.ViDoRe_V2:
        tasks = VIDORE_V2
    elif args.ViDoRe_V3:
        tasks = VIDORE_V3
#-----------------------------------------------------------------------------------------

    print(f"Tasks loaded: {tasks}")
    evaluator = mteb.MTEB(tasks=tasks)

    # --- Run evaluation ---
    encode_kwargs = {"batch_size": cfg.eval_config.batch_size}
    encode_kwargs.update(cfg.eval_config.encode_kwargs or {})

    evaluator.run(
        model=custom_model,
        verbosity=cfg.eval_config.verbosity,
        encode_kwargs=encode_kwargs,
        output_folder=cfg.eval_config.output_folder,
    )


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    main(cfg, args)
