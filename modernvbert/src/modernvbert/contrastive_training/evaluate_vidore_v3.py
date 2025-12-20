#!/usr/bin/env python
import argparse
import logging

import torch
import mteb
from mteb.get_tasks import get_tasks

from mteb.models.model_implementations import colmodernvbert_models, colqwen_models, colpali_models
from mteb.models import ModelMeta

from mteb.benchmarks.benchmark import VidoreBenchmark

from config import load_config

MODELS_MODULES = [colmodernvbert_models, colqwen_models, colpali_models]

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
def main(cfg) -> None:
    # --- Logging ---
    logging.getLogger("mteb").setLevel(logging.INFO)

    # --- Model metadata / loading ---
    model_class = str_to_model_class(cfg.eval_config.wrapper_cls)
    if cfg.eval_config.model_name_or_path:
        model_name_or_path = cfg.eval_config.model_name_or_path
    else:
        model_name_or_path = cfg.tr_args.output_dir + "/final"

    name = model_name_or_path 

    print(f"Model class: {model_class}")
    print(f"Model name:  {name}")

    custom_model_meta = ModelMeta(
        loader=model_class,
        name=name,
        modalities=["image", "text"],
        framework=cfg.eval_config.framework,
        similarity_fn_name=cfg.eval_config.similarity_fn_name_vidore_v3,
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
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

#-------------------------------------------------------------------------
    print("[INFO] Model loaded successfully:", custom_model)
#---------------------------------------------------------------------------

    if name == "ModernVBERT/colmodernvbert" or name == "ModernVBERT/bimodernvbert" or "colmodernvbert_reproduction" in name:
        print("[INFO] Adjusting image processor settings for ColModernVBert model...")
        custom_model.processor.image_processor.size["longest_edge"] = cfg.eval_config.encode_kwargs.pop("max_image_size", 2048)
        custom_model.processor.image_processor.do_resize = cfg.eval_config.encode_kwargs.pop("do_resize", True)

#-----------------------------------------------------------------------------------------
    
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
                # "Vidore3FinanceFrRetrieval", # NOTE: commented out since not English
                # "Vidore3PhysicsRetrieval", # NOTE: commented out since not English
                # "Vidore3EnergyRetrieval", # NOTE: commented out since not English
                # "Vidore3TelecomRetrieval", # NOTE: these two datasets are disabled since no acces to these private datasets
                # "Vidore3NuclearRetrieval",
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

    tasks = VIDORE_V3

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
    main(cfg)
