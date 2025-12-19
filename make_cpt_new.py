import torch
import argparse
from pathlib import Path
from typing import Dict, Type

from colpali_engine.models.modernvbert.modeling_modernvbert import ModernVBertForMaskedLM
from colpali_engine.models import (
    SparseModernVBertM2,
    SparseModernVBertM2Processor,
    SparseModernVBertMLM,
    SparseModernVBertMLMProcessor
)
from colpali_engine.models.modernvbert.configuration_modernvbert import ModernVBertConfig

m2_checkpoint_path = Path("/home/scur1709/modernvbert/models/sparsemodernvbertm2_initialization")
mlm_checkpoint_path = Path("/home/scur1709/modernvbert/models/sparsemodernvbertmlm_initialization")

def map_m2_weights(source_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Mapping logic for the M2 architecture."""
    new_sd = {}
    for key, value in source_sd.items():
        # A. Map to Vision Encoder (1:1 with prefix)
        new_sd[f"vision_encoder.model.{key}"] = value.clone()
        # B. Map to Text Encoder (Strip 'model.' prefix)
        if key.startswith("model."):
            new_sd[f"text_encoder.model.{key[6:]}"] = value.clone()
    return new_sd

def map_mlm_weights(source_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Mapping logic for the MLM architecture."""
    return {f"model.{key}": value.clone() for key, value in source_sd.items()}

# Configuration Registry
MODEL_REGISTRY = {
    "m2": {
        "model_cls": SparseModernVBertM2,
        "processor_cls": SparseModernVBertM2Processor,
        "mapper": map_m2_weights,
        "default_path": str(m2_checkpoint_path)
    },
    "mlm": {
        "model_cls": SparseModernVBertMLM,
        "processor_cls": SparseModernVBertMLMProcessor,
        "mapper": map_mlm_weights,
        "default_path": str(mlm_checkpoint_path)
    }
}

def initialize_sparse_model(model_type: str, base_ckpt: str = "ModernVBERT/modernvbert", output_path: str = None):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(MODEL_REGISTRY.keys())}")

    specs = MODEL_REGISTRY[model_type]
    save_path = output_path or specs["default_path"]

    print(f"--- Initializing {model_type.upper()} Model ---")

    # 1. Load source weights
    print(f"Loading source weights from {base_ckpt}...")
    source_state_dict = ModernVBertForMaskedLM.from_pretrained(base_ckpt).state_dict()

    # 2. Initialize empty target model
    config = ModernVBertConfig.from_pretrained(base_ckpt)
    model = specs["model_cls"](config)

    # 3. Apply custom mapping
    print(f"Mapping weights for {model_type}...")
    new_state_dict = specs["mapper"](source_state_dict)

    # 4. Load state dict
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")

    # 5. Save Model and Processor
    print(f"Saving to {save_path}...")
    model.save_pretrained(save_path)
    specs["processor_cls"].from_pretrained(base_ckpt).save_pretrained(save_path)

    print(f"Successfully initialized {model_type} at {save_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize Sparse ModernVBERT models.")
    parser.add_argument("--type", type=str, choices=["m2", "mlm", "all"], default="all",
                        help="Which model variant to initialize")
    parser.add_argument("--base_ckpt", type=str, default="ModernVBERT/modernvbert")
    args = parser.parse_args()

    if args.type == "all":
        for m_type in MODEL_REGISTRY.keys():
            initialize_sparse_model(m_type, args.base_ckpt)
    else:
        initialize_sparse_model(args.type, args.base_ckpt)

    print("Done!")