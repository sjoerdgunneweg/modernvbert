import torch
from colpali_engine.models.modernvbert.modeling_modernvbert import ModernVBertForMaskedLM
from colpali_engine.models import SparseModernVBertM2, SparseModernVBertM2Processor
from colpali_engine.models.modernvbert.configuration_modernvbert import ModernVBertConfig
# Import your custom class

# 1. Define paths
base_ckpt = "ModernVBERT/modernvbert"
save_path = "/home/scur1709/modernvbert/models/sparsemodernvbert_initialization"

# 2. Load the source weights
print("Loading source weights...")
source_state_dict = ModernVBertForMaskedLM.from_pretrained(base_ckpt).state_dict()

# 3. Initialize your empty custom model
config = ModernVBertConfig.from_pretrained(base_ckpt) # Assuming config is compatible
model = SparseModernVBertM2(config)

# 4. Map the weights manually (The logic we discussed)
new_state_dict = {}
for key, value in source_state_dict.items():
    # A. Map to Vision Encoder (MaskedLM -> MaskedLM)
    # The keys match 1:1, just add the prefix
    new_state_dict[f"vision_encoder.model.{key}"] = value

    # B. Map to Text Encoder (MaskedLM -> Base Model)
    # We only want keys starting with 'model.', and we strip that prefix
    if key.startswith("model."):
        stripped_key = key[6:] # remove 'model.'
        new_state_dict[f"text_encoder.model.{stripped_key}"] = value

# 5. Load this new constructed state dict
print("Loading constructed state dict...")
missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
print(f"Missing keys: {missing}")
print(f"Unexpected keys: {unexpected}")

# 6. Save as a standard Hugging Face model
print(f"Saving canonical checkpoint to {save_path}...")
model.save_pretrained(save_path)

# load and save processor:
SparseModernVBERTM2Processor
print("Done!")