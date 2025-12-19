import torch
from colpali_engine.models.modernvbert.modeling_modernvbert import ModernVBertForMaskedLM
from colpali_engine.models import SparseModernVBertM2, SparseModernVBertM2Processor, SparseModernVBertMLM, SparseModernVBertMLMProcessor
from colpali_engine.models.modernvbert.configuration_modernvbert import ModernVBertConfig
# Import your custom class

# 1. Define paths
base_ckpt = "ModernVBERT/modernvbert"
save_path_m2 = "/home/scur1709/modernvbert/models/sparsemodernvbertm2_initialization"
save_path_mlm = "/home/scur1709/modernvbert/models/sparsemodernvbertmlm_initialization"

# 2. Load the source weights
print("Loading source weights...")
source_state_dict = ModernVBertForMaskedLM.from_pretrained(base_ckpt).state_dict()

############
# M2 Model
############
# 3. Initialize your empty custom model
config_m2 = ModernVBertConfig.from_pretrained(base_ckpt) # Assuming config is compatible
model_m2 = SparseModernVBertM2(config_m2)

# 4. Map the weights manually (The logic we discussed)
new_state_dict_m2 = {}
for key, value in source_state_dict.items():
    # A. Map to Vision Encoder (MaskedLM -> MaskedLM)
    # The keys match 1:1, just add the prefix
    new_state_dict_m2[f"vision_encoder.model.{key}"] = value.clone()

    # B. Map to Text Encoder (MaskedLM -> Base Model)
    # We only want keys starting with 'model.', and we strip that prefix
    if key.startswith("model."):
        stripped_key = key[6:] # remove 'model.'
        new_state_dict_m2[f"text_encoder.model.{stripped_key}"] = value.clone()

# 5. Load this new constructed state dict
print("Loading constructed state dict for M2...")
missing, unexpected = model_m2.load_state_dict(new_state_dict_m2, strict=False)
print(f"Missing keys: {missing}")
print(f"Unexpected keys: {unexpected}")

# 6. Save as a standard Hugging Face model
print(f"Saving canonical checkpoint to {save_path_m2}...")
model_m2.save_pretrained(save_path_m2)

# load and save processor:
SparseModernVBertM2Processor.from_pretrained(base_ckpt).save_pretrained(save_path_m2)

############
# MLM Model
############
config_mlm = ModernVBertConfig.from_pretrained(base_ckpt) # Assuming config is compatible
model_mlm = SparseModernVBertMLM(config_mlm)

# 4. Map the weights manually (The logic we discussed)
new_state_dict_mlm = {}
for key, value in source_state_dict.items():
    new_state_dict_mlm[f"model.{key}"] = value.clone()

# 5. Load this new constructed state dict
print("Loading constructed state dict for MLM...")
missing, unexpected = model_mlm.load_state_dict(new_state_dict_mlm, strict=False)
print(f"Missing keys: {missing}")
print(f"Unexpected keys: {unexpected}")

# 6. Save as a standard Hugging Face model
print(f"Saving canonical checkpoint to {save_path_mlm}...")
model_mlm.save_pretrained(save_path_mlm)

# load and save processor:
SparseModernVBertMLMProcessor.from_pretrained(base_ckpt).save_pretrained(save_path_mlm)

print("All done!")








print("Done!")