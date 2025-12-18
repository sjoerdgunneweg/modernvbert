from torch import nn
import torch
from colpali_engine.models.modernvbert.modeling_modernvbert import (
    ModernVBertModel,
    ModernVBertPreTrainedModel,
)
from colpali_engine.utils.sparse_rep import SparseRep
import torch.nn.functional as F


class SparseModernVBertMLP(ModernVBertPreTrainedModel):
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def __init__(self, config, mask_non_image_embeddings: bool = False, scale=2.0, **kwargs):
        super().__init__(config=config)
        self.model = ModernVBertModel(config, **kwargs)
        self.mask_non_image_embeddings = mask_non_image_embeddings
        self.main_input_name = "doc_input_ids"
        self.vocab_size = self.model.config.text_config.vocab_size
        # Add the MLP head
        hidden_size = self.model.config.text_config.hidden_size
        self.mlp_head = nn.Linear(hidden_size, 1)
        self.scale = nn.Parameter(torch.tensor(float(scale)))

    def forward(self, *args, **kwargs):
        input_ids = kwargs.get("input_ids", None)

        # Forward ModernVBert
        output = self.model(*args, **kwargs)
        last_hidden_states = output[0]  # (B, L, H)
        token_weights = self.mlp_head(last_hidden_states).squeeze(-1)  # (B, L)

        # norm default: log(1 + Relu(x))
        token_weights = torch.log1p(F.softplus(token_weights))  # (B, L)

        invalid_mask = input_ids >= self.vocab_size
        token_weights = token_weights.masked_fill(invalid_mask, 0.0)
        safe_input_ids = input_ids.masked_fill(invalid_mask, 0)

        # Apply masks
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            token_weights = token_weights * attention_mask.to(token_weights.dtype)

        special_tokens_mask = kwargs.get("special_tokens_mask", None)
        if special_tokens_mask is not None:
            not_special = (~special_tokens_mask.bool()).to(token_weights.dtype)
            token_weights = token_weights * not_special

        # token_weights = token_weights * self.scale
        size = torch.tensor((token_weights.size(0), self.vocab_size), device=token_weights.device)

        return SparseRep(
            indices=safe_input_ids,
            values=token_weights,
            size=size,
        ).to_dense()
