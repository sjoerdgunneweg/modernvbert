from torch import nn
import torch
from colpali_engine.models.modernvbert.modeling_modernvbert import (
    ModernVBertForMaskedLM,
    ModernVBertPreTrainedModel,
)

from colpali_engine.utils.sparse_rep import SparseRep


class MaxPoolValue(nn.Module):
    """Max pooling over a specified dimension"""

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.max(inputs, dim=self.dim).values


class ColModernVBertSparse(ModernVBertPreTrainedModel):
    """
    SPLADE-style sparse retriever on top of ModernVBERT.

    Forward:
        - Runs ModernVBertModel to get last_hidden_state
        - Applies a linear "splade_head" to project to vocab space
        - Applies SPLADE-style log(1 + ReLU)
        - Masks padding (and optionally non-image tokens)
        - Max-pools over sequence dimension → [B, V] sparse scores
    """

    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def __init__(self, config, mask_non_image_embeddings: bool = False, **kwargs):
        super().__init__(config=config)

        # Backbone encoder
        self.model = ModernVBertForMaskedLM(config, **kwargs)


        for p in self.model.parameters():
            p.requires_grad_(True)

        self.mask_non_image_embeddings = mask_non_image_embeddings
        self.main_input_name = "doc_input_ids"
        self.max_pool = MaxPoolValue(dim=1)  # row-wise max pooling

    def forward(self, input_ids=None, attention_mask=None, special_tokens_mask=None, pixel_values=None, **kwargs) -> SparseRep:
        """
        Returns:
            SparseRep with dense shape [B, V]
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, **kwargs)
        logits = outputs[0]  # (B, L, V)

        token_scores = torch.log1p(torch.softplus(logits))  # (B, L, V)

        # Mask out padding tokens
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(token_scores.dtype)  # (B, L, 1)
            token_scores = token_scores * mask

        # Mask out special tokens
        if special_tokens_mask is not None:
            special_mask = (1 - special_tokens_mask).unsqueeze(-1).to(token_scores.dtype)  # (B, L, 1)
            token_scores = token_scores * special_mask

       
        if self.mask_non_image_embeddings and pixel_values is not None and input_ids is not None:
            # assuming config.image_token_id is the special token marking image features
            image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1)  # (B, L, 1)
            image_mask = image_mask.to(token_scores.dtype)
            token_scores = token_scores * image_mask

        lex_weights = torch.max(token_scores, dim=1).values  # (B, V)
        lex_weights = lex_weights.clamp(min=1e-6)

        return SparseRep(dense=lex_weights)