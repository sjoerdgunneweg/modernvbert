from torch import nn
import torch
from colpali_engine.models.modernvbert.modeling_modernvbert import (
    ModernVBertForMaskedLM,
    ModernVBertPreTrainedModel,
)
from colpali_engine.utils.sparse_rep import SparseRep
import torch.nn.functional as F


class MaxPoolValue(nn.Module):
    """Max pooling over a specified dimension"""

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.max(inputs, dim=self.dim).values


class SparseModernVBertMLM(ModernVBertPreTrainedModel):
    """
    Sparse ColBERT-style retriever on top of ModernVBERT.

    Forward:
        - Runs ModernVBertModel to get last_hidden_state
        - Applies a linear "splade_head" (mlm) to project to vocab space
        - Applies SPLADE-style log(1 + ReLU)
        - Masks padding (and optionally non-image tokens)
        - Max-pools over sequence dimension → [B, V] sparse scores
    """

    _tied_weights_keys = [
        "model.model.text_model.embeddings.tok_embeddings.weight",
        "model.lm_head.weight"
    ]
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def __init__(self, config, mask_non_image_embeddings: bool = False, **kwargs):
        super().__init__(config=config)
        self.model = ModernVBertForMaskedLM(config, **kwargs)
        self.mask_non_image_embeddings = mask_non_image_embeddings
        self.main_input_name = "doc_input_ids"

        # Add max_pool layer:
        self.max_pool = MaxPoolValue(dim=1)  # row-wise max pooling


    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass through ModernVBert + SPLADE head.

        Args:
            *args, **kwargs: passed directly to ModernVBertModel.

        Expected keyword args:
            - input_ids:      [B, L]
            - attention_mask: [B, L]
            - pixel_values:   optional, if multimodal
        Returns:
            token_scores: [B, V] SPLADE-style sparse scores (before any
                          external normalization).
        """
        output = self.model(*args, **kwargs)
        # logits = output[0]  # (B, L, V+additional_vocab_size)
        logits = output.logits  # (B, L, V+additional_vocab_size)
        # print("logits", logits.shape, logits.dtype, logits)
        # print("min logits", logits.min().item(), "max logits", logits.max().item())

        logits = logits[:, :, : self.model.config.text_config.vocab_size]  # (B, L, V)
        logits = torch.log1p(F.softplus(logits))  # (B, L, V)
        # logits = torch.log1p(torch.softplus(logits))  # (B, L, V)

        # Remove padding tokens:
        if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
            mask = kwargs["attention_mask"].unsqueeze(-1).to(logits.dtype)  # (B, L, 1)
            logits = logits * mask

        # keep only image tokens (if multimodal and specified):
        if self.mask_non_image_embeddings and "pixel_values" in kwargs and "input_ids" in kwargs:
            # assuming config.image_token_id is the special token marking image features
            image_mask = (kwargs["input_ids"] == self.config.image_token_id).unsqueeze(-1)  # (B, L, 1)
            image_mask = image_mask.to(logits.dtype)
            logits = logits * image_mask

        # Max-pool over sequence dimension → [B, V]
        lex_weights = self.max_pool(logits)  # (B, V)
        return SparseRep(dense=lex_weights).to_dense()