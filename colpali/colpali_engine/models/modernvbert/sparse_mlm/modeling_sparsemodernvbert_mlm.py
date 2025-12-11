from torch import nn
import torch
from colpali_engine.models.modernvbert.modeling_modernvbert import (
    ModernVBertModel,
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

    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def __init__(self, config, mask_non_image_embeddings: bool = False, **kwargs):
        super().__init__(config=config)
        self.model = ModernVBertModel(config, **kwargs)

        # Add the SPLADE head (mlm)
        hidden_size = self.model.config.text_config.hidden_size
        vocab_size = self.model.config.text_config.vocab_size
        self.splade_head = nn.Linear(hidden_size, vocab_size, bias=True)
        self.max_pool = MaxPoolValue(dim=1)  # row-wise max pooling

        self.mask_non_image_embeddings = mask_non_image_embeddings
        self.main_input_name = "doc_input_ids"


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
        last_hidden_states = output[0]  # (B, L, H)
        logits = self.splade_head(last_hidden_states)  # (B, L, V)

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

        # norm default: log(1 + Relu(x))
        logits = torch.log1p(torch.relu(logits))  # (B, L, V)
        # Max-pool over sequence dimension → [B, V]
        lex_weights = self.max_pool(logits)  # (B, V)
        return SparseRep(dense=lex_weights)