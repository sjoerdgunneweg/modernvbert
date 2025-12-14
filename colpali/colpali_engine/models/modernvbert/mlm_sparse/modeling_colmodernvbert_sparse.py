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


class ColModernVBertSparse(ModernVBertPreTrainedModel):
    """
    Sparse ColBERT-style retriever on top of ModernVBERT.

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
        self.model = ModernVBertModel(config, **kwargs)
        hidden_size = self.model.config.text_config.hidden_size
        vocab_size = self.model.config.text_config.vocab_size
        self.splade_head = nn.Linear(hidden_size, vocab_size, bias=True)

        for p in self.model.parameters():
            p.requires_grad_(True)
        for p in self.splade_head.parameters():
            p.requires_grad_(True)

        self.mask_non_image_embeddings = mask_non_image_embeddings
        self.main_input_name = "doc_input_ids"
        self.max_pool = MaxPoolValue(dim=1)  # row-wise max pooling

    def forward(self, input_ids=None, attention_mask=None, special_tokens_mask=None, pixel_values=None, **kwargs) -> SparseRep:
        """
        Forward pass through ModernVBert + SPLADE head.

        Args:
            input_ids:      [B, L]
            attention_mask: [B, L]
            pixel_values:   optional, if multimodal
            special_tokens_mask: [B, L] mask for special tokens

        Returns:
            SparseRep with indices=[B,L], values=[B,L], size=[B, V]
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, **kwargs)
        last_hidden_states = outputs[0]  # (B, L, H)
        logits = self.splade_head(last_hidden_states)  # (B, L, V)

        token_scores = torch.log1p(torch.relu(logits))  # (B, L, V)

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

        return SparseRep(dense=lex_weights)