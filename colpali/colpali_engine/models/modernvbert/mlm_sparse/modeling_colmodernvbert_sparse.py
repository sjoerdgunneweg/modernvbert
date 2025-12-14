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

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        special_tokens_mask=None,
        pixel_values=None,
        **kwargs,
    ) -> SparseRep:

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            **kwargs,
        )

        hidden = outputs.last_hidden_state          # (B, L, H)
        logits = self.splade_head(hidden)           # (B, L, V)

        scores = torch.log1p(torch.relu(logits)) 

        if attention_mask is not None:
            scores *= attention_mask.unsqueeze(-1)

        if special_tokens_mask is not None:
            scores *= (1 - special_tokens_mask).unsqueeze(-1)

        if self.mask_non_image_embeddings and pixel_values is not None:
            image_mask = (input_ids == self.config.image_token_id)
            scores *= image_mask.unsqueeze(-1)

        doc_scores = torch.max(scores, dim=1).values    # (B, V)

        vocab_size = doc_scores.size(1)
        size = torch.tensor(
            [doc_scores.size(0), vocab_size],
            device=doc_scores.device,
        )

        vocab_indices = torch.arange(
            vocab_size,
            device=doc_scores.device
        ).unsqueeze(0).expand(doc_scores.size(0), -1)

        return SparseRep(
            indices=vocab_indices,   
            values=doc_scores,       
            size=size,
        )
