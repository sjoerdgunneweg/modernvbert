from torch import nn
import torch
from colpali_engine.models.modernvbert.modeling_modernvbert import (
    ModernVBertForMaskedLM,
    ModernVBertPreTrainedModel,
)
from colpali_engine.utils.sparse_rep import SparseRep


class ColModernVBertSparse(ModernVBertPreTrainedModel):
    """
    MLM-head SPLADE-style sparse document encoder (LSR-style),
    adapted for in-batch contrastive training.
    """

    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def __init__(
        self,
        config,
        k: int = 1000,
        mask_non_image_embeddings: bool = False,
        **kwargs,
    ):
        super().__init__(config)

        self.model = ModernVBertForMaskedLM(config, **kwargs)
        self.k = k
        self.mask_non_image_embeddings = mask_non_image_embeddings
        self.main_input_name = "doc_input_ids"

        # ensure model is trainable
        for p in self.model.parameters():
            p.requires_grad_(True)

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

        # MLM logits: (B, L, V + extra)
        logits = outputs.logits

        # keep only text vocab
        vocab_size = self.config.text_config.vocab_size
        logits = logits[:, :, :vocab_size]   # (B, L, V)

        # SPLADE activation
        scores = torch.log1p(torch.relu(logits))  # (B, L, V)

        # mask padding
        if attention_mask is not None:
            scores = scores * attention_mask.unsqueeze(-1)

        # mask special tokens
        if special_tokens_mask is not None:
            scores = scores * (1 - special_tokens_mask).unsqueeze(-1)

        # optionally keep only image tokens
        if (
            self.mask_non_image_embeddings
            and pixel_values is not None
            and input_ids is not None
        ):
            image_mask = (input_ids == self.config.image_token_id)
            scores = scores * image_mask.unsqueeze(-1)

        # max-pool over sequence → (B, V)
        doc_scores = scores.max(dim=1).values

        topk_vals, topk_idx = torch.topk(doc_scores, self.k, dim=1)

        size = torch.tensor(
            [doc_scores.size(0), doc_scores.size(1)],
            device=doc_scores.device,
        )

        return SparseRep(
            indices=topk_idx,    # (B, k)
            values=topk_vals,   # (B, k)
            size=size,
        )
