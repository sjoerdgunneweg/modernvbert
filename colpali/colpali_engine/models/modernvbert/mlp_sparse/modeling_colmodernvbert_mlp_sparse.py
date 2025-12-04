import torch
from torch import nn
from colpali_engine.models.modernvbert.modeling_modernvbert import (
    ModernVBertModel,
    ModernVBertPreTrainedModel,
)
from colpali_engine.utils.sparse_rep import SparseRep


class ColModernVBertMLPSparse(ModernVBertPreTrainedModel):

    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def __init__(
        self,
        config,
        activation="relu",
        norm="log1p",
        scale=1.0,
        mask_non_image_embeddings=False,
        **kwargs,
    ):
        super().__init__(config)
        
        self.model = ModernVBertModel(config, **kwargs)

        hidden = self.model.config.text_config.hidden_size
        self.vocab_size = self.model.config.text_config.vocab_size

        self.linear = nn.Linear(hidden, 1)

        self.activation_fn = getattr(torch, activation)
        self.norm_fn = torch.log1p if norm == "log1p" else getattr(torch, norm)

        self.scale = nn.Parameter(torch.tensor(scale))
        self.mask_non_image_embeddings = mask_non_image_embeddings

        for p in self.model.parameters():
            p.requires_grad_(True)
        for p in self.linear.parameters():
            p.requires_grad_(True)

        self.main_input_name = "input_ids"

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        special_tokens_mask=None,
        pixel_values=None,
        to_scale=False,
        **kwargs,
    ):
        """
        Multimodal sparse encoder returning SparseRep.

        Output:
            SparseRep(indices=[B,L], values=[B,L], size=[B, vocab])
        """

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            **kwargs,
        )

        hidden = outputs.last_hidden_state          # (B, L, H)

        tok_scores = self.linear(hidden).squeeze(-1)    # (B, L)

        tok_scores = self.activation_fn(tok_scores)
        tok_scores = self.norm_fn(tok_scores)

        if attention_mask is not None:
            tok_scores = tok_scores * attention_mask     # (B, L)


        if special_tokens_mask is not None:
            tok_scores = tok_scores * (1 - special_tokens_mask)

        if (
            self.mask_non_image_embeddings and
            pixel_values is not None and
            input_ids is not None
        ):
            image_mask = (input_ids == self.config.image_token_id).to(tok_scores.dtype)
            tok_scores = tok_scores * image_mask          # (B, L)

        if to_scale:
            tok_scores = tok_scores * self.scale

        size = torch.tensor(
            [tok_scores.size(0), self.vocab_size],
            device=tok_scores.device,
        )

        return SparseRep(
            indices=input_ids,     # (B, L)
            values=tok_scores,     # (B, L)
            size=size,             # [batch, vocab]
        )
