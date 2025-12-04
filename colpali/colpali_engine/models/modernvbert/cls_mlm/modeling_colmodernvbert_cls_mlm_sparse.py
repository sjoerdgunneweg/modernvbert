import torch
from torch import nn
from colpali_engine.models.modernvbert.modeling_modernvbert import (
    ModernVBertModel,
    ModernVBertPreTrainedModel,
)
from colpali_engine.utils.sparse_rep import SparseRep


class ColModernVBertCLSMlMSparse(ModernVBertPreTrainedModel):

    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def __init__(
        self,
        config,
        activation="relu",
        norm="log1p",
        **kwargs,
    ):
        super().__init__(config)

        self.model = ModernVBertModel(config, **kwargs)

        hidden_size = self.model.config.text_config.hidden_size
        vocab_size = self.model.config.text_config.vocab_size

        self.mlm_head = nn.Linear(hidden_size, vocab_size, bias=True)

        self.activation_fn = getattr(torch, activation)
        self.norm_fn = torch.log1p if norm == "log1p" else getattr(torch, norm)

        for p in self.model.parameters():
            p.requires_grad_(True)
        for p in self.mlm_head.parameters():
            p.requires_grad_(True)

        self.main_input_name = "input_ids"

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        special_tokens_mask=None,
        pixel_values=None,
        **kwargs,
    ):
        """
        CLS_MLM sparse encoder:
          last_hidden_state[:,0,:] → MLM head → activation → norm
          returns: SparseRep(dense=[B,V])
        """

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            **kwargs,
        )

        hidden = outputs.last_hidden_state       # (B, L, H)
        cls = hidden[:, 0, :]                    # (B, H)

        logits = self.mlm_head(cls)              # (B, V)

        scores = self.activation_fn(logits)
        scores = self.norm_fn(scores)

        return SparseRep(dense=scores)
