from torch import nn
import torch
from colpali_engine.models.modernvbert.modeling_modernvbert import (
    ModernVBertModel,
    ModernVBertPreTrainedModel,
)
from colpali_engine.utils.sparse_rep import SparseRep


class SparseModernVBertMLP(ModernVBertPreTrainedModel):

    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def __init__(self, config, mask_non_image_embeddings: bool = False, **kwargs):
        super().__init__(config=config)
        self.model = ModernVBertModel(config, **kwargs)
        self.mask_non_image_embeddings = mask_non_image_embeddings
        self.main_input_name = "doc_input_ids"
        self.vocab_size = self.model.config.text_config.vocab_size

        # Add the MLP head
        hidden_size = self.model.config.text_config.hidden_size
        self.mlp_head = nn.Linear(hidden_size, 1, bias=True)




    def forward(self, *args, **kwargs) -> torch.Tensor:
        output = self.model(*args, **kwargs)
        last_hidden_states = output[0]  # (B, L, H)
        token_weights = self.mlp_head(last_hidden_states)  # (B, L, 1)
        token_weights = token_weights.squeeze(-1)  # (B, L)

        # Patch out of bound tokens their weights to zero
        invalid_mask = kwargs["input_ids"] >= self.vocab_limit

        # Step A: Set the VALUE of these tokens to 0 (so they have no effect)
        token_weights = token_weights.masked_fill(invalid_mask, 0.0)

        # Step B: Set the INDEX of these tokens to 0 (so scatter doesn't crash)
        # We use 0 because it's guaranteed to be a valid index.
        # Since the value is 0.0, it won't corrupt the actual embedding at index 0.
        safe_input_ids = kwargs["input_ids"].masked_fill(invalid_mask, 0)

        # Remove padding tokens:
        if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
            mask = kwargs["attention_mask"].to(token_weights.dtype)  # (B, L)
            token_weights = token_weights * mask

        # norm default: log(1 + Relu(x))
        token_weights = torch.log1p(torch.relu(token_weights))  # (B, L)

        size = torch.tensor((token_weights.size(0), self.vocab_size), device=token_weights.device)


        print("ids:")
        print(kwargs["input_ids"])
        print("size")
        print(size)
        print("values shape:")
        print(token_weights.shape)
        print("inputd ids shape:")
        print(kwargs["input_ids"].shape)

        return SparseRep(indices=kwargs["input_ids"], values=token_weights, size=size).to_dense()