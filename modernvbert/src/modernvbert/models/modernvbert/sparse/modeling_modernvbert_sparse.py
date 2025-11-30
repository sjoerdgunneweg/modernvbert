import torch
import torch.nn as nn
from torch import Tensor
from transformers import ModernVBertPreTrainedModel, ModernVBertModel


class ModernVBertSparse(ModernVBertPreTrainedModel):
    """
    Sparse CLS-based ModernVBert encoder (LSR).

    This model:
    - uses the CLS embedding
    - applies an MLM-trained encoder for token contextualization
    - projects CLS to a low-dimensional sparse vector
    - returns an L2-normalized sparse embedding
    """

    def __init__(self, config):
        super().__init__(config)

        # Backbone encoder (same as dense ModernVBert)
        self.model = ModernVBertModel(config)

        # Dimensionality of sparse embeddings
        self.sparse_dim = getattr(config, "sparse_dim", 128)

        # Linear projection CLS -> sparse latent space
        hidden = config.hidden_size
        self.proj = nn.Linear(hidden, self.sparse_dim)

        self.post_init()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        **kwargs,
    ) -> Tensor:
        """
        Forward pass for LSR ModernVBert.

        Steps:
        1. run ModernVBert encoder
        2. take CLS embedding
        3. project to sparse dim
        4. L2 normalize
        """

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        # shape: (batch, seq_len, hidden)
        last_hidden = outputs.last_hidden_state

        # 1) CLS pooling (this is the "CLS" part of CLS-MLM)
        cls_embed = last_hidden[:, 0]  # (batch, hidden)

        # 2) Sparse projection layer
        sparse_vec = self.proj(cls_embed)  # (batch, sparse_dim)

        # 3) L2 norm (required for retrieval)
        sparse_vec = sparse_vec / sparse_vec.norm(dim=-1, keepdim=True)

        return sparse_vec
