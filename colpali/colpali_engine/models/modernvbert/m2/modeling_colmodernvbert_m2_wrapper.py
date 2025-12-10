import torch
from torch import nn

from colpali_engine.utils.sparse_rep import SparseRep
from .modeling_colmodernvbert_mlp_sparse import ColModernVBertMLPSparse
from .modeling_colmodernvbert_sparse import ColModernVBertSparse


class ColModernVBertM2Wrapper(nn.Module):
    """
    MLSR Model M2:
        Query Encoder    = MLP
        Document Encoder = MLM
    """

    def __init__(
        self,
        config,
        mlp_kwargs=None,
        mlm_kwargs=None,
    ):
        super().__init__()

        mlp_kwargs = mlp_kwargs or {}
        mlm_kwargs = mlm_kwargs or {}

        self.query_encoder = ColModernVBertMLPSparse(config, **mlp_kwargs)

        self.doc_encoder = ColModernVBertSparse(config, **mlm_kwargs)

  
        self.config = config

  
    @torch.no_grad()
    def encode_query(self, **kwargs) -> SparseRep:
        return self.query_encoder(**kwargs)

    
    @torch.no_grad()
    def encode_document(self, **kwargs) -> SparseRep:
        return self.doc_encoder(**kwargs)

  
    def forward(self, is_query: bool = False, **kwargs):
        """
        Training forward pass
        """
        if is_query:
            return self.query_encoder(**kwargs)
        else:
            return self.doc_encoder(**kwargs)
