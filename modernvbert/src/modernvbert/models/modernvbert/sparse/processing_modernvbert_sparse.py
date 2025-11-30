from typing import List, Union
import torch
from transformers import BatchEncoding, BatchFeature
from modernvbert.models.modernvbert.processing_modernvbert import ModernVBertProcessor

class ModernVBertSparseProcessor(ModernVBertProcessor):
    """
    Processor for the LSR (CLS-MLM) ModernVBertSparse model.

    Responsibilities:
    - tokenize texts
    - batch them into tensors
    - compute similarity between query & passage embeddings
    """

    def process_texts(
        self,
        texts: List[str],
        return_tensors: str = "pt"
    ) -> Union[BatchEncoding, BatchFeature]:

        return self(
            text=texts,
            padding="longest",
            return_tensors=return_tensors,
        )

    def score(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device=None,
    ) -> torch.Tensor:
        """
        Cosine similarity between CLS sparse vectors.
        q: List[Tensor] – query embeddings
        p: List[Tensor] – passage embeddings
        """

        q = torch.stack(qs)  # (batch, dim)
        p = torch.stack(ps)  # (batch, dim)

        if device:
            q = q.to(device)
            p = p.to(device)

        # Cosine similarity (vectors already normalized)
        return (q * p).sum(dim=-1)
