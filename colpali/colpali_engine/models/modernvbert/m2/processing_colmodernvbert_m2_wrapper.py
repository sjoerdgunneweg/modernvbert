from typing import List, Optional, Union
import torch
from PIL import Image
from transformers import BatchEncoding, BatchFeature, Idefics3Processor

from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
from colpali_engine.utils.sparse_rep import SparseRep

from ..mlp_sparse.processing_colmodernvbert_mlp_sparse import ColModernVBertMLPSparseProcessor
from ..mlm_sparse.processing_colmodernvbert_sparse import ColModernVBertSparseProcessor


class ColModernVBertM2Processor(BaseVisualRetrieverProcessor, Idefics3Processor):
    """
    Processor for M2
    """

    def __init__(self, *args, image_seq_len=64, **kwargs):
        super().__init__(*args, image_seq_len=image_seq_len, **kwargs)

        self.query_processor = ColModernVBertMLPSparseProcessor(*args, **kwargs)

        self.doc_processor = ColModernVBertSparseProcessor(*args, **kwargs)

 
    def process_queries(self, texts: List[str]) -> BatchEncoding:
        return self.query_processor.process_texts(texts)


    def process_documents(self, images: List[Image.Image]) -> BatchEncoding:
        return self.doc_processor.process_images(images)

 
    def score(self, qs, ps, device=None, **kwargs):
        """
        qs → list of SparseReps or dense vectors from query_encoder
        ps → list of SparseReps or dense vectors from doc_encoder
        """
        return self.doc_processor.score(qs, ps, device=device)
