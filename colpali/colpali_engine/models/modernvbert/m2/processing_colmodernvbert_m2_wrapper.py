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
    Processor for M2 (MLP for queries, MLM for documents)
    """

    def __init__(self, *args, image_seq_len=64, **kwargs):
        super().__init__(*args, image_seq_len=image_seq_len, **kwargs)

        self.query_processor = ColModernVBertMLPSparseProcessor(*args, **kwargs)
        self.doc_processor   = ColModernVBertSparseProcessor(*args, **kwargs)
    
        self.query_augmentation_token = self.query_processor.query_augmentation_token
        self.image_token              = self.query_processor.image_token

        self.query_prefix = getattr(
            self.query_processor,
            "query_prefix",
            self.query_processor.visual_prompt_prefix,
        )

    def process_queries(self, texts: List[str]) -> BatchEncoding:
        return self.query_processor.process_texts(texts)

    def process_documents(self, images: List[Image.Image]) -> BatchEncoding:
        return self.doc_processor.process_images(images)

    def process_images(self, images: List[Image.Image]) -> BatchEncoding:
        return self.doc_processor.process_images(images)

    def process_texts(self, texts: List[str]) -> BatchEncoding:
        return self.query_processor.process_texts(texts)
    
    def score(self, qs, ps, device=None, **kwargs):
        return self.doc_processor.score(qs, ps, device=device)

 
    def get_n_patches(self, *args, **kwargs):
        return self.doc_processor.get_n_patches(*args, **kwargs)

    def get_query_len(self):
        if hasattr(self.query_processor, "get_query_len"):
            return self.query_processor.get_query_len()
        raise NotImplementedError

    def get_doc_len(self):
        if hasattr(self.doc_processor, "get_doc_len"):
            return self.doc_processor.get_doc_len()
        raise NotImplementedError
