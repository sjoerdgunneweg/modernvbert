from typing import List, Optional, Union, Tuple
import torch
from PIL import Image
from transformers import BatchEncoding, BatchFeature, Idefics3Processor

from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
from colpali_engine.utils.sparse_rep import SparseRep


class ColModernVBertMLPSparseProcessor(BaseVisualRetrieverProcessor, Idefics3Processor):
    """
    Processor for MLP sparse ModernVBERT.
    """

    query_augmentation_token = "<end_of_utterance>"
    image_token = "<image>"
    visual_prompt_prefix = (
        "<|begin_of_text|>User:<image>Describe the image.<end_of_utterance>\nAssistant:"
    )

    def __init__(self, *args, image_seq_len=64, **kwargs):
        super().__init__(*args, image_seq_len=image_seq_len, **kwargs)
        self.tokenizer.padding_side = "left"

    def process_images(self, images: List[Image.Image]):
        images = [img.convert("RGB") for img in images]

        return self(
            text=[self.visual_prompt_prefix] * len(images),
            images=images,
            padding="longest",
            return_tensors="pt",
        )


    def process_texts(self, texts: List[str]):
        return self(
            text=texts,
            padding="longest",
            return_tensors="pt",
        )

    def score(
        self,
        qs: List[SparseRep],
        ps: List[SparseRep],
        device=None,
        **kwargs,
    ):
        if device is None:
            device = qs[0].values.device

        q = SparseRep.merge(qs).to(device)
        p = SparseRep.merge(ps).to(device)

        return q.dot(p)

    def get_n_patches(self, *args, **kwargs):
        raise NotImplementedError

    def get_query_len(self):
        raise NotImplementedError

    def get_doc_len(self):
        raise NotImplementedError
