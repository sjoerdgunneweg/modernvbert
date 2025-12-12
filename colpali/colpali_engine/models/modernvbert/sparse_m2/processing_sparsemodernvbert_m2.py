from typing import ClassVar, List, Optional, Tuple, Union
import torch
from PIL import Image
from transformers import BatchEncoding, BatchFeature

from colpali_engine.models.modernvbert.colvbert import ColModernVBertProcessor



class SparseModernVBertM2Processor(ColModernVBertProcessor):
    """
    Processor for ColIdefics3.
    """

    query_augmentation_token: ClassVar[str] = "<end_of_utterance>"
    image_token: ClassVar[str] = "<image>"
    visual_prompt_prefix: ClassVar[str] = (
        "<|begin_of_text|>User:<image>Describe the image.<end_of_utterance>\nAssistant:"
    )

    def __init__(self, *args, image_seq_len=64, **kwargs):
        super().__init__(*args, image_seq_len=image_seq_len, **kwargs)
        self.tokenizer.padding_side = "left"

    def process_images(
        self,
        images: List[Image.Image],
    ) -> Union[BatchFeature, BatchEncoding]:
        """
        Process images for ColModernVBert.

        Args:
            images: List of PIL images.
        """
        images = [image.convert("RGB") for image in images]

        batch_doc = self(
            text=[self.visual_prompt_prefix] * len(images),
            images=images,
            padding="longest",
            return_tensors="pt",
        )
        return batch_doc

    def process_texts(self, texts: List[str]) -> Union[BatchFeature, BatchEncoding]:
        """
        Process texts for ColModernVBert.

        Args:
            texts: List of input texts.

        Returns:
            Union[BatchFeature, BatchEncoding]: Processed texts.
        """
        return self(
            text=texts,
            return_tensors="pt",
            padding="longest",
        )

    def score(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:

        # Convert SparseRep to dense tensors
        qs = [q.to_dense() for q in qs]
        ps = [p.to_dense() for p in ps]

        return self.score_single_vector(qs, ps, device=device)

    def get_query_len(self) -> int:
        raise NotImplementedError("Return mean length of sparse vectors")

    def get_doc_len(self) -> int:
        raise NotImplementedError("Return mean length of sparse vectors")