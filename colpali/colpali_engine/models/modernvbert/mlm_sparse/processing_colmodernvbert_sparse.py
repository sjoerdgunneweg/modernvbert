from typing import ClassVar, List, Optional, Tuple, Union
import torch
from PIL import Image
from transformers import BatchEncoding, BatchFeature, Idefics3Processor

from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


class ColModernVBertSparseProcessor(BaseVisualRetrieverProcessor, Idefics3Processor):
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
        # Move to device and concatenate chunks
        if device is None:
            device = qs[0].device

        q = torch.cat([q.to(device) for q in qs], dim=0)  # (Q, V)
        p = torch.cat([p.to(device) for p in ps], dim=0)  # (P, V)

        # Safety checks (optional but helpful while debugging)
        if q.dim() != 2 or p.dim() != 2:
            raise ValueError(
                f"SPLADE scoring expects 2D tensors (batch, vocab). "
                f"Got q.shape={q.shape}, p.shape={p.shape}"
            )
        if q.size(-1) != p.size(-1):
            raise ValueError(
                f"Query and passage vocab dims must match, "
                f"got {q.size(-1)} and {p.size(-1)}"
            )

        # SPLADE similarity = dot product between sparse vocab vectors  # (Q, V) @ (V, P) -> (Q, P)
        scores = q @ p.t()
        return scores


    def get_n_patches(
        self,
        image_size: Tuple[int, int],
        patch_size: int,
    ) -> Tuple[int, int]:
        raise NotImplementedError("This method is not implemented for ColIdefics3.")
    
    def get_query_len(self) -> int:
        raise NotImplementedError("Return mean length of sparse vectors")
    
    def get_doc_len(self) -> int:
        raise NotImplementedError("Return mean length of sparse vectors")