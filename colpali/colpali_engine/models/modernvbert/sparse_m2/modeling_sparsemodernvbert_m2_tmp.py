import torch
from colpali_engine.models.modernvbert.modeling_modernvbert import ModernVBertPreTrainedModel

from colpali_engine.models.modernvbert.sparse_mlp.modeling_sparsemodernvbert_mlp import SparseModernVBertMLP
from colpali_engine.models.modernvbert.sparse_mlm.modeling_sparsemodernvbert_mlm import SparseModernVBertMLM


class SparseModernVBertM2(ModernVBertPreTrainedModel):

    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def __init__(self, config, mask_non_image_embeddings: bool = False, **kwargs):
        super().__init__(config=config)
        self.text_encoder = SparseModernVBertMLP(config, mask_non_image_embeddings, **kwargs)
        self.vision_encoder = SparseModernVBertMLM(config, mask_non_image_embeddings, **kwargs)
        self.main_input_name = "doc_input_ids"



    def forward(self, *args, **kwargs) -> torch.Tensor:
        # vision_out = self.vision_encoder(*args, **kwargs)
        # text_out = self.text_encoder(*args, **kwargs)


        # vision = "pixel_values" in kwargs or "image_hidden_states" in kwargs
        # return vision_out * vision + text_out * (1 - vision)

        has_pixel_values = ("pixel_values" in kwargs) and (kwargs["pixel_values"] is not None)
        has_image_hs = ("image_hidden_states" in kwargs) and (kwargs["image_hidden_states"] is not None)
        vision = has_pixel_values or has_image_hs

        if vision:
            # print("Using vision encoder")
            print("vision", self.vision_encoder(*args, **kwargs))
            return self.vision_encoder(*args, **kwargs)
        else:
            # print("Using text encoder")
            print("text", self.text_encoder(*args, **kwargs))
            return self.text_encoder(*args, **kwargs)