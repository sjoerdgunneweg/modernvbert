import torch
from torch.nn.parallel import DistributedDataParallel
from colpali_engine.models.modernvbert.modeling_modernvbert import ModernVBertPreTrainedModel

from colpali_engine.models.modernvbert.sparse_mlp.modeling_sparsemodernvbert_mlp import (
    SparseModernVBertMLP
)
from colpali_engine.models.modernvbert.sparse_mlm.modeling_sparsemodernvbert_mlm import (
    SparseModernVBertMLM
)


class SparseModernVBertM2(ModernVBertPreTrainedModel):

    supports_gradient_checkpointing = False  # IMPORTANT
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def __init__(self, config, mask_non_image_embeddings: bool = False, **kwargs):
        super().__init__(config=config)
        self.text_encoder = SparseModernVBertMLP(config, mask_non_image_embeddings, **kwargs)
        self.vision_encoder = SparseModernVBertMLM(config, mask_non_image_embeddings, **kwargs)
        self.main_input_name = "doc_input_ids"

        # forward-call counter
        self._ddp_forward_count = 0

    def forward(self, *args, **kwargs) -> torch.Tensor:
        self._ddp_forward_count += 1

        is_ddp = isinstance(self, DistributedDataParallel)

        # Suppress sync on the first forward only
        if is_ddp and self._ddp_forward_count == 1:
            with self.no_sync():
                return self._forward_impl(*args, **kwargs)

        # Second forward (or single-GPU)
        out = self._forward_impl(*args, **kwargs)

        # Reset counter after second forward
        if self._ddp_forward_count >= 2:
            self._ddp_forward_count = 0

        return out

    def _forward_impl(self, *args, **kwargs) -> torch.Tensor:
        if "pixel_values" in kwargs or "image_hidden_states" in kwargs:
            return self.vision_encoder(*args, **kwargs)
        else:
            return self.text_encoder(*args, **kwargs)
