import torch
from typing import Any, Dict
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

        self.query_prefix = "query_"
        self.pos_prefix   = "doc_"
        self.neg_prefix   = "neg_doc_"

    # def _reshape_neg_doc_inputs(self, inputs):
    #     """
    #     inputs: already de-prefixed dict with shape [B, N, ...]
    #     """
    #     neg_doc_inputs = {}
    #     for k, v in inputs.items():
    #         # (B, N, ...) -> (B*N, ...)
    #         neg_doc_inputs[k] = v.view(-1, *v.shape[2:])
    #     return neg_doc_inputs

    # def _reshape_neg_doc_outputs(self, neg_doc_outputs, num_neg_docs):
    #     """
    #     neg_doc_outputs: Tensor [B*N, D]
    #     """
    #     return neg_doc_outputs.view(-1, num_neg_docs, *neg_doc_outputs.shape[1:])


    def _reshape_neg_doc_inputs(self, neg_inputs):
        """
        neg_inputs contains already-stripped keys:
        input_ids: [B, N, L], attention_mask: [B, N, L], ...
        returns:
        input_ids: [B*N, L], attention_mask: [B*N, L], ...
        """
        out = {}
        for k, v in neg_inputs.items():
            if v is None:
                continue
            # (B, N, ...) -> (B*N, ...)
            out[k] = v.view(-1, *v.shape[2:])
        return out


    def _reshape_neg_doc_outputs(self, neg_doc_outputs, num_neg_docs):
        """
        neg_doc_outputs: typically [B*N, D] (or a model output object)
        return: [B, N, D]
        """
        # If your encoder returns a tensor directly:
        return neg_doc_outputs.view(-1, num_neg_docs, *neg_doc_outputs.shape[1:])




    def forward(self, *args, **kwargs):
        query_inputs: Dict[str, Any] = {}
        doc_inputs: Dict[str, Any] = {}
        neg_inputs: Dict[str, Any] = {}

        for k, v in kwargs.items():
            if k.startswith(self.query_prefix):
                query_inputs[k[len(self.query_prefix):]] = v
            elif k.startswith(self.pos_prefix):
                doc_inputs[k[len(self.pos_prefix):]] = v
            elif k.startswith(self.neg_prefix):
                neg_inputs[k[len(self.neg_prefix):]] = v

        # -------------------
        # Encode query (text)
        # -------------------
        query_outputs = self.text_encoder(**query_inputs)

        # -------------------
        # Encode positive doc
        # -------------------
        has_pixel_values = ("pixel_values" in doc_inputs and doc_inputs["pixel_values"] is not None)
        has_image_hs = ("image_hidden_states" in doc_inputs and doc_inputs["image_hidden_states"] is not None)

        if has_pixel_values or has_image_hs:
            doc_outputs = self.vision_encoder(**doc_inputs)
        else:
            doc_outputs = self.text_encoder(**doc_inputs)


        # Encode negatives (optional)
        neg_doc_outputs = None
        if len(neg_inputs) > 0 and neg_inputs.get("input_ids", None) is not None:
            num_negs = neg_inputs["input_ids"].size(1)  # [B, N, L]
            flat_neg_inputs = self._reshape_neg_doc_inputs(neg_inputs)  # [B*N, L]
            flat_neg_outputs = self.vision_encoder(**flat_neg_inputs)   # [B*N, ...]
            neg_doc_outputs = self._reshape_neg_doc_outputs(flat_neg_outputs, num_negs)  # [B, N, ...]



        return {
            "q_out": query_outputs,
            "d_out": doc_outputs,
            "neg_d_out": neg_doc_outputs,
        }
