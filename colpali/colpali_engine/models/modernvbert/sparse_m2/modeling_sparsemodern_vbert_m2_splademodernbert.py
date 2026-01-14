import torch
from colpali_engine.models.modernvbert.modeling_modernvbert import ModernVBertPreTrainedModel

from colpali_engine.models.modernvbert.sparse_mlp.modeling_sparsemodernvbert_mlp import SparseModernVBertMLP
from colpali_engine.models.modernvbert.sparse_mlm.modeling_sparsemodernvbert_mlm import SparseModernVBertMLM
from transformers import ModernBertForMaskedLM




class SparseModernVBertM2SpladeModernBERT(ModernVBertPreTrainedModel):

    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def __init__(self, config, mask_non_image_embeddings: bool = False, **kwargs):
        super().__init__(config=config)
        self.text_encoder = ModernBertForMaskedLM.from_pretrained("sparse-encoder/splade-ModernBERT-nq-fresh-lq0.05-lc0.003_scale1_lr-1e-4_bs64")
        self.vision_encoder = SparseModernVBertMLM(config, mask_non_image_embeddings, **kwargs)
        self.main_input_name = "doc_input_ids"

        self.query_prefix = "query_"
        self.pos_prefix = "doc_"
        self.neg_prefix = "neg_doc_"


    def _reshape_neg_doc_inputs(self, inputs):
        """
        Helper function to reshape negative doc inputs to (batch_size * num_neg_docs, ...)
        """
        neg_doc_inputs = {k[len(self.neg_prefix) :]: v for k, v in inputs.items() if k.startswith(self.neg_prefix)}

        for k in neg_doc_inputs:
            # go from (batch_size, num_neg_docs, ...) to (batch_size * num_neg_docs, ...)
            neg_doc_inputs[k] = neg_doc_inputs[k].view(-1, *neg_doc_inputs[k].shape[2:])

        return neg_doc_inputs

    def _reshape_neg_doc_outputs(self, neg_doc_outputs, num_neg_docs):
        """
        Helper function to reshape negative doc outputs to (batch_size, num_neg_docs, ...)
        """
        neg_doc_outputs = neg_doc_outputs.view(-1, num_neg_docs, *neg_doc_outputs.shape[1:])

        return neg_doc_outputs

    def forward(self, *args, **kwargs) -> torch.Tensor:
        # === Extract inputs ===
        query_inputs = {k[len(self.query_prefix):]: v for k, v in kwargs.items() if k.startswith(self.query_prefix)}
        doc_inputs   = {k[len(self.pos_prefix):]:   v for k, v in kwargs.items() if k.startswith(self.pos_prefix)}

        # === Hard negatives ===
        neg_doc_inputs = None
        if "neg_doc_input_ids" in kwargs:
            num_negs = kwargs["neg_doc_input_ids"].size(1)
            neg_doc_inputs = self._reshape_neg_doc_inputs(kwargs)

        #== Encode query (text) ===
        print("query_inputs:", query_inputs)
        query_outputs = self.text_encoder(**query_inputs)

        print("query_outputs:", query_outputs)
        #=== Encode doc (vision) ===
        doc_outputs = self.vision_encoder(**doc_inputs)

        #=== Encode neg docs (vision) ===
        if neg_doc_inputs is not None:
            neg_doc_outputs = self.vision_encoder(**neg_doc_inputs)
            neg_doc_outputs = self._reshape_neg_doc_outputs(neg_doc_outputs, num_negs)

        return {
            "q_out": query_outputs,
            "d_out": doc_outputs,
            "neg_d_out": neg_doc_outputs if neg_doc_inputs is not None else None,
        }