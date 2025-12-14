import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from typing import Optional, Dict, Any

from colpali_engine.utils.sparse_rep import SparseRep


def num_active_terms(a, threshold: float = 1e-3) -> torch.Tensor:
    return (a.dense > threshold).float().sum(dim=1).mean()


class Regularizer(nn.Module):
    def __init__(self, weight: float = 0.1, T: int = 10000):
        super().__init__()
        self.weight_T = weight
        self.weight_t = 0.0
        self.T = T
        self.t = 0

    def step(self):
        if self.t < self.T:
            self.t += 1
            self.weight_t = self.weight_T * (self.t / self.T) ** 2

    def forward(self, reps):
        raise NotImplementedError


class FLOPs(Regularizer):
    def forward(self, reps):
        flops = F.softplus(reps.dense).sum(dim=1).mean()
        return flops * self.weight_t


class L1(Regularizer):
    def forward(self, reps):
        return reps.dense.abs().sum(dim=1).mean() * self.weight_t


class SparseBiEncoderModule(nn.Module):
    def __init__(self, temperature: float = 0.02):
        super().__init__()
        self.temperature = temperature


class SparseBiEncoderLoss(SparseBiEncoderModule):
    def __init__(
        self,
        temperature: float = 0.02,
        q_regularizer: Optional[Regularizer] = None,
        d_regularizer: Optional[Regularizer] = None,
    ):
        super().__init__(temperature=temperature)
        self.ce_loss = CrossEntropyLoss()
        self.q_regularizer = q_regularizer
        self.d_regularizer = d_regularizer

    def forward(self, q, d, offset: int = 0):
        scores = torch.einsum("bd,cd->bc", q.dense, d.dense)
        B = scores.size(0)
        labels = torch.arange(B, device=scores.device)

        if self.q_regularizer:
            self.q_regularizer.step()
        if self.d_regularizer:
            self.d_regularizer.step()

        reg_q = self.q_regularizer(q) if self.q_regularizer else 0.0
        reg_d = self.d_regularizer(d) if self.d_regularizer else 0.0

        loss = self.ce_loss(scores / self.temperature, labels)

        return loss + reg_q + reg_d


class SparseBiNegativeCELoss(SparseBiEncoderModule):
    def __init__(
        self,
        temperature: float = 0.02,
        in_batch_term_weight: float = 0.5,
        pos_aware_negative_filtering: bool = False,
        q_regularizer: Optional[Regularizer] = None,
        d_regularizer: Optional[Regularizer] = None,
        debug: bool = False,
        **kwargs,
    ):
        super().__init__(temperature=temperature)

        self.in_batch_term_weight = in_batch_term_weight
        self.q_regularizer = q_regularizer
        self.d_regularizer = d_regularizer
        self.debug = debug

        self.in_batch_loss = SparseBiEncoderLoss(
            temperature=temperature,
            q_regularizer=None,
            d_regularizer=None,
        )

    def forward(
        self,
        query_embeddings=None,
        doc_embeddings=None,
        neg_doc_embeddings=None,
        offset: int = 0,
        **kwargs,
    ) -> Dict[str, Any]:

        q = query_embeddings
        d = doc_embeddings
        neg_d = neg_doc_embeddings

        if neg_d is None:
            if self.q_regularizer:
                self.q_regularizer.step()
            if self.d_regularizer:
                self.d_regularizer.step()

            in_batch = self.in_batch_loss(q, d)

            reg_q = self.q_regularizer(q) if self.q_regularizer else 0.0
            reg_d = self.d_regularizer(d) if self.d_regularizer else 0.0

            total = in_batch + reg_q + reg_d

            return {
                "loss": total,
                "contrastive_loss": in_batch,
                "reg_q": reg_q,
                "reg_d": reg_d,
                "query_length": num_active_terms(q),
                "doc_length": num_active_terms(d),
            }

        B = q.dense.size(0)

        pos_scores = (q.dense * d.dense).sum(dim=1) / self.temperature
        neg_scores = torch.einsum(
            "bd,bnd->bn", q.dense, neg_d.dense
        ) / self.temperature

        hard_neg_loss = F.softplus(
            neg_scores - pos_scores.unsqueeze(1)
        ).mean()

        if self.in_batch_term_weight > 0:
            in_batch = self.in_batch_loss(q, d)
            contrastive_loss = (
                hard_neg_loss * (1 - self.in_batch_term_weight)
                + in_batch * self.in_batch_term_weight
            )
        else:
            contrastive_loss = hard_neg_loss

        if self.q_regularizer:
            self.q_regularizer.step()
            reg_q = self.q_regularizer(q)
        else:
            reg_q = 0.0

        if self.d_regularizer:
            self.d_regularizer.step()
            docs_all = torch.cat(
                [d.dense, neg_d.dense.view(B * neg_d.size(1), -1)],
                dim=0,
            )
            reg_d = self.d_regularizer(SparseRep(dense=docs_all))
        else:
            reg_d = 0.0

        total = contrastive_loss + reg_q + reg_d

        return {
            "loss": total,
            "contrastive_loss": contrastive_loss,
            "reg_q": reg_q,
            "reg_d": reg_d,
            "query_length": num_active_terms(q),
            "doc_length": num_active_terms(d),
        }