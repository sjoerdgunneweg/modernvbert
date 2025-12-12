import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch import nn
from typing import Optional, Dict, Any
from colpali_engine.utils.sparse_rep import SparseRep

def num_active_terms(a, threshold: float = 1e-3) -> torch.Tensor:
    if isinstance(a, SparseRep):
        return (a.values > threshold).float().sum(dim=1).mean()
    return (F.relu(a) > threshold).float().sum(dim=1).mean()


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
    """
    FLOPS LOSS
    """

    def forward(self, reps):
        if isinstance(reps, SparseRep):
            flops = F.softplus(reps.values).sum() / reps.batch_size()
            return flops * self.weight_t

        # dense fallback
        return F.softplus(reps).sum(dim=-1).mean() * self.weight_t


class L1(Regularizer):
    def forward(self, reps):
        if isinstance(reps, SparseRep):
            return reps.values.sum() / reps.batch_size() * self.weight_t
        return reps.sum(dim=-1).mean() * self.weight_t


class SparseBiEncoderModule(nn.Module):
    def __init__(
        self,
        max_batch_size: int = 1024,
        temperature: float = 0.02,
        filter_threshold: float = 0.95,
        filter_factor: float = 0.5,
    ):
        super().__init__()
        self.temperature = temperature
        self.filter_threshold = filter_threshold
        self.filter_factor = filter_factor

        self.register_buffer(
            "idx_buffer",
            torch.arange(max_batch_size),
            persistent=False,
        )

    def _get_idx(self, B: int, offset: int, device):
        idx = self.idx_buffer[:B].to(device)
        pos_idx = idx + offset
        return idx, pos_idx

    def _filter_high_negatives(self, scores, pos_idx):
        B = scores.size(0)
        idx = self.idx_buffer[:B].to(scores.device)

        pos_scores = scores[idx, pos_idx]
        thresh = self.filter_threshold * pos_scores.unsqueeze(1)

        mask = scores > thresh
        mask[idx, pos_idx] = False

        scores[mask] = scores[mask] * self.filter_factor



class SparseBiEncoderLoss(SparseBiEncoderModule):
    def __init__(
        self,
        temperature: float = 0.02,
        pos_aware_negative_filtering: bool = False,
        max_batch_size: int = 1024,
        filter_threshold: float = 0.95,
        filter_factor: float = 0.5,
        q_regularizer: Optional[Regularizer] = None,
        d_regularizer: Optional[Regularizer] = None,
    ):
        super().__init__(max_batch_size, temperature, filter_threshold, filter_factor)
        self.pos_aware_negative_filtering = pos_aware_negative_filtering
        self.ce_loss = CrossEntropyLoss()

        self.q_regularizer = q_regularizer
        self.d_regularizer = d_regularizer


    def forward(self, q, d, offset: int = 0):
      
        if isinstance(q, SparseRep):
            scores = q.cross_dot(d)  # [B, B]
        else:
            scores = torch.einsum("bd,cd->bc", q, d)

        B = scores.size(0)
        device = scores.device

        idx, pos_idx = self._get_idx(B, offset, device)

        if self.pos_aware_negative_filtering:
            self._filter_high_negatives(scores, pos_idx)

        # update regularization warmup
        if self.q_regularizer:
            self.q_regularizer.step()
        if self.d_regularizer:
            self.d_regularizer.step()

        # compute regularization
        reg_q = self.q_regularizer(q) if self.q_regularizer else 0
        reg_d = self.d_regularizer(d) if self.d_regularizer else 0

        # softmax cross-entropy
        scores = scores / self.temperature
        ce = self.ce_loss(scores, pos_idx)

        return ce + reg_q + reg_d


class SparseBiNegativeCELoss(SparseBiEncoderModule):
    """
    True SPLADE contrastive objective.

    Supports sparse query & doc reps.
    """

    def __init__(
        self,
        temperature: float = 0.02,
        in_batch_term_weight: float = 0.5,
        pos_aware_negative_filtering: bool = False,
        max_batch_size: int = 1024,
        filter_threshold: float = 0.95,
        filter_factor: float = 0.5,
        q_regularizer: Optional[Regularizer] = None,
        d_regularizer: Optional[Regularizer] = None,
        debug: bool = False,
    ):
        super().__init__(max_batch_size, temperature, filter_threshold, filter_factor)

        self.in_batch_term_weight = in_batch_term_weight
        self.pos_aware_negative_filtering = pos_aware_negative_filtering
        self.debug = debug

        self.q_regularizer = q_regularizer
        self.d_regularizer = d_regularizer

        self.in_batch_loss = SparseBiEncoderLoss(
            temperature=temperature,
            pos_aware_negative_filtering=pos_aware_negative_filtering,
            max_batch_size=max_batch_size,
            q_regularizer=None,
            d_regularizer=None,
        )

   
    def forward(self, q, d, neg_d=None, offset: int = 0):
            # training without negatives
            if neg_d is None:
                if self.q_regularizer:
                    self.q_regularizer.step()
                if self.d_regularizer:
                    self.d_regularizer.step()

                in_batch = self.in_batch_loss(q, d, offset)
                reg_q = self.q_regularizer(q) if self.q_regularizer else 0
                reg_d = self.d_regularizer(d) if self.d_regularizer else 0

                total = in_batch + reg_q + reg_d

                return {
                    "loss": total,
                    "contrastive_loss": in_batch,
                    "reg_q": reg_q,
                    "reg_d": reg_d,
                    "query_length": num_active_terms(q),
                    "doc_length": num_active_terms(d),
                }

