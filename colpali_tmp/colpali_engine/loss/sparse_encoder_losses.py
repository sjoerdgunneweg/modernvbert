import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch import nn
from typing import Optional, Tuple, Dict, Any


def num_active_terms(a: torch.Tensor, threshold: float = 1e-3) -> torch.Tensor:
    """
    Compute the average number of "active" (non-zero) dimensions per example.

    Args:
        a: Tensor of shape [B, D] (e.g., sparse-ish embedding).
        threshold: Values above this threshold are considered active.

    Returns:
        Scalar tensor: mean number of active dimensions over the batch.
    """
    return (F.relu(a) > threshold).float().sum(dim=1).mean()


class Regularizer(nn.Module):
    """
    Base class for sparse regularizers with warmup scheduling.

    Attributes
    ----------
    weight_T : float
        Target max weight of the regularizer after warmup.
    T : int
        Number of steps before reaching full weight (warmup length).
    t : int
        Current step.
    weight_t : float
        Current effective weight, evolves from 0 to weight_T over T steps.
    """

    def __init__(self, weight: float = 0.001, T: int = 50000) -> None:
        """
        Initialize regularizer with warmup schedule.

        Args:
            weight: Target regularizer weight when warmup completes.
            T: Number of steps over which to warm up.
        """
        super().__init__()
        self.weight_T = weight
        self.weight_t = 0.0
        self.T = T
        self.t = 0

    def step(self):
        """
        Advance the warmup schedule by one training step.

        The weight starts at 0 and increases quadratically up to weight_T
        between step 1 and step T. After T, weight_t stays at weight_T.
        """
        if self.t < self.T:
            self.t += 1
            self.weight_t = self.weight_T * (self.t / self.T) ** 2

    def forward(self, reps: torch.Tensor) -> torch.Tensor:
        """
        Placeholder forward definition; must be implemented in subclasses.

        Args:
            reps: batch representation tensor.

        Returns:
            Regularization loss term (scalar tensor).
        """
        raise NotImplementedError("This is an abstract regularizer only.")


class FLOPs(Regularizer):
    """
    SPLADE-style FLOPs regularizer (original SPLADE 2021).

    Paper: https://arxiv.org/abs/2104.05665

    Applies softplus to the representations and sums across dimensions.
    Encourages sparsity by penalizing the overall magnitude / activity.

    Input:
        reps: Tensor of shape [B, D]
    """

    def forward(self, reps: torch.Tensor) -> torch.Tensor:
        # softplus(reps) is like a smooth ReLU; summing gives an "activation mass" per example, then we mean over batch and scale by warmup weight.
        return F.softplus(reps).sum(dim=-1).mean() * self.weight_t

class SparseBiEncoderModule(nn.Module):
    """
    Base module for bi-encoder losses, handling buffer indexing and
    negative filtering hyperparameters.

    Args:
        max_batch_size: Maximum batch size for the pre-allocated index buffer.
        temperature: Scaling factor for logits (must be > 0).
        filter_threshold: Fraction of positive score above which negatives are down-weighted.
        filter_factor: Multiplicative factor applied to filtered negative scores.
    """

    def __init__(
        self,
        max_batch_size: int = 1024,
        temperature: float = 0.02,
        filter_threshold: float = 0.95,
        filter_factor: float = 0.5,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("Temperature must be strictly positive")

        self.temperature = temperature
        self.filter_threshold = filter_threshold
        self.filter_factor = filter_factor
        self.max_batch_size = max_batch_size

        # Pre-allocate indices for in-batch positives [0, 1, ..., max_batch_size-1].
        self.register_buffer(
            "idx_buffer",
            torch.arange(max_batch_size),
            persistent=False,
        )

    def _get_idx(self, batch_size: int, offset: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate index tensors for in-batch cross-entropy.

        Args:
            batch_size: Number of queries/docs in the batch.
            offset: Offset to apply for multi-GPU indexing.
            device: Target device of the indices.

        Returns:
            Tuple (idx, pos_idx), both of shape [batch_size].
            - idx: index of queries
            - pos_idx: index of corresponding positives (taking offset into account)
        """
        if batch_size > self.idx_buffer.size(0):
            raise ValueError(
                f"Batch size {batch_size} exceeds max_batch_size {self.idx_buffer.size(0)}. "
                f"Increase max_batch_size in SparseBiEncoderModule."
            )

        idx = self.idx_buffer[:batch_size].to(device)
        pos_idx = idx + offset
        return idx, pos_idx

    def _filter_high_negatives(self, scores: torch.Tensor, pos_idx: torch.Tensor) -> None:
        """
        In-place down-weighting of "too-high" in-batch negative scores.

        Args:
            scores: Tensor[B, B] — in-batch similarity matrix (query x doc).
            pos_idx: Tensor[B] — positive index for each query.

        Behavior:
            For each query b, we compute:
                thresh_b = filter_threshold * pos_scores_b
            All negatives with scores > thresh_b are multiplied by filter_factor.
        """
        batch_size = scores.size(0)
        idx = self.idx_buffer[:batch_size].to(scores.device)

        # Positive scores for each query in the batch.
        pos_scores = scores[idx, pos_idx]  # [B]

        # Broadcast threshold over columns: [B, 1] => [B, B].
        thresh = self.filter_threshold * pos_scores.unsqueeze(1)

        # Identify negatives that exceed the threshold.
        mask = scores > thresh

        # Ensure we do NOT touch the diagonal positives.
        mask[idx, pos_idx] = False

        # Down-weight "too high" negatives in-place.
        scores[mask] = scores[mask] * self.filter_factor


class SparseBiEncoderLoss(SparseBiEncoderModule):
    """
    Simple in-batch bi-encoder loss with optional SPLADE-style regularization.

    This corresponds to a standard InfoNCE over the in-batch scores, plus
    FLOPs-style regularization on queries and docs.
    """

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

        # Avoid using nn.Module instances as default args: create them here if None.
        self.q_regularizer = q_regularizer or FLOPs(weight=0.0001, T=100000)
        self.d_regularizer = d_regularizer or FLOPs(weight=0.0001, T=100000)

    def forward(self, query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Compute InfoNCE-style loss for a batch of query/doc embeddings.

        Args:
            query_embeddings: Tensor[B, D]
            doc_embeddings:   Tensor[B, D]
            offset:           Offset for positive indices (for multi-GPU setups).

        Returns:
            Scalar tensor: total loss = CE + q_regularizer + d_regularizer
        """
        # Similarity matrix: scores[b, c] = <q_b, d_c>.
        scores = torch.einsum("bd,cd->bc", query_embeddings, doc_embeddings)
        batch_size = scores.size(0)
        device = scores.device

        # Indices of queries and their corresponding positives.
        idx, pos_idx = self._get_idx(batch_size, offset, device)

        # Optionally filter out "too strong" in-batch negatives.
        if self.pos_aware_negative_filtering:
            self._filter_high_negatives(scores, pos_idx)

        # Update warmup weights BEFORE applying regularization.
        if self.q_regularizer is not None:
            self.q_regularizer.step()
        if self.d_regularizer is not None:
            self.d_regularizer.step()

        # SPLADE FLOPs regularization terms.
        reg_q = self.q_regularizer(query_embeddings) if self.q_regularizer is not None else 0.0
        reg_d = self.d_regularizer(doc_embeddings) if self.d_regularizer is not None else 0.0

        # Temperature scaling and cross-entropy.
        scores = scores / self.temperature
        ce_loss = self.ce_loss(scores, pos_idx)

        total_loss = ce_loss + reg_q + reg_d

        return total_loss


class SparseBiNegativeCELoss(SparseBiEncoderModule):
    """
    SPLADE-style contrastive loss with hard negatives + in-batch InfoNCE + FLOPs regularization.

    Output is a dict for logging, but the "loss" entry is a scalar tensor
    that MUST require grad (we assert it to catch issues early).
    """

    def __init__(
        self,
        temperature: float = 0.02,
        in_batch_term_weight: float = 0.5,
        pos_aware_negative_filtering: bool = False,
        max_batch_size: int = 1024,
        filter_threshold: float = 0.95,
        filter_factor: float = 0.5,
        q_regularizer: Optional[Regularizer] = None,  # Queries: mild sparsity
        d_regularizer: Optional[Regularizer] = None,  # Docs: stronger sparsity
        debug: bool = False,
    ):
        super().__init__(max_batch_size, temperature, filter_threshold, filter_factor)
        self.in_batch_term_weight = in_batch_term_weight
        self.pos_aware_negative_filtering = pos_aware_negative_filtering
        self.debug = debug

        # Avoid module instances as default args; create them per-loss instance.
        self.q_regularizer = q_regularizer or FLOPs(weight=0.0001, T=100000)
        self.d_regularizer = d_regularizer or FLOPs(weight=0.0001, T=100000)

        self.in_batch_loss_fn = SparseBiEncoderLoss(
            temperature=temperature,
            pos_aware_negative_filtering=pos_aware_negative_filtering,
            max_batch_size=max_batch_size,
            filter_threshold=filter_threshold,
            filter_factor=filter_factor,
            q_regularizer=None,
            d_regularizer=None,
        )

    def forward(
        self,
        query_embeddings: torch.Tensor,      # [B, D]
        doc_embeddings: torch.Tensor,        # [B, D] positives
        neg_doc_embeddings: torch.Tensor,    # [B, N, D] hard negatives
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        Compute SPLADE-style loss with hard negatives and in-batch negatives.

        Args:
            query_embeddings:    Tensor[B, D] query representations.
            doc_embeddings:      Tensor[B, D] positive documents.
            neg_doc_embeddings:  Tensor[B, N, D] hard negative documents per query.
            offset:              Offset for in-batch positive indices (multi-GPU).

        Returns:
            Dict with:
                "loss": total loss (contrastive + regularization) [scalar tensor]
                "contrastive_loss": hard_neg + in-batch mixture [scalar tensor]
        """
        B, N, D = neg_doc_embeddings.shape

        # === Hard negative loss ===
        pos_scores = (query_embeddings * doc_embeddings).sum(dim=-1) / self.temperature  # [B]
        neg_scores = torch.einsum("bd,bnd->bn", query_embeddings, neg_doc_embeddings) / self.temperature  # [B, N]
        hard_neg_loss = F.softplus(neg_scores - pos_scores.unsqueeze(1)).mean()

        # === In-batch InfoNCE ===
        if self.in_batch_term_weight > 0:
            in_batch_loss = self.in_batch_loss_fn(query_embeddings, doc_embeddings, offset)
            contrastive_loss = (
                hard_neg_loss * (1.0 - self.in_batch_term_weight)
                + in_batch_loss * self.in_batch_term_weight
            )
        else:
            # If no in-batch term, use only hard negatives.
            contrastive_loss = hard_neg_loss

        # === SPLADE FLOPs regularization ===
        # Update warmup weights before applying regularization.
        if self.q_regularizer is not None:
            self.q_regularizer.step()
        if self.d_regularizer is not None:
            self.d_regularizer.step()

        reg_q = self.q_regularizer(query_embeddings) if self.q_regularizer is not None else 0.0

        # Apply document regularizer to both positives and negatives.
        docs_all = torch.cat([doc_embeddings, neg_doc_embeddings.flatten(0, 1)], dim=0)  # [B + B*N, D]
        reg_d = self.d_regularizer(docs_all) if self.d_regularizer is not None else 0.0
        total_loss = contrastive_loss + reg_q + reg_d

        return {
            "loss": total_loss,
            "contrastive_loss": contrastive_loss,
            "reg_q": reg_q,
            "reg_d": reg_d,
            "query_length": num_active_terms(query_embeddings),
            "doc_length": num_active_terms(doc_embeddings),
        }
