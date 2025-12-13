import torch
from torch.distributed.nn.functional import all_gather
from colpali_engine.utils.sparse_rep import SparseRep

def gather_sparserep(rep: SparseRep) -> SparseRep:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return rep

    if hasattr(rep, "indices") and hasattr(rep, "values"):
        gathered_indices = all_gather(rep.indices)
        gathered_values  = all_gather(rep.values)
        return SparseRep(
            indices=torch.cat(gathered_indices, dim=0),
            values=torch.cat(gathered_values, dim=0),
            size=rep.size,
        )

    return rep
