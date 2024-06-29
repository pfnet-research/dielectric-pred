import torch

_use_mncore = False


def use_mncore():
    global _use_mncore
    _use_mncore = True


def index_select(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if not _use_mncore:
        return x.index_select(0, indices)

    # Use 2D indexing for better MN-Core utilization.
    shape = x.shape
    x = x.view((shape[0], -1))
    y = x.index_select(0, indices)
    y = y.view((y.shape[0],) + shape[1:])
    return y


def _index_add_by_scatter(x: torch.Tensor, indices: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    indices = indices.view(*(indices.shape + (1,) * (x.ndim - 1))).expand(*a.shape)
    return x.scatter_add(0, indices, a)


def _index_add_by_index_add(
    x: torch.Tensor, indices: torch.Tensor, a: torch.Tensor
) -> torch.Tensor:
    shape = x.shape
    # Use 2D indexing for better MN-Core utilization.
    x = x.view((shape[0], -1))
    a = a.view((a.shape[0], -1))
    if hasattr(indices, "mask"):
        # Eliminate the effect of padded edges.
        a = a * indices.mask.reshape(-1, 1)
    y = x.index_add(0, indices, a)
    return y.view(shape)


def index_add(x: torch.Tensor, indices: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    if _use_mncore:
        fn = _index_add_by_index_add
    else:
        fn = _index_add_by_scatter
    return fn(x, indices, a)
