import torch
from torch.autograd import Function


def _index_add_by_scatter(x, i, a):
    i = i.view(*(i.shape + (1, ) * (x.ndim - 1))).expand(*a.shape)
    return x.scatter_add(0, i, a)


class GatherMulSum(Function):
    @staticmethod
    def symbolic(g, x, i, f):
        return g.op("ChainerTeaNetGatherMulSum", x, i, f)

    @staticmethod
    def forward(ctx, x, i, f):
        ctx.save_for_backward(x, i, f)
        return (x[i] * torch.unsqueeze(f, dim=1)).sum(dim=2)

    @staticmethod
    def backward(ctx, gy):
        x, i, f = ctx.saved_tensors
        gy = torch.unsqueeze(gy, dim=2)
        gf = (gy * x[i]).sum(dim=1).sum_to_size(f.shape)
        gx = _index_add_by_scatter(torch.zeros_like(x), i, gy * torch.unsqueeze(f, dim=1))
        return gx, None, gf


class MulIndexAddAdd(Function):
    @staticmethod
    def symbolic(g, x, y, i, j, f):
        return g.op("ChainerTeaNetMulIndexAddAdd", x, y, i, j, f)

    @staticmethod
    def forward(ctx, x, y, i, j, f):
        ctx.save_for_backward(y, i, j, f)
        z = y * f
        r = _index_add_by_scatter(x, i, z)
        r = _index_add_by_scatter(r, j, z)
        return r

    @staticmethod
    def backward(ctx, gx):
        y, i, j, f = ctx.saved_tensors
        gz = gx[i] + gx[j]
        gy = (f * gz).sum_to_size(y.shape)
        gf = (y * gz).sum_to_size(f.shape)
        return None, gy, None, None, gf


class MulIndexAddSub(Function):
    @staticmethod
    def symbolic(g, x, y, i, j, f):
        return g.op("ChainerTeaNetMulIndexAddSub", x, y, i, j, f)

    @staticmethod
    def forward(ctx, x, y, i, j, f):
        ctx.save_for_backward(y, i, j, f)
        z = y * f
        r = -_index_add_by_scatter(x, j, z)
        r = _index_add_by_scatter(r, i, z)
        return r

    @staticmethod
    def backward(ctx, gx):
        y, i, j, f = ctx.saved_tensors
        gz = gx[i] + (-gx)[j]
        gy = (f * gz).sum_to_size(y.shape)
        gf = (y * gz).sum_to_size(f.shape)
        return None, gy, None, None, gf


# For slightly faster computation.
# gather_mul_sum = lambda x, i, f: (x[i] * f.unsqueeze(dim=1)).sum(dim=2)
gather_mul_sum = GatherMulSum.apply

mul_index_add_add = MulIndexAddAdd.apply
mul_index_add_sub = MulIndexAddSub.apply
