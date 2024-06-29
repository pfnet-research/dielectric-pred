import math

import torch
import torch.nn.functional as F


class Polylog(torch.autograd.Function):
    """Integral of softplus function

    Refer https://drive.google.com/file/d/1hT14gxxIB9ZQIugYxrPEU4ert_EFJ3tw/view
    """

    def __init__(self):
        super(Polylog, self).__init__()

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        absx = torch.abs(x)
        signx = torch.sign(x)
        lix = 1.0 / (1.0 + torch.exp(absx))
        li2_sum = torch.zeros_like(lix)
        for k in range(34, 0, -1):
            k_mul = 1.0 / k / k
            li2_sum = lix * (li2_sum + k_mul)
        softplus_absx = F.softplus(absx)
        neg_shift = 0.5 * (1.0 - signx) * (-math.pi * math.pi / 6.0 - 0.5 * x * x)
        li2_sum += -math.pi * math.pi / 6.0 + (-absx + 0.5 * softplus_absx) * softplus_absx
        li2_sum = signx * li2_sum + neg_shift
        return -li2_sum

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * F.softplus(x)


def polylog(x: torch.Tensor) -> torch.Tensor:
    """polylog function, whose backward is `softplus` function.

    Args:
        x (torch.Tensor): input

    Returns:
        output (torch.Tensor): output
    """
    return Polylog.apply(x)  # type: ignore


def polylog_shift(x: torch.Tensor) -> torch.Tensor:
    """polylog with shifted to take 0 at x=0, `polylog_shift(0.) = 0.`.

    Args:
        x (torch.Tensor): input

    Returns:
        output (torch.Tensor): output
    """
    return Polylog.apply(x) - 0.82246703342411320303  # type: ignore
