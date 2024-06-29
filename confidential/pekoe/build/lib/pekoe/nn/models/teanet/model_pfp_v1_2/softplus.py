import torch
import torch.nn.functional as F


def shifted_softplus(x: torch.Tensor) -> torch.Tensor:
    """Modified softplus function used by MEGNet and SchNet

    The original implementation is below.
    https://github.com/materialsvirtuallab/megnet/blob/f91773f0f3fa8402b494638af9ef2ed2807fcba7/megnet/activations.py#L6   # NOQA

    This function is mathematically equivalent to the shifted softplus function from SchNet.
    https://papers.nips.cc/paper/6700-schnet-a-continuous-filter-convolutional-neural-network-for-modeling-quantum-interactions  # NOQA

    .. math::
        y = \\ln(0.5 e^x + 0.5)

    Args:
        x (torch.Tensor): Input variable
    Returns:
        output (torch.Tensor): Output variable whose shape is same with `x`
    """
    return F.relu(x) + torch.log(0.5 * torch.exp(-torch.abs(x)) + 0.5)
