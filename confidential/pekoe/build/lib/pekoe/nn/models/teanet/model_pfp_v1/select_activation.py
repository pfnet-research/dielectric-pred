from enum import Enum
from typing import Callable, Tuple

import torch
import torch.nn.functional as F

try:
    from .polylog import polylog, polylog_shift
except ImportError:
    from chicle.functions.polylog import polylog, polylog_shift

try:
    from .softplus import shifted_softplus
except ImportError:
    from chicle.functions.softplus import shifted_softplus

# try:
#     from chicle.functions.cpp.cutoff.cutoff_function import invnegexp, xinvnegexp
#     from chicle.functions.cpp.teanet_activation.teanet_activation import ssp
# except ImportError:
#     from .cpp.cutoff.cutoff_function import invnegexp, xinvnegexp
#     from .cpp.teanet_activation.teanet_activation import ssp


class TeaNetActivation(Enum):
    SOFTPLUS = "softplus"
    SHIFTED_SOFTPLUS = "shifted_softplus"
    XSOFTPLUS = "xsoftplus"
    SSP = "ssp"
    POLYLOG = "polylog"
    SWISH = "swish"
    ORIGINAL = "original"


ActivationFunctionType = Callable[[torch.Tensor], torch.Tensor]


softplus_shift = shifted_softplus


def exp_shift(x: torch.Tensor) -> torch.Tensor:
    """exp(x) with shifted to take 0. at x=0."""
    return torch.exp(x) - 1.0


def xsoftplus(x: torch.Tensor) -> torch.Tensor:
    """x * softplus"""
    y: torch.Tensor = x * F.softplus(x)
    return y


def xsoftplus_grad(x: torch.Tensor) -> torch.Tensor:
    """Grad of xsoftplus"""
    y: torch.Tensor = x * torch.sigmoid(x) + F.softplus(x)
    return y


def swish(x: torch.Tensor) -> torch.Tensor:
    """Swish activation, x * sigmoid"""
    return x * torch.sigmoid(x)


def swish_grad(x: torch.Tensor) -> torch.Tensor:
    """Grad of swish activation"""
    sx = torch.sigmoid(x)
    return sx + x * sx * (1.0 - sx)  # type: ignore


def ssp(x: torch.Tensor) -> torch.Tensor:
    """It converges to x log(x) when x -> infty

    The term `1.0 + F.softplus(x)` instead of `x` is to make domain all real space,
    and it is differentiable.
    """
    return x * torch.log(1.0 + F.softplus(x))


def ssp_grad(x: torch.Tensor) -> torch.Tensor:
    """Grad of ssp"""
    y: torch.Tensor = torch.log(1.0 + F.softplus(x)) + x / (1.0 + torch.exp(-x)) / (
        1.0 + F.softplus(x)
    )
    return y


def teanet_select_activation(
    activation_id: TeaNetActivation, is_child: bool
) -> Tuple[ActivationFunctionType, ActivationFunctionType, ActivationFunctionType,]:
    """Select activation function for TeaNet

    3 activation function is returned. These are used to construct cutoff function in TeaNet.

    Args:
        activation_id (TeaNetActivation): ideitification of activation
        is_child (bool):

    Returns:
        activation (callable): activation function
        activation_shift (callable): activation function, shifted to return 0 when input x=0
        activation_grad (callable): grad of activation
    """

    if activation_id == TeaNetActivation.SOFTPLUS:
        activation: ActivationFunctionType = F.softplus
        activation_shift: ActivationFunctionType = softplus_shift
        activation_grad: ActivationFunctionType = torch.sigmoid
    elif activation_id == TeaNetActivation.SHIFTED_SOFTPLUS:
        activation = softplus_shift
        activation_shift = softplus_shift
        activation_grad = torch.sigmoid
    elif activation_id == TeaNetActivation.XSOFTPLUS:
        activation = xsoftplus
        activation_shift = xsoftplus
        activation_grad = xsoftplus_grad
    elif activation_id == TeaNetActivation.SSP:
        activation = ssp
        activation_shift = ssp
        activation_grad = ssp_grad
    elif activation_id == TeaNetActivation.POLYLOG:
        activation = polylog
        activation_shift = polylog_shift
        activation_grad = F.softplus
    elif activation_id == TeaNetActivation.SWISH:
        activation = swish
        activation_shift = swish
        activation_grad = swish_grad
    elif activation_id == TeaNetActivation.ORIGINAL:
        if is_child:
            activation = polylog
            activation_shift = polylog_shift
            activation_grad = F.softplus
        else:
            activation = torch.exp
            activation_shift = exp_shift
            activation_grad = torch.exp
    else:
        raise NotImplementedError(f"activation_id {activation_id} not supported")
    return activation, activation_shift, activation_grad


class TeaNetCutoffFunction(Enum):
    INVNEGEXP = "invnegexp"
    XINVNEGEXP = "xinvnegexp"
    TRANSITION = "transition"


CutoffFunctionType = Callable[[torch.Tensor, torch.nn.Linear], torch.Tensor]


def invnegexp(x: torch.Tensor) -> torch.Tensor:
    """Smooth cutoff function `exp(-1/x)`

    Args:
        x (torch.Tensor): input value

    Returns:
        torch.Tensor: cutoff value. 0.0 if x < 0.
    """
    y = torch.max(x, 1.0e-5 * torch.ones_like(x))
    res = torch.exp(-torch.reciprocal(y)) / 2.7182818285
    return res


def xinvnegexp(x: torch.Tensor) -> torch.Tensor:
    y = torch.max(x, 1.0e-5 * torch.ones_like(x))
    res = x * torch.exp(-torch.reciprocal(y)) / 2.7182818285 / 6.0
    return res


def smooth_transition(x: torch.Tensor) -> torch.Tensor:
    x1 = invnegexp(x)
    x2 = invnegexp(1.0 - x)
    return x1 / (x1 + x2)


def select_cutoff_function(
    cutoff_id: TeaNetCutoffFunction,
) -> CutoffFunctionType:
    if cutoff_id == TeaNetCutoffFunction.INVNEGEXP:
        nonlinear_function = invnegexp
    elif cutoff_id == TeaNetCutoffFunction.XINVNEGEXP:
        nonlinear_function = xinvnegexp
    elif cutoff_id == TeaNetCutoffFunction.TRANSITION:
        nonlinear_function = smooth_transition
    else:
        raise ValueError("Unknown cutoff function: {}".format(cutoff_id))

    def cutoff_function(length: torch.Tensor, linear: torch.nn.Linear) -> torch.Tensor:
        return nonlinear_function(linear(length))

    return cutoff_function
