from typing import Union

import cupy
import numpy
import torch

ndarray = Union[numpy.ndarray, cupy.ndarray]


def astensor(arr: ndarray) -> torch.Tensor:
    if isinstance(arr, numpy.ndarray):
        return torch.as_tensor(arr)
    elif isinstance(arr, cupy.ndarray):
        index = arr.device.id
        # TODO: fix "error: Module 'torch.cuda' has no attribute 'device';"
        with torch.cuda.device(index):  # type: ignore
            return torch.as_tensor(arr, device=torch.device(f"cuda:{index}"))
    else:
        raise TypeError()


def asarray(ten: torch.Tensor) -> ndarray:
    device = ten.device
    if device.type == "cpu":
        return numpy.asarray(ten)
    elif device.type == "cuda":
        with cupy.cuda.Device(device.index):
            return cupy.asarray(ten)
    else:
        raise ValueError()
