from typing import List

import torch


class TeaNetBase(torch.nn.Module):
    def __init__(self):
        super(TeaNetBase, self).__init__()
        self.cutoff_list: List[float] = list()
        self.device = torch.zeros(0).device
        self.is_codegen = False
        self.coord_dtype = torch.float64

    def to(self, device: torch.device, **kwargs) -> None:
        super(TeaNetBase, self).to(device, **kwargs)
        self.device = device

    def torch_device_from_str(self, device_str: str) -> torch.device:
        return torch.device(device_str)
