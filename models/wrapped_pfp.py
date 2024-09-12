from __future__ import annotations

import torch


class WrappedPFP(torch.nn.Module):
    def __init__(self, pretrained_pfp, **kwargs):
        """Interface to a pre-trained PFP model
        
        This inputs graphs built from materials to PFP,
        and outputs intermediate features of PFP for the equivariant readout module.

        Args:
            pretrained_pfp: This is our commercial neural network potential model, see details at https://matlantis.com/
        """
        super().__init__()
        self.pretrained_pfp = pretrained_pfp
        pass
    
    def forward(self, **pfp_inputs):
        pass