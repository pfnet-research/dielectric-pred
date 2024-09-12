
import torch

from release.models.wrapped_pfp import WrappedPFP
from models.equivariant_model import GatedEquivariantModel

class CombinedModel(torch.nn.Module):
    def __init__(self, pfp_wrapped: WrappedPFP, tensorial_model: GatedEquivariantModel):
        """Init CombineModel with the wrapped PFP and equivariant readout NN.
        
        Args:
            pfp_wrapped: a wrapped pre-trained PFP
            tensorial_model: an equivariant readout NN to be trained
        """
        super().__init__()
        self.pfp_wrapped = pfp_wrapped
        self.tensorial_model = tensorial_model

    def forward(self, **pfp_inputs):
        pfp_outputs = self.pfp_wrapped(
            **pfp_inputs
        )
        dielectric = self.tensorial_model(
            *pfp_outputs)
        
        return dielectric