
import torch

from models.equivariant_model import GatedEquivariantModel

class WrappedPFP(torch.nn.Module):
    def __init__(self, pfp, calc_mode: int = 1, return_layer: int=3):
        """Init Wrapped PFP convenient for calc_mode and return layer setting

        Args:
            pfp: pre-trained PFP
            calc_mode: calculation mode of PFP
            return_layer: output i-th intermediate layer from PFP
        """
        super().__init__()
        self.pfp = pfp
        self.repulsion_calc_mode_int = calc_mode
        self.return_layer = return_layer

    def forward(self, vecs, atomic_numbers, atom_index1, atom_index2, ob, ba, be, xa):
        calc_mode_type = self.repulsion_calc_mode_int * torch.ones(
            (atomic_numbers.size()[0],), dtype=torch.int64, device=vecs.device)
        return self.pfp(vecs, atomic_numbers, atom_index1, atom_index2, ob, ba, be, xa, calc_mode_type, self.return_layer)
    
class CombinedModel(torch.nn.Module):
    def __init__(self, pfp_wrapped: WrappedPFP, tensorial_model: GatedEquivariantModel):
        """Init CombineModel_v1_4 with wrapped Teaneat and equivariant readout NN.
        
        Args:
            pfp_wrapped: A wrapped pre-trained PFP
            tensorial_model: A equivariant readout NN to be trained
        """
        super().__init__()
        self.pfp_wrapped = pfp_wrapped
        self.tensorial_model = tensorial_model
        self.dropout_ratio = None   # should be removed

    def forward(self, vecs, atomic_numbers, atom_index1, atom_index2, ob, ba, be, xa, mask=None):
        h_ns, h_nv, h_nt, layer_es, layer_ev, h_ncharge, atom_index1, atom_index2 = self.pfp_wrapped(
            vecs, atomic_numbers, atom_index1, atom_index2, ob, ba, be, xa
        )
        dielectric = self.tensorial_model(
            h_ns, ba, h_nv, h_nt, layer_es, layer_ev, atom_index1, atom_index2, mask)
        return dielectric

    def to(self, device: torch.device, **kwargs) -> None:
        super().to(device, **kwargs)
        self.device = device