from __future__ import annotations

import torch
import torch.nn as nn

from layers.mlp import GatedMLP

class TensorReadout(nn.Module):
    def __init__(self, 
            ns_feat: int, 
            nv_feat: int, 
            nt_feat: int, 
            latent_feat: int, 
            mlp_layer: int =3):
        """Init TensorReadout for tensor output.

        Args:
            ns_feat: dim of node scalars
            nv_feat: dim of node vectors
            nt_feat: dim of node tensors
            latent_feat: dim of latent features
            mlp_layer: num of mlp layers for node scalars
        """
        super().__init__()
        self.linear_nv = nn.Linear(nv_feat, 1, bias=False)
        self.linear_nt = nn.Linear(nt_feat, 1, bias=False)
        self.dense_ns = GatedMLP(ns_feat, 1, latent_feat, n_layers=mlp_layer)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.linear_nv.weight)
        torch.nn.init.xavier_uniform_(self.linear_nt.weight)


    def forward(self, h_ns, h_nv, h_nt):

        h_nv = self.linear_nv(h_nv) # N x 3 x 1
        h_nv = h_nv @ h_nv.transpose(1,2)   # expand to N x 3 x 3
        h_nt = self.linear_nt(h_nt).squeeze(-1)     # N x 3 x 3

        h_ns = self.dense_ns(h_ns).expand(-1, 3)
        h_ns = h_ns.unsqueeze(-1) * torch.eye(3, device=h_ns.device).unsqueeze(0)
        # assert h_ns.shape == h_nv.shape == h_nt.shape

        res_node = h_ns + h_nv + h_nt

        return res_node
    
    
class ScalarReadout(nn.Module):
    def __init__(self, ns_feat: int, latent_feat: int, mlp_layer: int=3):
        """Init ScalarReadout for scalar output.
           Similar to the class TensorReadout, but this only outputs a scalar prediction.

        Args:
            ns_feat: dim of node scalars
            latent_feat: dim of latent features
            mlp_layer: num of mlp layers for node scalar readout
        """
        super(ScalarReadout, self).__init__()
        self.dense_ns = GatedMLP(ns_feat, 1, latent_feat, n_layers=mlp_layer)

    def forward(self, h_ns):

        h_ns = self.dense_ns(h_ns)

        return h_ns