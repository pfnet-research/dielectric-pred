from __future__ import annotations

import torch
import torch.nn as nn

from layers import (
    GatedEquivariantBlock,
    NodeEquivariantBlock,
    TensorReadout,
    ScalarReadout
)

class GatedEquivariantModel(nn.Module):
    def __init__(
        self,
        ns_feat: int,
        nv_feat: int,
        nt_feat: int,
        es_feat: int | None = None,
        ev_feat: int | None = None,
        latent_feat: int = 64,
        n_gate_layers: int = 2,
        dropout_rate: float = 0.,
        residual: bool = True,
        gate_sigmoid: bool = True,
        mlp_layer: int = 3,
        integrate_nv_nt=True,  
        integrate_es_ev=True,
        apply_mask=False):
        """Init GatedEquivariantModel with key parameters

        This model gets input from PFP intermediate layers, and processes the data
        with equivariant blocks, readout layers and node integration.

        Args:
            ns_feat: Dim of node scalars
            nv_feat: Dim of node vectors
            nt_feat: Dim of node tensors
            es_feat: Dim of edge scalars
            ev_feat: Dim of edge vectors
            latent_feat: Dim of latent features
            n_gate_layers: Num of stacked Gated Equivariant blocks
            dropout_rate: Dropout rate in Gated Equivariant blocks
            residue: Whether to use residual connection
            gate_sigmoid: Whether to add a sigmoid function for gating node vectors/tensors
            mlp_layer: Num of mlp layers for node scalar readout
            integrate_nv_nt: Whether to integrate node vectors/tensors information, the output will be a scalar if false
            integrate_es_ev: Whether to integrate edge scalars/vectors information
            apply_mask: Whether to mask off-diag elements during training
        """
        super().__init__()

        self.integrate_nv_nt = integrate_nv_nt
        self.integrate_es_ev = integrate_es_ev

        self.n_gate_layers = n_gate_layers

        if integrate_nv_nt:
            if integrate_es_ev:
                for i in range(0, n_gate_layers):
                    self.add_module("edge_gated_equi_block_%d" %i, 
                        GatedEquivariantBlock(
                            ns_feat=ns_feat,
                            nv_feat=nv_feat,
                            nt_feat=nt_feat,
                            es_feat=es_feat,
                            ev_feat=ev_feat,
                            act_gate=gate_sigmoid,
                            residue=residual,
                            dropout_rate=dropout_rate
                        ))
            else:
                for i in range(0, n_gate_layers):
                    self.add_module("node_gated_equi_block_%d" %i, 
                        NodeEquivariantBlock(
                            ns_feat=ns_feat,
                            nv_feat=nv_feat,
                            nt_feat=nt_feat,
                            act_gate=gate_sigmoid,
                            residue=residual,
                            dropout_rate=dropout_rate,
                        ))

            self.readout = TensorReadout(ns_feat, nv_feat, nt_feat, latent_feat, mlp_layer=mlp_layer)
        else:
            self.readout = ScalarReadout(ns_feat, latent_feat, mlp_layer)
        
        self.apply_mask = apply_mask


    @classmethod
    def from_config_dict(
        cls,
        **configs
        ):
        model = cls(**configs)
        return model

    def forward(
        self,
        h_ns: torch.Tensor,
        ba: torch.Tensor,
        h_nv: torch.Tensor = None,
        h_nt: torch.Tensor = None,
        h_es: torch.Tensor = None,
        h_ev: torch.Tensor = None,
        atom_index1: torch.Tensor = None,
        atom_index2: torch.Tensor = None,
        mask: torch.Tensor = None
        ):

        # Get graph information
        n_nodes = h_ns.shape[0]
        n_graph = ba.max() + 1
        n_count_zero = torch.zeros(n_graph, device=ba.device, dtype=ba.dtype)
        n_count = n_count_zero.scatter_add(0, ba, torch.ones_like(ba))

        h_nt = (h_nt + h_nt.transpose(1, 2)) / 2.0  # sym


        # Information exchange in gated equivariant blocks
        if self.integrate_es_ev:
            for i in range(self.n_gate_layers):
                h_ns, h_nv, h_nt = self._modules["edge_gated_equi_block_%d" %i](
                    h_ns, h_nv, h_nt, h_es, h_ev, atom_index1, atom_index2)
        else:
            if self.integrate_nv_nt:
                for i in range(self.n_gate_layers):
                    h_ns, h_nv, h_nt = self._modules["node_gated_equi_block_%d" %i](
                        h_ns, h_nv, h_nt)

        # Readout
        if self.integrate_nv_nt:
            res_node = self.readout(h_ns, h_nv, h_nt)

            # Aggregation for output
            out_node_zero = torch.zeros([n_graph, 3, 3], device=h_ns.device)
            output_node = out_node_zero.scatter_add(
                0, ba.view([-1, 1, 1]).expand([n_nodes, 3, 3]), res_node
            ) / n_count.view(-1, 1, 1)    # B x 3 x 3
            
            if self.apply_mask and mask is not None:
                assert output_node.shape == mask.shape
                output_node = output_node * mask

        else:
            res_node = self.readout(h_ns)
            out_node_zero = torch.zeros([n_graph, 1], device=h_ns.device)
            output_node = out_node_zero.scatter_add(
                0, ba.view([-1, 1]).expand([n_nodes, 1]), res_node
            ) / n_count.view(-1, 1)    # B x 1

        return output_node