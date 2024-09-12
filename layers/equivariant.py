
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
    

class NodeEquivariantBlock(nn.Module):
    def __init__(
        self,
        ns_feat: int,
        nv_feat: int,
        nt_feat: int,
        act_gate: bool = True,
        residue: bool = True,
        dropout_rate: float = 0.):
        """Init NodeEquivariantBlock with key parameters.
           Similar to the class GatedEquivariantBlock, but without edge information integrated.

        Args:
            ns_feat: dim of node scalars
            nv_feat: dim of node vectors
            nt_feat: dim of node tensors
            act_gate: whether to use a sigmoid activation function before gate
            residue: whether to use residual connection
            dropout_rate: use dropout if > 0
        """
        super().__init__()

        self.ns_feat = ns_feat
        self.nv_feat = nv_feat
        self.nt_feat = nt_feat

        self.residue = residue
        self.dropout_rate = dropout_rate
        if dropout_rate > 0.:
            self.dropout = nn.Dropout(dropout_rate)
        self.residue = residue
        self.act_gate = act_gate

        self.ns_dense1 = nn.Sequential(
            nn.Linear(ns_feat, ns_feat//2),
            nn.SiLU(),
            nn.Linear(ns_feat//2, ns_feat//2))
        
        self.nv_linear1 = nn.Linear(nv_feat, nv_feat, bias=False)
        self.nv_linear2 = nn.Linear(nv_feat, nv_feat, bias=False)

        self.nt_linear1 = nn.Linear(nt_feat, nt_feat, bias=False)
        self.nt_linear2 = nn.Linear(nt_feat, nt_feat, bias=False)

        cat_feat = ns_feat//2 + nv_feat + nt_feat
        out_feat = ns_feat + nv_feat + nt_feat
        self.cat_dense = nn.Sequential(
            nn.Linear(cat_feat, cat_feat//2),
            nn.SiLU(),
            nn.Linear(cat_feat//2, out_feat))

        self._reset_parameters()
    
    def _weights_init(self, layer):
        if isinstance(layer, nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

    def _reset_parameters(self):
        self.ns_dense1.apply(self._weights_init)

        torch.nn.init.xavier_uniform_(self.nv_linear1.weight)
        torch.nn.init.xavier_uniform_(self.nv_linear2.weight)

        torch.nn.init.xavier_uniform_(self.nt_linear1.weight)
        torch.nn.init.xavier_uniform_(self.nt_linear2.weight)

        self.cat_dense.apply(self._weights_init)


    def forward(
        self,
        h_ns: Tensor,
        h_nv: Tensor,
        h_nt: Tensor):

        # integrate h_ns
        h_ns_dense1 = self.ns_dense1(h_ns)

        h_ns_int = h_ns_dense1

        # integrate h_nv
        h_nv_linear1 = self.nv_linear1(h_nv)
        h_nv_linear2 = self.nv_linear2(h_nv)

        h_nv_int = h_nv_linear1
        h_nv_int_norm = torch.norm(h_nv_int, dim=1)

        h_nt_linear1 = self.nt_linear1(h_nt)
        h_nt_linear2 = self.nt_linear2(h_nt)
        h_nt_norm = torch.norm(h_nt_linear1, dim=(1,2))

        h_cat = torch.cat([h_ns_int, h_nv_int_norm, h_nt_norm], dim=1)
        ns_out, nv_gate, nt_gate = torch.split(self.cat_dense(h_cat),
            [self.ns_feat, self.nv_feat, self.nt_feat], dim=1)

        if self.act_gate:
            nv_gate = torch.sigmoid(nv_gate)
            nt_gate = torch.sigmoid(nt_gate)

        nv_out = nv_gate.view(-1, 1, self.nv_feat) * h_nv_linear2
        nt_out = nt_gate.view(-1, 1, 1, self.nt_feat) * h_nt_linear2

        if self.dropout_rate > 0:
            ns_out = self.dropout(ns_out)

        if self.residue:
            ns_out = ns_out + h_ns
            nv_out = nv_out + h_nv
            nt_out = nt_out + h_nt

        return ns_out, nv_out, nt_out


class GatedEquivariantBlock(NodeEquivariantBlock):
    def __init__(
        self,
        es_feat: int,
        ev_feat: int,
        **kwargs
        ):
        """Init GatedEquivariantBlock with key parameters.

        Args:
            ns_feat: dim of node scalars
            nv_feat: dim of node vectors
            nt_feat: dim of node tensors
            es_feat: dim of edge scalars
            ev_feat: dim of edge vectors
            act_gate: whether to use a sigmoid activation function before gate
            residue: whether to use residual connection
            dropout_rate: use dropout if > 0
        """

        super().__init__(**kwargs)

        self.es_feat = es_feat
        self.ev_feat = ev_feat

        self.es_dense1 = nn.Sequential(
            nn.Linear(es_feat, es_feat//2),
            nn.SiLU(),
            nn.Linear(es_feat//2, es_feat//2))
        
        self.ev_linear1 = nn.Linear(ev_feat, ev_feat, bias=False)

        self.reset_parameters()


    def _weights_init(self, layer):
        if isinstance(layer, nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

    def reset_parameters(self):
        super()._reset_parameters()

        self.es_dense1.apply(self._weights_init)
        torch.nn.init.xavier_uniform_(self.ev_linear1.weight)


    def forward(
        self,
        h_ns: Tensor,
        h_nv: Tensor,
        h_nt: Tensor,
        h_es: Tensor,
        h_ev: Tensor,
        atom_index1: Tensor,
        atom_index2: Tensor):

        # Integrate h_ns with h_es
        h_ns_dense1 = self.ns_dense1(h_ns)
        h_es_dense1 = self.es_dense1(h_es)

        h_ns_es1 = torch.zeros_like(h_ns_dense1).scatter_add(
            0, atom_index1.view(-1, 1).expand(-1, self.ns_feat//2), h_es_dense1
        )
        h_ns_es2 = torch.zeros_like(h_ns_dense1).scatter_add(
            0, atom_index2.view(-1, 1).expand(-1, self.ns_feat//2), h_es_dense1
        )
        # if not self.integrate_es_ev:
        #     h_ns_es1, h_ns_es2 = torch.zeros_like(h_ns_es1), torch.zeros_like(h_ns_es2)
        h_ns_int = h_ns_dense1 + h_ns_es1 + h_ns_es2

        # Integrate h_nv with h_ev
        h_nv_linear1 = self.nv_linear1(h_nv)
        h_nv_linear2 = self.nv_linear2(h_nv)
        h_ev_linear1 = self.ev_linear1(h_ev)

        h_nv_ev1 = torch.zeros_like(h_nv_linear1).scatter_add(
            0, atom_index1.view(-1, 1, 1).expand(-1, 3, self.nv_feat), h_ev_linear1
        )
        h_nv_ev2 = torch.zeros_like(h_nv_linear1).scatter_add(
            0, atom_index2.view(-1, 1, 1).expand(-1, 3, self.nv_feat), h_ev_linear1
        )

        h_nv_int = h_nv_linear1 + h_nv_ev1 + h_nv_ev2
        # Information interactions among node scalars/vectors/tensors
        h_nv_int_norm = torch.norm(h_nv_int, dim=1)

        h_nt_linear1 = self.nt_linear1(h_nt)
        h_nt_linear2 = self.nt_linear2(h_nt)
        h_nt_norm = torch.norm(h_nt_linear1, dim=(1,2))

        h_cat = torch.cat([h_ns_int, h_nv_int_norm, h_nt_norm], dim=1)
        ns_out, nv_gate, nt_gate = torch.split(self.cat_dense(h_cat),
            [self.ns_feat, self.nv_feat, self.nt_feat], dim=1)

        if self.act_gate:
            nv_gate = torch.sigmoid(nv_gate)
            nt_gate = torch.sigmoid(nt_gate)

        nv_out = nv_gate.view(-1, 1, self.nv_feat) * h_nv_linear2
        nt_out = nt_gate.view(-1, 1, 1, self.nt_feat) * h_nt_linear2

        if self.dropout_rate > 0:
            ns_out = self.dropout(ns_out)

        if self.residue:
            ns_out = ns_out + h_ns
            nv_out = nv_out + h_nv
            nt_out = nt_out + h_nt

        return ns_out, nv_out, nt_out