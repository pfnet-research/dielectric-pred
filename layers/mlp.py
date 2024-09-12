from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_feat: int,
        output_feat: int,
        latent_feat: int,
        n_layers: int = 3,
        activate_last: bool = False,):
        """Init GatedMLP for scalar readout NN.
        
        Args:
            input_feat: input dim of MLP
            output_feat: output dim of MLP
            latent_feat: latent dim of MLP
            n_layers: number of MLP layers
            activate_last: if activate the final layer
        """

        super(GatedMLP, self).__init__()

        self.mlp = nn.Sequential()
        dims = [input_feat]+[latent_feat]*(n_layers-1)+[output_feat]
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            self.mlp.append(nn.Linear(in_dim, out_dim))
            if i < n_layers-1:
                self.mlp.append(nn.SiLU())
        if activate_last:
            self.mlp.append(nn.SiLU())
        self.reset_parameters()

    def _weights_init(self, layer):
        if isinstance(layer, nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

    def reset_parameters(self):
        self.mlp.apply(self._weights_init)

    def forward(self, x):
        return self.mlp(x)
    

class GatedMLP(nn.Module):
    def __init__(
        self,
        input_feat: int,
        output_feat: int,
        latent_feat: int,
        n_layers: int = 3):
        """Init GatedMLP for scalar readout NN.
        
        Args:
            input_feat: Input dim of MLP
            output_feat: Output dim of MLP
            latent_feat: Latent dim of MLP
            n_layers: Num of MLP layers
        """

        super(GatedMLP, self).__init__()

        self.mlp = nn.Sequential()
        self.gate_mlp = nn.Sequential()
        dims = [input_feat]+[latent_feat]*(n_layers-1)+[output_feat]
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            self.mlp.append(nn.Linear(in_dim, out_dim))
            self.gate_mlp.append(nn.Linear(in_dim, out_dim))
            if i < n_layers-1:
                self.mlp.append(nn.SiLU())
                self.gate_mlp.append(nn.SiLU())
            else:
                self.gate_mlp.append(nn.Sigmoid())

        self.reset_parameters()

    def _weights_init(self, layer):
        if isinstance(layer, nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

    def reset_parameters(self):
        self.gate_mlp.apply(self._weights_init)
        self.mlp.apply(self._weights_init)

    def forward(self, x):
        return self.mlp(x) * self.gate_mlp(x)