import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple

import dacite
import torch

from pekoe.nn.models import DEFAULT_MODEL_DIRECTORY
from pekoe.nn.models.config import BaseConfig
from pekoe.nn.models.teanet.model_pfp_v1_3.indexing import index_add
from pekoe.nn.models.teanet.model_pfp_v1_3.select_activation import (
    TeaNetActivation,
    teanet_select_activation,
)
from pekoe.nn.models.teanet.model_pfp_v1_3.teanet import TeaNet_v1_3, TeaNetParameters_v1_3

_NODE_FEATURE_SIZE = 513

logger = logging.getLogger(__name__)


@dataclass
class TeaNetNodeFeatureMLPParameters_v1_3:
    activation: TeaNetActivation
    n_hiddens: List[int]

    @classmethod
    def from_dict(class_, d: Dict[Any, Any]) -> "TeaNetNodeFeatureMLPParameters_v1_3":
        return dacite.from_dict(
            data_class=class_,
            data=d,
            config=dacite.Config(cast=[TeaNetActivation]),
        )


class TeaNetNodeFeatureMLP_v1_3(TeaNet_v1_3):
    def __init__(self, parameters: TeaNetNodeFeatureMLPParameters_v1_3) -> None:

        base_config = BaseConfig.from_yaml(DEFAULT_MODEL_DIRECTORY / "model_v1_3_1.yaml")
        teanet_parameters = TeaNetParameters_v1_3.from_dict(base_config.parameters)
        teanet_parameters.return_node_feature = True
        super().__init__(teanet_parameters)
        state = torch.load(  # type: ignore [no-untyped-call]
            str(base_config.weights_path),
            map_location="cpu",
        )
        super().load_state_dict(state)
        self.mlp = _MLP(parameters)

    def forward(
        self,
        edge_vector: torch.Tensor,
        atomic_numbers: torch.Tensor,
        left_indices: torch.Tensor,
        right_indices: torch.Tensor,
        num_graphs_arr: torch.Tensor,
        batch: torch.Tensor,
        batch_edge: torch.Tensor,
        atomic_numbers_add: torch.Tensor,
        calc_mode_type: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        teanet_energy, charge, node_scalar_feature = super().forward(
            edge_vector,
            atomic_numbers,
            left_indices,
            right_indices,
            num_graphs_arr,
            batch,
            batch_edge,
            atomic_numbers_add,
            calc_mode_type=calc_mode_type,
        )
        assert node_scalar_feature.size(1) == _NODE_FEATURE_SIZE
        correction_energy = self.mlp(node_scalar_feature, num_graphs_arr, batch).view(-1, 1, 1)
        assert teanet_energy.size() == correction_energy.size()
        energy = teanet_energy + correction_energy
        return energy, charge

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        return self.mlp.load_state_dict(state_dict, strict)


class _MLP(torch.nn.Module):
    def __init__(self, parameters: TeaNetNodeFeatureMLPParameters_v1_3) -> None:
        super().__init__()
        self.activation, _, _ = teanet_select_activation(parameters.activation, False)
        layers = []
        prev_hiddens = _NODE_FEATURE_SIZE
        for n_hidden in parameters.n_hiddens:
            layers.append(torch.nn.Linear(prev_hiddens, n_hidden))
            prev_hiddens = n_hidden
        layers.append(torch.nn.Linear(prev_hiddens, 1))
        self.linears = torch.nn.ModuleList(layers)

    def forward(
        self,
        node_scalar_feature: torch.Tensor,
        num_graphs_arr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        assert node_scalar_feature.size(1) == _NODE_FEATURE_SIZE
        x = node_scalar_feature
        for i, l in enumerate(self.linears):
            if i != 0:
                x = self.activation(x)
            x = l(x)
        batch_size = int(num_graphs_arr.size(0))
        energy = torch.zeros(
            (batch_size, 1),
            dtype=x.dtype,
            device=x.device,
            requires_grad=True,
        )
        energy = index_add(energy, batch, x)  # (batch,)
        return energy
