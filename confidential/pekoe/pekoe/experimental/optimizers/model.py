import dataclasses
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import torch

import pekoe.nn.models.teanet.model_pfp_v1_2.teanet as teanet_mod

from . import cupytorch

if TYPE_CHECKING:
    from .data import CuPyMol


class MyTeaNet:
    def __init__(self, teanet: teanet_mod.TeaNet_v1_2):
        self.device = next(teanet.parameters()).device
        self.predictor = teanet
        # pekoe.nn.estimator_base.EstimatorCalcMode.CRYSTAL
        self.calc_mode = [0.0, 0.0]  # CRYSTAL

    def forward(self, mols: List["CuPyMol"]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if self.predictor is None:
            raise RuntimeError("model is not loaded")
        batch = [_atoms_to_data(mol) for mol in mols]
        xs = [data.pos for data in batch]
        collated = _collate_fn(batch)
        assert collated.pos.device == self.device
        collated.to(self.device)  # send `batch` and `batch_edge` to gpu
        collated.x_add = torch.tensor(
            self.calc_mode, dtype=torch.float32, device=self.device
        ).expand(collated.num_nodes, 2)
        c = teanet_mod.convert_data_to_tuple(collated)  # type: ignore
        energy, charges = self.predictor(*c)
        assert energy.ndim == 2  # (num_nodes, 1)
        energy = energy.squeeze(1)
        return energy, xs


# torch_geometric.data.Data or Batch
@dataclasses.dataclass
class _Data:
    x: torch.Tensor
    pos: torch.Tensor
    edge_index: torch.Tensor
    edge_vector: torch.Tensor
    num_graphs: Optional[int] = None
    batch: Optional[torch.Tensor] = None
    batch_edge: Optional[torch.Tensor] = None
    x_add: Optional[torch.Tensor] = None

    @property
    def num_nodes(self) -> int:
        return self.x.shape[0]  # type: ignore

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]  # type: ignore

    def to(self, device: Any) -> None:
        assert self.batch is not None
        self.batch = self.batch.to(device)
        assert self.batch_edge is not None
        self.batch_edge = self.batch_edge.to(device)


def _atoms_to_data(atoms: "CuPyMol") -> _Data:
    pos = cupytorch.astensor(atoms.positions).requires_grad_()
    # the sign of vec should be the same as `TeaNetPreprocessor` in chicle.
    vec = pos[:, None] - pos[None, :]  # vec[i, j] = pos[i] - pos[j]
    cutoff = 6.0
    edges = torch.nonzero(
        torch.triu((vec ** 2).sum(dim=-1) < cutoff ** 2, 1),  # i < j
        # Fixes "/pytorch/torch/csrc/utils/python_arg_parser.cpp:756:
        # UserWarning: This overload of nonzero is deprecated:"
        as_tuple=False,  # default
    ).t()  # shape: (2, m)

    numbers = cupytorch.astensor(atoms.atomic_numbers)
    x = numbers.to(torch.long)

    return _Data(x=x, pos=pos, edge_index=edges, edge_vector=vec[tuple(edges)])


def _collate_fn(data_list: List[_Data]) -> _Data:
    batch = torch.cat(
        [torch.full((d.num_nodes,), i, dtype=torch.long) for i, d in enumerate(data_list)],
        dim=0,
    )
    batch_edge = torch.cat(
        [torch.full((d.num_edges,), i, dtype=torch.long) for i, d in enumerate(data_list)],
        dim=0,
    )

    shift_index_array = torch.cumsum(torch.tensor([0] + [d.num_nodes for d in data_list]), dim=0)
    # d.edge_index: (2, num_nodes)
    edge_index = torch.cat(
        [d.edge_index + shift_index_array[i] for i, d in enumerate(data_list)],
        dim=1,
    )

    x = torch.cat([d.x for d in data_list], dim=0)
    pos = torch.cat([d.pos for d in data_list], dim=0)
    edge_vector = torch.cat([d.edge_vector for d in data_list], dim=0)
    return _Data(
        x=x,
        pos=pos,
        edge_index=edge_index,
        edge_vector=edge_vector,
        num_graphs=len(data_list),
        batch=batch,
        batch_edge=batch_edge,
    )
