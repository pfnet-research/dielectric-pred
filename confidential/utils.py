from typing import Dict, List

import numpy as np
import torch
from ase import Atoms

import pathlib

from confidential.pekoe.pekoe.nn.models.teanet.preprocessor.preprocessor import preprocess_pbc, wrap_coordinates
from confidential.pekoe.pekoe.nn.models import DEFAULT_MODEL as DEFAULT_PEKOE_MODEL
from confidential.pekoe.pekoe.nn.models.config import BaseConfig
from confidential.pekoe.pekoe.nn.models.teanet.model_pfp_v1_3.teanet import TeaNet_v1_3, TeaNetParameters_v1_3
from confidential.models import WrappedPFP

DEFAULT_MODEL = pathlib.Path(__file__).parent / "configs"


def get_pfp_inputs_dict(
    atoms: Atoms,
    cutoff: float = 6.0,
    n_elements: int = 128,
) -> Dict[str, torch.Tensor]:
    
    coordinates = torch.tensor(atoms.positions, dtype=torch.float32)
    cell = torch.tensor(atoms.cell[:], dtype=torch.float32)
    pbc = torch.tensor(atoms.pbc, dtype=torch.int64)
    atomic_numbers = torch.tensor(atoms.numbers, dtype=torch.int64)
    atom_pos, fractional = wrap_coordinates(coordinates, cell, pbc)
    atom_index1, atom_index2, shift_int = preprocess_pbc(
        atom_pos,
        cell,
        pbc,
        cutoff,
        max_atoms=-1,
    )
    shift = shift_int.to(dtype=torch.float32)
    vecs = atom_pos[atom_index1] - atom_pos[atom_index2] - torch.mm(shift, cell)
    ob = torch.zeros((1,), dtype=torch.float32)
    ba = torch.zeros((coordinates.size()[0],), dtype=torch.int64)
    be = torch.zeros((shift.size()[0],), dtype=torch.int64)
    x_add = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    x_add = torch.tensor(
        np.repeat(np.expand_dims(x_add, 0), n_elements, axis=0), dtype=torch.float32
    )   # Lable for computational method
    xa = x_add[atomic_numbers]
    vecs = atom_pos[atom_index1] - atom_pos[atom_index2] - torch.mm(shift, cell)
    # vecs, atomic_numbers, atom_index1, atom_index2, ob, ba, be, xa
    return {
        "vecs": vecs, 
        "atomic_numbers": atomic_numbers,
        "atom_index1": atom_index1,
        "atom_index2": atom_index2,
        "ob": ob, 
        "ba": ba, 
        "be": be, 
        "xa": xa}



def load_pfp(load_parameters=False, return_layer=4):
    base_config = BaseConfig.from_yaml(DEFAULT_PEKOE_MODEL)
    teanet_parameters = TeaNetParameters_v1_3.from_dict(base_config.parameters)
    teanet = TeaNet_v1_3(teanet_parameters)
    if load_parameters:
        teanet.load_state_dict(torch.load((DEFAULT_PEKOE_MODEL.parent / "model_v1_4_1.pt").as_posix(), map_location="cpu"))
    return WrappedPFP(teanet, return_layer=return_layer)



def collate_fn_dict(
    list_of_data: List[Dict],
) -> Dict:
    n_atoms_cumsum = torch.cumsum(
        torch.tensor([0] + [len(d["atomic_numbers"]) for d in list_of_data[:-1]]), dim=0
    )
    vecs_batch = torch.cat([d["vecs"] for d in list_of_data])
    atomic_numbers_batch = torch.cat([d["atomic_numbers"] for d in list_of_data])
    atom_index1_batch = torch.cat([d["atom_index1"] + n for d, n in zip(list_of_data, n_atoms_cumsum)])
    atom_index2_batch = torch.cat([d["atom_index2"] + n for d, n in zip(list_of_data, n_atoms_cumsum)])
    ob_batch = torch.cat([d["ob"] for d in list_of_data])
    ba_batch = torch.cat([torch.full_like(d["ba"], n) for n, d in enumerate(list_of_data)])
    be_batch = torch.cat([torch.full_like(d["be"], n) for n, d in enumerate(list_of_data)])
    xa_batch = torch.cat([d["xa"] for d in list_of_data])
    mask_batch = torch.stack([d["mask"] for d in list_of_data])

    dielectric_tensor_batch = torch.stack([d["label"] for d in list_of_data])

    mp_ids = [d["key"] for d in list_of_data]

    return {
        "vecs": vecs_batch,
        "atomic_numbers": atomic_numbers_batch,
        "atom_index1": atom_index1_batch,
        "atom_index2": atom_index2_batch,
        "ob": ob_batch,
        "ba": ba_batch,
        "be": be_batch,
        "xa": xa_batch,
        # "mask": mask_batch,
        "label": dielectric_tensor_batch,
        "id": mp_ids
    }