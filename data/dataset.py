from typing import List, Optional, Tuple, Union, Dict
import json
from monty.json import MontyDecoder
import torch
from ase import Atoms

from torch.utils.data import Dataset
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from confidential import get_pfp_inputs_dict


class JSONDataset(Dataset):
    def __init__(self,
        json_file,
        target: str,
        cutoff: float = 6.0,
        keys: List[str] = None):

        with open(json_file, "r") as f:
            struct_info = json.load(f, cls=MontyDecoder)

        if keys is None:
            self.keys = sorted(struct_info.keys())
        else:
            self.keys = keys
        
        self.atoms = []
        self.labels = []
        for key in self.keys:
            struct = struct_info[key]['structure']
            if isinstance(struct, Atoms):
                self.atoms.append(struct)
            elif isinstance(struct, Structure):
                self.atoms.append(AseAtomsAdaptor.get_atoms(struct))
            else:
                raise ValueError("Input structures should be ASE or Pymatgen")
            self.labels.append(struct_info[key][target])

        self.cutoff =cutoff

        assert len(self.atoms) == len(self.labels)
    
    def __len__(self):
        return len(self.atoms)

    def __getitem__(self, idx):
        atoms = self.atoms[idx]

        pfp_input_kwargs = get_pfp_inputs_dict(
            atoms, self.cutoff)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        pfp_input_kwargs["label"] = label
        pfp_input_kwargs["mask"] = torch.zeros((3, 3))  # placeholder
        pfp_input_kwargs["key"] = self.keys[idx]

        return pfp_input_kwargs