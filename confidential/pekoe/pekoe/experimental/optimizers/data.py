import dataclasses

import ase
import cupy

try:
    from rdkit import Chem
except ImportError:
    pass


@dataclasses.dataclass
class CuPyMol:
    positions: cupy.ndarray
    atomic_numbers: cupy.ndarray


def ase_get(atoms: ase.Atoms) -> CuPyMol:
    if atoms.pbc.any():
        raise NotImplementedError("pbc is not supported")
    positions = atoms.positions
    numbers = atoms.numbers
    return CuPyMol(
        cupy.array(positions, dtype=cupy.float32),
        cupy.array(numbers, dtype=cupy.uint8),
    )


def ase_set(atoms: ase.Atoms, cupy_mol: CuPyMol) -> None:
    atoms.positions = cupy.asnumpy(cupy_mol.positions)


def rdkit_get(mol: "Chem.Mol") -> CuPyMol:
    (conformer,) = mol.GetConformers()
    positions = conformer.GetPositions()
    numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    return CuPyMol(
        cupy.array(positions, dtype=cupy.float32),
        cupy.array(numbers, dtype=cupy.uint8),
    )


def rdkit_set(mol: "Chem.Mol", cupy_mol: CuPyMol) -> None:
    (conformer,) = mol.GetConformers()
    pos = cupy.asnumpy(cupy_mol.positions).astype("float64")
    for i, p in enumerate(pos):
        conformer.SetAtomPosition(i, p)
