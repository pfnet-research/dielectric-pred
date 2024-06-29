import argparse
import functools
import inspect
import json
import pathlib

import ase
import ase.neighborlist
import torch

from pekoe.experimental.optimizers.data import ase_get, ase_set
from pekoe.experimental.optimizers.scaled_bfgs import Executor
from pekoe.nn.models import DEFAULT_MODEL
from pekoe.nn.models.model_builder import build_estimator


def ase_job(atoms, *, optimize):
    mol = ase_get(atoms)
    info = yield from optimize(mol)
    new_atoms = atoms.copy()
    ase_set(new_atoms, mol)
    info.update(_check_bonds(atoms, new_atoms))
    return new_atoms, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_MODEL),
        help="path to configuration yaml",
    )
    parser.add_argument("--device", type=str, default="cpu", help="device (default: cpu)")
    parser.add_argument("--dataset-n", type=int)
    get_executor_kwargs = _add_kwargs(parser, Executor.__init__)
    args = parser.parse_args()

    device = torch.device(args.device)

    # create estimator and calculator:
    config_path = pathlib.Path(args.config)
    estimator = build_estimator(config_path, device=device)

    # {smiles: [[atomic_number, x, y, z], ...], ...}
    with open("input.json") as f:
        dataset = json.load(f)
    dataset = {k: _to_ase(v) for k, v in dataset.items()}
    if args.dataset_n is not None:
        dataset = {k: v for k, v in list(dataset.items())[: args.dataset_n]}

    print([len(mol) for mol in dataset.values()])

    ex = Executor(estimator=estimator, **get_executor_kwargs(args))
    results = ex.map(ase_job, dataset.values())
    results = dict(zip(dataset.keys(), results))

    output = {}
    for id_, (atoms, info) in sorted(results.items()):
        print(f"{id_}: {info}")
        output[id_] = _from_ase(atoms)

    with open("output.json", "w") as f:
        json.dump(output, f)


def _dist(atoms, bonds):
    x = atoms.positions
    vec = x[bonds[0]] - x[bonds[1]]
    return (vec ** 2).sum(1) ** 0.5


def _check_bonds(old_atoms, new_atoms):
    nl = ase.neighborlist.build_neighbor_list(old_atoms, self_interaction=False)
    bonds = nl.get_connectivity_matrix().nonzero()
    dist_change = _dist(new_atoms, bonds) - _dist(old_atoms, bonds)
    return {"bonds_diff": [dist_change.min(), dist_change.max()]}


def _to_ase(mol):
    numbers = [atom[0] for atom in mol]
    positions = [atom[1:] for atom in mol]
    return ase.Atoms(numbers=numbers, positions=positions)


def _from_ase(atoms):
    return [[n] + p for n, p in zip(atoms.numbers.tolist(), atoms.positions.tolist())]


def _add_kwargs(parser, func):
    names = []
    for name, p in inspect.signature(func).parameters.items():
        if p.kind == inspect.Parameter.KEYWORD_ONLY:
            kwargs = {}
            if p.default is inspect.Parameter.empty:
                kwargs["required"] = True
            else:
                kwargs["default"] = p.default
            if isinstance(p.annotation, type):
                kwargs["type"] = p.annotation
            help_str = "".join(
                (
                    name,
                    f": {kwargs['type'].__name__}" if "type" in kwargs else "",
                    f" (default: {kwargs['default']})" if "default" in kwargs else "",
                )
            )
            parser.add_argument(f"--{name}", help=help_str, **kwargs)
            names.append(name)
    return functools.partial(_get_kwargs, names=names)


def _get_kwargs(args, *, names):
    kwargs = {}
    for name in names:
        kwargs[name] = getattr(args, name)
    return kwargs


if __name__ == "__main__":
    main()
