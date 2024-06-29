from typing import List, Optional, Tuple

import ase
import ase.build
import numpy as np
import pytest
import torch

from pekoe.calculators.ase_calculator import ASECalculator
from pekoe.nn.estimator_base import EstimatorCalcMode
from pekoe.nn.models import DEFAULT_MODEL_DIRECTORY
from pekoe.nn.models.config import BaseConfig
from pekoe.nn.models.model_builder import build_estimator
from pekoe.nn.models.teanet.codegen_options import CodeGenOptions
from pekoe.nn.models.teanet.model_pfp_v1_3.teanet import TeaNet_v1_3, TeaNetParameters_v1_3
from pekoe.nn.models.teanet_estimator import batch_estimate_preprocess, calc_mode_to_vector


@pytest.fixture(
    params=[
        pytest.param(["cpu", "na"], marks=[pytest.mark.cpu], id="cpu"),
        pytest.param(["gpu", "na"], marks=[pytest.mark.gpu], id="gpu"),
    ]
)
def device_config(request, gpuid) -> Tuple[str, Optional[CodeGenOptions]]:
    device, codegen_options = request.param
    if device == "cpu":
        return "cpu", None
    else:
        return f"cuda:{gpuid}", None


@pytest.fixture()
def atoms_input() -> ase.Atoms:
    return ase.build.molecule("CH3CH2OCH3")


def test_from_yaml(pytestconfig):
    base_config = BaseConfig.from_yaml(DEFAULT_MODEL_DIRECTORY / "model_v1_3_0.yaml")

    parameters = TeaNetParameters_v1_3.from_dict(base_config.parameters)
    if isinstance(parameters.cutoff, float):
        assert parameters.cutoff >= 0.0
    else:
        assert min(parameters.cutoff) >= 0.0


def test_node_feature(device_config, atoms_input):
    device, _ = device_config
    model_yaml = "model_v1_3_1.yaml"

    # Load model
    base_config = BaseConfig.from_yaml(DEFAULT_MODEL_DIRECTORY / model_yaml)
    parameters = TeaNetParameters_v1_3.from_dict(base_config.parameters)
    parameters.return_node_feature = True
    model = TeaNet_v1_3(parameters)
    state = torch.load(str(base_config.weights_path), map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    max_cutoff = (
        max(parameters.cutoff) if isinstance(parameters.cutoff, List) else parameters.cutoff
    )
    calc_mode = EstimatorCalcMode.CRYSTAL_U0

    # Prepare inputs based on structure and model settings
    atomic_numbers = torch.tensor(
        atoms_input.get_atomic_numbers(), dtype=torch.int64, device=device
    )
    coordinates = torch.tensor(atoms_input.get_positions(), dtype=torch.float32, device=device)
    cell = torch.tensor(atoms_input.get_cell()[:], dtype=torch.float32, device=device)
    pbc = torch.tensor(atoms_input.get_pbc(), dtype=torch.int64, device=device)
    (
        n_nodes,
        n_edges,
        atom_index1,
        atom_index2,
        shift_int,
        coordinates_mod,
    ) = batch_estimate_preprocess(
        atomic_numbers,
        coordinates,
        cell,
        pbc,
        max_cutoff,
    )

    x_add_ref = calc_mode_to_vector(
        calc_mode, base_config.version, np.ones((120,), dtype=np.int32)
    )
    shift = torch.mm(shift_int.to(torch.float32), cell)
    vecs = coordinates_mod[atom_index1] - coordinates_mod[atom_index2] - shift
    num_graphs_arr = torch.zeros((1,), device=device)
    batch = torch.zeros((n_nodes,), dtype=torch.int64, device=device)
    batch_edge = torch.zeros((n_edges,), dtype=torch.int64, device=device)
    x_add = torch.tensor(x_add_ref, dtype=torch.float32, device=device)[atomic_numbers]
    repulsion_calc_mode_int = 1 if calc_mode == EstimatorCalcMode.CRYSTAL else 0
    calc_mode_type = repulsion_calc_mode_int * torch.ones(
        (n_nodes,), dtype=torch.int64, device=device
    )

    # Infer
    model_input = (
        vecs,
        atomic_numbers,
        atom_index1,
        atom_index2,
        num_graphs_arr,
        batch,
        batch_edge,
        x_add,
        calc_mode_type,
    )
    energy, charges, node_scalar_feature = model(*model_input)

    assert node_scalar_feature.size()[0] == n_nodes
    assert node_scalar_feature.size()[1] == 513

    # Check wheter inputs are correctly treated using estimator
    estimator = build_estimator(base_config.version, device=device)
    estimator.set_calc_mode(calc_mode)
    calculator = ASECalculator(estimator)

    atoms_input.calc = calculator
    energy_calc = atoms_input.get_potential_energy()
    charges_calc = atoms_input.get_charges()

    assert energy.item() == pytest.approx(energy_calc, abs=1.0e-3)
    assert np.allclose(charges.detach().cpu().numpy(), charges_calc, atol=1.0e-3)
