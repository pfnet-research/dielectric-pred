import numpy as np
import pytest
from ase import Atoms

from pfp.nn.estimator_base import EstimatorCalcMode


def make_batch_input(atoms, model):
    atomic_numbers_1 = atoms.get_atomic_numbers().astype(np.int64)
    atomic_numbers = np.concatenate((atomic_numbers_1, [1, 1]), axis=0)
    coordinates = atoms.get_positions().astype(np.float32)
    cell = atoms.get_cell(complete=True)
    if not isinstance(cell, np.ndarray):
        cell = cell.array
    pbc = atoms.get_pbc().astype(np.uint8)

    coordinates, _ = model.ppw(coordinates, cell, np.linalg.inv(cell), pbc)

    atom_index1, atom_index2, shift, _ = model.ppp(coordinates, cell, pbc, 6.0, 100000)

    atom_index1 = atom_index1.astype(np.int64)
    atom_index2 = atom_index2.astype(np.int64)

    n_original = len(coordinates)
    coordinates = np.concatenate((coordinates, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), axis=0)
    atom_index1 = np.concatenate((atom_index1, [n_original]), axis=0)
    atom_index2 = np.concatenate((atom_index2, [n_original + 1]), axis=0)
    shift = np.concatenate((shift, [[0.0, 0.0, 0.0]]), axis=0)

    out_size = 2
    out_batch = np.zeros((coordinates.shape[0] - 2,), dtype=np.int64)
    out_batch_edge = np.zeros((shift.shape[0] - 1,), dtype=np.int64)
    out_batch = np.concatenate((out_batch, np.ones((2,), dtype=np.int64)), axis=0)
    out_batch_edge = np.concatenate((out_batch_edge, np.ones((1,), dtype=np.int64)), axis=0)

    x_add = np.zeros((atomic_numbers.shape[0], 8), dtype=np.float32)
    calc_mode_type = np.zeros((coordinates.shape[0],), dtype=np.int64)
    return (
        (
            np.zeros_like(coordinates),
            atom_index1,
            atom_index2,
            shift,
            out_batch,
            out_batch_edge,
        ),
        (coordinates, atomic_numbers, cell, out_size, x_add, calc_mode_type, True),
    )


@pytest.fixture
def estimator():
    from pfp.nn.models.crystal.model_builder import build_estimator

    return build_estimator(device=0)


@pytest.mark.gpu
@pytest.mark.pfp
def test_batch_estimate(estimator, atom_data, calc_mode):
    if calc_mode is not EstimatorCalcMode.CRYSTAL:
        pytest.skip("skipped: calc_mode is not CRYSTAL")

    estimator_inputs, expected_results = atom_data
    atoms = Atoms(
        numbers=estimator_inputs.atomic_numbers,
        positions=estimator_inputs.coordinates,
        cell=estimator_inputs.cell,
        pbc=estimator_inputs.pbc,
    )

    m = estimator.model
    inputs_sp, inputs_run = make_batch_input(atoms, m)
    m.sp(*inputs_sp)
    energy, charges, forces, virial = m.r(*inputs_run)
    assert energy.ravel().shape == (2,)
    assert -1000.0 < float(energy[0]) and float(energy[0]) < -10.0
    assert float(energy[0]) == pytest.approx(expected_results["energy"], 1e-5)
    assert -10.0 < float(energy[1]) and float(energy[1]) < 0.0
    assert charges.shape == (inputs_run[0].shape[0], 1)
    assert -15.0 < np.min(forces) and np.max(forces) < 15.0
    assert forces.shape == inputs_run[0].shape
    assert np.max(np.abs(forces)) < 20.0
    assert virial.shape == (2, 6)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
