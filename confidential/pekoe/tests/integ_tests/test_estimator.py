import numpy as np
import pytest

from pfp.nn.estimator_base import AtomsTooManyError, EstimatorSystem, WithinBatchEstimationError
from pfp.utils.errors import PFPError


def test_batch_estimate(atom_data, calc_mode, estimator):
    estimator_inputs, expected_results = atom_data

    atomic_numbers = estimator_inputs.atomic_numbers
    coordinates = estimator_inputs.coordinates
    cell = estimator_inputs.cell
    pbc = estimator_inputs.pbc
    is_pbc = not (np.all(pbc == 0) or pbc is None)
    if is_pbc:
        pytest.skip()  # extreme test case using set_max_atoms does not work

    estimator.set_calc_mode(calc_mode)

    atom_single = estimator_inputs

    atom_long = EstimatorSystem(
        atomic_numbers=np.concatenate((atomic_numbers, atomic_numbers)),
        coordinates=np.concatenate((coordinates, np.array(([[50.0, 50.0, 50.0]])) + coordinates)),
        cell=cell,
        pbc=pbc,
        properties=estimator.implemented_properties,
    )

    atom_toolong = EstimatorSystem(
        atomic_numbers=np.concatenate((atomic_numbers, atomic_numbers, atomic_numbers)),
        coordinates=np.concatenate(
            (
                coordinates,
                np.array(([[30.0, 30.0, 30.0]])) + coordinates,
                np.array(([[60.0, 60.0, 60.0]])) + coordinates,
            )
        ),
        cell=cell,
        pbc=pbc,
        properties=estimator.implemented_properties,
    )

    atom_batch = [
        atom_single,
        atom_single,
        atom_long,
        atom_single,
        atom_toolong,
        atom_long,
        atom_single,
        atom_long,
        atom_single,
        atom_toolong,
        atom_single,
    ]
    estimator.set_max_atoms(len(atomic_numbers) * 2)
    results = [None] * len(atom_batch)
    pending_index = list(range(len(atom_batch)))
    pending_atom_batch = atom_batch
    while len(pending_index) > 0:
        next_batch = []
        next_pending_index = []

        partial_results = estimator.batch_estimate(pending_atom_batch)
        for i, res in zip(pending_index, partial_results):
            if isinstance(res, WithinBatchEstimationError):
                next_batch.append(atom_batch[i])
                next_pending_index.append(i)
            elif isinstance(res, PFPError):
                results[i] = res  # fail
            else:
                results[i] = res  # got result

        pending_atom_batch = next_batch
        pending_index = next_pending_index

    assert not isinstance(results[0], PFPError)
    assert not isinstance(results[1], PFPError)
    assert not isinstance(results[2], PFPError)
    assert not isinstance(results[3], PFPError)
    assert isinstance(results[4], AtomsTooManyError)
    assert not isinstance(results[5], PFPError)
    assert not isinstance(results[6], PFPError)
    assert not isinstance(results[7], PFPError)
    assert not isinstance(results[8], PFPError)
    assert isinstance(results[9], AtomsTooManyError)
    assert not isinstance(results[10], PFPError)

    assert results[0]["forces"].shape == atom_single.coordinates.shape
    assert results[2]["forces"].shape == atom_long.coordinates.shape
