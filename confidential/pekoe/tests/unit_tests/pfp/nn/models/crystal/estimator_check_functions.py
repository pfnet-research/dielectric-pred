import copy

import numpy as np
import pytest

from pfp.nn.estimator_base import (
    AtomsHardLimitExceededError,
    AtomsTooManyError,
    BatchAtomsTooManyError,
    BatchNeighborsTooManyError,
    CellTooSmallError,
    EstimatorCalcMode,
    EstimatorSystem,
    NeighborsHardLimitExceededError,
    NeighborsTooManyError,
)
from pfp.utils.messages import MessageEnum


def check_estimator_pbc(estimator, estimator_inputs_all):
    results = [estimator.estimate(input) for input in estimator_inputs_all]
    results_for_test_case_1_2_3 = results[1:4]
    for result in results_for_test_case_1_2_3:
        assert result["energy"] == pytest.approx(results[0]["energy"], 5.0e-5)
        assert np.allclose(np.sort(result["forces"]), np.sort(results[0]["forces"]), atol=1.0e-2)
        assert result["calc_stats"]["n_neighbors"] == results[0]["calc_stats"]["n_neighbors"]


def check_estimate(estimator, atom_data, calc_mode, use_batch):
    estimator_inputs, expected_results = atom_data
    if use_batch:

        def estimate(x):
            res_array = estimator.batch_estimate([x])
            res_0 = res_array[0]
            if isinstance(res_0, Exception):
                raise res_0
            return res_0

    else:

        def estimate(x):
            return estimator.estimate(x)

    expected_results_charges_np = (
        np.array(expected_results["charges"])
        if "charges" in estimator.implemented_properties
        else None
    )
    expected_results_forces_np = np.array(expected_results["forces"])
    atomic_numbers = estimator_inputs.atomic_numbers
    coordinates = estimator_inputs.coordinates
    cell = estimator_inputs.cell
    pbc = estimator_inputs.pbc

    volume = np.abs(np.dot(np.cross(cell[0], cell[1]), cell[2])) if cell is not None else None
    if not isinstance(cell, np.ndarray):
        cell = cell.array

    estimator.set_calc_mode(calc_mode)

    check_estimate_simple(
        estimate,
        (
            estimator.implemented_properties,
            use_batch,
            atomic_numbers,
            coordinates,
            cell,
            pbc,
            volume,
        ),
        (expected_results, expected_results_charges_np, expected_results_forces_np),
    )
    check_estimate_fp64(
        estimate,
        (
            estimator.implemented_properties,
            use_batch,
            atomic_numbers,
            coordinates,
            cell,
            pbc,
            volume,
        ),
        (expected_results, expected_results_charges_np, expected_results_forces_np),
    )
    check_estimate_shift(
        estimate,
        (
            estimator.implemented_properties,
            use_batch,
            atomic_numbers,
            coordinates,
            cell,
            pbc,
            volume,
        ),
        (expected_results, expected_results_charges_np, expected_results_forces_np),
    )
    check_estimate_energy_only(
        estimate,
        (
            estimator.implemented_properties,
            use_batch,
            atomic_numbers,
            coordinates,
            cell,
            pbc,
            volume,
        ),
        (expected_results, expected_results_charges_np, expected_results_forces_np),
    )
    check_estimate_element_status(
        estimator,
        calc_mode,
        estimate,
        (
            estimator.implemented_properties,
            use_batch,
            atomic_numbers,
            coordinates,
            cell,
            pbc,
            volume,
        ),
        (expected_results, expected_results_charges_np, expected_results_forces_np),
    )
    check_estimate_errors(
        estimator,
        estimate,
        (
            estimator.implemented_properties,
            use_batch,
            atomic_numbers,
            coordinates,
            cell,
            pbc,
            volume,
        ),
        (expected_results, expected_results_charges_np, expected_results_forces_np),
    )
    check_estimate_no_atoms(
        estimate,
        (
            estimator.implemented_properties,
            use_batch,
            atomic_numbers,
            coordinates,
            cell,
            pbc,
            volume,
        ),
        (expected_results, expected_results_charges_np, expected_results_forces_np),
    )


def check_estimate_simple(estimate_func, inputs, outputs):
    implemented_properties, use_batch, atomic_numbers, coordinates, cell, pbc, volume = inputs
    expected_results, expected_results_charges_np, expected_results_forces_np = outputs
    is_pbc = not (np.all(pbc == 0) or pbc is None)
    inputs = EstimatorSystem(
        atomic_numbers=atomic_numbers,
        coordinates=coordinates,
        cell=cell,
        pbc=pbc,
        properties=implemented_properties,
    )
    results = estimate_func(inputs)

    assert inputs.atomic_numbers.dtype == np.uint8
    assert inputs.coordinates.dtype == inputs.cell.dtype

    assert isinstance(results["energy"], float)
    assert results["energy"] == pytest.approx(expected_results["energy"], 5e-5)
    if "charges" in implemented_properties:
        assert np.allclose(results["charges"], expected_results_charges_np, atol=1.0e-2)
    assert np.allclose(results["forces"], expected_results_forces_np, atol=1.0e-2)
    assert len(results["virial"]) == 6
    if is_pbc:
        for actual, expect in zip(results["virial"], expected_results["stress"]):
            assert abs(actual - expect * volume) < 1e-3
    assert len(results["messages"]) == 0
    assert results["calc_stats"]["n_neighbors"] == expected_results["calc_stats"]["n_neighbors"]
    if "TeaNetEstimator" in str(estimate_func):
        # TODO: Is there better way to distinguish estimator?
        assert "elapsed_usec_preprocess" in results["calc_stats"]
        assert "elapsed_usec_infer" in results["calc_stats"]
    elif "D3Estimator" in str(estimate_func):
        assert "elapsed_usec_d3_preprocess" in results["calc_stats"]
        assert "elapsed_usec_d3_infer" in results["calc_stats"]


def check_estimate_fp64(estimate_func, inputs, outputs):
    implemented_properties, use_batch, atomic_numbers, coordinates, cell, pbc, volume = inputs
    expected_results, expected_results_charges_np, expected_results_forces_np = outputs
    is_pbc = not (np.all(pbc == 0) or pbc is None)
    inputs = EstimatorSystem(
        atomic_numbers=atomic_numbers,
        coordinates=coordinates.astype(np.float64),
        cell=cell.astype(np.float32),
        pbc=pbc,
        properties=implemented_properties,
    )
    results = estimate_func(inputs)

    assert inputs.atomic_numbers.dtype == np.uint8
    assert inputs.coordinates.dtype == inputs.cell.dtype

    assert isinstance(results["energy"], float)
    assert results["energy"] == pytest.approx(expected_results["energy"], 5e-5)
    if "charges" in implemented_properties:
        assert np.allclose(results["charges"], expected_results_charges_np, atol=1.0e-2)
    assert np.allclose(results["forces"], expected_results_forces_np, atol=1.0e-2)
    assert len(results["virial"]) == 6
    if is_pbc:
        for actual, expect in zip(results["virial"], expected_results["stress"]):
            assert abs(actual - expect * volume) < 1e-3
    assert len(results["messages"]) == 0
    assert results["calc_stats"]["n_neighbors"] == expected_results["calc_stats"]["n_neighbors"]


def check_estimate_shift(estimate_func, inputs, outputs):
    implemented_properties, use_batch, atomic_numbers, coordinates, cell, pbc, volume = inputs
    expected_results, expected_results_charges_np, expected_results_forces_np = outputs
    is_pbc = not (np.all(pbc == 0) or pbc is None)
    inputs_shift = EstimatorSystem(
        atomic_numbers=atomic_numbers,
        coordinates=coordinates + np.array([[100.0, 200.0, 300.0]]),
        cell=cell,
        pbc=pbc,
        properties=implemented_properties,
    )
    results_shift = estimate_func(inputs_shift)

    assert isinstance(results_shift["energy"], float)
    assert results_shift["energy"] == pytest.approx(expected_results["energy"], 5e-5)
    if "charges" in implemented_properties:
        assert np.allclose(results_shift["charges"], expected_results_charges_np, atol=1.0e-2)
    assert np.allclose(results_shift["forces"], expected_results_forces_np, atol=1.0e-2)
    assert len(results_shift["virial"]) == 6
    if is_pbc:
        for actual, expect in zip(results_shift["virial"], expected_results["stress"]):
            assert (
                abs(actual - expect * volume) < 1e-2
            )  # The numerical accuracy can be degraded for larger shift.
    assert len(results_shift["messages"]) == 0
    assert (
        results_shift["calc_stats"]["n_neighbors"] == expected_results["calc_stats"]["n_neighbors"]
    )


def check_estimate_energy_only(estimate_func, inputs, outputs):
    implemented_properties, use_batch, atomic_numbers, coordinates, cell, pbc, volume = inputs
    expected_results, expected_results_charges_np, expected_results_forces_np = outputs
    inputs_energy_only = EstimatorSystem(
        atomic_numbers=atomic_numbers,
        coordinates=coordinates + np.array([[1000.0, 2000.0, 3000.0]]),
        cell=cell,
        pbc=pbc,
        properties=["energy"],
    )
    results_energy_only = estimate_func(inputs_energy_only)

    assert isinstance(results_energy_only["energy"], float)
    assert results_energy_only["energy"] == pytest.approx(expected_results["energy"], 5e-5)
    if "charges" in implemented_properties:
        assert np.allclose(
            results_energy_only["charges"], expected_results_charges_np, atol=1.0e-2
        )
    if not use_batch:
        assert (
            len(
                set(results_energy_only.keys())
                - set(["energy", "charges", "messages", "calc_stats"])
            )
            == 0
        )
    assert len(results_energy_only["messages"]) == 0
    assert (
        results_energy_only["calc_stats"]["n_neighbors"]
        == expected_results["calc_stats"]["n_neighbors"]
    )


def check_estimate_element_status(estimator, calc_mode, estimate_func, inputs, outputs):
    implemented_properties, use_batch, atomic_numbers, coordinates, cell, pbc, volume = inputs
    expected_results, expected_results_charges_np, expected_results_forces_np = outputs
    is_pbc = not (np.all(pbc == 0) or pbc is None)

    atomic_numbers_unexpected = copy.deepcopy(atomic_numbers)
    atomic_numbers_unexpected[0] = 119

    inputs_unexpected = EstimatorSystem(
        atomic_numbers=atomic_numbers_unexpected,
        coordinates=coordinates,
        cell=cell,
        pbc=pbc,
        properties=implemented_properties,
    )
    results_unexpected = estimate_func(inputs_unexpected)

    assert results_unexpected["messages"] == [MessageEnum.UnexpectedElementWarning]

    estimator.set_message_status(MessageEnum.UnexpectedElementWarning, False)

    results_unexpected = estimate_func(inputs_unexpected)

    assert results_unexpected["messages"] == []

    estimator.set_message_status(MessageEnum.UnexpectedElementWarning, True)

    results_unexpected = estimate_func(inputs_unexpected)

    assert results_unexpected["messages"] == [MessageEnum.UnexpectedElementWarning]

    if len(estimator.available_calc_modes()) > 1:
        if (
            calc_mode is EstimatorCalcMode.CRYSTAL
            and EstimatorCalcMode.MOLECULE in estimator.available_calc_modes()
        ):
            estimator.set_calc_mode(EstimatorCalcMode.MOLECULE)
        else:
            estimator.set_calc_mode(EstimatorCalcMode.CRYSTAL)

        inputs = EstimatorSystem(
            atomic_numbers=atomic_numbers,
            coordinates=coordinates,
            cell=cell,
            pbc=pbc,
            properties=estimator.implemented_properties,
            calc_mode=calc_mode,
        )
        results = estimate_func(inputs)

        assert results["energy"] == pytest.approx(expected_results["energy"], 5e-5)
        if "charges" in estimator.implemented_properties:
            assert np.allclose(results["charges"], expected_results_charges_np, atol=1.0e-2)
        assert np.allclose(results["forces"], expected_results_forces_np, atol=1.0e-2)
        assert len(results["virial"]) == 6
        if is_pbc:
            for actual, expect in zip(results["virial"], expected_results["stress"]):
                assert abs(actual - expect * volume) < 1e-3
        assert (
            results["calc_stats"]["n_neighbors"] == expected_results["calc_stats"]["n_neighbors"]
        )

    estimator.set_calc_mode(calc_mode)


def check_estimate_errors(estimator, estimate_func, inputs, outputs):
    implemented_properties, use_batch, atomic_numbers, coordinates, cell, pbc, volume = inputs
    expected_results, expected_results_charges_np, expected_results_forces_np = outputs
    is_pbc = not (np.all(pbc == 0) or pbc is None)

    if is_pbc:
        max_atoms_prev = estimator.max_atoms
        estimator.set_max_atoms(10000)
        inputs = EstimatorSystem(
            atomic_numbers=atomic_numbers,
            coordinates=coordinates,
            cell=np.array([[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]]),
            pbc=pbc,
            properties=implemented_properties,
        )

        with pytest.raises(CellTooSmallError):
            _ = estimate_func(inputs)

        # NOTE: n_atoms=2 > max_atoms=1
        estimator.set_max_atoms(1)
        inputs = EstimatorSystem(
            atomic_numbers=atomic_numbers[:2],
            coordinates=np.array([[0.01, 0.01, 0.01]] * 2),
            cell=cell,
            pbc=pbc,
            properties=implemented_properties,
        )

        with pytest.raises(AtomsTooManyError):
            _ = estimate_func(inputs)

        estimator.set_max_atoms(max_atoms_prev)


def check_estimate_no_atoms(estimate_func, inputs, outputs):
    implemented_properties, use_batch, atomic_numbers, coordinates, cell, pbc, volume = inputs
    expected_results, expected_results_charges_np, expected_results_forces_np = outputs

    inputs_single_atom = EstimatorSystem(
        atomic_numbers=atomic_numbers[:1],
        coordinates=coordinates[:1],
        cell=cell,
        pbc=np.array([0, 0, 0], dtype=np.int32),
        properties=implemented_properties,
    )
    _ = estimate_func(inputs_single_atom)

    inputs_single_atom_pbc = EstimatorSystem(
        atomic_numbers=atomic_numbers[:1],
        coordinates=coordinates[:1],
        cell=np.array([[1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0], [0.0, 0.0, 1000.0]]),
        pbc=np.array([1, 1, 1], dtype=np.int32),
        properties=implemented_properties,
    )
    _ = estimate_func(inputs_single_atom_pbc)

    inputs_zero_atom = EstimatorSystem(
        atomic_numbers=atomic_numbers[:0],
        coordinates=coordinates[:0],
        cell=cell,
        pbc=np.array([0, 0, 0], dtype=np.int32),
        properties=implemented_properties,
    )
    _ = estimate_func(inputs_zero_atom)

    inputs_zero_atom_pbc = EstimatorSystem(
        atomic_numbers=atomic_numbers[:0],
        coordinates=coordinates[:0],
        cell=np.array([[1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0], [0.0, 0.0, 1000.0]]),
        pbc=np.array([1, 1, 1], dtype=np.int32),
        properties=implemented_properties,
    )
    _ = estimate_func(inputs_zero_atom_pbc)


def check_batch_estimate(estimator, atom_data, calc_mode):
    estimator_inputs, expected_results = atom_data
    if "charges" in estimator.implemented_properties:
        expected_results_charges_np = np.array(expected_results["charges"])
    expected_results_forces_np = np.array(expected_results["forces"])
    atomic_numbers = estimator_inputs.atomic_numbers
    coordinates = estimator_inputs.coordinates
    cell = estimator_inputs.cell
    pbc = estimator_inputs.pbc

    is_pbc = not (np.all(pbc == 0) or pbc is None)
    volume = np.abs(np.dot(np.cross(cell[0], cell[1]), cell[2])) if cell is not None else None
    if not isinstance(cell, np.ndarray):
        cell = cell.array

    estimator.set_calc_mode(calc_mode)

    batch = [estimator_inputs, estimator_inputs]
    results = estimator.batch_estimate(batch)

    assert len(results) == 2
    assert abs(results[0]["energy"] - results[1]["energy"]) < 5e-5
    assert results[0]["energy"] == pytest.approx(expected_results["energy"], 5e-5)
    if "charges" in estimator.implemented_properties:
        assert np.allclose(results[0]["charges"], expected_results_charges_np, atol=1.0e-2)
        assert np.allclose(results[0]["charges"], results[1]["charges"], atol=1.0e-2)
    assert np.allclose(results[0]["forces"], expected_results_forces_np, atol=1.0e-2)
    assert np.allclose(results[0]["forces"], results[1]["forces"], atol=1.0e-2)

    force_diff = results[0]["forces"] - results[1]["forces"]
    assert np.sum(np.abs(force_diff)) < 1e-3

    assert len(results[0]["virial"]) == 6
    if is_pbc:
        for actual0, actual1, expect in zip(
            results[0]["virial"], results[1]["virial"], expected_results["stress"]
        ):
            assert abs(actual0 - expect * volume) < 1e-3
            assert abs(actual1 - expect * volume) < 1e-3

    assert results[0]["calc_stats"]["n_neighbors"] == expected_results["calc_stats"]["n_neighbors"]
    assert results[0]["calc_stats"]["n_neighbors"] == results[1]["calc_stats"]["n_neighbors"]

    if "TeaNetEstimator" in str(estimator.__class__):
        # Avoid importing pekoe package directly
        assert "elapsed_usec_preprocess" in results[0]["calc_stats"]
        assert "elapsed_usec_infer" in results[0]["calc_stats"]
    elif "D3Estimator" in str(estimator.__class__):
        assert "elapsed_usec_d3_preprocess" in results[0]["calc_stats"]
        assert "elapsed_usec_d3_infer" in results[0]["calc_stats"]

    if not is_pbc:
        estimator.set_max_atoms(len(atomic_numbers) * 2)
        results_within_batch = estimator.batch_estimate(batch)
        assert isinstance(results_within_batch[0], dict)
        assert isinstance(results_within_batch[1], dict)
        estimator.set_max_atoms(len(atomic_numbers) * 2 - 1)
        results_atoms_over = estimator.batch_estimate(batch)
        assert isinstance(results_atoms_over[0], dict)
        assert isinstance(results_atoms_over[1], BatchAtomsTooManyError)
        estimator.set_max_atoms(len(atomic_numbers) - 1)
        results_atoms_too_small = estimator.batch_estimate(batch)
        assert isinstance(results_atoms_too_small[0], AtomsTooManyError)
        assert isinstance(results_atoms_too_small[1], AtomsTooManyError)

        atom_long = EstimatorSystem(
            atomic_numbers=np.concatenate((atomic_numbers, atomic_numbers)),
            coordinates=np.concatenate(
                (coordinates, np.array(([[50.0, 50.0, 50.0]])) + coordinates)
            ),
            cell=cell,
            pbc=pbc,
            properties=estimator.implemented_properties,
        )

        batch_long = [estimator_inputs, atom_long, estimator_inputs, estimator_inputs]
        estimator.set_max_atoms(len(atomic_numbers) * 2 - 1)
        results_atoms_long = estimator.batch_estimate(batch_long)
        assert isinstance(results_atoms_long[0], dict)
        assert isinstance(results_atoms_long[1], AtomsTooManyError)
        assert isinstance(results_atoms_long[2], BatchAtomsTooManyError)
        assert isinstance(results_atoms_long[3], BatchAtomsTooManyError)
        estimator.set_max_atoms(len(atomic_numbers) * 2)
        results_atoms_long = estimator.batch_estimate(batch_long)
        assert isinstance(results_atoms_long[0], dict)
        assert isinstance(results_atoms_long[1], BatchAtomsTooManyError)
        assert isinstance(results_atoms_long[2], dict)
        assert isinstance(results_atoms_long[3], BatchAtomsTooManyError)

        num_neighbors = (
            np.sum(
                np.sum(
                    np.square(np.expand_dims(coordinates, 0) - np.expand_dims(coordinates, 1)),
                    axis=2,
                ).flatten()
                <= estimator.cutoff ** 2
            ).item()
            - len(atomic_numbers)
        ) // 2

        estimator.set_max_neighbors(num_neighbors * 2)
        results_within_batch_2 = estimator.batch_estimate(batch)
        assert isinstance(results_within_batch_2[0], dict)
        assert isinstance(results_within_batch_2[1], dict)
        estimator.set_max_neighbors(num_neighbors * 2 - 1)
        results_neighbors_over = estimator.batch_estimate(batch)
        assert isinstance(results_neighbors_over[0], dict)
        assert isinstance(results_neighbors_over[1], BatchNeighborsTooManyError)
        estimator.set_max_neighbors(num_neighbors - 1)
        results_neighbors_over = estimator.batch_estimate(batch)
        assert isinstance(results_neighbors_over[0], NeighborsTooManyError)
        assert isinstance(results_neighbors_over[1], NeighborsTooManyError)


def check_estimate_book_keeping(estimator, atom_data):
    np.random.seed(12345)

    estimator_inputs, _ = atom_data

    atomic_numbers = estimator_inputs.atomic_numbers
    atom_positions = estimator_inputs.coordinates
    cell = estimator_inputs.cell
    pbc = estimator_inputs.pbc

    fluctuations = 0.5 * np.random.rand(*atom_positions.shape)
    cell_deform = cell + np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]])

    base_params = EstimatorSystem(
        atomic_numbers=atomic_numbers,
        coordinates=atom_positions,
        cell=cell,
        pbc=pbc,
        properties=estimator.implemented_properties,
    )

    original_params = copy.deepcopy(base_params)
    results_original = estimator.estimate(original_params)
    assert estimator.previous_book_keeping is False

    near_params = copy.deepcopy(base_params)
    near_params.coordinates += fluctuations
    results_near_wo_book_keeping = estimator.estimate(near_params)
    assert estimator.previous_book_keeping is False

    far_params = copy.deepcopy(base_params)
    far_params.coordinates += 100.0 * fluctuations
    results_far_wo_book_keeping = estimator.estimate(far_params)
    assert estimator.previous_book_keeping is False

    estimator.set_book_keeping(use_book_keeping=True, book_keeping_skin=2.0)

    estimator.estimate(original_params)
    assert estimator.previous_book_keeping is False

    results_near_with_book_keeping = estimator.estimate(near_params)
    assert estimator.previous_book_keeping is True

    assert (
        abs(results_near_wo_book_keeping["energy"] - results_near_with_book_keeping["energy"])
        < 1.0e-3
    )

    results_far_with_book_keeping = estimator.estimate(far_params)
    assert estimator.previous_book_keeping is False
    assert (
        abs(results_far_wo_book_keeping["energy"] - results_far_with_book_keeping["energy"])
        < 1.0e-3
    )

    deform_params = copy.deepcopy(base_params)
    deform_params.cell = cell_deform

    estimator.estimate(original_params)
    assert estimator.previous_book_keeping is False
    results_deform_with_book_keeping = estimator.estimate(deform_params)
    assert estimator.previous_book_keeping is True
    assert abs(results_original["energy"] - results_deform_with_book_keeping["energy"]) < 1.0e-3


def check_estimate_book_keeping_extreme_cases(estimator):
    cutoff = estimator.cutoff
    skin_default = 2.0
    estimator.set_book_keeping(use_book_keeping=True, book_keeping_skin=skin_default)

    initial_coordinates = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    far_coordinates = np.array(
        [[0.0, 0.0, 0.0], [100.0 - cutoff - skin_default - 0.01, 0.0, 0.0]],
        dtype=np.float32,
    )
    initial_cell = np.array([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]])
    initial_settings = EstimatorSystem(
        atomic_numbers=np.array([1, 1]),
        coordinates=initial_coordinates,
        cell=initial_cell,
        pbc=np.array([1, 1, 1]),
        properties=estimator.implemented_properties,
    )

    # Too small cell (Throw Error)
    small_fail_settings = copy.deepcopy(initial_settings)
    small_fail_settings.cell = np.array(
        [
            [100.0, 0.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 0.0, cutoff + 1.5 * skin_default - 0.01],
        ]
    )

    with pytest.raises(CellTooSmallError):
        estimator.estimate(small_fail_settings)

    # Large enough cell (Pass)
    small_safe_settings = copy.deepcopy(initial_settings)
    small_safe_settings.cell = np.array(
        [
            [100.0, 0.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 0.0, cutoff + 1.5 * skin_default + 0.01],
        ]
    )

    estimator.estimate(small_safe_settings)

    # Large cell with atoms, initial (Pass, book-keeping yes)
    large_initial_settings = copy.deepcopy(initial_settings)
    large_initial_settings.coordinates = far_coordinates

    estimator.estimate(large_initial_settings)
    assert estimator.previous_book_keeping is False

    # Large cell with atoms, small cell deform (Pass, book-keeping no)
    large_cell_safe_settings = copy.deepcopy(large_initial_settings)
    large_cell_safe_settings.cell = np.array(
        [
            [100.0, 0.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 0.0, 100.0 - 0.5 * skin_default + 0.01],
        ]
    )

    estimator.estimate(large_cell_safe_settings)
    assert estimator.previous_book_keeping is True

    # Large cell with atoms, large cell deform (Pass, book-keeping yes)
    large_cell_rebuild_settings = copy.deepcopy(large_initial_settings)
    large_cell_rebuild_settings.cell = np.array(
        [
            [100.0, 0.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 0.0, 100.0 - 0.5 * skin_default - 0.01],
        ]
    )

    estimator.estimate(large_cell_rebuild_settings)
    assert estimator.previous_book_keeping is False

    # Large cell with atoms, large cell deform with outbound atom (Pass, book-keeping yes)
    large_cell_outbound_settings = copy.deepcopy(large_initial_settings)
    large_cell_outbound_settings.cell = np.array(
        [
            [100.0 - cutoff - skin_default - 0.02, 0.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 0.0, 100.0],
        ]
    )

    # Large cell with atoms, small atom move (Pass, book-keeping no)
    large_atom_safe_settings = copy.deepcopy(large_initial_settings)
    large_atom_safe_settings.coordinates = np.array(
        [
            [-0.5 * skin_default + 0.01, 0.0, 0.0],
            [100.0 - cutoff - 0.5 * skin_default - 0.02, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    estimator.estimate(large_initial_settings)
    estimator.estimate(large_atom_safe_settings)
    assert estimator.previous_book_keeping is True

    # Large cell with atoms, large atom move (Pass, book-keeping yes)
    large_atom_rebuild_settings = copy.deepcopy(large_initial_settings)
    large_atom_rebuild_settings.coordinates = np.array(
        [[0.0, 0.0, 0.0], [100.0 - cutoff - 0.5 * skin_default + 0.01, 0.0, 0.0]],
        dtype=np.float32,
    )

    estimator.estimate(large_atom_rebuild_settings)
    assert estimator.previous_book_keeping is False

    # Large cell with atoms, large atom move with outbound (Pass, book-keeping yes)
    large_atom_outbound_settings = copy.deepcopy(large_initial_settings)
    large_atom_outbound_settings.coordinates = np.array(
        [
            [-0.5 * skin_default + 0.01, 0.0, 0.0],
            [100.0 - cutoff - 0.5 * skin_default + 0.01, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    estimator.estimate(large_initial_settings)
    estimator.estimate(large_atom_outbound_settings)
    assert estimator.previous_book_keeping is False


def check_estimate_max_atoms_and_neighbors(estimator):
    for n_atoms in range(10, 50, 10):
        atomic_numbers = np.array([1] * n_atoms)
        coordinates = np.zeros((n_atoms, 3))
        cell = np.array([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]])
        pbc = np.array([0, 0, 0])
        inputs = EstimatorSystem(
            atomic_numbers=atomic_numbers,
            coordinates=coordinates,
            cell=cell,
            pbc=pbc,
            properties=estimator.implemented_properties,
        )
        # Set hard max neighbors.
        estimator.set_max_neighbors(n_atoms * (n_atoms - 1) // 2)
        estimator.estimate(inputs)
        estimator.set_max_neighbors(n_atoms * (n_atoms - 1) // 2 - 1)
        with pytest.raises(NeighborsTooManyError) as excinfo:
            estimator.estimate(inputs)
        neighbors_too_many_error = excinfo.value
        assert neighbors_too_many_error.max_neighbors == (n_atoms * (n_atoms - 1) // 2 - 1)
        assert neighbors_too_many_error.n_neighbors > neighbors_too_many_error.max_neighbors
        estimator.set_max_neighbors(None)

        # Set soft max neighbors.
        inputs.input_max_neighbors = n_atoms * (n_atoms - 1) // 2
        estimator.estimate(inputs)
        inputs.input_max_neighbors = n_atoms * (n_atoms - 1) // 2 - 1
        with pytest.raises(NeighborsTooManyError) as excinfo:
            estimator.estimate(inputs)
        neighbors_too_many_error = excinfo.value
        assert neighbors_too_many_error.max_neighbors == (n_atoms * (n_atoms - 1) // 2 - 1)
        assert neighbors_too_many_error.n_neighbors > neighbors_too_many_error.max_neighbors

        inputs.input_max_neighbors = None
        estimator.set_max_neighbors(None)

        # Set hard max atoms.
        estimator.set_max_atoms(n_atoms)
        estimator.estimate(inputs)
        estimator.set_max_atoms(n_atoms - 1)
        with pytest.raises(AtomsTooManyError) as excinfo:
            estimator.estimate(inputs)
        atoms_too_many_error = excinfo.value
        assert atoms_too_many_error.n_atoms == n_atoms
        assert atoms_too_many_error.max_atoms == n_atoms - 1
        estimator.set_max_atoms(None)

        # Set soft max atoms.
        inputs.input_max_atoms = n_atoms
        estimator.estimate(inputs)
        inputs.input_max_atoms = n_atoms - 1
        with pytest.raises(AtomsTooManyError) as excinfo:
            estimator.estimate(inputs)
        atoms_too_many_error = excinfo.value
        assert atoms_too_many_error.n_atoms == n_atoms
        assert atoms_too_many_error.max_atoms == n_atoms - 1
        estimator.set_max_atoms(None)


def check_estimate_max_atoms_and_neighbors_hard_limit(estimator):
    for n_atoms in range(10, 50, 10):
        atomic_numbers = np.array([1] * n_atoms)
        coordinates = np.zeros((n_atoms, 3))
        cell = np.array([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]])
        pbc = np.array([0, 0, 0])
        inputs = EstimatorSystem(
            atomic_numbers=atomic_numbers,
            coordinates=coordinates,
            cell=cell,
            pbc=pbc,
            properties=estimator.implemented_properties,
        )

        estimator.set_max_neighbors(n_atoms * (n_atoms - 1) // 2 - 1)
        inputs.input_max_neighbors = n_atoms * (n_atoms - 1) // 2 + 1
        with pytest.raises(NeighborsHardLimitExceededError) as excinfo:
            estimator.estimate(inputs)
        neighbors_too_many_error = excinfo.value
        assert neighbors_too_many_error.soft_max_neighbors == (n_atoms * (n_atoms - 1) // 2 + 1)
        assert neighbors_too_many_error.hard_max_neighbors == (n_atoms * (n_atoms - 1) // 2 - 1)
        assert neighbors_too_many_error.n_neighbors > neighbors_too_many_error.hard_max_neighbors
        assert neighbors_too_many_error.n_neighbors < neighbors_too_many_error.soft_max_neighbors

        inputs.input_max_neighbors = None
        estimator.set_max_neighbors(None)

        estimator.set_max_atoms(n_atoms - 1)
        inputs.input_max_atoms = n_atoms + 1
        with pytest.raises(AtomsHardLimitExceededError) as excinfo:
            estimator.estimate(inputs)
        atoms_too_many_error = excinfo.value
        assert atoms_too_many_error.n_atoms == n_atoms
        assert atoms_too_many_error.soft_max_atoms == n_atoms + 1
        assert atoms_too_many_error.hard_max_atoms == n_atoms - 1
        estimator.set_max_atoms(None)


def check_element_status(estimator, default_element_status):
    (
        accepted_atomic_numbers,
        experimental_atomic_numbers,
        unexpected_atomic_numbers,
    ) = default_element_status
    res_arr = [[], [], [], []]
    for i in range(130):
        e_status = estimator.element_status(np.array([i]))
        res_arr[e_status].append(i)
    assert res_arr[0] == accepted_atomic_numbers
    assert res_arr[1] == experimental_atomic_numbers
    assert res_arr[2] == unexpected_atomic_numbers
    assert res_arr[3] == [0, 128, 129]
