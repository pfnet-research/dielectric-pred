import itertools

import numpy as np
import pytest
from ase.geometry import wrap_positions

from pfp.nn.models.crystal.model_builder import build_estimator


@pytest.mark.libinfer
def test_preprocess_pbc():
    estimator = build_estimator(device=0)
    seed = 1
    coordinates_raw = np.random.RandomState(seed).uniform(0, 1, (2, 3))
    cell_raw = np.array([[0.0, 10.1, 10.2], [10.3, 0.0, 10.4], [10.5, 10.6, 0.0]])
    pbc = np.array([1, 1, 1], dtype=np.int32)
    # Answer for fixed seed
    edge_ans = np.array(
        [
            6.279273146377722,
            8.679736666758957,
            11.497088713713824,
            12.381413246589316,
            13.510146222750308,
            13.798105154994394,
            14.354441821262155,
            14.354441821262155,
            14.427057912131636,
            14.427057912131636,
            14.428604030520185,
            14.637281168304447,
            14.637281168304447,
            14.647184029703457,
            14.647184029703457,
            14.851262572589578,
            14.851262572589578,
            14.920120642943877,
            14.920120642943877,
            15.755417215262908,
            17.31340313814986,
            17.562303138641763,
            18.15463678032028,
            18.20775893902259,
            19.051390857431773,
            19.255626411800353,
            19.60692750658464,
        ]
    )
    expanded_coordinates_ans = 160

    for x_mirror, y_mirror, z_mirror in itertools.product([-1.0, 1.0], repeat=3):
        cell_mirror_arr = np.array([[x_mirror, y_mirror, z_mirror]], dtype=cell_raw.dtype)
        cell = cell_raw * cell_mirror_arr
        coordinates = coordinates_raw @ cell
        atom_index1, atom_index2, shift_int, _ = estimator.model.ppp(
            coordinates, cell, pbc, 20.0, expanded_coordinates_ans
        )
        assert atom_index1.dtype == np.int64
        assert atom_index2.dtype == np.int64
        assert shift_int.dtype == np.int64
        shift = shift_int.astype(np.float32)
        edge = coordinates[atom_index1] - coordinates[atom_index2] - np.matmul(shift, cell)
        edge_length = np.sort(np.sqrt(np.sum(edge * edge, axis=1)))

        assert np.allclose(edge_length, edge_ans, rtol=1.0e-5, atol=1.0e-5)

        # unlimited max_atoms case
        atom_index1, atom_index2, shift_int, _ = estimator.model.ppp(
            coordinates, cell, pbc, 20.0, -1
        )
        shift = shift_int.astype(np.float32)
        edge = coordinates[atom_index1] - coordinates[atom_index2] - np.matmul(shift, cell)
        edge_length = np.sort(np.sqrt(np.sum(edge * edge, axis=1)))

        assert np.allclose(edge_length, edge_ans, rtol=1.0e-5, atol=1.0e-5)

        _, _, _, error_value = estimator.model.ppp(
            coordinates, cell, pbc, 20.0, coordinates.shape[0]
        )

        assert error_value == estimator.model.ErrorEnumCC.CellTooSmallError

    cell_zero = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    atom_index1, atom_index2, shift_int, error_value = estimator.model.ppp(
        np.array([[0.0, 0.0, 0.0]], dtype=np.float32), cell_zero, pbc, 1.0, 1000000
    )
    assert error_value == estimator.model.ErrorEnumCC.CellTooSmallError

    cell_tooshort = np.array(
        [[0.001, 0.0, 0.0], [0.0, 0.001, 0.0], [0.0, 0.0, 0.001]], dtype=np.float32
    )
    atom_index1, atom_index2, shift_int, error_value = estimator.model.ppp(
        np.array([[0.0, 0.0, 0.0]], dtype=np.float32), cell_tooshort, pbc, 1.0, 1000
    )
    assert error_value == estimator.model.ErrorEnumCC.CellTooSmallError


cube = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
hexahedron = np.array([[1.0, 0.01, 0.02], [0.03, 1.0, 0.04], [0.05, 0.06, 1.0]])


@pytest.mark.parametrize(
    "cell,coordinate,pbc,expected",
    [
        (cube, [0.5, 0.5, 0.5], False, ([[0.5, 0.5, 0.5]], False)),
        (cube, [0.5, 0.5, 0.5], True, ([[0.5, 0.5, 0.5]], False)),
        (cube, [1.5, 1.5, 1.5], False, ([[1.5, 1.5, 1.5]], False)),
        (cube, [1.5, 1.5, 1.5], True, ([[0.5, 0.5, 0.5]], True)),
        (hexahedron, [0.5, 0.5, 0.5], False, ([[0.5, 0.5, 0.5]], False)),
        (hexahedron, [0.5, 0.5, 0.5], True, ([[0.5, 0.5, 0.5]], False)),
        (hexahedron, [1.5, 1.5, 1.5], False, ([[1.5, 1.5, 1.5]], False)),
        (
            hexahedron,
            [1.5, 1.5, 1.5],
            True,
            (wrap_positions(np.array([[1.5, 1.5, 1.5]]), hexahedron), True),
        ),
    ],
)
@pytest.mark.libinfer
def test_wrap_coordinates(cell, coordinate, pbc, expected):
    estimator = build_estimator(device=0)
    pbcs = np.array([pbc] * 3)
    coordinates = np.array([coordinate])

    new_coordinates, fractional = estimator.model.ppw(coordinates, cell, np.linalg.inv(cell), pbcs)
    is_out_of_bounds = np.any(np.logical_or(fractional < 0.0, fractional >= 1.0))
    assert is_out_of_bounds == expected[1]
    expected_coordinates = expected[0]
    assert np.allclose(new_coordinates, expected_coordinates, rtol=1.0e-5, atol=1.0e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
