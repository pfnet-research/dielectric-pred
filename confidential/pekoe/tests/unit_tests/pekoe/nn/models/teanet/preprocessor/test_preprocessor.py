import itertools
from typing import Tuple

import numpy as np
import pytest
import torch
from ase.geometry import wrap_positions
from torch import Tensor

from pekoe.nn.models.teanet.preprocessor.preprocessor import (
    CellTooSmallError,
    GhostAtomsTooManyError,
    calc_neighbors,
    preprocess_pbc,
    wrap_coordinates,
)
from pekoe.utils.dummy_inputs import DUMMY_INPUT


def _get_torch_device(device_config):
    device, codegen_options = device_config
    if device.startswith(("mncore", "emu")):
        device = "cpu"
    elif device.startswith("pfvm"):
        # E.g., pfvm:cuda:0 => cuda:0
        device = ":".join(device.split(":")[1:])
    return device


def test_preprocess_pbc(device_config):
    device = _get_torch_device(device_config)
    seed = 1
    coordinates_raw = torch.tensor(
        np.random.RandomState(seed).uniform(0, 1, (2, 3)), dtype=torch.float32, device=device
    )
    cell_raw = torch.tensor(
        [[0.0, 10.1, 10.2], [10.3, 0.0, 10.4], [10.5, 10.6, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    pbc = torch.tensor([1, 1, 1], dtype=torch.int64, device=device)
    # Answer for fixed seed
    edge_ans = torch.tensor(
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
        ],
        dtype=torch.float32,
        device=device,
    )
    expanded_coordinates_ans = 160

    for x_mirror, y_mirror, z_mirror in itertools.product([-1.0, 1.0], repeat=3):
        cell_mirror_arr = torch.tensor(
            [[x_mirror, y_mirror, z_mirror]], dtype=cell_raw.dtype, device=device
        )
        cell = cell_raw * cell_mirror_arr
        coordinates = coordinates_raw @ cell

        atom_index1, atom_index2, shift_int = preprocess_pbc(
            coordinates, cell, pbc, 20.0, expanded_coordinates_ans
        )
        assert atom_index1.dtype == torch.int64
        assert atom_index2.dtype == torch.int64
        assert shift_int.dtype == torch.int64
        shift = shift_int.to(torch.float32)
        edge = coordinates[atom_index1] - coordinates[atom_index2] - torch.matmul(shift, cell)
        edge_length, _ = torch.sort(torch.sqrt(torch.sum(edge * edge, dim=1)))

        assert torch.allclose(edge_length, edge_ans, rtol=1.0e-5, atol=1.0e-5)

        # unlimited max_atoms case
        atom_index1, atom_index2, shift_int = preprocess_pbc(coordinates, cell, pbc, 20.0, -1)
        shift = shift_int.to(torch.float32)
        edge = coordinates[atom_index1] - coordinates[atom_index2] - torch.matmul(shift, cell)
        edge_length, _ = torch.sort(torch.sqrt(torch.sum(edge * edge, dim=1)))

        assert torch.allclose(edge_length, edge_ans, rtol=1.0e-5, atol=1.0e-5)

        with pytest.raises(CellTooSmallError):
            _, _, _ = preprocess_pbc(coordinates, cell, pbc, 20.0, coordinates.size()[0])

    cell_zero = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device
    )
    with pytest.raises(CellTooSmallError):
        preprocess_pbc(
            torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device=device),
            cell_zero,
            pbc,
            1.0,
            1000,
        )

    cell_tooshort = torch.tensor(
        [[0.001, 0.0, 0.0], [0.0, 0.001, 0.0], [0.0, 0.0, 0.001]],
        dtype=torch.float32,
        device=device,
    )
    with pytest.raises(CellTooSmallError):
        preprocess_pbc(
            torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device=device),
            cell_tooshort,
            pbc,
            1.0,
            1000,
        )


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
def test_wrap_coordinates(cell, coordinate, pbc, expected, device_config):
    device = _get_torch_device(device_config)
    pbcs = np.array([pbc] * 3)
    coordinates = np.array([coordinate])

    new_coordinates, fractional = wrap_coordinates(
        torch.tensor(coordinates, dtype=torch.float32, device=device),
        torch.tensor(cell, dtype=torch.float32, device=device),
        torch.tensor(pbcs, dtype=torch.int64, device=device),
    )
    is_out_of_bounds = np.any(
        np.logical_or(fractional.cpu().numpy() < 0.0, fractional.cpu().numpy() >= 1.0)
    )
    assert is_out_of_bounds == expected[1]
    expected_coordinates = expected[0]
    assert np.allclose(
        new_coordinates.cpu().numpy(), expected_coordinates, rtol=1.0e-5, atol=1.0e-5
    )


@pytest.mark.parametrize("pbc_x", [False, True])
@pytest.mark.parametrize("pbc_y", [False, True])
@pytest.mark.parametrize("pbc_z", [False, True])
def test_wrap_coordinates_random_cell_with_various_pbc(pbc_x, pbc_y, pbc_z):
    """Test random structure with various Periodic Boundary Conditions
    (ex. pbc=[True, True, False] for slab)"""
    n_nodes = 10

    # Randomly generated cell.
    cell = torch.tensor([[0.3919, 0.3857, 0.5021], [0.0, 1.0503, 0.3564], [0.0, 0.0, 0.4538]])
    pbc = torch.tensor([pbc_x, pbc_y, pbc_z])
    coordinates = (torch.rand((n_nodes, 3)) - 1.0) * 10.0

    ret, fractional = wrap_coordinates(coordinates, cell, pbc)

    cell_expanded = cell.clone()
    for i, pbc_xyz in enumerate(pbc):
        if not pbc_xyz:
            # Make cell sufficiently large in this direction.
            cell_expanded[i] = cell_expanded[i] * 1000.0
    ret_expected, fractional_expected = wrap_coordinates(coordinates, cell_expanded, pbc)
    assert torch.allclose(ret, ret_expected)
    assert torch.allclose(fractional, fractional_expected)


@pytest.mark.parametrize("pbc_x", [False, True])
@pytest.mark.parametrize("pbc_y", [False, True])
@pytest.mark.parametrize("pbc_z", [False, True])
def test_wrap_coordinates_unit_cell_with_various_pbc(pbc_x, pbc_y, pbc_z, device_config):
    """Test unit cube cell with various Periodic Boundary Conditions
    (ex. pbc=[True, True, False] for slab)"""
    device = _get_torch_device(device_config)
    cell = torch.eye(3, device=device)
    pbc = torch.tensor([pbc_x, pbc_y, pbc_z], device=device)
    coordinates = torch.tensor([[1.2, 1.3, 1.4]], device=device)

    ret, fractional = wrap_coordinates(coordinates, cell, pbc)
    exp_x = 0.2 if pbc_x else 1.2
    exp_y = 0.3 if pbc_y else 1.3
    exp_z = 0.4 if pbc_z else 1.4
    ret_expected = torch.tensor([[exp_x, exp_y, exp_z]], device=device)
    fractional_expected = coordinates - ret_expected
    assert torch.allclose(ret, ret_expected), f"ret {ret}, exp {ret_expected}"
    assert torch.allclose(
        fractional, fractional_expected
    ), f"frac {fractional}, exp {fractional_expected}"


def _sort_edge(atom_index1: Tensor, atom_index2: Tensor, shift: Tensor) -> Tuple[Tensor, Tensor]:
    """Uniquely sort `atom_index1`, `atom_index2` & `shift`
    Args:
        atom_index1 (Tensor): (n_edges,)
        atom_index2 (Tensor): (n_edges,)
        shift (Tensor): (n_edges, 3)
    Returns:
        edge_index (Tensor): (2, n_edges) sorted edge_index
        shift (Tensor): (n_edges, 3) sorted shift
    """
    edge_index = torch.stack((atom_index1, atom_index2))
    n_nodes = torch.max(edge_index) + 1
    src, dst = edge_index[0], edge_index[1]
    sx, sy, sz = shift[:, 0], shift[:, 1], shift[:, 2]

    # n_sx = torch.max(sx) - torch.min(sx) + 1
    n_sy = torch.max(sy) - torch.min(sy) + 1
    n_sz = torch.max(sz) - torch.min(sz) + 1

    # sort based on (sx, sy, sz, src, dst) whose dim is (n_sx, n_sy, n_sz, n_nodes, n_nodes).
    rank = sx
    rank = rank * n_sy + sy
    rank = rank * n_sz + sz
    rank = rank * n_nodes + src
    rank = rank * n_nodes + dst
    _, indices = torch.sort(rank)
    return edge_index[:, indices], shift[indices]


@pytest.mark.parametrize("pbc_x", [False, True])
@pytest.mark.parametrize("pbc_y", [False, True])
@pytest.mark.parametrize("pbc_z", [False, True])
def test_preprocess_pbc_with_various_pbc(pbc_x, pbc_y, pbc_z, device_config):
    """Test random structure with various Periodic Boundary Conditions
    (ex. pbc=[True, True, False] for slab)"""
    device = _get_torch_device(device_config)
    n_nodes = 10

    # seed = 77
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # cell = torch.rand(3, 3) + 0.1
    # Randomly generated cell.
    cell = torch.tensor(
        [[0.3919, 0.3857, 0.5021], [0.0, 1.0503, 0.3564], [0.0, 0.0, 0.4538]], device=device
    )
    pbc = torch.tensor([pbc_x, pbc_y, pbc_z], device=device)
    coordinates = torch.rand((n_nodes, 3), device=device) @ cell

    cutoff = 0.9
    atom_index1, atom_index2, shift_int = preprocess_pbc(
        coordinates=coordinates,
        cell=cell,
        pbc=pbc,
        cutoff=cutoff,
        max_atoms=1000,
    )

    cell_expanded = cell.clone()
    for i, pbc_xyz in enumerate(pbc):
        if not pbc_xyz:
            # Make cell sufficiently large in this direction.
            cell_expanded[i] = cell_expanded[i] * 1000.0
    all_direction_pbc = torch.tensor([True, True, True], device=device)
    atom_index1_expected, atom_index2_expected, shift_int_expected = preprocess_pbc(
        coordinates=coordinates,
        cell=cell_expanded,
        pbc=all_direction_pbc,
        cutoff=cutoff,
        max_atoms=1000,
    )
    # Sort edge indices for comparision
    edge_index_sorted, shift_sorted = _sort_edge(atom_index1, atom_index2, shift_int)
    edge_index_expected_sorted, shift_expected_sorted = _sort_edge(
        atom_index1_expected, atom_index2_expected, shift_int_expected
    )
    assert torch.all(edge_index_sorted == edge_index_expected_sorted)
    assert torch.all(shift_sorted == shift_expected_sorted)


def test_calc_neighbors(device_config):
    device = _get_torch_device(device_config)
    coordinates = torch.rand(20, 3, device=device)
    total_coordinates = torch.rand(50, 3, device=device)
    cutoff = 0.1
    right_ind1, left_ind1 = calc_neighbors(
        coordinates, total_coordinates, cutoff, max_matrix_size=100
    )
    right_ind2, left_ind2 = calc_neighbors(
        coordinates, total_coordinates, cutoff, max_matrix_size=200
    )
    assert torch.all(right_ind1 == right_ind2)
    assert torch.all(left_ind1 == left_ind2)


def test_preprocess_pbc_GhostAtomsTooManyError(device_config):
    device = _get_torch_device(device_config)
    n_atoms = DUMMY_INPUT.coordinates.size(0)
    too_small_max_atoms_plus_ghost_atoms = n_atoms + 1
    with pytest.raises(GhostAtomsTooManyError) as ex:
        preprocess_pbc(
            coordinates=DUMMY_INPUT.coordinates.to(device),
            cell=DUMMY_INPUT.cell[0].to(device),  # NOTE: (1, 3, 3) -> (3, 3)
            pbc=torch.tensor([True, True, True]),
            cutoff=0.9,
            max_atoms=n_atoms,
            max_atoms_plus_ghost_atoms=too_small_max_atoms_plus_ghost_atoms,
        )
    assert ex.value.max_atoms == too_small_max_atoms_plus_ghost_atoms
    assert ex.value.n_atoms > too_small_max_atoms_plus_ghost_atoms


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
