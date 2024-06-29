from typing import Optional, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors


def preprocess_pbc(
    species: np.ndarray,
    coordinates: np.ndarray,
    cell: Optional[np.ndarray],
    pbc: np.ndarray,
    cutoff: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if np.all(pbc == 0) or pbc is None:
        ghost_shift = np.zeros((0, 3), dtype=coordinates.dtype)
        ghost_refs = np.zeros((0,), dtype=np.int32)
    elif np.all(pbc == 1):
        assert isinstance(cell, np.ndarray)
        ghost_shift, ghost_refs = ghost_copy(coordinates, cell, pbc, cutoff)
        ghost_shift = np.inner(ghost_shift, cell.transpose())
    else:
        raise NotImplementedError("Only all 0 or 1 can be available for pbc now.")

    num_atoms = len(coordinates)
    shift_coordinates = np.concatenate([np.zeros_like(coordinates), ghost_shift])
    total_coordinates = np.concatenate([coordinates, coordinates[ghost_refs] + ghost_shift])
    ghost_refs_all = np.concatenate([np.arange(num_atoms), ghost_refs], axis=0)

    atom_index1, atom_index2 = get_neighbors_single(
        total_coordinates, num_atoms, ghost_refs_all, cutoff
    )

    shift = shift_coordinates[atom_index2]
    atom_index2 = ghost_refs_all[atom_index2]

    return atom_index1, atom_index2, shift


def get_neighbors_single(
    coordinates: np.ndarray, num_atoms: int, ghost_refs_all: np.ndarray, cutoff: float
) -> Tuple[np.ndarray, np.ndarray]:
    # get neighbor site
    nn = NearestNeighbors(radius=cutoff)
    nn.fit(coordinates)
    neigh_ind = nn.radius_neighbors(coordinates[:num_atoms], return_distance=False)
    # n_total = 0
    left_ind = []
    for i, n in enumerate(neigh_ind):
        left_ind.append(i * np.ones(n.shape[0], dtype=np.int32))
        # n_total += n.shape[0]
    left_ind = np.concatenate(left_ind, axis=0)
    right_ind = np.concatenate(neigh_ind, axis=0)
    neigh_unique_ind = ghost_refs_all[left_ind] < ghost_refs_all[right_ind]
    neigh_same_ind = ghost_refs_all[left_ind] == ghost_refs_all[right_ind]
    l_same_ind = []
    r_same_ind = []
    for li, ri in zip(left_ind[neigh_same_ind], right_ind[neigh_same_ind]):
        lc = coordinates[li]
        rc = coordinates[ri]
        if lc[0] < rc[0]:
            l_same_ind.append(li)
            r_same_ind.append(ri)
        elif lc[0] == rc[0]:
            if lc[1] < rc[1]:
                l_same_ind.append(li)
                r_same_ind.append(ri)
            elif lc[1] == rc[1]:
                if lc[2] < rc[2]:
                    l_same_ind.append(li)
                    r_same_ind.append(ri)
    atom_index1 = left_ind[neigh_unique_ind]
    atom_index2 = right_ind[neigh_unique_ind]

    if len(l_same_ind) > 0:
        atom_index1 = np.concatenate((atom_index1, np.array(l_same_ind)))
        atom_index2 = np.concatenate((atom_index2, np.array(r_same_ind)))
    # distances = coordinates[atom_index1]-coordinates[atom_index2]

    return atom_index1, atom_index2


def ghost_copy(
    coordinates: np.ndarray, cell: np.ndarray, pbc: np.ndarray, cutoff: float
) -> Tuple[np.ndarray, np.ndarray]:
    # Calculate distances between points and 6 planes
    # (d = |(x,n)-h| where the plane is defined as (x,n)-h=0)
    xn = np.cross(cell[1, :], cell[2, :])
    yn = np.cross(cell[0, :], cell[2, :])
    zn = np.cross(cell[0, :], cell[1, :])
    xn = xn / np.linalg.norm(xn)
    yn = yn / np.linalg.norm(yn)
    zn = zn / np.linalg.norm(zn)

    cell_dx = np.abs(np.inner(cell[0, :], xn))
    cell_dy = np.abs(np.inner(cell[1, :], yn))
    cell_dz = np.abs(np.inner(cell[2, :], zn))
    rep_x = int(cutoff / cell_dx) + 1
    rep_y = int(cutoff / cell_dy) + 1
    rep_z = int(cutoff / cell_dz) + 1

    xdl = np.abs(np.inner(coordinates, xn))
    ydl = np.abs(np.inner(coordinates, yn))
    zdl = np.abs(np.inner(coordinates, zn))

    n_coords = len(coordinates)

    ghost_coordinates = np.zeros((0, 3), dtype=coordinates.dtype)
    ghost_refs = np.zeros((0,), dtype=np.int32)

    # x_axis replicate
    if pbc[0]:
        for cell_step in range(1, rep_x + 1):
            xd_idx = np.where(xdl + (cell_step - 1) * cell_dx < cutoff)
            ghost_refs_add1 = np.arange(n_coords)[xd_idx]
            ghost_coordinates_add1 = (np.zeros_like(coordinates) + np.array([[cell_step, 0, 0]]))[
                xd_idx
            ]
            xd_idx = np.where((cell_dx - xdl) + (cell_step - 1) * cell_dx < cutoff)
            ghost_refs_add2 = np.arange(n_coords)[xd_idx]
            ghost_coordinates_add2 = (np.zeros_like(coordinates) + np.array([[-cell_step, 0, 0]]))[
                xd_idx
            ]

            ghost_refs = np.concatenate((ghost_refs, ghost_refs_add1, ghost_refs_add2))
            ghost_coordinates = np.concatenate(
                (ghost_coordinates, ghost_coordinates_add1, ghost_coordinates_add2)
            )

    # y_axis replicate
    if pbc[1]:
        ng = len(ghost_refs)
        for cell_step in range(1, rep_y + 1):
            yd_idx = np.where(ydl + (cell_step - 1) * cell_dy < cutoff)
            ghost_refs_add1 = np.arange(n_coords)[yd_idx]
            ghost_coordinates_add1 = (np.zeros_like(coordinates) + np.array([[0, cell_step, 0]]))[
                yd_idx
            ]
            yd_idx = np.where((cell_dy - ydl) + (cell_step - 1) * cell_dy < cutoff)
            ghost_refs_add2 = np.arange(n_coords)[yd_idx]
            ghost_coordinates_add2 = (np.zeros_like(coordinates) + np.array([[0, -cell_step, 0]]))[
                yd_idx
            ]

            yd_idx = np.where(ydl[ghost_refs[:ng]] + (cell_step - 1) * cell_dy < cutoff)
            ghost_refs_add3 = np.arange(n_coords)[np.array(ghost_refs)[yd_idx]]
            ghost_coordinates_add3 = (ghost_coordinates[:ng] + np.array([[0, cell_step, 0]]))[
                yd_idx
            ]
            yd_idx = np.where(
                (cell_dy - ydl[ghost_refs[:ng]]) + (cell_step - 1) * cell_dy < cutoff
            )
            ghost_refs_add4 = np.arange(n_coords)[np.array(ghost_refs)[yd_idx]]
            ghost_coordinates_add4 = (ghost_coordinates[:ng] + np.array([[0, -cell_step, 0]]))[
                yd_idx
            ]

            ghost_refs = np.concatenate(
                (
                    ghost_refs,
                    ghost_refs_add1,
                    ghost_refs_add2,
                    ghost_refs_add3,
                    ghost_refs_add4,
                )
            )
            ghost_coordinates = np.concatenate(
                (
                    ghost_coordinates,
                    ghost_coordinates_add1,
                    ghost_coordinates_add2,
                    ghost_coordinates_add3,
                    ghost_coordinates_add4,
                )
            )

    # z_axis replicate
    if pbc[2]:
        ng = len(ghost_refs)
        for cell_step in range(1, rep_z + 1):
            zd_idx = np.where(zdl + (cell_step - 1) * cell_dz < cutoff)
            ghost_refs_add1 = np.arange(n_coords)[zd_idx]
            ghost_coordinates_add1 = (np.zeros_like(coordinates) + np.array([[0, 0, cell_step]]))[
                zd_idx
            ]
            zd_idx = np.where((cell_dz - zdl) + (cell_step - 1) * cell_dz < cutoff)
            ghost_refs_add2 = np.arange(n_coords)[zd_idx]
            ghost_coordinates_add2 = (np.zeros_like(coordinates) + np.array([[0, 0, -cell_step]]))[
                zd_idx
            ]

            zd_idx = np.where(zdl[ghost_refs[:ng]] + (cell_step - 1) * cell_dz < cutoff)
            ghost_refs_add3 = np.arange(n_coords)[np.array(ghost_refs)[zd_idx]]
            ghost_coordinates_add3 = (ghost_coordinates[:ng] + np.array([[0, 0, cell_step]]))[
                zd_idx
            ]
            zd_idx = np.where(
                (cell_dz - zdl[ghost_refs[:ng]]) + (cell_step - 1) * cell_dz < cutoff
            )
            ghost_refs_add4 = np.arange(n_coords)[np.array(ghost_refs)[zd_idx]]
            ghost_coordinates_add4 = (ghost_coordinates[:ng] + np.array([[0, 0, -cell_step]]))[
                zd_idx
            ]

            ghost_refs = np.concatenate(
                (
                    ghost_refs,
                    ghost_refs_add1,
                    ghost_refs_add2,
                    ghost_refs_add3,
                    ghost_refs_add4,
                )
            )
            ghost_coordinates = np.concatenate(
                (
                    ghost_coordinates,
                    ghost_coordinates_add1,
                    ghost_coordinates_add2,
                    ghost_coordinates_add3,
                    ghost_coordinates_add4,
                )
            )

    if len(ghost_coordinates) == 0:
        return np.zeros((0, 3), dtype=np.int32), np.zeros((0,), dtype=np.int32)
    return ghost_coordinates, ghost_refs
