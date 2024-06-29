import logging
from typing import Optional, Tuple

import torch

# You can opt in faster preprocessor for CPU like
# https://github.pfidev.jp/hamaji/teanet_preproc
try:
    import teanet_preprocessor
except ImportError:
    teanet_preprocessor = None

logger = logging.getLogger(__file__)


def preprocess_pbc(
    coordinates: torch.Tensor,
    cell: Optional[torch.Tensor],
    pbc: torch.Tensor,
    cutoff: float,
    max_atoms: int,
    # NOTE: For the compatibility of non-Python implementations, None is not used for its default value
    max_atoms_plus_ghost_atoms: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """max_atoms: -1 for unlimited"""
    if coordinates.device.type == "cpu" and teanet_preprocessor is not None:
        atom_index1, atom_index2, shift = teanet_preprocessor.preprocess_pbc(
            coordinates, cell, pbc, cutoff, max_atoms, max_atoms_plus_ghost_atoms
        )
        logger.info(
            "[preprocess] n_atoms=%s\tn_edges=%s", coordinates.size(0), atom_index1.size(0)
        )
        return atom_index1, atom_index2, shift

    if torch.all(pbc == 0):
        right_ind, left_ind = calc_neighbors(coordinates, coordinates, cutoff)
        is_larger = left_ind < right_ind
        atom_index1 = left_ind[is_larger]
        atom_index2 = right_ind[is_larger]
        shift = torch.zeros(
            (atom_index1.size()[0], 3), device=coordinates.device, dtype=torch.int64
        )
        logger.info(
            "[preprocess] n_atoms=%s\tn_edges=%s", coordinates.size(0), atom_index1.size(0)
        )
        return atom_index1, atom_index2, shift

    assert cell is not None
    assert pbc.shape == (3,)
    ghost_shift, ghost_refs = ghost_copy(coordinates, cell, pbc, cutoff, max_atoms)
    total_coordinates = coordinates[ghost_refs] + torch.matmul(
        ghost_shift.to(dtype=coordinates.dtype), cell
    )

    n_atoms_plus_ghost_atoms = total_coordinates.size(0)

    # NOTE: To avoid the time-overrun of get_neighbors_single, the max number of atoms and ghost_atoms is
    # limited up to max_atoms_plus_ghost_atoms.
    if max_atoms_plus_ghost_atoms == -1:
        if max_atoms != -1:
            max_atoms_plus_ghost_atoms = max_atoms * 10
        else:
            # NOTE: If no max_atoms is specified, we do not limit n_atoms_plus_ghost_atoms.
            max_atoms_plus_ghost_atoms = n_atoms_plus_ghost_atoms

    if max_atoms != -1 and n_atoms_plus_ghost_atoms > max_atoms_plus_ghost_atoms:
        logger.info(
            f"GhostAtomsTooManyError has occurred (limit={max_atoms_plus_ghost_atoms}, "
            f"actual: {n_atoms_plus_ghost_atoms})"
        )
        raise GhostAtomsTooManyError(
            n_atoms_plus_ghost_atoms,
            max_atoms_plus_ghost_atoms,
            f"GhostAtomsTooManyError has occurred (limit={max_atoms_plus_ghost_atoms}, "
            f"actual: {n_atoms_plus_ghost_atoms})",
        )

    atom_index1, atom_index2, shift = get_neighbors_single(
        coordinates, total_coordinates, ghost_refs, ghost_shift, cutoff
    )
    logger.info("[preprocess] n_atoms=%s\tn_edges=%s", coordinates.size(0), atom_index1.size(0))
    return atom_index1, atom_index2, shift


def wrap_coordinates(
    coordinates: torch.Tensor,
    cell: Optional[torch.Tensor],
    pbc: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if torch.all(pbc == 0):
        return coordinates, torch.zeros_like(coordinates)
    else:
        assert cell is not None
        assert pbc.shape == (3,)
        fractional = torch.floor(torch.matmul(coordinates, torch.inverse(cell)))
        if not pbc[0]:
            fractional[:, 0] = 0.0
        if not pbc[1]:
            fractional[:, 1] = 0.0
        if not pbc[2]:
            fractional[:, 2] = 0.0
        wrap_shift: torch.Tensor = -1.0 * torch.matmul(fractional, cell)
        ret: torch.Tensor = coordinates + wrap_shift.type(coordinates.dtype)
        return ret, fractional


def calc_neighbors(
    coordinates: torch.Tensor,
    total_coordinates: torch.Tensor,
    cutoff: float,
    max_matrix_size: int = 100 * 1000 * 1000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate `edge_index` within `cutoff`
    Args:
        coordinates (Tensor): (n_atoms_in_cell, 3)
        total_coordinates (Tensor): (n_total_atoms, 3)
        cutoff (float): cutoff distance
        max_matrix_size (int): distance matrix larger than this will be split
    Returns:
        left_ind (Tensor): (n_edges,)
        right_ind (Tensor): (n_edges,)
    """

    if coordinates.shape[0] > 0:
        max_total_coordinates_len = max_matrix_size // coordinates.shape[0]
    else:
        max_total_coordinates_len = max_matrix_size

    right_inds = []
    left_inds = []
    offset = 0

    cut_sq = cutoff * cutoff
    for split_coordinates in total_coordinates.split(max_total_coordinates_len):
        distances = torch.sum(
            (coordinates.unsqueeze(0) - split_coordinates.unsqueeze(1)).pow_(2), dim=2
        )
        assert distances.numel() <= max_matrix_size
        right_ind, left_ind = torch.where(distances < cut_sq)
        right_ind += offset
        offset += split_coordinates.shape[0]
        right_inds.append(right_ind)
        left_inds.append(left_ind)

    return torch.cat(right_inds), torch.cat(left_inds)


def get_neighbors_single(
    coordinates: torch.Tensor,
    total_coordinates: torch.Tensor,
    ghost_refs: torch.Tensor,
    shift_coordinates: torch.Tensor,
    cutoff: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    right_ind, left_ind = calc_neighbors(coordinates, total_coordinates, cutoff)

    right_ind_refs = ghost_refs[right_ind]

    larger_ind = left_ind < right_ind_refs
    left_ind_larger = left_ind[larger_ind]
    right_ind_larger = right_ind[larger_ind]

    same_ind = left_ind == right_ind_refs
    left_ind_same = left_ind[same_ind]
    right_ind_same = right_ind[same_ind]
    shift_diff_same_atom = shift_coordinates[left_ind_same] - shift_coordinates[right_ind_same]
    half_same_atom = (shift_diff_same_atom[:, 0] > 0) | (
        (shift_diff_same_atom[:, 0] == 0)
        & (
            (shift_diff_same_atom[:, 1] > 0)
            | ((shift_diff_same_atom[:, 1] == 0) & (shift_diff_same_atom[:, 2] > 0))
        )
    )
    atom_index1 = torch.cat([left_ind_larger, left_ind_same[half_same_atom]])
    atom_index2_raw = torch.cat([right_ind_larger, right_ind_same[half_same_atom]])
    shift = shift_coordinates[atom_index2_raw]

    atom_index2 = ghost_refs[atom_index2_raw]

    return atom_index1, atom_index2, shift


def cell_width(
    cell: torch.Tensor,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[float, float, float]]:
    # (xn, yn, zn): unit vectors of three planes
    # (cell_dx, cell_dy, cell_dz): width of cell along three plane vectors
    xn = torch.cross(cell[1, :], cell[2, :])
    yn = torch.cross(cell[2, :], cell[0, :])
    zn = torch.cross(cell[0, :], cell[1, :])
    xn = xn / torch.norm(xn)
    yn = yn / torch.norm(yn)
    zn = zn / torch.norm(zn)

    cell_dx = float(torch.abs(torch.matmul(cell[0, :], xn)))
    cell_dy = float(torch.abs(torch.matmul(cell[1, :], yn)))
    cell_dz = float(torch.abs(torch.matmul(cell[2, :], zn)))

    return (xn, yn, zn), (cell_dx, cell_dy, cell_dz)


def _extend_ghost_along_axis(
    current_refs: torch.Tensor,
    current_shift: torch.Tensor,
    rep_x: int,
    xdl: torch.Tensor,
    cell_dx: float,
    shift_x: torch.Tensor,
    cutoff: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extend ghost atoms along specific axis
    Args:
        current_refs: (n_current_atoms,) indices of atoms in base cell
        current_shift:  (n_current_atoms, 3) shift of this atom
        rep_x:
        xdl: (n_atoms,) number of atoms in base cell
        cell_dx:
        shift_x: (1, 3) one hot vector representing which axis to expand.
        cutoff (float): cutoff distance
    Returns:
        ghost_refs: (n_total_atoms,) Extended indices of atoms
        ghost_shift:  (n_total_atoms, 3) Extended shift of atoms
    """
    n_atoms: int = current_refs.shape[0]
    # current_refs (n_all_extended_atoms = (2*rep_x+1) * n_atoms,)
    repeated_current_refs = current_refs.repeat(2 * rep_x + 1)
    repeated_current_shift = current_shift.repeat(2 * rep_x + 1, 1)
    # xdl_refs (n_all_extended_atoms,)
    xdl_refs = xdl[repeated_current_refs]
    # cell_step (n_all_extended_atoms,)
    # We would like to align cell_step starting from 0, so that original cell always comes first.
    # Ex. [0, 1, 2, -1, -2] for n_atoms=2.
    rep_x_range = torch.arange(2 * rep_x + 1, device=repeated_current_refs.device)
    rep_x_range[rep_x + 1 :] = -torch.arange(1, rep_x + 1, device=repeated_current_refs.device)
    cell_step = rep_x_range.repeat_interleave(n_atoms)
    # 1st eq for cell_step > 0, 2nd eq for cell_step < 0.
    dists = torch.max((cell_step - 1) * cell_dx + xdl_refs, -cell_step * cell_dx - xdl_refs)
    within_cutoff = dists < cutoff
    ghost_refs = repeated_current_refs[within_cutoff]
    ghost_shift = (repeated_current_shift + cell_step.unsqueeze(1) * shift_x)[within_cutoff]
    return ghost_refs, ghost_shift


def ghost_copy(
    coordinates: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: float,
    max_atoms: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Calculate distances between points and 6 planes
    # (d = |(x,n)-h| where the plane is defined as (x,n)-h=0)
    (xn, yn, zn), (cell_dx, cell_dy, cell_dz) = cell_width(cell)

    if cell_dx == 0.0 or cell_dy == 0.0 or cell_dz == 0.0:
        # Infinite ghost atoms
        raise CellTooSmallError

    rep_x = int(cutoff / cell_dx) + 1
    rep_y = int(cutoff / cell_dy) + 1
    rep_z = int(cutoff / cell_dz) + 1

    # Considering extreme case, like n_real_atoms = 1
    n_atoms_margin = 10
    n_max_atoms_matrix = (max_atoms + n_atoms_margin) * (max_atoms + n_atoms_margin)
    n_real_atoms_with_margin = coordinates.shape[0] + n_atoms_margin
    # Optimistic (at least) estimated atom number
    # Please see preprocess_pbc (around GhostAtomsTooManyError) as well
    n_ghost_atoms_optimistic = (
        (2 * rep_x - 1) * (2 * rep_y - 1) * (2 * rep_z - 1) * n_real_atoms_with_margin
    )
    if (
        max_atoms != -1
        and n_real_atoms_with_margin * n_ghost_atoms_optimistic > n_max_atoms_matrix
    ):
        raise CellTooSmallError

    xdl = torch.abs(torch.matmul(coordinates, xn))
    ydl = torch.abs(torch.matmul(coordinates, yn))
    zdl = torch.abs(torch.matmul(coordinates, zn))

    n_atoms = coordinates.shape[0]
    ghost_shift = torch.zeros((n_atoms, 3), dtype=torch.int64, device=coordinates.device)
    ghost_refs = torch.arange(n_atoms, dtype=torch.int64, device=coordinates.device)

    shift_x = torch.tensor([[1, 0, 0]], dtype=torch.int64, device=coordinates.device)
    shift_y = torch.tensor([[0, 1, 0]], dtype=torch.int64, device=coordinates.device)
    shift_z = torch.tensor([[0, 0, 1]], dtype=torch.int64, device=coordinates.device)

    # x_axis replicate
    if pbc[0]:
        ghost_refs, ghost_shift = _extend_ghost_along_axis(
            ghost_refs, ghost_shift, rep_x, xdl, cell_dx, shift_x, cutoff
        )

    # y_axis replicate
    if pbc[1]:
        ghost_refs, ghost_shift = _extend_ghost_along_axis(
            ghost_refs, ghost_shift, rep_y, ydl, cell_dy, shift_y, cutoff
        )

    # z_axis replicate
    if pbc[2]:
        ghost_refs, ghost_shift = _extend_ghost_along_axis(
            ghost_refs, ghost_shift, rep_z, zdl, cell_dz, shift_z, cutoff
        )

    return ghost_shift, ghost_refs


class CellTooSmallError(Exception):
    """The cell shape is too small."""

    pass


class GhostAtomsTooManyError(Exception):
    """Too many ghost atoms."""

    def __init__(self, n_atoms: int, max_atoms: int, message: str = ""):
        self.n_atoms = n_atoms
        self.max_atoms = max_atoms
        super().__init__(message)
