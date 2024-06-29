import itertools
from typing import Any, Dict, List, Optional, Union

import numpy as np

from pfp.nn.estimator_base import (
    AtomsFarAwayError,
    AtomsTooManyError,
    BaseEstimator,
    BatchAtomsTooManyError,
    BatchNeighborsTooManyError,
    CellTooSmallError,
    EstimatorCalcMode,
    EstimatorSystem,
    IllegalElementError,
    ModeInvalidError,
    NeighborsTooManyError,
    PFPError,
)
from pfp.utils.atoms import ElementStatusEnum
from pfp.utils.messages import MessageEnum


def raise_exception(ErrorEnumCC: Any, error_code: Any) -> None:
    if error_code == ErrorEnumCC.NoError:
        return
    if error_code == ErrorEnumCC.AtomsFarAwayError:
        raise AtomsFarAwayError
    if error_code == ErrorEnumCC.CellTooSmallError:
        raise CellTooSmallError
    if error_code == ErrorEnumCC.GhostAtomsTooManyError:
        # TODO: set n_atoms and max_atoms
        raise AtomsTooManyError(n_atoms=-1, max_atoms=-1)


def calc_mode_to_vector(
    calc_mode: EstimatorCalcMode, model_version: str, element_supported_np: np.ndarray
) -> np.ndarray:
    u_param_elements_list = np.array([23, 24, 25, 26, 27, 28, 29, 42, 74])
    if model_version.startswith(("v0.", "v1.0.", "v1.1.")):
        if calc_mode is EstimatorCalcMode.CRYSTAL:
            x_add = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        elif calc_mode is EstimatorCalcMode.CRYSTAL_U0:
            x_add = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        elif calc_mode is EstimatorCalcMode.MOLECULE:
            x_add = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        elif calc_mode is EstimatorCalcMode.OC20:
            x_add = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            raise ValueError("Unknown calc_mode.")
        x_add = np.repeat(np.expand_dims(x_add, 0), element_supported_np.shape[0], axis=0)
    elif model_version.startswith(("v1.2.", "v1.3.", "v1.4")):
        # CRYSTAL_U0 corresponds to 0
        if calc_mode is EstimatorCalcMode.CRYSTAL_U0:
            x_add = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        elif calc_mode is EstimatorCalcMode.CRYSTAL:
            # Special consideration for CRYSTAL
            x_add = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        elif calc_mode is EstimatorCalcMode.MOLECULE:
            x_add = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        elif calc_mode is EstimatorCalcMode.OC20:
            x_add = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            raise ValueError("Unknown calc_mode.")
        x_add = np.repeat(np.expand_dims(x_add, 0), element_supported_np.shape[0], axis=0)

        if calc_mode is EstimatorCalcMode.CRYSTAL:
            x_add[u_param_elements_list] = np.expand_dims(
                np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 0
            )
    else:
        raise ValueError("Unknown model")
    return x_add


class EnergyEstimator(BaseEstimator):
    """ """

    implemented_properties = ["energy", "forces", "virial", "charges"]

    def __init__(
        self,
        model: Any,
        cutoff: float,
        element_energy: Optional[Dict[str, float]] = None,
        calc_mode: EstimatorCalcMode = EstimatorCalcMode.CRYSTAL_U0,
        available_calc_modes: Optional[List[EstimatorCalcMode]] = None,
        max_neighbors: Optional[int] = None,
        max_atoms: Optional[int] = None,
        version: Optional[str] = None,
    ):
        """ """
        super(EnergyEstimator, self).__init__()
        self.model = model
        self.cutoff: float = cutoff
        self.use_book_keeping: bool = False
        self.book_keeping_skin: float = 0.0
        self.calc_mode: EstimatorCalcMode = calc_mode
        self.available_calc_modes_value: List[EstimatorCalcMode] = (
            available_calc_modes if available_calc_modes else [self.calc_mode]
        )
        self.preprocess_bruteforce_threshold: int = 30
        self.previous_book_keeping: float = False
        self.max_neighbors: Optional[int] = max_neighbors
        self.max_atoms: Optional[int] = max_atoms
        self.version = version if version else "UNDEFINED"

        element_energy_arr = [0.0 for _ in range(120)]
        if element_energy is not None:
            for k, v in element_energy.items():
                element_energy_arr[int(k)] = float(v)
        self.element_energy = np.array(element_energy_arr, dtype=np.float32)

        self.x_add_dict: Dict[EstimatorCalcMode, np.ndarray] = {
            e: calc_mode_to_vector(e, self.version, self.element_supported_np)
            for e in EstimatorCalcMode
        }

    def set_book_keeping(
        self, use_book_keeping: bool = True, book_keeping_skin: float = 2.0
    ) -> None:
        self.use_book_keeping = use_book_keeping
        if use_book_keeping:
            assert book_keeping_skin >= 0.0
            self.book_keeping_skin = book_keeping_skin
        else:
            self.book_keeping_skin = 0.0
        self.model.bks(use_book_keeping, book_keeping_skin)

    def reset_book_keeping(self) -> None:
        self.model.bkr()

    def set_small_system_threshold(self, threshold: int = 30) -> None:
        self.preprocess_bruteforce_threshold = threshold

    def set_calc_mode(self, calc_mode: EstimatorCalcMode) -> None:
        if calc_mode not in self.available_calc_modes():
            raise ModeInvalidError
        self.calc_mode = calc_mode

    def available_calc_modes(self) -> List[EstimatorCalcMode]:
        return self.available_calc_modes_value

    def set_max_atoms(self, max_atoms: Optional[int]) -> None:
        self.max_atoms = max_atoms

    def set_max_neighbors(self, max_neighbors: Optional[int]) -> None:
        self.max_neighbors = max_neighbors

    def get_version(self) -> str:
        return self.version

    def estimate(
        self, args: EstimatorSystem
    ) -> Dict[str, Union[float, Dict[str, int], np.ndarray, List[MessageEnum]]]:
        """ """
        properties = args.properties
        args.format()

        coord_dtype = np.dtype(np.float64)
        if self.version.startswith(("v0.", "v1.0.", "v1.1.")):
            coord_dtype = np.dtype(np.float32)

        assert args.atomic_numbers is not None
        if args.calc_mode is None:
            calc_mode = self.calc_mode
        else:
            calc_mode = args.calc_mode
        pbc = args.pbc
        if pbc is None:
            pbc = np.array([0, 0, 0], dtype=np.int32)
        if np.all(pbc == 0):
            is_pbc = False
        else:
            is_pbc = True
        atomic_numbers = args.atomic_numbers.astype(np.int64)
        coordinates = args.coordinates.astype(coord_dtype)
        cell = args.cell.astype(coord_dtype)

        n_atoms = atomic_numbers.shape[0]
        soft_max_atoms = args.input_max_atoms
        hard_max_atoms = self.max_atoms
        # (jettan) this version does not raise HardLimitExceededError.
        if soft_max_atoms is not None and n_atoms > soft_max_atoms:
            raise AtomsTooManyError(n_atoms, soft_max_atoms)
        elif hard_max_atoms is not None and n_atoms > hard_max_atoms:
            raise AtomsTooManyError(n_atoms, hard_max_atoms)

        message_output: List[MessageEnum] = []
        elements_condition = self.element_status(atomic_numbers)
        if elements_condition == ElementStatusEnum.Illegal:
            raise IllegalElementError
        if elements_condition == ElementStatusEnum.Unexpected:
            self._append_message_if_active(MessageEnum.UnexpectedElementWarning, message_output)
        if elements_condition == ElementStatusEnum.Experimental:
            self._append_message_if_active(MessageEnum.ExperimentalElementWarning, message_output)

        refresh_neighbor, refresh_exception_id = (True, self.model.ErrorEnumCC.NoError)
        n_neighbors = -1  # Will be updated later
        if self.use_book_keeping:
            refresh_neighbor, refresh_exception_id, n_neighbors = self.model.bkc(
                coordinates, cell, is_pbc
            )
            self.previous_book_keeping = not refresh_neighbor

        if refresh_neighbor:
            raise_exception(self.model.ErrorEnumCC, refresh_exception_id)
            atom_pos, fractional = self.model.ppw(coordinates, cell, np.linalg.inv(cell), pbc)
            if (
                self.use_book_keeping
                and len(atom_pos) <= self.preprocess_bruteforce_threshold
                and not is_pbc
            ):
                n_atoms = len(atom_pos)
                a12_wide = np.array(
                    list(itertools.combinations(range(n_atoms), 2)),
                    dtype=np.int64,
                )
                a1 = np.ascontiguousarray(a12_wide[:, 0])
                a2 = np.ascontiguousarray(a12_wide[:, 1])
                sh = np.zeros((a1.shape[0], atom_pos.shape[1]), dtype=coord_dtype)
            else:
                max_atoms_int = self.max_atoms if self.max_atoms else -1
                a1, a2, sh, error_code = self.model.ppp(
                    atom_pos, cell, pbc, self.cutoff + self.book_keeping_skin, max_atoms_int
                )
                raise_exception(self.model.ErrorEnumCC, error_code)
                a1 = a1.astype(np.int64)
                a2 = a2.astype(np.int64)
                sh = sh.astype(coord_dtype)

            n_neighbors = sh.shape[0]
            soft_max_neighbors = args.input_max_neighbors
            hard_max_neighbors = self.max_neighbors

            # (jettan) this version does not raise HardLimitExceededError.
            if soft_max_neighbors is not None and n_neighbors > soft_max_neighbors:
                raise NeighborsTooManyError(n_neighbors, soft_max_neighbors)
            elif hard_max_neighbors is not None and n_neighbors > hard_max_neighbors:
                raise NeighborsTooManyError(n_neighbors, hard_max_neighbors)

            ba = np.zeros((coordinates.shape[0],), dtype=np.int64)
            be = np.zeros((sh.shape[0],), dtype=np.int64)
            if self.use_book_keeping:
                self.model.spb(fractional, a1, a2, sh, ba, be, coordinates, cell, is_pbc)
            else:
                self.model.sp(fractional, a1, a2, sh, ba, be)
            n_neighbors = sh.shape[0]

        shift_energy = np.sum(self.element_energy[atomic_numbers])
        results: Dict[str, Union[float, Dict[str, int], np.ndarray, List[MessageEnum]]] = dict()
        results["messages"] = message_output

        ob = 1
        xa = self.x_add_dict[calc_mode][atomic_numbers].astype(coord_dtype)

        repulsion_calc_mode_int = 1 if calc_mode == EstimatorCalcMode.CRYSTAL else 0
        cm = repulsion_calc_mode_int * np.ones((coordinates.shape[0],), dtype=np.int64)

        if set(properties) <= set(["energy", "charges"]):
            energies, charges = self.model.r(coordinates, atomic_numbers, cell, ob, xa, cm, False)
            results["energy"] = float((energies + shift_energy).item())
            results["charges"] = charges
        else:
            energies, charges, forces, virial = self.model.r(
                coordinates, atomic_numbers, cell, ob, xa, cm, True
            )
            results["energy"] = float((energies + shift_energy).item())
            results["charges"] = charges
            results["forces"] = forces
            if "virial" in properties:
                results["virial"] = virial.flatten()

        results["calc_stats"] = {
            "n_neighbors": n_neighbors,
        }
        assert isinstance(results["energy"], float)
        return results

    def batch_estimate(
        self, args_list: List[EstimatorSystem]
    ) -> List[
        Union[Dict[str, Union[np.ndarray, float, Dict[str, int], List[MessageEnum]]], PFPError]
    ]:
        """ """
        coord_dtype = np.dtype(np.float64)
        if self.version.startswith(("v0.", "v1.0.", "v1.1.")):
            coord_dtype = np.dtype(np.float32)
        atomic_numbers_arr = []
        coordinates_arr = []
        a1_arr = []
        a2_arr = []
        sh_arr = []
        ba_arr = []
        be_arr = []
        xa_arr = []
        cm_arr = []
        n_atoms_sum = 0
        n_atoms_arr = [0]
        shift_energy_arr = []
        message_output_arr: List[List[MessageEnum]] = []
        n_neighbors_list = []

        results_valid_arr: List[Union[int, PFPError]] = list()
        n_valid = 0
        total_n_atoms = 0
        total_n_neighbors = 0
        for args in args_list:
            try:
                args.format()
            except PFPError as e:
                results_valid_arr.append(e)
                continue

            assert args.atomic_numbers is not None
            if args.calc_mode is None:
                calc_mode = self.calc_mode
            else:
                calc_mode = args.calc_mode
            atomic_numbers = args.atomic_numbers.astype(np.int64)
            coordinates = args.coordinates.astype(coord_dtype)
            cell = args.cell.astype(coord_dtype)
            pbc = args.pbc
            if pbc is None:
                pbc = np.array([0, 0, 0], dtype=np.int32)

            atom_pos, fractional = self.model.ppw(coordinates, cell, np.linalg.inv(cell), pbc)

            message_output: List[MessageEnum] = []
            elements_condition = self.element_status(atomic_numbers)
            if elements_condition == ElementStatusEnum.Illegal:
                results_valid_arr.append(IllegalElementError())
                continue
            if elements_condition == ElementStatusEnum.Unexpected:
                self._append_message_if_active(
                    MessageEnum.UnexpectedElementWarning, message_output
                )
            if elements_condition == ElementStatusEnum.Experimental:
                self._append_message_if_active(
                    MessageEnum.ExperimentalElementWarning, message_output
                )

            n_atoms = len(atomic_numbers)
            soft_max_atoms = args.input_max_atoms
            hard_max_atoms = self.max_atoms
            if soft_max_atoms is not None and n_atoms > soft_max_atoms:
                results_valid_arr.append(AtomsTooManyError(n_atoms, soft_max_atoms))
                continue
            if hard_max_atoms is not None and n_atoms > hard_max_atoms:
                results_valid_arr.append(AtomsTooManyError(n_atoms, hard_max_atoms))
                continue
            if hard_max_atoms is not None and total_n_atoms + n_atoms > hard_max_atoms:
                results_valid_arr.append(
                    BatchAtomsTooManyError(total_n_atoms + n_atoms, hard_max_atoms)
                )
                continue

            if (
                self.use_book_keeping
                and len(atom_pos) <= self.preprocess_bruteforce_threshold
                and (np.all(pbc == 0) or pbc is None)
            ):
                a12_wide = np.array(
                    list(itertools.combinations(range(n_atoms), 2)),
                    dtype=np.int64,
                )
                a1 = np.ascontiguousarray(a12_wide[:, 0])
                a2 = np.ascontiguousarray(a12_wide[:, 1])
                sh = np.zeros((a1.shape[0], atom_pos.shape[1]), dtype=coord_dtype)
            else:
                max_atoms_int = self.max_atoms if self.max_atoms else -1
                a1, a2, sh, error_code = self.model.ppp(
                    atom_pos, cell, pbc, self.cutoff + self.book_keeping_skin, max_atoms_int
                )
                try:
                    raise_exception(self.model.ErrorEnumCC, error_code)
                except PFPError as e:
                    results_valid_arr.append(e)
                    continue
                a1 = a1.astype(np.int64)
                a2 = a2.astype(np.int64)
                sh = sh.astype(coord_dtype)
                sh = np.matmul(sh, cell).astype(coord_dtype)

            n_neighbors = sh.shape[0]
            soft_max_neighbors = args.input_max_neighbors
            hard_max_neighbors = self.max_neighbors
            if soft_max_neighbors is not None and n_neighbors > soft_max_neighbors:
                results_valid_arr.append(NeighborsTooManyError(n_neighbors, soft_max_neighbors))
                continue
            if hard_max_neighbors is not None and n_neighbors > hard_max_neighbors:
                results_valid_arr.append(NeighborsTooManyError(n_neighbors, hard_max_neighbors))
                continue
            if (
                hard_max_neighbors is not None
                and total_n_neighbors + n_neighbors > hard_max_neighbors
            ):
                results_valid_arr.append(
                    BatchNeighborsTooManyError(total_n_neighbors + n_neighbors, hard_max_neighbors)
                )
                continue

            total_n_atoms += n_atoms
            total_n_neighbors += n_neighbors
            n_neighbors_list.append(n_neighbors)

            atomic_numbers_arr.append(atomic_numbers)
            coordinates_arr.append(atom_pos)
            a1_arr.append(a1 + n_atoms_sum)
            a2_arr.append(a2 + n_atoms_sum)
            sh_arr.append(sh)
            ba_arr.append(n_valid * np.ones((coordinates.shape[0],), dtype=np.int64))
            be_arr.append(n_valid * np.ones((sh.shape[0],), dtype=np.int64))
            xa_arr.extend(self.x_add_dict[calc_mode][atomic_numbers].astype(coord_dtype))
            repulsion_calc_mode_int = 1 if calc_mode == EstimatorCalcMode.CRYSTAL else 0
            cm_arr.append(
                repulsion_calc_mode_int * np.ones((coordinates.shape[0],), dtype=np.int64)
            )
            n_atoms_sum += atomic_numbers.shape[0]
            n_atoms_arr.append(n_atoms_sum)
            shift_energy_arr.append(np.sum(self.element_energy[atomic_numbers]))
            results_valid_arr.append(n_valid)
            n_valid += 1

            message_output_arr.append(message_output)

        if len(coordinates_arr) == 0:
            results_succeeded: List[
                Dict[str, Union[np.ndarray, float, Dict[str, int], List[MessageEnum]]]
            ] = list()
        else:
            sh_cat = np.concatenate(sh_arr, axis=0)
            atomic_numbers_cat = np.concatenate(atomic_numbers_arr, axis=0)
            if self.max_atoms is not None:
                assert atomic_numbers_cat.shape[0] <= self.max_atoms
            if self.max_neighbors is not None:
                assert sh_cat.shape[0] <= self.max_neighbors
            if len(xa_arr) == 0:
                xa_stack = np.zeros(
                    (0, self.x_add_dict[EstimatorCalcMode.CRYSTAL].shape[1]), dtype=coord_dtype
                )
            else:
                xa_stack = np.stack(xa_arr, axis=0)

            coordinates_cat = np.concatenate(coordinates_arr, axis=0).astype(coord_dtype)
            self.model.sp(
                np.zeros_like(coordinates_cat),
                np.concatenate(a1_arr, axis=0),
                np.concatenate(a2_arr, axis=0),
                sh_cat,
                np.concatenate(ba_arr, axis=0),
                np.concatenate(be_arr, axis=0),
            )
            energies, charges, forces, virial = self.model.r(
                coordinates_cat,
                atomic_numbers_cat,
                np.identity(3).astype(coord_dtype),
                len(atomic_numbers_arr),
                xa_stack,
                np.concatenate(cm_arr, axis=0),
                True,
            )

            shift_energy_np = np.stack(shift_energy_arr)
            results_succeeded = [
                {
                    "energy": float(e),
                    "charges": c,
                    "forces": f,
                    "virial": v,
                    "messages": m,
                    "calc_stats": {
                        "n_neighbors": n,
                    },
                }
                for e, c, f, v, m, n in zip(
                    (energies.flatten() + shift_energy_np).tolist(),
                    np.split(charges, n_atoms_arr[1:-1]),
                    np.split(forces, n_atoms_arr[1:-1]),
                    virial.tolist(),
                    message_output_arr,
                    n_neighbors_list,
                )
            ]

        results: List[
            Union[Dict[str, Union[np.ndarray, float, Dict[str, int], List[MessageEnum]]], PFPError]
        ] = [results_succeeded[x] if isinstance(x, int) else x for x in results_valid_arr]

        return results
