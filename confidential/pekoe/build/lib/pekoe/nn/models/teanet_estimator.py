import itertools
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

import pekoe.nn.models.onnx
from pekoe.nn.estimator_base import (
    AtomsHardLimitExceededError,
    AtomsTooManyError,
    BaseEstimator,
    BatchAtomsTooManyError,
    BatchNeighborsTooManyError,
    CellTooSmallError,
    EstimatorCalcMode,
    EstimatorKind,
    IllegalElementError,
    ModeInvalidError,
    NeighborsHardLimitExceededError,
    NeighborsTooManyError,
    PFPError,
    max_atoms_from_input,
)
from pekoe.nn.models.teanet.preprocessor.preprocessor import (
    CellTooSmallError as TeanetCellTooSmallError,
)
from pekoe.nn.models.teanet.preprocessor.preprocessor import (
    GhostAtomsTooManyError as TeanetGhostAtomsTooManyError,
)
from pekoe.nn.models.teanet.preprocessor.preprocessor import (
    cell_width,
    preprocess_pbc,
    wrap_coordinates,
)
from pekoe.nn.models.teanet.teanet_base import TeaNetBase
from pfp.nn.estimator_base import EstimatorSystem
from pfp.utils.atoms import ElementStatusEnum
from pfp.utils.messages import MessageEnum


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
    elif model_version.startswith(("v1.2.", "v1.3.", "v1.4.")):
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


def batch_estimate_preprocess(
    atomic_numbers_t: torch.Tensor,
    coordinates_t: torch.Tensor,
    cell_t: torch.Tensor,
    pbc_t: torch.Tensor,
    cutoff: float,
    soft_max_atoms_single: Optional[int] = None,
    soft_max_neighbors_single: Optional[int] = None,
    hard_max_atoms_single: Optional[int] = None,
    hard_max_neighbors_single: Optional[int] = None,
    max_atoms_batch: Optional[int] = None,
    max_neighbors_batch: Optional[int] = None,
    current_n_atoms: int = 0,
    current_n_neighbors: int = 0,
    coord_dtype: torch.dtype = torch.float64,
    model_dtype: torch.dtype = torch.float32,
    estimator_kind: EstimatorKind = EstimatorKind.PFP,
) -> Tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_atoms = atomic_numbers_t.size(0)
    if soft_max_atoms_single is not None:
        if n_atoms > soft_max_atoms_single:
            raise AtomsTooManyError(n_atoms, soft_max_atoms_single, estimator_kind=estimator_kind)
        elif hard_max_atoms_single is not None and n_atoms > hard_max_atoms_single:
            raise AtomsHardLimitExceededError(
                n_atoms,
                soft_max_atoms_single,
                hard_max_atoms_single,
                estimator_kind=estimator_kind,
            )
    else:
        if hard_max_atoms_single is not None and n_atoms > hard_max_atoms_single:
            raise AtomsTooManyError(n_atoms, hard_max_atoms_single, estimator_kind=estimator_kind)

    if max_atoms_batch is not None:
        if current_n_atoms + n_atoms > max_atoms_batch:
            raise BatchAtomsTooManyError(
                current_n_atoms + n_atoms, max_atoms_batch, estimator_kind=estimator_kind
            )

    coordinates_t, _ = wrap_coordinates(coordinates_t, cell_t, pbc_t)
    try:
        max_atoms_single = max_atoms_from_input(soft_max_atoms_single, hard_max_atoms_single)
        a1, a2, sh_int = preprocess_pbc(
            coordinates_t.to(model_dtype),
            cell_t.to(model_dtype),
            pbc_t,
            cutoff,
            max_atoms=max_atoms_single if max_atoms_single else -1,
        )
    except TeanetCellTooSmallError as e:
        raise CellTooSmallError(str(e))
    except TeanetGhostAtomsTooManyError as e:
        raise AtomsTooManyError(e.n_atoms, e.max_atoms)

    n_neighbors = int(sh_int.size()[0])
    if soft_max_neighbors_single is not None:
        if n_neighbors > soft_max_neighbors_single:
            raise NeighborsTooManyError(
                n_neighbors, soft_max_neighbors_single, estimator_kind=estimator_kind
            )
        elif hard_max_neighbors_single is not None and n_neighbors > hard_max_neighbors_single:
            raise NeighborsHardLimitExceededError(
                n_neighbors,
                soft_max_neighbors_single,
                hard_max_neighbors_single,
                estimator_kind=estimator_kind,
            )
    else:
        if hard_max_neighbors_single is not None and n_neighbors > hard_max_neighbors_single:
            raise NeighborsTooManyError(
                n_neighbors, hard_max_neighbors_single, estimator_kind=estimator_kind
            )
    if max_neighbors_batch is not None:
        if current_n_neighbors + n_neighbors > max_neighbors_batch:
            raise BatchNeighborsTooManyError(
                current_n_neighbors + n_neighbors,
                max_neighbors_batch,
                estimator_kind=estimator_kind,
            )
    return n_atoms, n_neighbors, a1, a2, sh_int, coordinates_t


def elements_status_check(
    estimator: BaseEstimator, elements_condition: ElementStatusEnum
) -> List[MessageEnum]:
    message_output: List[MessageEnum] = []
    if elements_condition == ElementStatusEnum.Illegal:
        raise IllegalElementError
    if elements_condition == ElementStatusEnum.Unexpected:
        estimator._append_message_if_active(MessageEnum.UnexpectedElementWarning, message_output)
    if elements_condition == ElementStatusEnum.Experimental:
        estimator._append_message_if_active(MessageEnum.ExperimentalElementWarning, message_output)
    return message_output


class TeaNetEstimator(BaseEstimator):
    """ """

    implemented_properties = ["energy", "forces", "virial", "charges"]

    def __init__(
        self,
        model: TeaNetBase,
        element_energy: Optional[Dict[str, str]] = None,
        calc_mode: EstimatorCalcMode = EstimatorCalcMode.CRYSTAL,
        available_calc_modes: Optional[List[EstimatorCalcMode]] = None,
        output_onnx: Optional[str] = None,
        output_onnx_first_forward_only: bool = True,
        max_neighbors: Optional[int] = None,
        max_atoms: Optional[int] = None,
        version: Optional[str] = None,
    ):
        super(TeaNetEstimator, self).__init__()
        self.model = model
        self.cutoff: float = max(model.cutoff_list)
        self.atom_index1_wide: Optional[torch.Tensor] = None
        self.atom_index2_wide: Optional[torch.Tensor] = None
        self.shift_wide: Optional[torch.Tensor] = None
        self.book_keeping_skin: float = 0.0
        self.use_book_keeping: bool = False
        self.atom_pos_prev: Optional[torch.Tensor] = None
        self.corner_pos_prev: Optional[torch.Tensor] = None
        self.fraction_prev: Optional[torch.Tensor] = None
        self.CORNERS: torch.Tensor = torch.tensor(
            [list(x) for x in itertools.product([0.0, 1.0], repeat=3)],
            dtype=torch.float64,
        )
        self.calc_mode: EstimatorCalcMode = calc_mode
        self.available_calc_modes_value: List[EstimatorCalcMode] = (
            available_calc_modes if available_calc_modes else [self.calc_mode]
        )
        self.previous_book_keeping: bool = False
        self.output_onnx = output_onnx
        self.output_onnx_first_forward_only = output_onnx_first_forward_only
        self.output_onnx_count: int = 0
        self.max_neighbors: Optional[int] = max_neighbors
        self.max_atoms: Optional[int] = max_atoms
        self.version = version if version else "UNDEFINED"

        element_energy_arr = [0.0 for _ in range(120)]
        if element_energy is not None:
            for k, v in element_energy.items():
                element_energy_arr[int(k)] = float(v)
        self.element_energy: np.ndarray = np.array(element_energy_arr, dtype=np.float64)
        if not isinstance(self.model.device, (torch.device, str)):
            raise ValueError("Device cannot be specified")

        self.x_add_dict: Dict[EstimatorCalcMode, torch.Tensor] = {
            e: torch.tensor(
                calc_mode_to_vector(e, self.version, self.element_supported_np),
                dtype=torch.float32,
            )
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

    def reset_book_keeping(self) -> None:
        self.atom_pos_prev = None

    def shifted_energy(self, atomic_numbers: np.ndarray) -> float:
        return float(np.sum(self.element_energy[atomic_numbers]).item())

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

    def to(self, device_str: str) -> None:
        if device_str == "auto":
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = self.model.torch_device_from_str(device_str)
        self.model.to(device)
        self.CORNERS = self.CORNERS.to(device)
        for k in self.x_add_dict.keys():
            self.x_add_dict[k] = self.x_add_dict[k].to(device)

    def eval(self) -> None:
        self.model.eval()

    def forward(
        self, *inputs: Any, calc_mode_type: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.output_onnx and (
            not self.output_onnx_first_forward_only or self.output_onnx_count == 0
        ):
            self.output_onnx_count += 1
            onnx_inputs = (*inputs, calc_mode_type)
            pekoe.nn.models.onnx.export_with_testcase(self.model, onnx_inputs, self.output_onnx)
        energy, charges = self.model(*inputs, calc_mode_type=calc_mode_type)
        e_ret: torch.Tensor = energy.to(torch.float64)
        return e_ret, charges

    def estimate(
        self, args: EstimatorSystem
    ) -> Dict[str, Union[float, Dict[str, int], np.ndarray, List[MessageEnum]]]:
        """ """
        properties = args.properties
        args.format()
        assert args.atomic_numbers is not None
        if args.calc_mode is None:
            calc_mode = self.calc_mode
        else:
            if args.calc_mode not in self.available_calc_modes():
                raise ModeInvalidError
            calc_mode = args.calc_mode
        atomic_numbers = args.atomic_numbers.astype(np.int64)
        coordinates = args.coordinates
        cell = args.cell
        pbc = args.pbc
        if pbc is None:
            pbc = np.array([0, 0, 0], dtype=np.int32)
        if np.all(pbc == 0):
            is_pbc = False
        else:
            is_pbc = True

        n_atoms = atomic_numbers.shape[0]
        soft_max_atoms = args.input_max_atoms
        hard_max_atoms = self.max_atoms
        if soft_max_atoms is not None:
            if n_atoms > soft_max_atoms:
                raise AtomsTooManyError(n_atoms, soft_max_atoms)
            elif hard_max_atoms is not None and n_atoms > hard_max_atoms:
                raise AtomsHardLimitExceededError(n_atoms, soft_max_atoms, hard_max_atoms)
        elif hard_max_atoms is not None and n_atoms > hard_max_atoms:
            raise AtomsTooManyError(n_atoms, hard_max_atoms)

        coord_dtype = self.model.coord_dtype
        model_dtype = torch.float32
        device = self.model.device
        coordinates_t = torch.tensor(coordinates, dtype=coord_dtype, device=device)
        cell_t: torch.Tensor = torch.tensor(cell, dtype=coord_dtype, device=device)
        pbc_t: torch.Tensor = torch.tensor(pbc, dtype=torch.int64, device=device)

        message_output: List[MessageEnum] = []
        elements_condition = self.element_status(atomic_numbers)
        if elements_condition == ElementStatusEnum.Illegal:
            raise IllegalElementError
        if elements_condition == ElementStatusEnum.Unexpected:
            self._append_message_if_active(MessageEnum.UnexpectedElementWarning, message_output)
        if elements_condition == ElementStatusEnum.Experimental:
            self._append_message_if_active(MessageEnum.ExperimentalElementWarning, message_output)

        elapsed_preprocess = -1.0  # put dummy time for when preprocess was not performed

        refresh_neighbor = True
        if self.use_book_keeping:
            if self.atom_pos_prev is not None:
                assert self.corner_pos_prev is not None
                assert self.fraction_prev is not None
                atom_pos_t = coordinates_t - torch.mm(self.fraction_prev, cell_t)
                if len(atom_pos_t) == len(self.atom_pos_prev):
                    atom_diff = atom_pos_t - self.atom_pos_prev
                    atom_diffsq = float(torch.max(torch.sum(atom_diff * atom_diff, dim=1)).item())
                    corner_pos_margin = 0.0
                    if is_pbc:
                        corner_pos_diff = (
                            torch.mm(self.CORNERS, cell_t.to(torch.float64)) - self.corner_pos_prev
                        )
                        corner_pos_margin = float(
                            torch.sqrt(
                                torch.max(torch.sum(corner_pos_diff * corner_pos_diff, dim=1))
                            ).item()
                        )
                    skin = max(0.0, self.book_keeping_skin - 2.0 * corner_pos_margin)
                    if atom_diffsq < (0.5 * skin) ** 2:
                        refresh_neighbor = False
            self.previous_book_keeping = not refresh_neighbor

        if refresh_neighbor:
            # Replace atoms into the simulation box.
            # Replacing after book-keeping decision allow atoms move over boundaries
            # without refresh neighboring.
            # However, at the very moment of book-keeping reneighboring, it ensures
            # that all atoms are within the simulation box.
            atom_pos_t, fractional_t = wrap_coordinates(coordinates_t, cell_t, pbc_t)
            if self.use_book_keeping:
                _, (cell_dx, cell_dy, cell_dz) = cell_width(cell_t)
                # cell acceptable deformation margin (skin)
                #  + atom acceptable fluctuation margin (0.5 * skin)
                if (
                    is_pbc
                    and min(cell_dx, cell_dy, cell_dz) < self.cutoff + 1.5 * self.book_keeping_skin
                ):
                    raise CellTooSmallError
                self.atom_pos_prev = atom_pos_t
                self.corner_pos_prev = torch.mm(self.CORNERS, cell_t.to(torch.float64))
                self.fraction_prev = fractional_t
            try:
                max_atoms = max_atoms_from_input(soft_max_atoms, hard_max_atoms)
                start_t = time.perf_counter()
                atom_index1_ar, atom_index2_ar, shift_int_ar = preprocess_pbc(
                    atom_pos_t.to(model_dtype),
                    cell_t.to(model_dtype),
                    pbc_t,
                    self.cutoff + self.book_keeping_skin,
                    max_atoms=max_atoms if max_atoms else -1,
                )
                elapsed_preprocess = time.perf_counter() - start_t
            except TeanetCellTooSmallError:
                raise CellTooSmallError
            except TeanetGhostAtomsTooManyError as e:
                raise AtomsTooManyError(e.n_atoms, e.max_atoms)
            shift_ar = shift_int_ar.to(dtype=coord_dtype)
            self.atom_index1_wide = atom_index1_ar
            self.atom_index2_wide = atom_index2_ar
            self.shift_wide = shift_ar
            n_neighbors = self.shift_wide.size(0)
            soft_max_neighbors = args.input_max_neighbors
            hard_max_neighbors = self.max_neighbors
            if soft_max_neighbors is not None:
                if n_neighbors > soft_max_neighbors:
                    raise NeighborsTooManyError(n_neighbors, soft_max_neighbors)
                elif hard_max_neighbors is not None and n_neighbors > hard_max_neighbors:
                    raise NeighborsHardLimitExceededError(
                        n_neighbors, soft_max_neighbors, hard_max_neighbors
                    )
            # soft limit not defined, check hard limit and throw as usual
            elif hard_max_neighbors is not None and n_neighbors > hard_max_neighbors:
                raise NeighborsTooManyError(n_neighbors, hard_max_neighbors)
        assert self.atom_index1_wide is not None
        assert self.atom_index2_wide is not None
        assert self.shift_wide is not None

        atomic_numbers_t = torch.tensor(atomic_numbers, dtype=torch.int64, device=device)

        if self.use_book_keeping:
            vecs_all = (
                atom_pos_t[self.atom_index1_wide]
                - atom_pos_t[self.atom_index2_wide]
                - torch.mm(self.shift_wide, cell_t)
            )
            rsq_all = torch.sum(vecs_all * vecs_all, dim=1)
            within_cutoff = rsq_all < (self.cutoff * self.cutoff)
            atom_index1 = self.atom_index1_wide[within_cutoff]
            atom_index2 = self.atom_index2_wide[within_cutoff]
            shift = self.shift_wide[within_cutoff]
            if not self.model.is_codegen:
                vecs = vecs_all[within_cutoff]
        else:
            atom_index1 = self.atom_index1_wide
            atom_index2 = self.atom_index2_wide
            shift = self.shift_wide
            if not self.model.is_codegen:
                vecs = atom_pos_t[atom_index1] - atom_pos_t[atom_index2] - torch.mm(shift, cell_t)

        ob = torch.zeros((1,), dtype=torch.float32, device=device)
        ba = torch.zeros((coordinates_t.size()[0],), dtype=torch.int64, device=device)
        be = torch.zeros((shift.size()[0],), dtype=torch.int64, device=device)
        xa = self.x_add_dict[calc_mode][atomic_numbers_t]

        repulsion_calc_mode_int = 1 if calc_mode == EstimatorCalcMode.CRYSTAL else 0
        cm = repulsion_calc_mode_int * torch.ones(
            (coordinates_t.size()[0],), dtype=torch.int64, device=device
        )

        results: Dict[str, Union[float, Dict[str, int], np.ndarray, List[MessageEnum]]] = dict()
        results["messages"] = message_output

        start_t = time.perf_counter()

        if self.model.is_codegen:
            energy, forces, virial, charges = self.model(
                atom_pos_t,
                atomic_numbers_t,
                atom_index1,
                atom_index2,
                shift.long(),
                cell_t.unsqueeze(0),
                ob,
                ba,
                be,
                xa,
                cm,
            )
            results["energy"] = float(energy.item()) + self.shifted_energy(atomic_numbers)
            results["charges"] = charges.detach().cpu().numpy()
            if not (set(properties) <= set(["energy", "charges"])):
                results["forces"] = forces.detach().cpu().numpy()
            if "virial" in properties:
                results["virial"] = virial.detach().cpu().numpy().squeeze(0)
        elif set(properties) <= set(["energy", "charges"]):
            if self.version.startswith(("v0.", "v1.0.", "v1.1.", "v1.2")):
                energy, charges = self.forward(
                    vecs.to(model_dtype),
                    atomic_numbers_t,
                    atom_index1,
                    atom_index2,
                    ob,
                    ba,
                    be,
                    xa,
                )
            else:
                energy, charges = self.forward(
                    vecs.to(model_dtype),
                    atomic_numbers_t,
                    atom_index1,
                    atom_index2,
                    ob,
                    ba,
                    be,
                    xa,
                    calc_mode_type=cm,
                )
            results["energy"] = float(energy.item()) + self.shifted_energy(atomic_numbers)
            results["charges"] = charges.detach().cpu().numpy()

        else:
            with torch.enable_grad():
                vecs = vecs.requires_grad_(True)
                if self.version.startswith(("v0.", "v1.0.", "v1.1.", "v1.2")):
                    energy, charges = self.forward(
                        vecs.to(model_dtype),
                        atomic_numbers_t,
                        atom_index1,
                        atom_index2,
                        ob,
                        ba,
                        be,
                        xa,
                    )
                else:
                    energy, charges = self.forward(
                        vecs.to(model_dtype),
                        atomic_numbers_t,
                        atom_index1,
                        atom_index2,
                        ob,
                        ba,
                        be,
                        xa,
                        calc_mode_type=cm,
                    )
                forces_raw = torch.autograd.grad(energy, vecs)[0]

            n_edge = vecs.size()[0]
            forces_zeros = torch.zeros_like(atom_pos_t)
            forces = -forces_zeros.scatter_add(
                0, atom_index1.view(n_edge, 1).expand(n_edge, 3), forces_raw
            ) + forces_zeros.scatter_add(
                0, atom_index2.view(n_edge, 1).expand(n_edge, 3), forces_raw
            )

            results["energy"] = float(energy.item()) + self.shifted_energy(atomic_numbers)
            results["charges"] = charges.detach().cpu().numpy()
            results["forces"] = forces.detach().cpu().numpy()

            if "virial" in properties:
                virial_raw = forces_raw[:, [0, 1, 2, 1, 2, 0]] * vecs[:, [0, 1, 2, 2, 0, 1]]
                virial = virial_raw.sum(dim=0)

                results["virial"] = virial.detach().cpu().numpy()
        elapsed_infer = time.perf_counter() - start_t

        results["calc_stats"] = {
            "n_neighbors": self.shift_wide.size(0),
            "elapsed_usec_preprocess": int(1e6 * elapsed_preprocess),  # returns in microseconds
            "elapsed_usec_infer": int(1e6 * elapsed_infer),
        }
        return results

    def batch_estimate(
        self, args_list: List[EstimatorSystem]
    ) -> List[
        Union[Dict[str, Union[np.ndarray, float, Dict[str, int], List[MessageEnum]]], PFPError]
    ]:
        """ """
        coord_dtype = torch.float64
        if self.version.startswith(("v0.", "v1.0.", "v1.1.")):
            coord_dtype = torch.float32
        model_dtype = torch.float32
        device = self.model.device
        atomic_numbers_arr = []
        coordinates_t_arr = []
        a1_arr = []
        a2_arr = []
        sh_arr = []
        sh_int_arr = []
        cell_arr = []
        ba_arr = []
        be_arr = []
        xa_arr: List[torch.Tensor] = []
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
        elapsed_preprocess_total = 0.0
        for args in args_list:
            try:
                args.format()

                assert args.atomic_numbers is not None
                if args.calc_mode is None:
                    calc_mode = self.calc_mode
                else:
                    if args.calc_mode not in self.available_calc_modes():
                        raise ModeInvalidError
                    calc_mode = args.calc_mode
                atomic_numbers = args.atomic_numbers.astype(np.int64)
                coordinates = args.coordinates
                cell = args.cell
                pbc = args.pbc
                if pbc is None:
                    pbc = np.array([0, 0, 0], dtype=np.int32)

                elements_condition = self.element_status(atomic_numbers)
                message_output = elements_status_check(self, elements_condition)

                atomic_numbers_t = torch.tensor(atomic_numbers, dtype=torch.int64, device=device)
                coordinates_t = torch.tensor(coordinates, dtype=coord_dtype, device=device)
                cell_t: torch.Tensor = torch.tensor(cell, dtype=coord_dtype, device=device)
                pbc_t: torch.Tensor = torch.tensor(pbc, dtype=torch.int64, device=device)

                start_t = time.perf_counter()
                batch_estimate_preprocess_results = batch_estimate_preprocess(
                    atomic_numbers_t,
                    coordinates_t,
                    cell_t,
                    pbc_t,
                    self.cutoff,
                    soft_max_atoms_single=args.input_max_atoms,
                    soft_max_neighbors_single=args.input_max_neighbors,
                    hard_max_atoms_single=self.max_atoms,
                    hard_max_neighbors_single=self.max_neighbors,
                    max_atoms_batch=self.max_atoms,
                    max_neighbors_batch=self.max_neighbors,
                    current_n_atoms=total_n_atoms,
                    current_n_neighbors=total_n_neighbors,
                    coord_dtype=coord_dtype,
                    model_dtype=model_dtype,
                )
                elapsed_preprocess_total += time.perf_counter() - start_t

                (
                    n_atoms,
                    n_neighbors,
                    a1,
                    a2,
                    sh_int,
                    coordinates_t,
                ) = batch_estimate_preprocess_results
                if not self.model.is_codegen:
                    sh = sh_int.to(dtype=coord_dtype)
                    sh = torch.mm(sh, cell_t)  # remove cell dependency here

            except PFPError as e:
                results_valid_arr.append(e)
                continue

            total_n_atoms += n_atoms
            total_n_neighbors += n_neighbors
            n_neighbors_list.append(n_neighbors)

            atomic_numbers_arr.append(atomic_numbers_t)
            coordinates_t_arr.append(coordinates_t)
            a1_arr.append(a1 + n_atoms_sum)
            a2_arr.append(a2 + n_atoms_sum)
            if not self.model.is_codegen:
                sh_arr.append(sh)
            sh_int_arr.append(sh_int)
            cell_arr.append(cell_t)
            ba_arr.append(
                n_valid * torch.ones((coordinates_t.size()[0],), dtype=torch.int64, device=device)
            )
            be_arr.append(
                n_valid * torch.ones((sh_int.size()[0],), dtype=torch.int64, device=device)
            )
            xa_arr.extend(self.x_add_dict[calc_mode][atomic_numbers_t])
            repulsion_calc_mode_int = 1 if calc_mode == EstimatorCalcMode.CRYSTAL else 0
            cm_arr.append(
                repulsion_calc_mode_int
                * torch.ones((coordinates_t.size()[0],), dtype=torch.int64, device=device)
            )
            n_atoms_sum += atomic_numbers.shape[0]
            n_atoms_arr.append(n_atoms_sum)
            shift_energy_arr.append(self.shifted_energy(atomic_numbers))
            results_valid_arr.append(n_valid)
            n_valid += 1

            message_output_arr.append(message_output)

        if len(coordinates_t_arr) == 0:
            results_succeeded: List[
                Dict[str, Union[np.ndarray, float, Dict[str, int], List[MessageEnum]]]
            ] = list()
        else:
            start_t = time.perf_counter()
            coordinates_t = torch.cat(coordinates_t_arr, dim=0)
            atomic_numbers_t = torch.cat(atomic_numbers_arr, dim=0)
            atom_index1 = torch.cat(a1_arr, dim=0)
            atom_index2 = torch.cat(a2_arr, dim=0)
            if self.max_atoms is not None:
                assert atomic_numbers_t.size()[0] <= self.max_atoms
            if self.max_neighbors is not None:
                assert atom_index1.size()[0] <= self.max_neighbors

            ob = torch.zeros((len(atomic_numbers_arr),), dtype=torch.float32, device=device)
            ba = torch.cat(ba_arr, dim=0)
            be = torch.cat(be_arr, dim=0)
            if len(xa_arr) == 0:
                xa = torch.zeros(
                    (0, self.x_add_dict[EstimatorCalcMode.CRYSTAL].size()[1]),
                    dtype=torch.float32,
                    device=device,
                )
            else:
                xa = torch.stack(xa_arr, dim=0)
            cm = torch.cat(cm_arr, dim=0)

            if self.model.is_codegen:
                shift_int = torch.cat(sh_int_arr, dim=0)
                energy, forces, virial, charges = self.model(
                    coordinates_t,
                    atomic_numbers_t,
                    atom_index1,
                    atom_index2,
                    shift_int.long(),
                    torch.stack(cell_arr),
                    ob,
                    ba,
                    be,
                    xa,
                    cm,
                )
            else:
                shift = torch.cat(sh_arr, dim=0)
                vecs = coordinates_t[atom_index1] - coordinates_t[atom_index2] - shift

                n_edge = vecs.size()[0]
                with torch.enable_grad():
                    vecs = vecs.requires_grad_(True)
                    if self.version.startswith(("v0.", "v1.0.", "v1.1.", "v1.2")):
                        energy, charges = self.forward(
                            vecs.to(model_dtype),
                            atomic_numbers_t,
                            atom_index1,
                            atom_index2,
                            ob,
                            ba,
                            be,
                            xa,
                        )
                    else:
                        energy, charges = self.forward(
                            vecs.to(model_dtype),
                            atomic_numbers_t,
                            atom_index1,
                            atom_index2,
                            ob,
                            ba,
                            be,
                            xa,
                            calc_mode_type=cm,
                        )
                    forces_raw = torch.autograd.grad(torch.sum(energy), vecs)[0]

                forces_zeros = torch.zeros_like(coordinates_t)
                forces = -forces_zeros.scatter_add(
                    0, atom_index1.view(n_edge, 1).expand(n_edge, 3), forces_raw
                ) + forces_zeros.scatter_add(
                    0, atom_index2.view(n_edge, 1).expand(n_edge, 3), forces_raw
                )

                virial = torch.zeros(
                    (len(atomic_numbers_arr), 6), dtype=coord_dtype, device=device
                )
                virial_raw = forces_raw[:, [0, 1, 2, 1, 2, 0]] * vecs[:, [0, 1, 2, 2, 0, 1]]
                virial = virial.scatter_add(
                    0, be.reshape((be.size()[0], 1)).expand((be.size()[0], 6)), virial_raw
                )

            shift_energy_t = torch.tensor(
                shift_energy_arr, dtype=torch.float64, device=energy.device
            )
            n_results_succeeded = len(n_neighbors_list)
            elapsed_calc = (time.perf_counter() - start_t) / n_results_succeeded
            elapsed_preprocess = elapsed_preprocess_total / n_results_succeeded

            results_succeeded = [
                {
                    "energy": float(e),
                    "charges": c,
                    "forces": f,
                    "virial": v,
                    "messages": m,
                    "calc_stats": {
                        "n_neighbors": n,
                        "elapsed_usec_preprocess": int(1e6 * elapsed_preprocess),
                        "elapsed_usec_infer": int(1e6 * elapsed_calc),
                    },
                }
                for e, c, f, v, m, n in zip(
                    (energy.flatten() + shift_energy_t).cpu().tolist(),
                    np.split(charges.detach().cpu().numpy(), n_atoms_arr[1:-1]),
                    np.split(forces.detach().cpu().numpy(), n_atoms_arr[1:-1]),
                    virial.detach().cpu().numpy(),
                    message_output_arr,
                    n_neighbors_list,
                )
            ]

        results: List[
            Union[Dict[str, Union[np.ndarray, float, Dict[str, int], List[MessageEnum]]], PFPError]
        ] = [results_succeeded[x] if isinstance(x, int) else x for x in results_valid_arr]

        return results
