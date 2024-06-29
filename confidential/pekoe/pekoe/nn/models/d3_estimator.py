import time
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import dacite
import numpy as np
import torch
from ase.units import Bohr
from torch_dftd.dftd3_xc_params import get_dftd3_default_params
from torch_dftd.functions.triplets_kernel import _calc_triplets_core_gpu_kernel
from torch_dftd.nn.base_dftd_module import BaseDFTDModule
from torch_dftd.nn.dftd2_module import DFTD2Module
from torch_dftd.nn.dftd3_module import DFTD3Module

from pekoe.nn.estimator_base import (
    BaseEstimator,
    CellTooSmallError,
    EstimatorCalcMode,
    EstimatorKind,
    ModeInvalidError,
    PFPError,
)
from pekoe.nn.models.teanet_estimator import batch_estimate_preprocess, elements_status_check
from pfp.nn.estimator_base import EstimatorSystem
from pfp.utils.messages import MessageEnum


class DampingEnum(Enum):
    ZERO = "zero"
    BJ = "bj"
    ZEROM = "zerom"
    BJM = "bjm"
    DFTD2 = "dftd2"


class XCEnum(Enum):
    SLATER_DIRAC_EXCHANGE = "slater-dirac-exchange"
    B_LYP = "b-lyp"
    B_P = "b-p"
    B97_D = "b97-d"
    REVPBE = "revpbe"
    PBE = "pbe"
    PBESOL = "pbesol"
    RPW86_PBE = "rpw86-pbe"
    RPBE = "rpbe"
    TPSS = "tpss"
    B3_LYP = "b3-lyp"
    PBE0 = "pbe0"
    HSE06 = "hse06"
    REVPBE38 = "revpbe38"
    PW6B95 = "pw6b95"
    TPSS0 = "tpss0"
    B2_PLYP = "b2-plyp"
    PWPB95 = "pwpb95"
    B2GP_PLYP = "b2gp-plyp"
    PTPSS = "ptpss"
    HF = "hf"
    MPWLYP = "mpwlyp"
    BPBE = "bpbe"
    BH_LYP = "bh-lyp"
    TPSSH = "tpssh"
    PWB6K = "pwb6k"
    B1B95 = "b1b95"
    BOP = "bop"
    O_LYP = "o-lyp"
    O_PBE = "o-pbe"
    SSB = "ssb"
    REVSSB = "revssb"
    OTPSS = "otpss"
    B3PW91 = "b3pw91"
    REVPBE0 = "revpbe0"
    PBE38 = "pbe38"
    MPW1B95 = "mpw1b95"
    MPWB1K = "mpwb1k"
    BMK = "bmk"
    CAM_B3LYP = "cam-b3lyp"
    LC_WPBE = "lc-wpbe"
    M05 = "m05"
    M052X = "m052x"
    M06L = "m06l"
    M06 = "m06"
    M062X = "m062x"
    M06HF = "m06hf"
    HCTH120 = "hcth120"
    DSD_BLYP = "dsd-blyp"
    DSD_BLYP_FC = "dsd-blyp-fc"
    OPBE = "opbe"
    HF_MIXED = "hf/mixed"
    HF_SV = "hf/sv"
    HF_MINIS = "hf/minis"
    B3_LYP_6_31GD = "b3-lyp/6-31gd"
    DFTB3 = "dftb3"
    PW1PW = "pw1pw"
    PWGGA = "pwgga"
    HSESOL = "hsesol"
    HF3C = "hf3c"
    HF3CV = "hf3cv"
    PBEH3C = "pbeh3c"
    PBEH_3C = "pbeh-3c"


class CutoffSmoothingEnum(Enum):
    NONE = "none"
    POLY = "poly"


@dataclass
class D3EstimatorParameters:
    damping: DampingEnum = DampingEnum.ZERO
    xc: XCEnum = XCEnum.PBE
    old: bool = False
    abc: bool = False
    cutoff: float = 95.0 * Bohr
    cnthr: float = 40.0 * Bohr
    cutoff_smoothing: CutoffSmoothingEnum = CutoffSmoothingEnum.NONE
    bidirectional: bool = False
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if isinstance(self.damping, str):
            self.damping = DampingEnum(self.damping)
        if isinstance(self.xc, str):
            self.xc = XCEnum(self.xc)
        if isinstance(self.cutoff_smoothing, str):
            self.cutoff_smoothing = CutoffSmoothingEnum(self.cutoff_smoothing)

    @classmethod
    def from_dict(class_, d: Dict[Any, Any]) -> "D3EstimatorParameters":
        if "dtype" in d:
            dtype = d["dtype"]
            assert dtype in ["float16", "float32", "float64"]
            d["dtype"] = getattr(torch, dtype)
        return dacite.from_dict(
            data_class=class_,
            data=d,
            config=dacite.Config(cast=[DampingEnum, XCEnum, CutoffSmoothingEnum]),
        )


class D3Estimator(BaseEstimator):
    """
    Settings different from torch-dftd:
        `bidirectional=False`
        `dft` option was omitted.
    """

    implemented_properties = ["energy", "forces", "virial"]

    def __init__(
        self,
        model_parameters: Optional[D3EstimatorParameters] = None,
        # pekoe settings
        max_neighbors: Optional[int] = None,
        max_atoms: Optional[int] = None,
        version: Optional[str] = None,
    ):
        super(D3Estimator, self).__init__()
        self.model_parameters = (
            model_parameters if model_parameters is not None else D3EstimatorParameters()
        )
        self.params = get_dftd3_default_params(
            damping=self.model_parameters.damping.value,
            xc=self.model_parameters.xc.value,
            old=self.model_parameters.old,
        )
        self.cutoff = self.model_parameters.cutoff
        if self.model_parameters.old:
            self.dftd_module: BaseDFTDModule = DFTD2Module(
                self.params,
                cutoff=self.model_parameters.cutoff,
                dtype=self.model_parameters.dtype,
                bidirectional=self.model_parameters.bidirectional,
                cutoff_smoothing=self.model_parameters.cutoff_smoothing,
            )
        else:
            self.dftd_module = DFTD3Module(
                self.params,
                cutoff=self.model_parameters.cutoff,
                cnthr=self.model_parameters.cnthr,
                abc=self.model_parameters.abc,
                dtype=self.model_parameters.dtype,
                bidirectional=self.model_parameters.bidirectional,
                cutoff_smoothing=self.model_parameters.cutoff_smoothing,
            )

        self.max_neighbors: Optional[int] = max_neighbors
        self.max_atoms: Optional[int] = max_atoms
        self.version = version if version else "UNDEFINED"
        self.available_calc_modes_value: List[EstimatorCalcMode] = [EstimatorCalcMode.CRYSTAL]

        self.device_str = "cpu"

    def to(self, device_str: str) -> None:
        if device_str == "auto":
            device_str = "cuda:0" if torch.cuda.is_available() else "cpu"

        device = torch.device(device_str)
        self.device_str = device_str
        self.dftd_module.to(device)

        if self.model_parameters.abc and device_str != "cpu":
            if _calc_triplets_core_gpu_kernel is None:
                warnings.warn(
                    "Dev warning: D3 with abc (three-body term) in GPU is implemented by CuPy.",
                    UserWarning,
                )

    def eval(self) -> None:
        self.dftd_module.eval()

    def estimate(
        self, args: EstimatorSystem
    ) -> Dict[str, Union[float, Dict[str, int], np.ndarray, List[MessageEnum]]]:
        results_arr = self.batch_estimate([args])
        assert len(results_arr) == 1
        results = results_arr[0]
        if isinstance(results, PFPError):
            raise results
        return results

    def batch_estimate(
        self, args_list: List[EstimatorSystem]
    ) -> List[
        Union[Dict[str, Union[np.ndarray, float, Dict[str, int], List[MessageEnum]]], PFPError]
    ]:
        """ """
        device = torch.device(self.device_str)
        atomic_numbers_arr = []
        coordinates_t_arr = []
        cell_arr = []
        cell_volume_arr = []
        pbc_arr = []
        edge_index_arr = []
        sh_arr = []
        ba_arr = []
        be_arr = []
        n_atoms_sum = 0
        n_atoms_arr = [0]
        message_output_arr: List[List[MessageEnum]] = []
        n_neighbors_list = []
        batch_calculate_force = False

        results_valid_arr: List[Union[int, PFPError]] = list()
        n_valid = 0
        total_n_atoms = 0
        total_n_neighbors = 0
        elapsed_d3_preprocess_total = 0.0
        for args in args_list:
            try:
                args.format()

                assert args.atomic_numbers is not None
                atomic_numbers = args.atomic_numbers.astype(np.int64)
                coordinates = args.coordinates
                cell = args.cell
                cell_volume = np.abs(np.linalg.det(cell))
                if cell_volume < 1.0e-8:
                    raise CellTooSmallError
                pbc = args.pbc
                if pbc is None:
                    pbc = np.array([0, 0, 0], dtype=np.int32)

                elements_condition = self.element_status(atomic_numbers)
                message_output = elements_status_check(self, elements_condition)

                atomic_numbers_t = torch.tensor(atomic_numbers, dtype=torch.int64, device=device)
                # Make illegal element work.
                # Should be altered by torch.minimum after torch version is updated
                atomic_numbers_t[atomic_numbers_t > 94] = 94

                coordinates_t = torch.tensor(
                    coordinates, dtype=self.model_parameters.dtype, device=device
                )
                cell_t: torch.Tensor = torch.tensor(
                    cell, dtype=self.model_parameters.dtype, device=device
                )
                pbc_t: torch.Tensor = torch.tensor(pbc, dtype=torch.int64, device=device)

                start_t = time.perf_counter()
                batch_estimate_preprocess_results = batch_estimate_preprocess(
                    atomic_numbers_t,
                    coordinates_t,
                    cell_t,
                    pbc_t,
                    self.model_parameters.cutoff,
                    soft_max_atoms_single=args.input_max_atoms,
                    soft_max_neighbors_single=args.input_max_neighbors,
                    hard_max_atoms_single=self.max_atoms,
                    hard_max_neighbors_single=self.max_neighbors,
                    max_atoms_batch=self.max_atoms,
                    max_neighbors_batch=self.max_neighbors,
                    current_n_atoms=total_n_atoms,
                    current_n_neighbors=total_n_neighbors,
                    estimator_kind=EstimatorKind.D3,
                )
                elapsed_d3_preprocess_total += time.perf_counter() - start_t

                (
                    n_atoms,
                    n_neighbors,
                    a1,
                    a2,
                    sh_int,
                    coordinates_t,
                ) = batch_estimate_preprocess_results
            except PFPError as e:
                results_valid_arr.append(e)
                continue

            total_n_atoms += n_atoms
            total_n_neighbors += n_neighbors
            n_neighbors_list.append(n_neighbors)

            atomic_numbers_arr.append(atomic_numbers_t)
            coordinates_t_arr.append(coordinates_t)
            cell_arr.append(cell_t)
            cell_volume_arr.append(cell_volume)
            pbc_arr.append(pbc_t)
            ba_arr.append(
                n_valid * torch.ones((coordinates_t.size()[0],), dtype=torch.int64, device=device)
            )
            sh = torch.mm(sh_int.to(self.model_parameters.dtype), cell_t)
            if self.model_parameters.bidirectional:
                edge_index_arr.append(
                    torch.stack([torch.cat([a1, a2]), torch.cat([a2, a1])], dim=0) + n_atoms_sum
                )
                sh_arr.append(torch.cat([sh, -sh]).to(self.model_parameters.dtype))
                be_arr.append(
                    n_valid * torch.ones((2 * sh.size()[0],), dtype=torch.int64, device=device)
                )
            else:
                edge_index_arr.append(torch.stack([a1, a2], dim=0) + n_atoms_sum)
                sh_arr.append(sh)
                be_arr.append(
                    n_valid * torch.ones((sh.size()[0],), dtype=torch.int64, device=device)
                )
            n_atoms_sum += atomic_numbers.shape[0]
            n_atoms_arr.append(n_atoms_sum)
            results_valid_arr.append(n_valid)
            if "forces" in args.properties or "virial" in args.properties:
                batch_calculate_force = True
            n_valid += 1

            message_output_arr.append(message_output)

        if len(coordinates_t_arr) == 0:
            results_succeeded: List[
                Dict[str, Union[np.ndarray, float, Dict[str, int], List[MessageEnum]]]
            ] = list()
        else:
            coordinates_t = torch.cat(coordinates_t_arr, dim=0).requires_grad_(True)
            atomic_numbers_t = torch.cat(atomic_numbers_arr, dim=0)
            cell_t = torch.stack(cell_arr).requires_grad_(True)
            pbc_t = torch.stack(pbc_arr)
            shift = torch.cat(sh_arr, dim=0)
            edge_index = torch.cat(edge_index_arr, dim=1)
            batch = None if n_valid == 1 else torch.cat(ba_arr, dim=0)
            batch_edge = None if n_valid == 1 else torch.cat(be_arr, dim=0)

            if self.max_atoms is not None:
                assert atomic_numbers_t.size()[0] <= self.max_atoms
            if self.max_neighbors is not None:
                assert shift.size()[0] <= self.max_neighbors

            batch_dicts = dict(
                Z=atomic_numbers_t,  # (n_nodes,)
                pos=coordinates_t,  # (n_nodes,)
                cell=cell_t,  # (bs, 3, 3)
                pbc=pbc_t,  # (bs, 3)
                shift_pos=shift,  # (n_nodes, 3)
                edge_index=edge_index,  # (2, n_edges)
                batch=batch,  # (n_nodes)
                batch_edge=batch_edge,  # (n_edges)
            )

            start_t = time.perf_counter()
            if batch_calculate_force:
                results_succeeded = self.dftd_module.calc_energy_and_forces(
                    **batch_dicts, damping=self.model_parameters.damping.value
                )
                for result, cell_volume in zip(results_succeeded, cell_volume_arr):
                    result["virial"] = result["stress"] * cell_volume
                    del result["stress"]

            else:
                results_succeeded = self.dftd_module.calc_energy(
                    **batch_dicts, damping=self.model_parameters.damping.value
                )
            n_results_succeeded = len(results_succeeded)
            elapsed_d3_infer = (time.perf_counter() - start_t) / n_results_succeeded
            elapsed_d3_preprocess = elapsed_d3_preprocess_total / n_results_succeeded

            for result, message, n_neighbors in zip(
                results_succeeded, message_output_arr, n_neighbors_list
            ):
                result["messages"] = message
                result["calc_stats"] = {
                    "n_neighbors": n_neighbors,
                    # average elapsed time of preprocessing
                    "elapsed_usec_d3_preprocess": int(1e6 * elapsed_d3_preprocess),
                    # average elapsed time for infering by batch estimate
                    "elapsed_usec_d3_infer": int(1e6 * elapsed_d3_infer),
                }

        results: List[
            Union[Dict[str, Union[np.ndarray, float, Dict[str, int], List[MessageEnum]]], PFPError]
        ] = [results_succeeded[x] if isinstance(x, int) else x for x in results_valid_arr]

        return results

    def set_book_keeping(
        self, use_book_keeping: bool = True, book_keeping_skin: float = 2.0
    ) -> None:
        raise NotImplementedError

    def reset_book_keeping(self) -> None:
        raise NotImplementedError

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
