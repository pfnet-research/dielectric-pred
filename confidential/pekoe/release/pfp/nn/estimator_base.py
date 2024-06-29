from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union

import numpy as np

from pfp.utils.atoms import (
    ElementStatusEnum,
    InvalidAtomicSymbolError,
    atomic_number,
    max_atomic_number,
)
from pfp.utils.errors import PFPError
from pfp.utils.messages import MessageEnum


class EstimatorCalcMode(Enum):
    CRYSTAL = "crystal"         # GGA+U
    CRYSTAL_U0 = "crystal_u0"   # GGA  see MP for their details
    MOLECULE = "molecule"
    VASP = "crystal"  # alias for CRYSTAL
    GAUSSIAN = "molecule"  # alias for MOLECULE
    OC20 = "oc20"


@dataclass(init=False)
class EstimatorSystem:
    """
    Input argument of `Estimator.estimate` function.
    Parameters
    --------
    properties: List[str]
        What properties should be calculated. Current implemented properties are "energy", \
        "forces", "charges", and "virial". If gradient-based parameters (forces, virial) are not \
        requested, the estimator does not calculate them.
    coordinates: np.ndarray (dtype: Union[np.float32, np.float64])
        Coordination of atoms. The shape is `(n_atoms, 3)`.
    cell: np.ndarray (dtype: Union[np.float32, np.float64])
        Simulation cell of the system. The shape is `(3, 3)`
    species: Optional[np.ndarray] (dtype="<U3")
        Element list of atoms as string. Either species or atomic_numbers should be provided. \
        The shape is `(n_atoms, )`
    atomic_numbers: Optional[np.ndarray] (dtype=np.uint8)
        Element list of atoms as atomic number. Either species or atomic_numbers should be \
        provided. The shape is `(n_atoms, )`
    pbc: Optional[np.ndarray] (dtype=np.uint8)
        Whether periodic boundary conditions are applied or not. Currently no pbc mode \
        `([0, 0, 0])` and all pbc mode `([1, 1, 1])` are supported.
    calc_mode: Optional[EstimatorCalcMode]
        Calculate results with given calculation mode.
    input_max_atoms: Optional[int]
        Maximum number of atoms.
    input_max_neighbors: Optional[int]
        Maximum number of neighbors of the system.
    """

    properties: List[str]
    coordinates: np.ndarray
    cell: np.ndarray
    species: Optional[np.ndarray] = None
    atomic_numbers: Optional[np.ndarray] = None
    pbc: Optional[np.ndarray] = None
    calc_mode: Optional[EstimatorCalcMode] = None
    input_max_atoms: Optional[int] = None
    input_max_neighbors: Optional[int] = None

    def __init__(
        self,
        properties: List[str],
        coordinates: np.ndarray,
        cell: np.ndarray,
        species: Optional[np.ndarray] = None,
        atomic_numbers: Optional[np.ndarray] = None,
        pbc: Optional[np.ndarray] = None,
        calc_mode: Optional[EstimatorCalcMode] = None,
        input_max_atoms: Optional[int] = None,
        input_max_neighbors: Optional[int] = None,
    ) -> None:
        self.properties = properties
        if coordinates.dtype.type is np.float64:
            float_dtype = np.dtype(np.float64)
        else:
            float_dtype = np.dtype(np.float32)
        self.coordinates = coordinates.astype(float_dtype)
        self.cell = cell.astype(float_dtype)
        if isinstance(pbc, np.ndarray):
            pbc = pbc.astype(np.uint8)
        if isinstance(species, np.ndarray):
            species = species.astype("<U3")
        elif isinstance(atomic_numbers, np.ndarray):
            atomic_numbers = atomic_numbers.astype(np.uint8)
        self.pbc = pbc
        self.species = species
        self.atomic_numbers = atomic_numbers
        self.calc_mode = calc_mode
        self.input_max_atoms = input_max_atoms
        self.input_max_neighbors = input_max_neighbors

    def format(self) -> None:
        if self.atomic_numbers is not None:
            if self.atomic_numbers.shape[0] > 0 and (
                np.min(self.atomic_numbers) < 1
                or np.max(self.atomic_numbers) >= max_atomic_number()
            ):
                raise InputInvalidError(
                    f"The entry of atomic_numbers={self.atomic_numbers} "
                    f"should be between 1 and {max_atomic_number()}."
                )
        elif self.species is not None:
            try:
                self.atomic_numbers = np.fromiter(map(atomic_number, self.species), dtype=np.uint8)
            except InvalidAtomicSymbolError:
                raise InputInvalidError(
                    f"species={self.species} (chemical_symbols) contains invalid symbols."
                )
        else:
            raise InputNoElementInformationError(
                "Neither species (chemical_symbols) nor atomic_numbers is provided."
            )

        if self.coordinates.shape != (self.atomic_numbers.shape[0], 3):
            raise InputInvalidError(
                f"coordinates.shape={self.coordinates.shape} (positions) "
                f"is not {(self.atomic_numbers.shape[0], 3)}."
            )
        if not np.isfinite(self.coordinates).all():
            raise InputInvalidError("coordinates (positions) contains infinite values.")
        if not np.isfinite(self.cell).all():
            raise InputInvalidError("cell contains infinite values.")
        if self.cell.shape != (3, 3):
            raise InputInvalidError(f"cell.shape={self.cell.shape} is not (3, 3).")
        if self.pbc is not None:
            if self.pbc.shape != (3,):
                raise InputInvalidError(f"pbc.shape={self.pbc.shape} is not (3,).")
            if not np.all(np.isin(self.pbc, [0, 1])):
                raise InputInvalidError(f"pbc={self.pbc} are not 0 or 1.")
            if not np.all(self.pbc == 0):
                cell_size = np.abs(np.linalg.det(self.cell))
                if cell_size < 1.0e-7:
                    raise InputInvalidError(f"cell_size={cell_size} is too small (<1.0e-7).")


class BaseEstimator(object):
    """ """

    implemented_properties: List[str] = []
    element_supported_np: np.ndarray = 2 * np.ones((128,), dtype=np.int32)

    def __init__(self) -> None:
        super(BaseEstimator, self).__init__()
        self.message_isactive: Dict[MessageEnum, bool] = {m: True for m in MessageEnum}

    def _append_message_if_active(self, message: MessageEnum, messages: List[MessageEnum]) -> None:
        if self.message_isactive[message]:
            messages.append(message)

    def set_message_status(self, message: MessageEnum, message_enable: bool) -> None:
        self.message_isactive[message] = message_enable

    def estimate(
        self, args: EstimatorSystem
    ) -> Dict[str, Union[float, Dict[str, int], np.ndarray, List[MessageEnum]]]:
        """ """
        raise NotImplementedError

    def batch_estimate(
        self, args_list: List[EstimatorSystem]
    ) -> List[
        Union[Dict[str, Union[np.ndarray, float, Dict[str, int], List[MessageEnum]]], PFPError]
    ]:
        """ """
        raise NotImplementedError

    def set_book_keeping(
        self, use_book_keeping: bool = True, book_keeping_skin: float = 2.0
    ) -> None:
        """
        Set book-keeping feature.
        `use_book_keeping`: whether to use book-keeping feature
        `book_keeping_skin`: set book-keeping skin size (length scale)
        """
        raise NotImplementedError

    def set_calc_mode(self, calc_mode: EstimatorCalcMode) -> None:
        """
        Set estimator calculation mode.
        `calc_mode`: calculation mode (enum)
        """
        raise NotImplementedError

    def available_calc_modes(self) -> List[EstimatorCalcMode]:
        """
        Returns available calculation mode.
        """
        raise NotImplementedError

    def set_max_neighbors(self, max_neighbors: Optional[int]) -> None:
        """
        Set max number of neighbors.
        `max_neighbors`: Optional[int]
        """
        raise NotImplementedError

    def set_max_atoms(self, max_atoms: Optional[int]) -> None:
        """
        Set max number of atoms.
        `max_atoms`: Optional[int]
        """
        raise NotImplementedError

    def reset_book_keeping(self) -> None:
        """
        Reset internal cache used for book-keeping.
        This should be called before applying estimator to the new atomic system.
        """
        raise NotImplementedError

    def element_status(self, atomic_numbers: Union[np.ndarray, int]) -> ElementStatusEnum:
        max_atomic_num = self.element_supported_np.shape[0] - 1
        if isinstance(atomic_numbers, int):
            atomic_numbers_np = np.array([atomic_numbers], dtype=np.int32)
        else:
            atomic_numbers_np = atomic_numbers.astype(np.int32)
        if atomic_numbers_np.shape[0] == 0:
            return ElementStatusEnum.Expected
        if (
            np.min(atomic_numbers_np).item() < 1
            or max_atomic_num < np.max(atomic_numbers_np).item()
        ):
            return ElementStatusEnum.Illegal
        elements_status_int: int = np.max(self.element_supported_np[atomic_numbers_np]).item()

        return ElementStatusEnum(elements_status_int)

    def get_version(self) -> str:
        """
        Return version
        """
        raise NotImplementedError


class WithinSingleEstimationError(PFPError):
    """Errors related to single calculation"""

    pass


class AtomsFarAwayError(WithinSingleEstimationError):
    """Atoms are far away from the primitive cell.
    Please correct positions before calling estimate(),
    or turn off book keeping feature by `estimator.set_book_keeping(false)`
    """

    pass


class CellTooSmallError(WithinSingleEstimationError):
    """The cell shape is too small."""

    pass


class IllegalElementError(WithinSingleEstimationError):
    """Illegal atomic number was detected.
    Please check the input structure.
    """

    pass


class EstimatorKind(Enum):
    """
    Used to distinguish a D3 estimator and other kinds of estimators in error arguments
    """

    PFP = "PFP"
    D3 = "D3"


class AtomsTooManyError(WithinSingleEstimationError):
    """Too many atoms are detected in the system.
    Please check the input structure.
    """

    def __init__(
        self,
        n_atoms: int,
        max_atoms: int,
        estimator_kind: EstimatorKind = EstimatorKind.PFP,
        message: str = "",
    ):
        self.n_atoms = n_atoms
        self.max_atoms = max_atoms
        self.estimator_kind = estimator_kind
        if not message:
            message = (
                f"Too many atoms (input_atoms={n_atoms}, max_atoms={max_atoms},"
                f" estimator_kind={estimator_kind})"
            )
        super().__init__(message)


class AtomsHardLimitExceededError(WithinSingleEstimationError):
    """Atoms detected in the system exceeded hard limit.
    Please check the input structure.
    """

    def __init__(
        self,
        n_atoms: int,
        soft_max_atoms: int,
        hard_max_atoms: int,
        estimator_kind: EstimatorKind = EstimatorKind.PFP,
        message: str = "",
    ):
        self.n_atoms = n_atoms
        self.soft_max_atoms = soft_max_atoms
        self.hard_max_atoms = hard_max_atoms
        self.estimator_kind = estimator_kind
        if not message:
            message = (
                f"Atoms hard limit exceeded (input_atoms={n_atoms},"
                f" soft_max_atoms={soft_max_atoms}, hard_max_atoms={hard_max_atoms},"
                f" estimator_kind={estimator_kind})"
            )
        super().__init__(message)


class NeighborsTooManyError(WithinSingleEstimationError):
    """Too many neighbors are detected in the system.
    Please check the input structure.
    """

    def __init__(
        self,
        n_neighbors: int,
        max_neighbors: int,
        estimator_kind: EstimatorKind = EstimatorKind.PFP,
        message: str = "",
    ):
        self.n_neighbors = n_neighbors
        self.max_neighbors = max_neighbors
        self.estimator_kind = estimator_kind
        if not message:
            message = (
                f"Too many neighbors (input_neighbors={n_neighbors},"
                f" max_neighbors={max_neighbors},"
                f" estimator_kind={estimator_kind})"
            )
        super().__init__(message)


class NeighborsHardLimitExceededError(WithinSingleEstimationError):
    """Neighbors detected in the system exceeded hard limit.
    Please check the input structure.
    """

    def __init__(
        self,
        n_neighbors: int,
        soft_max_neighbors: int,
        hard_max_neighbors: int,
        estimator_kind: EstimatorKind = EstimatorKind.PFP,
        message: str = "",
    ):
        self.n_neighbors = n_neighbors
        self.soft_max_neighbors = soft_max_neighbors
        self.hard_max_neighbors = hard_max_neighbors
        self.estimator_kind = estimator_kind
        if not message:
            message = (
                f"Neighbors hard limit exceeded (input_neighbors={n_neighbors},"
                f" soft_max_neighbors={soft_max_neighbors}, hard_max_neighbors={hard_max_neighbors},"
                f" estimator_kind={estimator_kind})"
            )
        super().__init__(message)


def max_atoms_from_input(
    soft_max_atoms: Optional[int], hard_max_atoms: Optional[int]
) -> Optional[int]:
    return hard_max_atoms if soft_max_atoms is None else soft_max_atoms


def max_neighbors_from_input(
    soft_max_neighbors: Optional[int], hard_max_neighbors: Optional[int]
) -> Optional[int]:
    return hard_max_neighbors if soft_max_neighbors is None else soft_max_neighbors


class PBCNotAvailableError(WithinSingleEstimationError):
    """[Deprecated] Only all 0 or 1 can be available for pbc now."""

    pass


class WithinBatchEstimationError(PFPError):
    """Errors related to batch calculation limitation"""


class BatchAtomsTooManyError(WithinBatchEstimationError):
    """The total number of atoms for batch estimator exceeded the limit."""

    def __init__(
        self,
        n_atoms: int,
        max_atoms: int,
        estimator_kind: EstimatorKind = EstimatorKind.PFP,
        message: str = "",
    ):
        self.n_atoms = n_atoms
        self.max_atoms = max_atoms
        self.estimator_kind = estimator_kind
        if message == "":
            message = (
                f"Too many atoms (input_atoms={n_atoms}, max_atoms={max_atoms},"
                f" estimator_kind={estimator_kind})"
            )
        super().__init__(message)


class BatchNeighborsTooManyError(WithinBatchEstimationError):
    """The total number of neighbors for batch estimator exceeded the limit."""

    def __init__(
        self,
        n_neighbors: int,
        max_neighbors: int,
        estimator_kind: EstimatorKind = EstimatorKind.PFP,
        message: str = "",
    ):
        self.n_neighbors = n_neighbors
        self.max_neighbors = max_neighbors
        self.estimator_kind = estimator_kind
        if not message:
            message = (
                f"Too many neighbors (input_neighbors={n_neighbors}, max_neighbors={max_neighbors},"
                f" estimator_kind={estimator_kind})"
            )
        super().__init__(message)


class WithinInputError(PFPError):
    """Errors related to input values"""

    pass


class InputNoElementInformationError(WithinInputError):
    """No element information is provided.
    Please check the input structure.
    """

    pass


class InputInvalidError(WithinInputError):
    """Invalid input value is detected.
    Please check the input structure.
    """

    pass


class ModelUnknownError(PFPError):
    """Unknown model is specified."""

    pass


class ModeInvalidError(PFPError):
    """Invalid mode is specified."""

    pass
