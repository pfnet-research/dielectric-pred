from types import TracebackType
from typing import Dict, List, Optional, Sequence, Set, Type

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from pfp.nn.estimator_base import BaseEstimator, EstimatorSystem
from pfp.utils.messages import MessageEnum, message_info


class ASECalculator(Calculator):  # type: ignore
    """ """

    def __init__(self, estimator: BaseEstimator):
        """ """
        super(ASECalculator, self).__init__()
        self.estimator = estimator
        self.implemented_properties = self.convert_properties(estimator.implemented_properties)
        self.current_messages: Set[MessageEnum] = set()
        self.default_properties: Sequence[str] = [
            prop for prop in ["energy", "charges", "forces"] if prop in self.implemented_properties
        ]

    def __enter__(self) -> "ASECalculator":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        pass

    def set_default_properties(self, default_properties: Sequence[str]) -> None:
        """
        Specify the properties that are always calculated.
        """
        for prop in default_properties:
            assert prop in self.implemented_properties

        # make a copy to prevent issues with outside modifications of the parameter
        self.default_properties = list(default_properties)

    def convert_properties(self, properties: Sequence[str]) -> List[str]:
        def convert(s: str) -> str:
            if s == "virial":
                return "stress"
            else:
                return s

        return [convert(s) for s in properties]

    def reverse_convert_properties(self, properties: Sequence[str]) -> List[str]:
        def convert(s: str) -> str:
            if s == "stress":
                return "virial"
            else:
                return s

        return [convert(s) for s in properties]

    def calculate(
        self,
        atoms: Optional[Atoms] = None,
        properties: Optional[Sequence[str]] = None,
        system_changes: Sequence[str] = all_changes,
    ) -> None:
        """ """
        if properties is None:
            properties = ["energy"]

        super(ASECalculator, self).calculate(atoms, properties, system_changes)

        pbc = self.atoms.get_pbc().astype(np.uint8)

        _properties = list(properties)  # work on a copy of the passed value
        if len(self.default_properties) > 0:
            for prop in self.default_properties:
                if prop not in _properties:
                    _properties.append(prop)

        if (
            "forces" in _properties
            and np.all(pbc)
            and "stress" not in _properties
            and "stress" in self.implemented_properties
        ):
            # Automatically add stress when force is requested
            _properties.append("stress")

        cell = self.atoms.get_cell(complete=True)
        if not isinstance(cell, np.ndarray):
            cell = cell.array

        atomic_numbers = self.atoms.get_atomic_numbers()
        coordinates = self.atoms.get_positions()
        properties_estimator = self.reverse_convert_properties(_properties)

        self.results = self.estimator.estimate(
            EstimatorSystem(
                atomic_numbers=atomic_numbers,
                coordinates=coordinates,
                cell=cell,
                pbc=pbc,
                properties=properties_estimator,
            )
        )

        # NOTE (himkt): to make compatible with ``SinglePointCalculator``.
        for key, value in self.results.items():
            if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating):
                self.results[key] = value.astype(np.float64)

        if "energy" in self.results:
            self.results["free_energy"] = self.results["energy"]

        if "virial" in self.results and np.all(pbc):
            assert isinstance(self.results["virial"], np.ndarray)
            self.results["stress"] = self.calculate_stress(
                self.results["virial"],
                self.atoms.get_volume(),
            )
            del self.results["virial"]

        if "messages" in self.results:
            assert isinstance(self.results["messages"], list)
            for message in self.results["messages"]:
                print(message)
                self.current_messages.add(message)
            del self.results["messages"]

    @staticmethod
    def calculate_stress(virial: np.ndarray, atom_volume: np.ndarray) -> np.ndarray:
        """ """
        stress: np.ndarray = virial / atom_volume
        return stress

    def pop_messages(self) -> Dict[int, str]:
        res = {int(message): message_info(message)[1] for message in self.current_messages}
        self.current_messages = set()
        return res
