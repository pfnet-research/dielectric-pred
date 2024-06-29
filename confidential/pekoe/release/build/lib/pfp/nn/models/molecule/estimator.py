from typing import Any, Dict, List, Optional, Union

import numpy as np

from pfp.nn.estimator_base import BaseEstimator, EstimatorSystem
from pfp.nn.models.molecule.preprocessor import preprocess_pbc
from pfp.utils.messages import MessageEnum


class EnergyEstimator(BaseEstimator):
    """ """

    implemented_properties = ["energy", "forces", "virial"]

    def __init__(
        self,
        model: Any,
        cutoff: float,
        element_energy: Optional[Dict[str, float]] = None,
    ):
        """ """
        super(EnergyEstimator, self).__init__()
        self.model = model
        self.cutoff = cutoff

        element_energy_arr = [0.0 for _ in range(120)]
        if element_energy is not None:
            for k, v in element_energy.items():
                element_energy_arr[int(k)] = float(v)
        self.element_energy = np.array(element_energy_arr, dtype=np.float32)

    def estimate(
        self, args: EstimatorSystem
    ) -> Dict[str, Union[float, Dict[str, int], np.ndarray, List[MessageEnum]]]:
        """ """
        pbc = args.pbc
        args.format()
        assert args.atomic_numbers is not None
        if pbc is None:
            pbc = np.array([0, 0, 0], dtype=np.int32)
        if np.all(pbc == 0):
            cell = None
        elif np.all(pbc == 1):
            cell = args.cell
        coordinates = args.coordinates
        atomic_numbers = args.atomic_numbers
        a1, a2, sh = preprocess_pbc(atomic_numbers, coordinates, cell, pbc, self.cutoff)
        a1 = a1.astype(np.int64)
        a2 = a2.astype(np.int64)
        sh = sh.astype(np.float32)

        energies, forces, virial = self.model.r(coordinates, atomic_numbers, a1, a2, sh)

        shift_energy = np.sum(self.element_energy[atomic_numbers])

        results = {
            "energy": float(energies + shift_energy),
            "forces": forces,
            "virial": virial,
        }
        assert isinstance(results["energy"], float)
        return results
