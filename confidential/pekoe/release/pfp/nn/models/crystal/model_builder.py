import json
from importlib import import_module
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml

from pfp.nn.models.crystal.estimator import EnergyEstimator, EstimatorCalcMode


def build_estimator(
    device: int = 0,
    path_lib: str = "pfp.nn.models.crystal.lib.libinfer",
    available_calc_modes: Optional[List[EstimatorCalcMode]] = None,
    max_neighbors: Optional[int] = None,
    max_atoms: Optional[int] = None,
) -> EnergyEstimator:
    """ """
    abs_path = str(Path(__file__).parent)
    # load setting
    model_settings = yaml.load(
        open(abs_path + "/lib/model_settings.yaml", "r"),
        Loader=yaml.FullLoader,
    )
    version = model_settings.get("version", "v0.0.0")
    print(f"Loading PFP {version}.")

    cutoff = model_settings["preprocess_settings"]["cutoff"]
    element_energy = json.load(open(abs_path + "/lib/" + model_settings["shift_energies"], "r"))

    if device < 0:
        raise ValueError("This model should be used with gpu.")
    model = import_module(path_lib).l(device)  # type:ignore
    estimator = EnergyEstimator(
        model,
        cutoff=cutoff,
        element_energy=element_energy,
        available_calc_modes=(
            available_calc_modes if available_calc_modes else [e for e in EstimatorCalcMode]
        ),
        max_neighbors=max_neighbors,
        max_atoms=max_atoms,
        version=version,
    )
    estimator.element_supported_np = np.array(model_settings["elements_supported"], dtype=np.int32)
    return estimator
