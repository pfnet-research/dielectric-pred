import json
from importlib import import_module
from pathlib import Path

import yaml

from pfp.nn.models.molecule.estimator import EnergyEstimator


def build_estimator(
    device: int = 0, path_lib: str = "pfp.nn.models.molecule.lib.libinfer"
) -> EnergyEstimator:
    """ """
    abs_path = str(Path(__file__).parent)
    # load setting
    model_settings = yaml.load(
        open(abs_path + "/lib/model_settings.yaml", "r"), Loader=yaml.FullLoader
    )
    version = model_settings.get("version", "v0.0.0")
    print(f"Loading PFP {version}.")

    cutoff = model_settings["preprocess_settings"]["cutoff"]
    element_energy = json.load(open(abs_path + "/lib/" + model_settings["shift_energies"], "r"))
    element_energy_original = json.load(
        open(abs_path + "/lib/" + model_settings["shift_energies_original"], "r")
    )
    for k, v in element_energy_original.items():
        element_energy.setdefault(k, 0.0)
        element_energy[k] -= v

    if device < 0:
        raise ValueError("This model should be used with gpu.")
    model = import_module(path_lib).l(device)  # type: ignore
    estimator = EnergyEstimator(model, cutoff=cutoff, element_energy=element_energy)
    return estimator
