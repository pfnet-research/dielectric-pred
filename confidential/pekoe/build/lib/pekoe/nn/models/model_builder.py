import pathlib
import warnings
from pickle import UnpicklingError
from typing import Dict, Optional, Union

import numpy as np
import torch

from pekoe.nn.estimator_base import BaseEstimator, EstimatorCalcMode, ModelUnknownError
from pekoe.nn.models import DEFAULT_MODEL, DEFAULT_MODEL_DIRECTORY
from pekoe.nn.models.config import BaseConfig
from pekoe.nn.models.teanet import (
    TeaNet_v1,
    TeaNet_v1_1,
    TeaNet_v1_2,
    TeaNet_v1_3,
    TeaNetNodeFeatureMLP_v1_3,
    TeaNetNodeFeatureMLPParameters_v1_3,
    TeaNetParameters_v1,
    TeaNetParameters_v1_1,
    TeaNetParameters_v1_2,
    TeaNetParameters_v1_3,
)
from pekoe.nn.models.teanet.codegen_options import CodeGenOptions
from pekoe.nn.models.teanet.teanet_base import TeaNetBase
from pekoe.nn.models.teanet_estimator import TeaNetEstimator
from pfp.nn.models.crystal.estimator import EnergyEstimator
from pfp.nn.models.crystal.model_builder import build_estimator as pfp_build_estimator

_model_strings: Dict[str, pathlib.Path] = {
    "v0.10.0": DEFAULT_MODEL_DIRECTORY / "model_v0_10_0.yaml",
    "v1.0.0": DEFAULT_MODEL_DIRECTORY / "model_v1_0_0.yaml",
    "v1.1.0": DEFAULT_MODEL_DIRECTORY / "model_v1_1_0.yaml",
    "v1.2.0": DEFAULT_MODEL_DIRECTORY / "model_v1_2_0.yaml",
    "v1.2.1": DEFAULT_MODEL_DIRECTORY / "model_v1_2_1.yaml",
    "v1.2.2": DEFAULT_MODEL_DIRECTORY / "model_v1_2_2.yaml",
    "v1.3.0": DEFAULT_MODEL_DIRECTORY / "model_v1_3_0.yaml",
    "v1.3.1": DEFAULT_MODEL_DIRECTORY / "model_v1_3_1.yaml",
    "v1.4.0": DEFAULT_MODEL_DIRECTORY / "model_v1_4_0.yaml",
    "v1.4.1": DEFAULT_MODEL,
    "default": DEFAULT_MODEL,
    "d3_pbe": DEFAULT_MODEL_DIRECTORY / "model_d3_pbe.yaml",
    "v1.3.1+ccsd(t)_correction_0.0.1": DEFAULT_MODEL_DIRECTORY
    / "model_v1_3_1_ccsd_t_correction_0_0_1.yaml",
    "v1.3.1+ccsd(t)_correction_0.0.2": DEFAULT_MODEL_DIRECTORY
    / "model_v1_3_1_ccsd_t_correction_0_0_2.yaml",
    "v1.3.1+ccsd(t)_correction_0.0.3": DEFAULT_MODEL_DIRECTORY
    / "model_v1_3_1_ccsd_t_correction_0_0_3.yaml",
    "v1.3.1+ccsd(t)_correction_0.0.4": DEFAULT_MODEL_DIRECTORY
    / "model_v1_3_1_ccsd_t_correction_0_0_4.yaml",
}


def build_estimator(
    version_or_config_path: Union[pathlib.Path, str],
    device: str = "auto",
    max_neighbors: Optional[int] = None,
    max_atoms: Optional[int] = None,
    output_onnx: Optional[str] = None,
    codegen_options: Optional[CodeGenOptions] = None,
) -> BaseEstimator:
    """ """

    if isinstance(version_or_config_path, str):
        if version_or_config_path not in _model_strings:
            raise ModelUnknownError
        version_or_config_path = _model_strings[version_or_config_path]

    base_config = BaseConfig.from_yaml(version_or_config_path)
    if base_config.arch in [
        "teanet_v1",
        "teanet_v1_1",
        "teanet_v1_2",
        "teanet_v1_3",
        "teanet_v1_3_node_feature_mlp",
    ]:
        # TODO(takamoto) should be polished.
        if base_config.arch == "teanet_v1":
            model: TeaNetBase = TeaNet_v1(TeaNetParameters_v1.from_dict(base_config.parameters))
        elif base_config.arch == "teanet_v1_1":
            model = TeaNet_v1_1(TeaNetParameters_v1_1.from_dict(base_config.parameters))
        elif base_config.arch == "teanet_v1_2":
            model = TeaNet_v1_2(TeaNetParameters_v1_2.from_dict(base_config.parameters))
        elif base_config.arch == "teanet_v1_3":
            model = TeaNet_v1_3(TeaNetParameters_v1_3.from_dict(base_config.parameters))
        elif base_config.arch == "teanet_v1_3_node_feature_mlp":
            model = TeaNetNodeFeatureMLP_v1_3(
                TeaNetNodeFeatureMLPParameters_v1_3.from_dict(base_config.parameters)
            )
            if device.startswith("pfvm:"):
                device = device[len("pfvm:") :]
                warnings.warn(
                    "TeaNetNodeFeatureMLPParameters_v1_3 does not support pfvm device, "
                    f"fall back to {device}"
                )
        try:
            state = torch.load(str(base_config.weights_path), map_location="cpu")
        except UnpicklingError:
            print(
                f"""
UnpicklingError has been detected while loading model parameter file: {base_config.weights_path}
If you ran with default settings, possible cause is that you couldn't pull git lfs correctly.
Please check the model file size shown above. It should be ~MB order size.
"""
            )
            raise
        model.load_state_dict(state)

        if device.startswith(("pfvm:", "mncore:", "mncore2:", "torch:", "emu:", "emu2:")):
            from pekoe.nn.models.teanet.codegen_teanet import CodeGenTeaNet

            if codegen_options is None:
                codegen_options = CodeGenOptions()

            if (
                device.startswith("pfvm:")
                and not codegen_options.mncore_options.allow_missing_teanet_conv_helper
            ):
                # NOTE: Since CodeGenTeaNet w/o teanet_conv_helper is not
                # supported, raise an ImportError here.
                import teanet_conv_helper as _  # NOQA F401

            if codegen_options.outdir is None:
                codegen_options.outdir = pathlib.Path("/tmp/codegen_teanet")
            if device.startswith(("mncore:", "mncore2:", "emu:", "emu2:")):
                codegen_options.mncore_options.pad_edge = True
            config_name = base_config.version.replace(".", "_") + "-" + base_config.checksum
            model = CodeGenTeaNet(config_name, model, device, codegen_options)

        teanet_estimator: TeaNetEstimator = TeaNetEstimator(
            model,
            calc_mode=EstimatorCalcMode(base_config.default_calc_mode),
            max_neighbors=max_neighbors,
            max_atoms=max_atoms,
            available_calc_modes=[EstimatorCalcMode(s) for s in base_config.calc_modes],
            output_onnx=output_onnx,
            version=base_config.version,
        )
        teanet_estimator.to(device)
        teanet_estimator.eval()
        estimator: BaseEstimator = teanet_estimator
    elif base_config.arch == "pfp_binary":
        path_lib = base_config.parameters["path_lib"]
        device_id = 0
        if device == "auto":
            pass
        elif device.startswith("cuda:"):
            device_id = int(device[5:])
        else:
            raise ValueError("Unknown device for pfp_binary")
        assert output_onnx is None
        pfp_estimator: EnergyEstimator = pfp_build_estimator(
            device=device_id,
            path_lib=path_lib,
            available_calc_modes=[EstimatorCalcMode(s) for s in base_config.calc_modes],
            max_neighbors=max_neighbors,
            max_atoms=max_atoms,
        )
        pfp_estimator.set_calc_mode(EstimatorCalcMode(base_config.default_calc_mode))
        estimator = pfp_estimator
    elif base_config.arch == "d3":
        if device.startswith("pfvm:"):
            device = device[len("pfvm:") :]
            warnings.warn(f"D3Estimator do not support pfvm device, fall back to {device}")

        from pekoe.nn.models.d3_estimator import D3Estimator, D3EstimatorParameters

        d3_estimator = D3Estimator(
            model_parameters=D3EstimatorParameters.from_dict(base_config.parameters),
            version=base_config.version,
            max_neighbors=max_neighbors,
            max_atoms=max_atoms,
        )
        d3_estimator.to(device)
        d3_estimator.eval()
        estimator = d3_estimator
    else:
        raise ValueError(base_config.arch)

    assert len(base_config.elements_supported) == 128
    estimator.element_supported_np = np.array(base_config.elements_supported, dtype=np.int32)

    return estimator
