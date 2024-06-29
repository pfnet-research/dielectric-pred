from typing import Any, Type

import pytest
import torch
from torch_dftd.nn.dftd3_module import DFTD3Module

from pekoe.nn.estimator_base import EstimatorCalcMode
from pekoe.nn.models.model_builder import build_estimator
from pekoe.nn.models.teanet import TeaNetNodeFeatureMLP_v1_3
from pekoe.nn.models.teanet.model_pfp_v1.teanet import TeaNet_v1
from pekoe.nn.models.teanet.model_pfp_v1_1.teanet import TeaNet_v1_1
from pekoe.nn.models.teanet.model_pfp_v1_2.teanet import TeaNet_v1_2
from pekoe.nn.models.teanet.model_pfp_v1_3.teanet import TeaNet_v1_3


def _assert_model_and_device(model: Any, model_class: Type, device: str) -> None:
    is_pfvm = device.startswith("pfvm:")
    if is_pfvm:
        from pekoe.nn.models.teanet.codegen_teanet import CodeGenTeaNet

        if model_class in [TeaNet_v1, TeaNet_v1_1, TeaNet_v1_2, TeaNet_v1_3]:
            pfvm_model_class = CodeGenTeaNet
        elif model_class == DFTD3Module:
            raise ValueError(f"{model_class} is not supported by this function")
        else:
            raise ValueError(f"{model_class} is unknown")
        assert isinstance(model, pfvm_model_class)
        org_device = device[len("pfvm:") :]
        assert model.device == torch.device(org_device)
    else:
        assert isinstance(model, model_class)
        if model_class == DFTD3Module:
            assert model.device_str == torch.device(device)
        else:
            assert model.device == torch.device(device)


def test_build_estimator(device_config):
    device, codegen_options = device_config
    teanet_estimator = build_estimator("default", device, codegen_options=codegen_options)
    _assert_model_and_device(teanet_estimator.model, TeaNet_v1_3, device)
    del teanet_estimator

    teanet_estimator_v1 = build_estimator("v1.0.0", device, codegen_options=codegen_options)
    _assert_model_and_device(teanet_estimator_v1.model, TeaNet_v1, device)
    del teanet_estimator_v1

    teanet_estimator_v1_1 = build_estimator("v1.1.0", device, codegen_options=codegen_options)
    _assert_model_and_device(teanet_estimator_v1_1.model, TeaNet_v1_1, device)
    del teanet_estimator_v1_1

    teanet_estimator_v1_2 = build_estimator("v1.2.0", device, codegen_options=codegen_options)
    _assert_model_and_device(teanet_estimator_v1_2.model, TeaNet_v1_2, device)
    del teanet_estimator_v1_2

    teanet_estimator_v1_3 = build_estimator("v1.3.0", device, codegen_options=codegen_options)
    _assert_model_and_device(teanet_estimator_v1_3.model, TeaNet_v1_3, device)
    del teanet_estimator_v1_3

    # PFP v1.4 series used TeaNet model v1.3
    teanet_estimator_v1_4 = build_estimator("v1.4.0", device, codegen_options=codegen_options)
    _assert_model_and_device(teanet_estimator_v1_4.model, TeaNet_v1_3, device)
    del teanet_estimator_v1_4

    if device.startswith("pfvm:"):
        pytest.skip("pfvm device does not support d3 estimator")

    d3_estimator = build_estimator("d3_pbe", device)
    assert isinstance(d3_estimator.dftd_module, DFTD3Module)
    assert torch.device(d3_estimator.device_str) == torch.device(device)
    del d3_estimator

    for correction_model_version in ["0.0.1", "0.0.2", "0.0.3", "0.0.4"]:
        pfp_correction_estimator_v1_3_1 = build_estimator(
            f"v1.3.1+ccsd(t)_correction_{correction_model_version}", device
        )
        _assert_model_and_device(
            pfp_correction_estimator_v1_3_1.model, TeaNetNodeFeatureMLP_v1_3, device
        )
        del pfp_correction_estimator_v1_3_1


@pytest.mark.no_tenet_conv_helper
def test_build_estimator_with_pfvm(gpuid):
    device = f"pfvm:cuda:{gpuid}"
    with pytest.raises(ImportError):
        build_estimator("default", device)
    with pytest.raises(ImportError):
        build_estimator("v1.0.0", device)
    with pytest.raises(ImportError):
        build_estimator("v1.1.0", device)
    with pytest.raises(ImportError):
        build_estimator("v1.2.0", device)
    with pytest.raises(ImportError):
        build_estimator("v1.3.0", device)
    with pytest.raises(ImportError):
        build_estimator("v1.4.0", device)

    with pytest.warns(UserWarning):
        d3_estimator = build_estimator("d3_pbe", device)
        assert torch.device(d3_estimator.device_str) == torch.device(f"cuda:{gpuid}")

    for correction_model_version in ["0.0.1", "0.0.2", "0.0.3", "0.0.4"]:
        with pytest.warns(UserWarning):
            pfp_correction_estimator_v1_3_1 = build_estimator(
                f"v1.3.1+ccsd(t)_correction_{correction_model_version}", device
            )
            assert pfp_correction_estimator_v1_3_1.model.device == torch.device(f"cuda:{gpuid}")


def test_default_calc_mode(device_config):
    device, codegen_options = device_config
    teanet_estimator_v1_4_0 = build_estimator("v1.4.0", device, codegen_options=codegen_options)
    assert teanet_estimator_v1_4_0.calc_mode == EstimatorCalcMode.CRYSTAL
    del teanet_estimator_v1_4_0

    teanet_estimator_v1_4_0 = build_estimator("v1.4.1", device, codegen_options=codegen_options)
    assert teanet_estimator_v1_4_0.calc_mode == EstimatorCalcMode.CRYSTAL_U0
    del teanet_estimator_v1_4_0
