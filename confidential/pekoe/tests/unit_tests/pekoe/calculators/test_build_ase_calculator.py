import pytest

from pekoe import build_ase_calculator as build_ase_calculator_from_root
from pekoe.calculators import build_ase_calculator
from pekoe.calculators.ase_calculator import ASECalculator


@pytest.fixture(
    params=[
        pytest.param("cpu", marks=[pytest.mark.cpu], id="cpu"),
        pytest.param("gpu", marks=[pytest.mark.gpu], id="gpu"),
        pytest.param("pfvm", marks=[pytest.mark.pfvm], id="pfvm"),
        pytest.param("mncore", marks=[pytest.mark.mncore], id="mncore"),
    ]
)
def device(request, gpuid):
    if request.param == "cpu":
        return "cpu"
    elif request.param == "pfvm":
        return f"pfvm:cuda:{gpuid}"
    else:
        return f"cuda:{gpuid}"


def test_build_ase_calculator(device_config):
    device, codegen_options = device_config
    calc = build_ase_calculator()
    assert isinstance(calc, ASECalculator)
    calc = build_ase_calculator(device=device, codegen_options=codegen_options)
    assert isinstance(calc, ASECalculator)
    calc = build_ase_calculator_from_root(device=device, codegen_options=codegen_options)
    assert isinstance(calc, ASECalculator)
