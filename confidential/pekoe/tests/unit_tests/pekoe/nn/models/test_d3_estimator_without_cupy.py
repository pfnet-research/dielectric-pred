import warnings

import pytest

from pekoe.nn.models.config import BaseConfig
from pekoe.nn.models.d3_estimator import D3Estimator, D3EstimatorParameters
from pekoe.nn.models.model_builder import _model_strings


@pytest.fixture(
    params=[
        pytest.param("cpu", marks=[pytest.mark.cpu], id="cpu"),
        pytest.param("gpu", marks=[pytest.mark.gpu], id="gpu"),
    ]
)
def device(request, gpuid):
    if request.param == "cpu":
        return "cpu"
    else:
        return f"cuda:{gpuid}"


def test_d3_estimator_without_cupy(device):
    try:
        import cupy  # NOQA
    except ImportError:
        pass
    else:
        pytest.skip("cupy is installed")

    # d3 estimator with abc=True, without CuPy
    d3_config = BaseConfig.from_yaml(_model_strings["d3_pbe"])
    d3_config.parameters["abc"] = True
    d3_estimator = D3Estimator(
        model_parameters=D3EstimatorParameters.from_dict(d3_config.parameters),
        version=d3_config.version,
    )
    if device == "cpu":
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            d3_estimator.to(device)
    else:
        with pytest.warns(UserWarning):
            d3_estimator.to(device)
