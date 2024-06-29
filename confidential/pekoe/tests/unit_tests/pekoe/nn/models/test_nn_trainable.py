import copy

import pytest
import torch

from pekoe.nn.models import EDGE_FULL_MODEL
from pekoe.nn.models.model_builder import build_estimator
from pekoe.nn.models.teanet.teanet_base import TeaNetBase


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


@pytest.fixture(
    params=[
        pytest.param("default", marks=[pytest.mark.default_model], id="default model"),
        pytest.param(EDGE_FULL_MODEL, marks=[pytest.mark.edge_full_model], id="edge full model"),
    ]
)
def model_config_path(request):
    return request.param


def test_nn_trainable(model_config_path, device, estimator_inputs_all):
    estimator_input = estimator_inputs_all[0]

    input_list = list()

    class ModelMock(TeaNetBase):
        def __call__(self, *inputs, **kwargs):
            for i in inputs:
                input_list.append(copy.copy(i))
            for i in kwargs.values():
                input_list.append(copy.copy(i))
            raise NotImplementedError

    estimator = build_estimator(model_config_path, device=device)
    model = estimator.model

    model_mock = ModelMock()
    model_mock.to(model.device)
    model_mock.device = model.device
    estimator.model = model_mock

    with pytest.raises(NotImplementedError):
        estimator.estimate(estimator_input)

    with torch.enable_grad():
        # Modify model to fit the training situation
        model.is_inference = False
        model.output_every_layer = True

        model.requires_grad_(True)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-4)
        optimizer.zero_grad()
        vecs = input_list[0]
        model_results = model(*tuple(input_list))
        energy = model_results[0]
        charges = model_results[1]
        forces_raw = torch.autograd.grad(torch.sum(energy), vecs, create_graph=True)[0]
        (torch.sum(energy) + torch.sum(charges) + torch.sum(forces_raw)).backward()
        optimizer.step()
