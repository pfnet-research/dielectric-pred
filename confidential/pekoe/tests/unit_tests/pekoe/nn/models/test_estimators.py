import copy

import pytest

from pekoe.nn.models import DEFAULT_MODEL, EDGE_FULL_MODEL
from pekoe.nn.models.config import BaseConfig
from pekoe.nn.models.model_builder import build_estimator
from tests.unit_tests.pfp.nn.models.crystal.estimator_check_functions import (
    check_batch_estimate,
    check_element_status,
    check_estimate,
    check_estimate_book_keeping,
    check_estimate_book_keeping_extreme_cases,
    check_estimate_max_atoms_and_neighbors,
    check_estimate_max_atoms_and_neighbors_hard_limit,
    check_estimator_pbc,
)


@pytest.fixture(
    params=[
        pytest.param("default", marks=[pytest.mark.default_model], id="default model"),
        pytest.param(EDGE_FULL_MODEL, marks=[pytest.mark.edge_full_model], id="edge full model"),
        pytest.param("v1.4.1", marks=[pytest.mark.v1_4_1], id="v1.4.1"),
        pytest.param("v1.4.0", marks=[pytest.mark.v1_4_0], id="v1.4.0"),
        pytest.param("v1.3.1", marks=[pytest.mark.v1_3_1], id="v1.3.1"),
        pytest.param("v1.3.0", marks=[pytest.mark.v1_3_0], id="v1.3.0"),
        pytest.param("v1.2.2", marks=[pytest.mark.v1_2_2], id="v1.2.2"),
        pytest.param("v1.2.1", marks=[pytest.mark.v1_2_1], id="v1.2.1"),
        pytest.param("v1.2.0", marks=[pytest.mark.v1_2_0], id="v1.2.0"),
        pytest.param("v1.1.0", marks=[pytest.mark.v1_1_0], id="v1.1.0"),
        pytest.param("v1.0.0", marks=[pytest.mark.v1_0_0], id="v1.0.0"),
        pytest.param("v0.10.0", marks=[pytest.mark.v0_10_0], id="v0.10.0"),
        pytest.param("d3_pbe", marks=[pytest.mark.d3_pbe], id="d3 pbe model"),
        pytest.param(
            "v1.3.1+ccsd(t)_correction_0.0.1",
            marks=[pytest.mark.v1_3_1_ccsd_t_correction_0_0_1],
            id="v1.3.1+ccsd(t)_correction_0.0.1 model",
        ),
        pytest.param(
            "v1.3.1+ccsd(t)_correction_0.0.2",
            marks=[pytest.mark.v1_3_1_ccsd_t_correction_0_0_2],
            id="v1.3.1+ccsd(t)_correction_0.0.2 model",
        ),
        pytest.param(
            "v1.3.1+ccsd(t)_correction_0.0.3",
            marks=[pytest.mark.v1_3_1_ccsd_t_correction_0_0_3],
            id="v1.3.1+ccsd(t)_correction_0.0.3 model",
        ),
        pytest.param(
            "v1.3.1+ccsd(t)_correction_0.0.4",
            marks=[pytest.mark.v1_3_1_ccsd_t_correction_0_0_4],
            id="v1.3.1+ccsd(t)_correction_0.0.4 model",
        ),
    ]
)
def model_config_path(request):
    return request.param


@pytest.fixture
def estimator(model_config_path, device_config):
    device, codegen_options = device_config
    if str(model_config_path).startswith("d3") and device.startswith("pfvm"):
        pytest.skip("pfvm device does not support d3 estimator")
    return build_estimator(
        model_config_path,
        device=device,
        codegen_options=codegen_options,
    )


@pytest.fixture
def book_keeping_available(estimator):
    try:
        estimator.set_book_keeping(True)
    except NotImplementedError:
        return False

    estimator.set_book_keeping(False)
    return True


def test_estimate_pbc(estimator, estimator_inputs_all):
    check_estimator_pbc(estimator, estimator_inputs_all)


@pytest.mark.parametrize("use_batch", [True, False], ids=["use batch", "don't use batch"])
def test_estimate(estimator, atom_data, calc_mode, use_batch):
    check_estimate(estimator, atom_data, calc_mode, use_batch)


def test_batch_estimate(estimator, atom_data, calc_mode):
    check_batch_estimate(estimator, atom_data, calc_mode)


def test_estimate_book_keeping(estimator, atom_data, book_keeping_available):
    if not book_keeping_available:
        pytest.skip("Book-keeping is not implemented in this estimator.")
    check_estimate_book_keeping(estimator, atom_data)


def test_estimate_book_keeping_extreme_cases(estimator, book_keeping_available):
    if not book_keeping_available:
        pytest.skip("Book-keeping is not implemented in this estimator.")
    check_estimate_book_keeping_extreme_cases(estimator)


def test_edge_full_model_yaml():
    default_model_config = BaseConfig.from_yaml(DEFAULT_MODEL)
    edge_full_model_config = BaseConfig.from_yaml(EDGE_FULL_MODEL)
    assert edge_full_model_config.parameters["shrink_edge"] is False
    equivalent_default_model_config = copy.deepcopy(edge_full_model_config)
    equivalent_default_model_config.parameters["shrink_edge"] = True
    # Check if they are equivalent except their checksum.
    equivalent_default_model_config.checksum = default_model_config.checksum
    assert equivalent_default_model_config == default_model_config


def test_estimate_max_atoms_and_neighbors(model_config_path, device_config):
    device, codegen_options = device_config
    if str(model_config_path).startswith("d3") and device.startswith("pfvm"):
        pytest.skip("pfvm device does not support d3 estimator")

    estimator = build_estimator(
        model_config_path,
        device=device,
        max_neighbors=1000,
        max_atoms=1000,
        codegen_options=codegen_options,
    )
    assert estimator.max_neighbors == 1000
    assert estimator.max_atoms == 1000

    estimator.set_max_neighbors(None)
    estimator.set_max_atoms(None)
    assert estimator.max_neighbors is None
    assert estimator.max_atoms is None

    check_estimate_max_atoms_and_neighbors(estimator)

    # Redirects are not thrown in d3 batch estimators
    if not str(model_config_path).startswith("d3"):
        check_estimate_max_atoms_and_neighbors_hard_limit(estimator)


def test_element_status(estimator, default_element_status):
    check_element_status(estimator, default_element_status)


def test_version(estimator):
    estimator.get_version()
