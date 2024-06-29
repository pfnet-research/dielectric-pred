import pytest

from pfp.nn.models.crystal.model_builder import build_estimator

from .estimator_check_functions import (
    check_batch_estimate,
    check_element_status,
    check_estimate,
    check_estimate_book_keeping,
    check_estimate_book_keeping_extreme_cases,
    check_estimate_max_atoms_and_neighbors,
    check_estimator_pbc,
)


@pytest.fixture
def estimator():
    return build_estimator(device=0)


@pytest.mark.libinfer
def test_estimate_pbc(estimator, estimator_inputs_all):
    check_estimator_pbc(estimator, estimator_inputs_all)


@pytest.mark.gpu
@pytest.mark.libinfer
@pytest.mark.parametrize("use_batch", [True, False])
def test_estimate(atom_data, calc_mode, estimator, use_batch):
    check_estimate(estimator, atom_data, calc_mode, use_batch)


@pytest.mark.gpu
@pytest.mark.libinfer
def test_batch_estimate(atom_data, calc_mode, estimator):
    check_batch_estimate(estimator, atom_data, calc_mode)


@pytest.mark.libinfer
def test_estimate_book_keeping(estimator, atom_data):
    check_estimate_book_keeping(estimator, atom_data)


@pytest.mark.libinfer
def test_estimate_book_keeping_extreme_cases(estimator):
    check_estimate_book_keeping_extreme_cases(estimator)


@pytest.mark.libinfer
def test_estimate_max_atoms_and_neighbors():

    estimator = build_estimator(device=0, max_neighbors=1000, max_atoms=1000)
    assert estimator.max_neighbors == 1000
    assert estimator.max_atoms == 1000

    estimator.set_max_neighbors(None)
    estimator.set_max_atoms(None)
    assert estimator.max_neighbors is None
    assert estimator.max_atoms is None

    check_estimate_max_atoms_and_neighbors(estimator)


@pytest.mark.libinfer
def test_element_status(estimator, default_element_status):
    check_element_status(estimator, default_element_status)


@pytest.mark.libinfer
def test_version(estimator):
    estimator.get_version()
