import json
import pathlib

import numpy as np
import pytest
from ase import Atoms

from pfp.nn.estimator_base import EstimatorCalcMode, EstimatorSystem


def pytest_configure(config):
    config.addinivalue_line("markers", "experimental: experimental test")
    config.addinivalue_line("markers", "slow: mark to skip slow tests")
    config.addinivalue_line("markers", "cpu: mark test to run on cpu")
    config.addinivalue_line("markers", "gpu: mark test to run when a gpu is available")
    config.addinivalue_line("markers", "pfvm: mark test to run when pfvm is available")
    config.addinivalue_line("markers", "mncore: mark test to run when mncore is available")
    config.addinivalue_line(
        "markers", "pfvm_recomp: mark test to run when pfvm with recomp is available"
    )
    config.addinivalue_line("markers", "multi: mark test to run multiprocess")
    config.addinivalue_line("markers", "libinfer: mark test to require libinfer.so")
    config.addinivalue_line(
        "markers", "no_tenet_conv_helper: mark test to run when tenet_conv_helper is not available"
    )


def pytest_addoption(parser):
    parser.addoption("--gpuid", type=int, default=0, help="gpu device id for tests")


def pytest_collection_modifyitems(session, config, items):
    def add_markers_if_not_specified(item, marks):
        item_markers = set(map(lambda m: m.name, item.iter_markers()))
        # NOTE: Do not add other marks to the test with no_tenet_conv_helper mark
        exclusive_marks = marks + ["no_tenet_conv_helper"]
        if all(mark not in item_markers for mark in exclusive_marks):
            for mark in marks:
                item.add_marker(mark)

    for item in items:
        add_markers_if_not_specified(item, ["cpu", "gpu", "pfvm", "pfvm_recomp", "mncore"])


@pytest.fixture
def gpuid(pytestconfig):
    return pytestconfig.getoption("gpuid")


@pytest.fixture
def model_version(estimator):
    return estimator.get_version()


@pytest.fixture(
    params=[
        pytest.param(EstimatorCalcMode.CRYSTAL, id="CRYSTAL mode"),
        pytest.param(EstimatorCalcMode.CRYSTAL_U0, id="CRYSTAL U0 mode"),
        pytest.param(EstimatorCalcMode.MOLECULE, id="MOLECULE mode"),
        pytest.param(EstimatorCalcMode.OC20, id="OC20 mode"),
    ]
)
def calc_mode(request, estimator):
    if request.param not in estimator.available_calc_modes():
        pytest.skip()
    return request.param


def pbc_check_input(test_case):
    position = np.array(
        [
            [5.481891e00, -2.889420e-01, -4.510000e-04],
            [-3.706765e00, -1.605052e00, 2.049000e-03],
            [-2.905558e00, 6.122530e-01, -4.510000e-04],
            [3.784449e00, -1.067110e-01, 4.900000e-05],
            [-1.518376e00, 4.630520e-01, -5.510000e-04],
            [-9.540000e-01, -8.124600e-01, -1.051000e-03],
            [-6.959750e-01, 1.589554e00, 5.490000e-04],
            [4.329800e-01, -9.615650e-01, -3.510000e-04],
            [6.909990e-01, 1.440649e00, 1.049000e-03],
            [1.255378e00, 1.650370e-01, 5.490000e-04],
            [-3.883689e00, -3.911680e-01, -7.510000e-04],
            [-5.277011e00, 1.860060e-01, -4.510000e-04],
            [2.676251e00, 1.232600e-02, 2.490000e-04],
            [-1.524821e00, -1.730502e00, -2.151000e-03],
            [-1.122155e00, 2.589732e00, 1.049000e-03],
            [-3.238813e00, 1.574100e00, -2.510000e-04],
            [8.594640e-01, -1.961936e00, -8.510000e-04],
            [1.319151e00, 2.328228e00, 1.649000e-03],
            [-5.432002e00, 7.763330e-01, 9.069490e-01],
            [-6.013055e00, -6.221830e-01, -2.915100e-02],
            [-5.415477e00, 8.223330e-01, -8.790510e-01],
        ]
    )
    if test_case == 2:
        position += np.array([[20.0, 20.0, 20.0]])

    cell = None
    pbc = False
    if test_case in (1, 2, 3):
        pbc = True
        cell = [100, 100, 100]
    atoms = Atoms("SON2C9H8", position, pbc=pbc, cell=cell)
    if test_case == 3:
        atoms.pbc = np.array([True, True, False], dtype=np.bool_)
        assert np.all(atoms.get_pbc() == [True, True, False])

    inputs = EstimatorSystem(
        atomic_numbers=atoms.get_atomic_numbers(),
        coordinates=atoms.get_positions(),
        cell=atoms.get_cell(complete=True),
        pbc=atoms.get_pbc().astype(np.uint8),
        properties=["energy", "forces"],
    )
    return inputs


@pytest.fixture
def estimator_inputs_all():
    return [pbc_check_input(i) for i in range(4)]


@pytest.fixture(
    params=[
        pytest.param(0, id="No PBC"),
        pytest.param(1, id="PBC with overlap"),
        pytest.param(2, id="PBC without overlap"),
        pytest.param(3, id="Slab-type PBC"),
    ]
)
def estimator_inputs(request, estimator_inputs_all):
    return estimator_inputs_all[request.param]


@pytest.fixture
def atom_data(request, model_version, calc_mode, estimator_inputs):
    # currently this is same to
    atoms = estimator_inputs
    properties_path = (
        pathlib.Path(__file__).parent
        / f"assets/model_{model_version}/mol_properties_{calc_mode}.json"
    )
    expected = json.load(open(properties_path))

    return atoms, expected


@pytest.fixture
def default_element_status(model_version):
    element_status_path = (
        pathlib.Path(__file__).parent / f"assets/model_{model_version}/element_status.json"
    )
    element_status = json.load(open(element_status_path))
    return (
        element_status["accepted_atomic_numbers"],
        element_status["experimental_atomic_numbers"],
        element_status["unexpected_atomic_numbers"],
    )


@pytest.fixture()
def cupy():
    try:
        import cupy
    except ImportError:
        pytest.skip("cupy is not installed")
    return cupy
