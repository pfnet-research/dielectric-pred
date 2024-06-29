"""
DFTD3 program need to be installed to test this method.
Original file is `test_torch_dftd3_calculator.py` in https://github.com/pfnet-research/torch-dftd
"""
import tempfile

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk, fcc111, molecule
from ase.calculators.dftd3 import DFTD3

from pekoe.calculators import ASECalculator
from pekoe.nn.models.d3_estimator import D3Estimator, D3EstimatorParameters

from .damping import damping_method_list, damping_xc_combination_list


@pytest.fixture(
    params=[
        pytest.param("mol", id="mol"),
        pytest.param("slab", id="slab"),
        pytest.param("large", marks=[pytest.mark.slow], id="large"),
    ]
)
def atoms(request) -> Atoms:
    """Initialization"""
    mol = molecule("CH3CH2OCH3")

    slab = fcc111("Au", size=(2, 1, 3), vacuum=80.0)
    slab.set_cell(
        slab.get_cell().array @ np.array([[1.0, 0.1, 0.2], [0.0, 1.0, 0.3], [0.0, 0.0, 1.0]])
    )
    slab.pbc = np.array([True, True, True])

    large_bulk = bulk("Pt", "fcc") * (4, 4, 4)

    atoms_dict = {"mol": mol, "slab": slab, "large": large_bulk}

    return atoms_dict[request.param]


def _assert_energy_equal(calc1, calc2, atoms: Atoms):
    calc1.reset()
    atoms.calc = calc1
    e1 = atoms.get_potential_energy()

    calc2.reset()
    atoms.calc = calc2
    e2 = atoms.get_potential_energy()
    assert np.allclose(e1, e2, atol=1e-4, rtol=1e-4)


def _test_calc_energy(damping, xc, old, atoms, device="cpu", dtype=torch.float64, abc=False):
    cutoff = 25.0  # Make test faster
    with tempfile.TemporaryDirectory() as tmpdirname:
        dftd3_calc = DFTD3(
            damping=damping,
            xc=xc,
            grad=True,
            old=old,
            cutoff=cutoff,
            directory=tmpdirname,
            abc=abc,
        )
        estimator = D3Estimator(
            D3EstimatorParameters(
                damping=damping, xc=xc, old=old, abc=abc, dtype=dtype, cutoff=cutoff
            ),
        )
        estimator.to(device)
        calculator = ASECalculator(estimator)
        _assert_energy_equal(dftd3_calc, calculator, atoms)


def _assert_energy_force_stress_equal(calc1, calc2, atoms: Atoms):
    calc1.reset()
    atoms.calc = calc1
    f1 = atoms.get_forces()
    e1 = atoms.get_potential_energy()
    atoms_pbc = False
    if np.all(atoms.pbc == np.array([True, True, True])):
        atoms_pbc = True
        s1 = atoms.get_stress()

    calc2.reset()
    atoms.calc = calc2
    f2 = atoms.get_forces()
    e2 = atoms.get_potential_energy()
    assert np.allclose(e1, e2, atol=1e-4, rtol=1e-4)
    assert np.allclose(f1, f2, atol=1e-5, rtol=1e-5)
    if atoms_pbc:
        s2 = atoms.get_stress()
        assert np.allclose(s1, s2, atol=1e-5, rtol=1e-5)


def _test_calc_energy_force_stress(
    damping,
    xc,
    old,
    atoms,
    device="cpu",
    dtype=torch.float64,
    abc=False,
    cnthr=15.0,
    cutoff_smoothing="none",
    bidirectional=False,
):
    cutoff = 22.0  # Make test faster
    with tempfile.TemporaryDirectory() as tmpdirname:
        dftd3_calc = DFTD3(
            damping=damping,
            xc=xc,
            grad=True,
            old=old,
            cutoff=cutoff,
            cnthr=cnthr,
            directory=tmpdirname,
            abc=abc,
        )
        estimator = D3Estimator(
            D3EstimatorParameters(
                damping=damping,
                xc=xc,
                old=old,
                abc=abc,
                dtype=dtype,
                cutoff=cutoff,
                cnthr=cnthr,
                cutoff_smoothing=cutoff_smoothing,
                bidirectional=bidirectional,
            ),
        )
        estimator.to(device)
        calculator = ASECalculator(estimator)
        _assert_energy_force_stress_equal(dftd3_calc, calculator, atoms)


@pytest.mark.parametrize("damping,xc,old", damping_xc_combination_list)
def test_calc_energy_force_stress(damping, xc, old, atoms):
    _test_calc_energy(damping, xc, old, atoms, device="cpu")
    _test_calc_energy_force_stress(damping, xc, old, atoms, device="cpu")


@pytest.mark.parametrize("damping,old", damping_method_list)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("abc", [True, False])
@pytest.mark.parametrize("cutoff_smoothing", ["none", "poly"])
@pytest.mark.parametrize("bidirectional", [True, False])
def test_device_dtype_abc_smoothing_bidirectional(
    damping, old, atoms, device, dtype, abc, cutoff_smoothing, bidirectional
):
    """Test: check tri-partite calc with device, dtype dependency."""
    xc = "pbe"
    _test_calc_energy_force_stress(
        damping,
        xc,
        old,
        atoms,
        device=device,
        dtype=dtype,
        abc=abc,
        cnthr=7.0,
        cutoff_smoothing=cutoff_smoothing,
        bidirectional=bidirectional,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
