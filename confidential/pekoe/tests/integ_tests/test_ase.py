import numpy as np
import pytest
from ase import Atoms, units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet
from ase.optimize import BFGS


def test_ase(calculator, atom_data):
    estimator_inputs, expected_results = atom_data
    atoms = Atoms(
        numbers=estimator_inputs.atomic_numbers,
        positions=estimator_inputs.coordinates,
        cell=estimator_inputs.cell,
        pbc=estimator_inputs.pbc,
    )
    atoms.calc = calculator

    assert atoms.get_potential_energy() == pytest.approx(expected_results["energy"], 1e-5)

    forces = atoms.get_forces()
    assert len(forces) == len(expected_results["forces"])

    for actual, expected in zip(forces, expected_results["forces"]):
        all(map(lambda ae: abs(ae[0] - ae[1]) < 1e-3, zip(actual, expected)))

    charges = atoms.get_charges()
    assert len(charges) == len(expected_results["charges"])

    for actual, expected in zip(charges, expected_results["charges"]):
        all(map(lambda ae: abs(ae[0] - ae[1]) < 1e-3, zip(actual, expected)))

    if all(atoms.pbc):
        stress = atoms.get_stress()
        assert len(stress) == 6
        for actual, expect in zip(stress, expected_results["stress"]):
            assert abs(actual - expect) < 1e-9
    else:
        assert "stress" not in calculator.results

    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=0.001, steps=200)

    np.random.seed(123456)
    MaxwellBoltzmannDistribution(atoms, 500.0 * units.kB)
    Stationary(atoms)
    dyn = VelocityVerlet(atoms, 0.2 * units.fs)

    start_energy = atoms.get_total_energy()
    dyn.run(50)
    end_energy = atoms.get_total_energy()
    assert end_energy - start_energy < 1.0e-3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
