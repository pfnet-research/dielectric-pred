import time

import numpy as np
import pytest
from ase import Atoms

from pekoe.calculators.batch_calculator import BatchCalculator, DynProcess
from pfp.nn.estimator_base import BaseEstimator, EstimatorCalcMode, EstimatorSystem


class MockEstimator(BaseEstimator):

    implemented_properties = ["energy", "forces", "virial", "charges"]

    def set_atoms(self, ase_data):
        self.atoms, self.expected_results = ase_data

    def estimate(self, args: EstimatorSystem):
        assert isinstance(args, EstimatorSystem)
        results = {}
        for key in ["energy", "forces", "charges"]:
            results[key] = self.expected_results[key]
        if np.all(args.pbc):
            results["virial"] = np.array(
                [s * self.atoms.get_volume() for s in self.expected_results["stress"]]
            )
        else:
            results["virial"] = np.zeros_like(self.expected_results["stress"])
        results["messages"] = []
        return results

    def batch_estimate(self, args_list):
        results = []
        for args in args_list:
            results.append(self.estimate(args))
        return results

    def available_calc_modes(self):
        return [EstimatorCalcMode.CRYSTAL]

    def get_version(self):
        return "v1.0.0"


@pytest.fixture
def estimator():
    return MockEstimator()


@pytest.mark.multi
def test_batch_calculator(estimator, atom_data):
    estimator_inputs, expected_results = atom_data
    atoms = Atoms(
        numbers=estimator_inputs.atomic_numbers,
        positions=estimator_inputs.coordinates,
        cell=estimator_inputs.cell,
        pbc=estimator_inputs.pbc,
    )
    ase_data = (atoms, expected_results)

    n_jobs = 3

    estimator.set_atoms(ase_data)
    calculator = BatchCalculator(estimator, n_jobs)

    def script(atoms, expected_results):
        for _ in range(10):
            time.sleep(0.1)
            energy = atoms.get_potential_energy()

        assert energy == pytest.approx(expected_results["energy"], 1e-5)

        if np.all(atoms.pbc):
            stress = atoms.get_stress()
            assert len(stress) == 6
            assert np.allclose(stress, expected_results["stress"], rtol=0.0, atol=1e-3)

        assert "charges" in atoms._calc.results

    process_list = []
    for _ in range(4):
        atoms1 = atoms.copy()
        atoms1.set_calculator(calculator)
        process_list.append(DynProcess(target=script, args=(atoms1, expected_results)))

    for process in process_list:
        process.start()

    try:
        calculator.run_calculate()
    except Exception as e:
        for process in process_list:
            process.kill()
        raise e

    for process in process_list:
        process.join()

    print("Finish !")
