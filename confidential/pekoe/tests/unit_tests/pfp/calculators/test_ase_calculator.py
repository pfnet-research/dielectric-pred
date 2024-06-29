import numpy as np
import pytest
from ase import Atoms

from pfp.calculators.ase_calculator import ASECalculator
from pfp.nn.estimator_base import BaseEstimator, EstimatorCalcMode, EstimatorSystem


class MockEstimator(BaseEstimator):

    implemented_properties = ["energy", "forces", "virial", "charges"]

    def set_atoms(self, ase_data):
        self.atoms, self.expected_results = ase_data

    def estimate(self, args: EstimatorSystem):
        properties = args.properties
        results = {}
        for key in ["energy", "forces", "charges"]:
            if key not in args.properties:
                continue
            results[key] = self.expected_results[key]
        if "virial" in properties:
            if np.all(args.pbc):
                results["virial"] = np.array(
                    [s * self.atoms.get_volume() for s in self.expected_results["stress"]]
                )
            else:
                results["virial"] = np.zeros_like(self.expected_results["stress"])
        results["messages"] = []
        return results

    def available_calc_modes(self):
        return [EstimatorCalcMode.CRYSTAL]

    def get_version(self):
        return "v1.0.0"


@pytest.fixture
def estimator():
    return MockEstimator()


def test_ase_calculator(estimator, atom_data):
    estimator_inputs, expected_results = atom_data
    atoms = Atoms(
        numbers=estimator_inputs.atomic_numbers,
        positions=estimator_inputs.coordinates,
        cell=estimator_inputs.cell,
        pbc=estimator_inputs.pbc,
    )
    ase_data = (atoms, expected_results)
    estimator.set_atoms(ase_data)
    calculator = ASECalculator(estimator)

    calculator.calculate(atoms, properties=None)
    assert calculator.results["energy"] == pytest.approx(expected_results["energy"], 1e-5)
    if all(atoms.pbc):
        assert "stress" in calculator.results
        assert len(calculator.results["stress"]) == 6
        for actual, expected in zip(calculator.results["stress"], expected_results["stress"]):
            assert abs(actual - expected) < 1e-3
    else:
        assert "stress" not in calculator.results
    assert "charges" in calculator.results

    messages = calculator.pop_messages()
    assert isinstance(messages, dict)
    assert len(messages) == 0

    calculator.set_default_properties(list())
    calculator.calculate(atoms, properties=["energy"])
    assert "energy" in calculator.results
    assert "forces" not in calculator.results

    calculator.set_default_properties(["charges"])
    calculator.calculate(atoms, properties=["energy"])
    assert "energy" in calculator.results
    assert "charges" in calculator.results
    assert "forces" not in calculator.results

    # check all ndarray of float is ndarray of np.float64
    # Note: ``ASECalculator`` converts all ndarray properties to be ndarray of np.float64.
    for _, value in calculator.results.items():
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating):
            assert np.issubdtype(value.dtype, np.float64)
