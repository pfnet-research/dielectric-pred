import numpy as np
import pytest

from pfp.nn.estimator_base import (
    EstimatorSystem,
    InputInvalidError,
    InputNoElementInformationError,
)
from pfp.utils.atoms import max_atomic_number


def test_estimator_system():
    properties = ["energy"]
    species = ["H", "He"]
    atomic_numbers = np.array([1, 2])
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    cell = np.array([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]])
    pbc = np.array([0, 0, 0])

    inputs_sp = EstimatorSystem(
        properties=properties,
        species=species,
        coordinates=coordinates,
        cell=cell,
        pbc=pbc,
    )

    inputs_an = EstimatorSystem(
        properties=properties,
        atomic_numbers=atomic_numbers,
        coordinates=coordinates,
        cell=cell,
        pbc=pbc,
    )

    inputs_sp.format()
    assert np.all(inputs_sp.atomic_numbers == inputs_an.atomic_numbers)

    with pytest.raises(
        InputNoElementInformationError,
        match=r"Neither species \(chemical_symbols\) nor atomic_numbers is provided.",
    ):
        inputs = EstimatorSystem(
            properties=properties,
            coordinates=coordinates,
            cell=cell,
            pbc=pbc,
        )
        inputs.format()

    with pytest.raises(InputInvalidError, match=r"coordinates\.shape=.* \(positions\) is not "):
        inputs = EstimatorSystem(
            properties=properties,
            atomic_numbers=atomic_numbers,
            coordinates=coordinates[:, :2],
            cell=cell,
            pbc=pbc,
        )
        inputs.format()

    with pytest.raises(InputInvalidError, match=r"coordinates\.shape=.* \(positions\) is not "):
        inputs = EstimatorSystem(
            properties=properties,
            atomic_numbers=atomic_numbers,
            coordinates=coordinates[:1, :],
            cell=cell,
            pbc=pbc,
        )
        inputs.format()

    with pytest.raises(
        InputInvalidError, match=r"coordinates \(positions\) contains infinite values."
    ):
        inputs = EstimatorSystem(
            properties=properties,
            atomic_numbers=atomic_numbers,
            coordinates=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, float("NaN")]]),
            cell=cell,
            pbc=pbc,
        )
        inputs.format()

    with pytest.raises(InputInvalidError, match=r"cell\.shape=.* is not \(3, 3\)."):
        inputs = EstimatorSystem(
            properties=properties,
            atomic_numbers=atomic_numbers,
            coordinates=coordinates,
            cell=cell[:, :2],
            pbc=pbc,
        )
        inputs.format()

    with pytest.raises(InputInvalidError, match=r"cell contains infinite values."):
        inputs = EstimatorSystem(
            properties=properties,
            atomic_numbers=atomic_numbers,
            coordinates=coordinates,
            cell=np.array([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, float("NaN")]]),
            pbc=pbc,
        )
        inputs.format()

    with pytest.raises(
        InputInvalidError, match=r"species=.* \(chemical_symbols\) contains invalid symbols"
    ):
        inputs = EstimatorSystem(
            properties=properties,
            species=["H", "He", "_"],
            coordinates=coordinates,
            cell=cell,
            pbc=pbc,
        )
        inputs.format()

    with pytest.raises(
        InputInvalidError, match=r"The entry of atomic_numbers=.* should be between 1 and "
    ):
        inputs = EstimatorSystem(
            properties=properties,
            atomic_numbers=np.array([0, 1]),
            coordinates=coordinates,
            cell=cell,
            pbc=pbc,
        )
        inputs.format()

    with pytest.raises(
        InputInvalidError, match=r"The entry of atomic_numbers=.* should be between 1 and "
    ):
        inputs = EstimatorSystem(
            properties=properties,
            atomic_numbers=np.array([1, max_atomic_number()]),
            coordinates=coordinates,
            cell=cell,
            pbc=pbc,
        )
        inputs.format()

    with pytest.raises(InputInvalidError, match=r"pbc.shape=.* is not \(3,\)."):
        inputs = EstimatorSystem(
            properties=properties,
            atomic_numbers=atomic_numbers,
            coordinates=coordinates,
            cell=cell,
            pbc=np.array([0, 0, 0, 0]),
        )
        inputs.format()

    with pytest.raises(InputInvalidError, match=r"pbc=.* are not 0 or 1."):
        inputs = EstimatorSystem(
            properties=properties,
            atomic_numbers=atomic_numbers,
            coordinates=coordinates,
            cell=cell,
            pbc=np.array([2, 2, 2]),
        )
        inputs.format()

    with pytest.raises(InputInvalidError, match=r"cell_size=.* is too small \(<1.0e-7\)."):
        inputs = EstimatorSystem(
            properties=properties,
            atomic_numbers=atomic_numbers,
            coordinates=coordinates,
            cell=np.array([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [100.0, 100.0, 0.0]]),
            pbc=np.array([1, 1, 1]),
        )
        inputs.format()
