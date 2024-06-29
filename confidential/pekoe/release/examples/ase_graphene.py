import math

import numpy as np
from ase import Atoms, units
from ase.md.langevin import Langevin
from ase.optimize import BFGS

from pfp.calculators.ase_calculator import ASECalculator
from pfp.nn.models.crystal import model_builder


def main() -> None:
    gpu = 0

    estimator = model_builder.build_estimator(gpu)
    calculator = ASECalculator(estimator)

    # set up a crystal
    c_coord = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0 / 2, 1.0 / (2 * math.sqrt(3)), 0.0],
            [1.0 / 2.0, 3.0 / (2 * math.sqrt(3)), 0.0],
            [0.0, 4.0 / (2 * math.sqrt(3)), 0.0],
        ]
    )
    lattice_a = 1.4 * 1.7
    coord = lattice_a * c_coord

    # coord_fluctuate = 0.1*np.random.randn(coord.shape[0], coord.shape[1])
    atoms = Atoms(
        "C4",
        coord,
        pbc=True,
        cell=[lattice_a, math.sqrt(3) * lattice_a, 100.0],
    )
    print(len(atoms), "atoms in the cell")
    atoms.calc = calculator

    # minimize the structure:
    print("Begin minimizing...")
    # opt = BFGS(atoms, trajectory="qn.traj")
    opt = BFGS(atoms)
    import time

    start_time = time.time()
    opt.run(fmax=0.001)
    print("elapsed : {}".format(time.time() - start_time))

    def printenergy(a: Atoms = atoms) -> None:
        """Function to print the potential, kinetic and total energy."""
        epot = a.get_potential_energy() / len(a)
        ekin = a.get_kinetic_energy() / len(a)
        print(
            "Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  "
            "Etot = %.3feV" % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin)
        )

    dyn = Langevin(atoms, 0.1 * units.fs, 300 * units.kB, 50 * units.fs)
    dyn.attach(printenergy, interval=50)

    print("Beginning dynamics...")
    printenergy(atoms)
    dyn.run(500)


if __name__ == "__main__":
    main()
