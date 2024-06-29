import argparse
import time

import numpy as np
from ase import Atoms, units
from ase.md.langevin import Langevin
from ase.optimize import BFGS

from pfp.calculators.ase_calculator import ASECalculator
from pfp.nn.estimator_base import EstimatorCalcMode
from pfp.nn.models.crystal import model_builder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="device id")
    parser.add_argument("--book-keeping", action="store_true", help="Enable book-keeping feature")
    parser.add_argument(
        "--calc-mode",
        type=EstimatorCalcMode,
        default=EstimatorCalcMode.CRYSTAL,
        help="Calculation mode (currently CRYSTAL, MOLECULE and OC20)",
    )
    parser.add_argument(
        "--pbc", action="store_true", help="Calculate under periodic-boundary condition"
    )
    parser.add_argument(
        "--large-md", action="store_true", help="Multiply system for MD calculation"
    )
    args = parser.parse_args()
    device = args.device

    estimator = model_builder.build_estimator(device)
    if args.book_keeping:
        estimator.set_book_keeping(True, 2.0)
    estimator.set_calc_mode(args.calc_mode)

    calculator = ASECalculator(estimator)

    # set up a crystal
    coord = np.array(
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
    atoms = Atoms("SON2C9H8", coord, pbc=args.pbc, cell=[100, 100, 100])
    print(len(atoms), "atoms in the cell")
    atoms.calc = calculator

    # calculate energy:
    print("One-shot energy calculation")
    start_time = time.time()
    energy = atoms.get_potential_energy()
    print("elapsed : {}".format(time.time() - start_time))
    print("Energy: {:.5f}".format(energy))

    # minimize the structure:
    print("Begin minimizing...")
    # opt = BFGS(atoms, trajectory="qn.traj")
    opt = BFGS(atoms)

    start_time = time.time()
    opt.run(fmax=0.001)
    print("elapsed : {}".format(time.time() - start_time))

    if args.large_md:
        atoms = atoms.repeat((3, 3, 3))
        atoms.calc = calculator

    def printenergy(a: Atoms = atoms) -> None:
        """Function to print the potential, kinetic and total energy."""
        epot = a.get_potential_energy() / len(a)
        ekin = a.get_kinetic_energy() / len(a)
        print(
            "Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  "
            "Etot = %.3feV" % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin)
        )

    dyn = Langevin(atoms, 1 * units.fs, 300 * units.kB, 0.2)
    dyn.attach(printenergy, interval=50)

    print("Beginning dynamics...")
    printenergy(atoms)
    start_time = time.time()
    dyn.run(500)
    print("elapsed : {}".format(time.time() - start_time))


if __name__ == "__main__":
    main()
