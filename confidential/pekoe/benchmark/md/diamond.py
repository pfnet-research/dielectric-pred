import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase import Atoms, units
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.optimize import BFGS
from matplotlib.backends.backend_pdf import PdfPages


def parse():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--gpu", "-g", type=int, default=0, help="GPU ID (=-1 means PU)"
    )

    return parser.parse_args()


def draw(result, figs, title, x=1, y=1, n=1):

    ax = figs.add_subplot(1, 1, 1)
    df = pd.DataFrame(
        result, columns=["name", "lattice", "epot", "ekin", "vol", "dens"]
    )
    keys = df["name"].unique()
    for key in keys:
        data = df[df["name"] == key]
        x = data["lattice"]
        y = data["epot"]
        ax.plot(x, y, marker="o", label=key)

    ax.set_title("diamond EXP=3.56683")
    ax.set_ylabel("energy [eV]")
    ax.set_xlabel("lattice [ang]")
    ax.grid()
    ax.legend()


class Logger:
    def __init__(self, atoms, name):
        self.name = name
        self.atoms = atoms
        self.result = []

    def __call__(self):
        self.result.append(self.printdata(self.atoms, self.name))

    @staticmethod
    def printdata(atoms, name):
        epot = atoms.get_potential_energy() / len(atoms)
        ekin = atoms.get_kinetic_energy() / len(atoms)
        vol = atoms.get_volume()
        stress = atoms.get_stress()
        press = -(stress[0] + stress[1] + stress[2]) / 3.0 / units.GPa
        cell = atoms.get_cell()
        lattice = np.average([cell[0, 0], cell[1, 1], cell[2, 2]]) / 2
        dens = (
            np.sum(atoms.get_masses()) / (6.02 * 10.0 ** 23) / (vol / 10 ** 27)
        )  # kg/m3 = g/L
        temp = ekin / (1.5 * units.kB)
        tot = epot + ekin
        print(
            f"Energy per atom: Epot={epot:.3f}eV  Ekin={ekin:.3f}eV (T={temp:.3f}K)  "
            f"Etot = {tot:.3f}eV  Vol={vol:.3f}  Press={press:.6f} "
            f"Lx,Ly,Lz = {cell[0, 0]:.3f}, {cell[1, 1]:.3f}, {cell[2, 2]:.3f}"
        )
        return [name, lattice, epot, ekin, vol, dens]


def calculate(
    coord_unit, name, calculator, lmin=3.45, lmax=3.8, stepw=0.02, fluctuation=0.0,
):
    result = []
    for lattice_a in np.arange(lmin, lmax, stepw):
        coord = lattice_a * np.array(coord_unit)
        coord_fluctuate = fluctuation * np.random.randn(coord.shape[0], coord.shape[1])
        atoms = Atoms(
            "C{:}".format(len(coord)),
            coord + coord_fluctuate,
            pbc=True,
            cell=[lattice_a, lattice_a, lattice_a],
        )
        atoms.translate([0, 0, 0])
        atoms.wrap()
        atoms = atoms.repeat((2, 2, 2))
        atoms.set_calculator(calculator)

        if fluctuation > 0.0:
            opt = BFGS(atoms, maxstep=0.01, logfile=None)
            opt.run(fmax=0.01, steps=200)
        result.append(Logger.printdata(atoms, name))

    return result


def md(coord_unit, calculator, lattice_a=3.7, fluctuation=0.001):
    coord = lattice_a * np.array(coord_unit)
    coord_fluctuate = fluctuation * np.random.randn(coord.shape[0], coord.shape[1])
    atoms = Atoms(
        "C{:}".format(len(coord)),
        coord + coord_fluctuate,
        pbc=True,
        cell=[lattice_a, lattice_a, lattice_a],
    )
    atoms.translate([0, 0, 0])
    atoms.wrap()
    atoms.set_calculator(calculator)
    # Structure minimization
    opt = BFGS(atoms, maxstep=0.01)
    opt.run(fmax=0.001, steps=50)
    atoms = atoms.repeat((2, 2, 2))
    atoms.set_calculator(calculator)
    MaxwellBoltzmannDistribution(atoms, 300.0 * units.kB)
    Stationary(atoms)

    dyn = NPT(
        atoms,
        1.0 * units.fs,
        300.0 * units.kB,
        1.0 * 100000.0 * units.Pascal,
        25.0 * units.fs,
        0.6 * ((75.0 * units.fs) ** 2),
        (1, 1, 1),
    )
    lg = Logger(atoms, "MD:NPT1bar300K")
    dyn.attach(lg, interval=10)
    dyn.run(2000)

    result = lg.result
    return result


def main(calculator, pdf_pages):

    result = []

    c_coord_unit = [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
        [0.25, 0.25, 0.25],
        [0.25, 0.75, 0.75],
        [0.75, 0.25, 0.75],
        [0.75, 0.75, 0.25],
    ]

    print("diamond")
    data = calculate(c_coord_unit, "diamond structure", calculator, fluctuation=0.0)
    result += data

    print("opt_cubic")
    data = calculate(c_coord_unit, "opt cubic", calculator, fluctuation=0.001)
    result += data

    print("MD")
    data = md(c_coord_unit, calculator)
    result += data

    figs = plt.figure(figsize=(9.0, 6.0), dpi=100)
    draw(result, figs, title="diamond EXP=3.56683")
    plt.tight_layout()
    pdf_pages.savefig(figs)
    figs.clear()


if __name__ == "__main__":
    args = parse()

    # need to change import model if use pre deploy model
    from pfp.calculators.ase_calculator import ASECalculator
    from pfp.nn.models.crystal import model_builder

    estimator = model_builder.build_estimator(args.gpu)
    calculator = ASECalculator(estimator)
    # ---------

    pdf_pages = PdfPages("diamond.pdf")
    main(calculator, pdf_pages)
    pdf_pages.close()

#
# python diamond.py -g 0
#
