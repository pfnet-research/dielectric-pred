import argparse
import os

import ase
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.optimize import BFGS
from matplotlib.backends.backend_pdf import PdfPages


def parse():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--gpu", "-g", type=int, default=0, help="GPU ID (=-1 means PU)"
    )
    parser.add_argument("--model_path", "-m", type=str, default="")
    parser.add_argument("--data_path", "-d", type=str, default="")

    return parser.parse_args()


def calculate(calculator, filename, fmax=0.001, dstep=0.002, dmin=3.0, dmax=6.0):

    mol = ase.io.read(filename)
    mol.set_calculator(calculator)
    opt = BFGS(mol)
    opt.run(fmax=fmax)
    name = os.path.splitext(os.path.basename(filename))[0] + "_2"

    result = []
    molmol = None
    minE = 1000.0
    print("calculating energy among different distances")
    for i, dis in enumerate(np.arange(dmin, dmax, dstep)):
        mol2 = mol.copy()
        tempmol = mol.copy()
        tempmol.set_positions(tempmol.get_positions() + [0, 0, dis])
        mol2.extend(tempmol)
        mol2.set_calculator(calculator)
        pot = mol2.get_potential_energy()
        fmax = np.max(np.abs(mol2.get_forces()))
        result.append([name, dis, pot, fmax])
        if pot < minE:
            molmol = mol2
            minE = pot
    result = pd.DataFrame(result, columns=["name", "dis", "pot", "fmax"])

    return result, mol, molmol


def draw(df, figs, x=1, y=2, n=1):

    name = df["name"][0]
    ax = figs.add_subplot(x, y, n)
    ax.plot(df["dis"], df["pot"], marker="o", markersize=2)
    ax.set_title(name + " energy")
    ax.set_xlabel("distance [ang]")
    ax.set_ylabel("potential_energy [eV]")
    ax.grid()

    ax = figs.add_subplot(x, y, n + 1)
    ax.plot(df["dis"], df["fmax"], marker="o", markersize=2)
    ax.set_title(name + " max force")
    ax.set_xlabel("distance [ang]")
    ax.set_ylabel("max force [eV/ang]")
    ax.grid()


def main(calculator, pdf_pages, data_path, fmax=0.001):
    # create directory for outputs
    os.makedirs("./graphene", exist_ok=True)

    figs = plt.figure(figsize=(9.0, 5.0), dpi=100)
    # benzene
    filename = data_path + "/241benzene.com"
    dstep = 0.002
    dmin = 3.0
    dmax = 6.0
    results, mol, mol2 = calculate(calculator, filename, fmax, dstep, dmin, dmax)
    ase.io.write("./graphene/benzen1.traj", mol)
    ase.io.write("./graphene/benzen2.traj", mol2)
    draw(results, figs, x=2, y=2, n=1)

    # graphene sheet
    filename = data_path + "/C150H30.com"
    dstep = 0.002
    dmin = 3.0
    dmax = 6.0
    results, mol, mol2 = calculate(calculator, filename, fmax, dstep, dmin, dmax)
    ase.io.write("./graphene/graphene.traj", mol)
    ase.io.write("./graphene/graphene2.traj", mol2)
    draw(results, figs, x=2, y=2, n=3)
    plt.tight_layout()
    pdf_pages.savefig(figs)
    figs.clear()


if __name__ == "__main__":
    args = parse()

    # need to change import model if use pre deploy model
    from pfp.calculators.ase_calculator import ASECalculator
    from pfp.nn.models.molecule import model_builder

    estimator = model_builder.build_estimator(args.gpu)
    calculator = ASECalculator(estimator)
    # ---------

    pdf_pages = PdfPages("graphene_sheets.pdf")
    fmax = 0.001
    data_path = args.data_path
    main(calculator, pdf_pages, data_path, fmax=fmax)
    pdf_pages.close()

#
# python graphene_sheets.py -g 0 -d ../assets/benchmark_data/two_mol
#
