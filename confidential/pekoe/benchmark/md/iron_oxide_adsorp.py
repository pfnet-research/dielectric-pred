import argparse
import os

import ase
import matplotlib.pyplot as plt
import numpy as np
from ase.constraints import FixAtoms
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


def calculate_FeOx_AcOH(calculator, data_path, fmax=0.01):
    """
    """
    filename = data_path + "/iron_oxide-acetic_acid.cif"
    atoms = ase.io.read(filename)
    idx_divide_1 = 120
    idx_divide_2 = idx_divide_1 + 8
    idx_divide_3 = idx_divide_2 + 8
    slab_idx = np.arange(len(atoms))[:idx_divide_1]
    mol1_idx = np.arange(len(atoms))[idx_divide_1:idx_divide_2]
    mol1_first_C_idx = 1
    mol2_idx = np.arange(len(atoms))[idx_divide_2:idx_divide_3]
    mol2_first_C_idx = 1
    slab_top_z = 12.0

    atoms.set_calculator(calculator)
    atoms.set_constraint()
    maxf = np.sqrt(((atoms.get_forces()) ** 2).sum(axis=1).max())
    print("pot : {:.4f}, maxforce : {:.4f}".format(atoms.get_potential_energy(), maxf))

    # FeOx + 2 AcOH
    print("opt of all system")
    constraintatoms = [atom.index for atom in atoms if atom.z < slab_top_z]
    atoms = constraint_opt(atoms, calculator, constraintatoms, fmax=fmax)

    # only FeOx
    print("opt of slab only")
    slab = atoms[slab_idx]
    constraintatoms = [atom.index for atom in slab if atom.z < slab_top_z]
    slab = constraint_opt(slab, calculator, constraintatoms, fmax=fmax)

    # each AcOH
    print("opt of each AcOH")
    mol1 = atoms[mol1_idx]
    mol1 = constraint_opt(mol1, calculator)

    mol2 = atoms[mol2_idx]
    mol2 = constraint_opt(mol2, calculator)

    org_energy = atoms.get_potential_energy()
    slab_energy = slab.get_potential_energy()
    mol1_energy = mol1.get_potential_energy()
    mol2_energy = mol2.get_potential_energy()

    sepe = slab_energy + mol1_energy + mol2_energy
    ade = org_energy - sepe
    print("slab energy : {:.4f}eV".format(slab_energy))
    print("mol1 energy : {:.4f}eV".format(mol1_energy))
    print("mol2 energy : {:.4f}eV".format(mol2_energy))

    print("slab + mol1 + mol2 individual energy : {:.4f}eV".format(sepe))
    print("energy after adsorption : {:.4f}eV".format(org_energy))
    print("adsorption energy : {:.4f}eV".format(ade))

    # energy scan with opt along z distance
    print("start scan along z distance")
    ini = atoms.get_positions()
    scanatoms = atoms.copy()
    scanatoms.set_calculator(calculator)
    result = []
    scanr = []
    mol_idx = np.concatenate([mol1_idx, mol2_idx])
    for distance in np.arange(-1, 2.5, 0.25):
        print(distance)
        scanatoms.set_constraint()
        scanatoms.set_positions(ini)
        for i in mol_idx:
            scanatoms[i].z += distance
        scanatoms = constraint_opt(
            scanatoms,
            calculator,
            constraintatoms + [mol1_idx[mol1_first_C_idx], mol2_idx[mol2_first_C_idx]],
            3,
            fmax,
        )
        scanr += [scanatoms.copy()]
        delta = np.min(scanatoms[mol_idx].get_positions()[:, 2]) - np.max(
            scanatoms[slab_idx].get_positions()[:, 2]
        )
        result.append([distance, scanatoms.get_potential_energy(), delta])

    result_dict = {
        "scan_structures": scanr,
        "scan_energies": result,
        "org_energy": org_energy,
        "slab_energy": slab_energy,
        "mol1_energy": mol1_energy,
        "mol2_energy": mol2_energy,
    }
    return result_dict


def constraint_opt(
    atoms, calculator, constraintatoms=None, ncycles=10, fmax=0.01, nsteps=100
):
    """
    """
    if constraintatoms is None:
        constraintatoms = []
    sc = FixAtoms(indices=constraintatoms)
    atoms.set_constraint(sc)
    atoms.set_calculator(calculator)
    maxf = np.sqrt(((atoms.get_forces()) ** 2).sum(axis=1).max())
    print("pot : {:.4f}, maxforce : {:.4f}".format(atoms.get_potential_energy(), maxf))
    de = -1
    s = 1
    while (de < -0.01 or de > 0.01) and s < ncycles:
        opt = BFGS(atoms, maxstep=0.03 * (0.9 ** s), logfile=None)
        old = atoms.get_potential_energy()
        opt.run(fmax=fmax, steps=nsteps)
        maxf = np.sqrt(((atoms.get_forces()) ** 2).sum(axis=1).max())
        de = atoms.get_potential_energy() - old
        print(
            "after {}opt pot : {:.4f}, maxforce : {:.4f}, delta : {:.4f}".format(
                s * nsteps, atoms.get_potential_energy(), maxf, de
            )
        )
        s += 1
    return atoms


def draw(results, figs, name):

    data = np.array(results["scan_energies"])

    ax = figs.add_subplot(2, 1, 1)
    ax.plot(data[:, 0], data[:, 1], marker="o", markersize=2)
    ax.set_title(f"{name} energy of distance of C from adsorption state")
    ax.set_xlabel("distance [ang]")
    ax.set_ylabel("potential_energy [eV]")
    ax.grid()

    ax = figs.add_subplot(2, 1, 2)
    ax.plot(data[:, 2], data[:, 1], marker="o", markersize=2)
    ax.set_title(f"{name} energy of distance between slab to and molecule bottom")
    ax.set_xlabel("distance [ang]")
    ax.set_ylabel("potential_energy [eV]")
    ax.grid()


def main(calculator, pdf_pages, data_path, fmax=0.01):
    # create directory for outputs
    os.makedirs("./adsorp", exist_ok=True)

    figs = plt.figure(figsize=(8.0, 5.0), dpi=100)
    # iron_oxide
    results = calculate_FeOx_AcOH(calculator, data_path)
    draw(results, figs, "FeOx-AcOH")
    for i, atoms in enumerate(results["scan_structures"]):
        ase.io.write(f"./adsorp/FeOx-AcOH_{i}.traj", atoms)

    plt.tight_layout()
    pdf_pages.savefig(figs)


if __name__ == "__main__":
    args = parse()

    # need to change import model if use pre deploy model
    from pfp.calculators.ase_calculator import ASECalculator
    from pfp.nn.models.molecule import model_builder

    estimator = model_builder.build_estimator(args.gpu)
    calculator = ASECalculator(estimator)
    # ---------

    pdf_pages = PdfPages("FeOx-AcOH_adsorption.pdf")
    fmax = 0.01
    data_path = args.data_path
    main(calculator, pdf_pages, data_path, fmax=fmax)
    pdf_pages.close()

#
# python iron_oxide_adsorp.py -g 0 -d path-to-assets/benchmark_data/adsorption
#
