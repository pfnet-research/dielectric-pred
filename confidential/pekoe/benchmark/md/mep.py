import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.autoneb import AutoNEB
from ase.neb import NEB
from ase.optimize import FIRE
from cclib.parser import Gaussian
from matplotlib.backends.backend_pdf import PdfPages


def parse():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--gpu", "-g", type=int, default=0, help="GPU ID (=-1 means PU)"
    )
    parser.add_argument("--model_path", "-m", type=str, default="")
    parser.add_argument("--data_path", "-d", type=str, default="")

    return parser.parse_args()


def draw(energies, figs, title, x=1, y=1, n=1):

    ax = figs.add_subplot(x, y, n)
    for label, values in energies:
        plot_values = [x - values[0] for x in values]
        steps = np.arange(len(plot_values))
        ax.plot(steps, plot_values, marker="o", label=label)

    ax.set_title(title)
    ax.set_ylabel("relative energy [eV]")
    ax.set_xlabel("step")
    ax.grid()
    ax.legend()


def load_gt(path):
    print(f"drawing {path}")
    filelist = list(Path(path).glob("*.log"))
    filelist = sorted(filelist, key=lambda x: int(x.stem))
    gts = list()
    for filepath in filelist:
        filepath = filepath.as_posix()
        parser = Gaussian(filepath,)
        data = parser.parse()
        coord = data.atomcoords[0]
        atomnos = data.atomnos
        energy = data.scfenergies
        basis_set = data.metadata["basis_set"]
        functional = data.metadata["functional"]
        gts.append(
            {
                "coord": coord,
                "atomnos": atomnos,
                "energy": energy,
                "meta": f"{functional}/{basis_set}",
            }
        )
    return gts


def calculate_neb(
    calculator, initial, final, name, fmax_set, nsteps=5, nebtype="default"
):
    initial.set_calculator(calculator)
    final.set_calculator(calculator.__class__(calculator.estimator))

    print(f"optimizing {name} initial")
    opt = FIRE(initial, trajectory=f"./mep/{name}{0:03}.traj")
    # opt = FIRE(initial)
    opt.run(fmax=fmax_set[0])

    print(f"optimizing {name} final")
    opt = FIRE(final, trajectory=f"./mep/{name}{1:03}.traj")
    # opt = FIRE(final)
    opt.run(fmax=fmax_set[1])

    images = [initial]
    for i in range(nsteps):
        image = initial.copy()
        image.set_calculator(calculator.__class__(calculator.estimator))
        images.append(image)
    images.append(final)

    # neb = SingleCalculatorNEB(images)
    # neb.climb = True

    if nebtype == "default":
        neb = NEB(images, climb=True, dynamic_relaxation=True)
        neb.interpolate("idpp")
        opt = FIRE(neb, trajectory=f"./mep/{name}_neb.traj")
        # opt = FIRE(neb)
        opt.run(fmax=fmax_set[2])
        energies = [image.get_potential_energy() for image in images]

    elif nebtype == "auto":

        def attach_calculator(images):
            for image in images:
                image.set_calculator(calculator.__class__(calculator.estimator))
            return images

        neb = AutoNEB(
            attach_calculator, name, 1, nsteps + 2, fmax=fmax_set[2], parallel=False
        )
        neb.run()
        energies = neb.get_energies()

    return energies


def main(calculator, pdf_pages, datadirs, fmax_list, nsteps_list, nebtype="auto"):

    # create directory for outputs
    os.makedirs("./mep", exist_ok=True)

    figs = plt.figure(figsize=(9.0, 6.0), dpi=100)
    for i, path in enumerate(datadirs):
        # load gaussian data
        gts = load_gt(path)
        gt_energies = [gt["energy"] for gt in gts]
        mol_list = [
            Atoms(numbers=gts[i]["atomnos"], positions=gts[i]["coord"])
            for i in range(len(gts))
        ]

        # calculate with pfp
        pfp_energies = list()
        for mol in mol_list:
            calculator.calculate(mol)
            energy = calculator.results["energy"]
            pfp_energies.append(energy)

        values = [[gts[0]["meta"], gt_energies], ["pfp estimation", pfp_energies]]

        # neb with pfp
        initial = mol_list[0]
        final = mol_list[-1]
        if nsteps_list[i] is not None:
            pfp_neb_energies = calculate_neb(
                calculator,
                initial,
                final,
                path.stem,
                fmax_list[i],
                nsteps_list[i],
                nebtype,
            )
            values.append(["pfp neb", pfp_neb_energies])

        draw(values, figs, path.stem, x=2, y=1, n=i + 1)
        print(values)

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

    pdf_pages = PdfPages("mep.pdf")
    data_path = args.data_path
    datadirs = list(Path(data_path).glob("*"))
    fmax_list = [(0.05, 0.05, 0.1), (0.05, 0.05, 0.1)]
    # nsteps_list = [29, 13]
    nsteps_list = [None, 13]
    main(calculator, pdf_pages, datadirs, fmax_list, nsteps_list, nebtype="default")
    pdf_pages.close()

#
# python mep.py -g 0 -d ../assets/benchmark_data/reaction
#
