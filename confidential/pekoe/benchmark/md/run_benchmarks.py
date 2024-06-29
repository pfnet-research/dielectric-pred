import argparse
from pathlib import Path

from matplotlib.backends.backend_pdf import PdfPages

import diamond
import graphene_sheets
import iron_oxide_adsorp
import mep


def parse():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--gpu", "-g", type=int, default=0, help="GPU ID (=-1 means PU)"
    )
    parser.add_argument("--data_path", "-d", type=str, default="")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse()

    # need to change import model if use pre deploy model
    from pfp.calculators.ase_calculator import ASECalculator
    from pfp.nn.models.crystal import model_builder

    estimator = model_builder.build_estimator(args.gpu)
    calculator = ASECalculator(estimator)
    # ---------

    pdf_pages = PdfPages("benchmarks.pdf")

    # benzen & graphene sheet
    fmax = 0.001
    data_path = args.data_path
    graphene_sheets.main(calculator, pdf_pages, data_path + "/two_mol", fmax=fmax)

    # minimum energy path
    data_path = args.data_path
    datadirs = list(Path(data_path + "/reaction").glob("*"))
    neb_fmax_list = [(0.05, 0.05, 0.5), (0.01, 0.01, 0.1)]
    neb_nsteps_list = [None, None]
    mep.main(
        calculator, pdf_pages, datadirs, neb_fmax_list, neb_nsteps_list, nebtype="auto"
    )

    # diamond
    diamond.main(calculator, pdf_pages)

    # FeOx-AcOH adsorption
    fmax = 0.01
    data_path = args.data_path
    iron_oxide_adsorp.main(calculator, pdf_pages, data_path + "/adsorp", fmax=fmax)

    pdf_pages.close()

#
# python run_benchmarks.py -g 0 -d /mnt/vol12/chem/pfnconfidential/pfp/benchmark_data
#
