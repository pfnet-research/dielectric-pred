import argparse
import json
from typing import Any, Dict, List

import h5py
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from progressbar import ProgressBar

from batch_calculate import calculate
from pfp.nn.models.crystal.model_builder import build_estimator


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="training model")
    parser.add_argument(
        "--batchsize", "-b", type=int, default=32, help="The number of mini-batch",
    )
    parser.add_argument(
        "--gpu", "-g", type=int, default=0, help="GPU ID (=-1 means PU)"
    )
    parser.add_argument(
        "--data_path", "-d", type=str, default="ssbs_cce.h5", help="path to data file",
    )
    parser.add_argument(
        "--outfilename",
        "-o",
        type=str,
        default="ssbs_cce.pdf",
        help="output file name",
    )

    return parser.parse_args()


def draw_potential_surface(
    pred: List[List[float]],
    gt: List[List[float]],
    figs: Any,
    sub_x: int,
    sub_y: int,
    n_plot: int,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    """
    """
    ax = figs.add_subplot(sub_x, sub_y, n_plot)
    ax.plot(gt[0], gt[1], "o", label="dft", markersize=1)
    ax.plot(pred[0], pred[1], "o", label="pred", markersize=1)
    ax.legend()
    minv = min(np.min(gt[1]), np.min(pred[1])) - 0.1
    maxv = max(np.max(gt[1]), np.max(pred[1])) + 0.1
    ax.set_ylim(minv, maxv)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid()


def draw_energy(
    gts: List[float], preds: List[float], figs: Any, n_plot: int, title: str = ""
) -> None:
    # keys = gts.keys()
    gt = np.array(gts)
    pred = np.array(preds)

    mae = round(np.mean(np.absolute(gt - pred)), 5)
    mse = round(np.mean(np.square(gt - pred)), 5)
    rmse = round(np.sqrt(np.mean(np.square(gt - pred))), 5)
    n_data = len(gt)

    ax = figs.add_subplot(2, 1, n_plot)
    H = ax.hist2d(gt.flatten(), pred.flatten(), bins=200, cmap=cm.jet, norm=LogNorm())
    ax.set_xlabel("gt [eV]")
    ax.set_ylabel("prediction [eV]")
    ax.set_title(title)
    figs.colorbar(H[3], ax=ax)
    ax.grid()
    ax.text(
        0.02,
        0.98,
        f"RMSE: {rmse}\nMSE: {mse}\nMAE: {mae}\nn data: {n_data}",
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
    )


def main(args: argparse.Namespace) -> None:
    estimator = build_estimator(device=args.gpu)

    shift_energy_dict = json.load(open("vasp_energies.json"))
    shift_energy_list = [0.0 for _ in range(120)]
    for k, v in shift_energy_dict.items():
        shift_energy_list[int(k)] = float(v)
    shift_energy = np.array(shift_energy_list, dtype=np.float64)

    dataset = h5py.File(args.data_path, "r")
    dataset_key_list = dataset["keys"][()]
    dataset_key_list = np.unique(dataset_key_list[:, 0])

    estimate_results = {}
    for dataset_key in dataset_key_list:
        single_dataset = dataset[dataset_key]

        coordinates = single_dataset["coordinates"][()]
        atomic_number = single_dataset["species"][()]  # single for one dataset_key
        energies = single_dataset["energies"][()]
        shift = np.sum(shift_energy[single_dataset["species"][()]])
        energies -= shift
        forces = single_dataset["forces"][()]
        cell = single_dataset["cell"][()]
        data = [
            {
                "atomic_numbers": atomic_number,
                "coordinates": co,
                "energies": en,
                "forces": fo,
                "cell": ce,
            }
            for co, en, fo, ce in zip(coordinates, energies, forces, cell)
        ]

        print(f"Calculating: {dataset_key}")
        calculate(estimator, data, args.batchsize, calc=True)
        estimate_results[dataset_key] = data

    print(f"Writing: {args.outfilename}")
    estimate_results_total: Dict[str, List[float]] = {
        "p_energies_per_atom": [],
        "energies_per_atom": [],
    }
    for estimate_result in estimate_results.values():
        estimate_results_total["p_energies_per_atom"].extend(
            [r["p_energies"] / len(r["atomic_numbers"]) for r in estimate_result]
        )
        estimate_results_total["energies_per_atom"].extend(
            [r["energies"] / len(r["atomic_numbers"]) for r in estimate_result]
        )

    pdf_pages = PdfPages(args.outfilename)
    figs = plt.figure(figsize=(6.0, 9.0), dpi=100)
    n_page = 0
    n_plot = 1
    draw_energy(
        estimate_results_total["energies_per_atom"],
        estimate_results_total["p_energies_per_atom"],
        figs,
        n_plot,
    )
    plt.tight_layout()
    pdf_pages.savefig(figs)
    figs.clear()

    # pdf_pages = PdfPages(args.outfilename)
    # figs = plt.figure(figsize=(6.0, 9.0), dpi=100)
    n_page += 1
    n_plot = 1

    x_divide = 2
    y_divide = 4

    # energy
    for dataset_key in ProgressBar()(dataset_key_list):
        result = estimate_results[dataset_key]
        sample_energy = [r["p_energies"] / len(r["atomic_numbers"]) for r in result]
        dft_energy = [r["energies"] / len(r["atomic_numbers"]) for r in result]
        volume = [r["volume"] / len(r["atomic_numbers"]) for r in result]

        title = dataset_key.decode()
        title_element_pos = title.rfind("/") + 1
        title = title[title_element_pos:]
        draw_potential_surface(
            [volume, sample_energy],
            [volume, dft_energy],
            figs,
            y_divide,
            x_divide,
            n_plot,
            title,
            "Volume [ang^3]",
            "Energy [eV/atom]",
        )
        n_plot += 1
        if n_plot > x_divide * y_divide:
            plt.tight_layout()
            pdf_pages.savefig(figs)
            n_page += 1
            figs.clear()
            n_plot = 1

    plt.tight_layout()
    pdf_pages.savefig(figs)
    n_page += 1
    figs.clear()
    n_plot = 1

    pdf_pages.close()
    plt.close("all")


if __name__ == "__main__":
    main(parse())
