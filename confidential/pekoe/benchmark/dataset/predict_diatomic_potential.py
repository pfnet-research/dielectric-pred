import argparse
import json
from typing import Any, List

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from progressbar import ProgressBar

from batch_calculate import calculate
from pfp.nn.models.crystal.model_builder import build_estimator


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="training model")
    parser.add_argument(
        "--batchsize", "-b", type=int, default=256, help="The number of mini-batch",
    )
    parser.add_argument(
        "--gpu", "-g", type=int, default=0, help="GPU ID (=-1 means PU)"
    )
    parser.add_argument(
        "--data_path",
        "-d",
        type=str,
        default="diatomic_molecule_filtered.h5",
        help="path to data file",
    )
    parser.add_argument(
        "--outfilename",
        "-o",
        type=str,
        default="diatomic_molecule_filtered.pdf",
        help="output file name",
    )
    parser.add_argument("--small", "-s", action="store_true")

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
    ax = figs.add_subplot(6, 2, n_plot)
    ax.plot(gt[0], gt[1], "o", label="dft", markersize=2)
    ax.plot(pred[0], pred[1], "o", label="pred", markersize=1)
    ax.legend()
    minv = min(np.min(gt[1]), np.min(pred[1])) - 0.1
    ax.set_ylim(minv, 5.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid()


def main(args: argparse.Namespace) -> None:
    estimator = build_estimator(device=args.gpu)
    # estimator = build_estimator(device=args.gpu)
    shift_energy_dict = json.load(open("vasp_energies.json"))
    shift_energy_list = [0.0 for _ in range(120)]
    for k, v in shift_energy_dict.items():
        shift_energy_list[int(k)] = float(v)
    shift_energy = np.array(shift_energy_list, dtype=np.float64)

    dataset = h5py.File(args.data_path, "r")
    dataset_key_list = dataset["keys"][()]
    dataset_key_list = np.unique(dataset_key_list[:, 0])

    if args.small:
        dataset_key_list = dataset_key_list[:10]
    # dataset_key_list = [b"others/diatomic_molecule/HC"]

    dataset_results = {}
    for dataset_key in dataset_key_list:
        single_dataset = dataset[dataset_key]

        coordinates = single_dataset["coordinates"][()]
        atomic_number = single_dataset["species"][()]
        # species = [NumToElementDict[n] for n in atomic_number]
        energies = single_dataset["energies"][()]
        shift = np.sum(shift_energy[single_dataset["species"][()]])
        energies -= shift
        forces = single_dataset["forces"][()]
        cell = single_dataset["cell"][()]
        if cell is not None:
            cell_xyz = np.stack(
                [
                    np.zeros_like(cell[:, 0, 0]),
                    np.zeros_like(cell[:, 1, 1]),
                    cell[:, 2, 2],
                ],
                axis=1,
            )
            # TODO precise calculation
            coordinates[:, 1] -= cell_xyz

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

        print(f"Accumulating: {dataset_key}")
        calculate(estimator, data, args.batchsize, calc=False)
        all_distances = np.linalg.norm(coordinates[:, 0] - coordinates[:, 1], axis=-1)
        for d, distance in zip(data, all_distances.tolist()):
            d["distance"] = distance
        dataset_results[dataset_key] = data

    # predictor intermidiates
    radii = np.arange(0.2, 7.0, 0.05) / 2.0
    radii = radii[:, None]
    radii = np.concatenate([radii, -1.0 * radii], axis=-1)
    coordinates = np.zeros((len(radii), 2, 3), dtype="float32")
    coordinates[:, :, -1] = radii

    estimate_results = {}
    for dataset_key in dataset_key_list:
        atomic_number = dataset_results[dataset_key][0]["atomic_numbers"]
        data = [
            {
                "atomic_numbers": atomic_number,
                "coordinates": co,
                "energies": en,
                "forces": fo,
                "cell": ce,
            }
            for co, en, fo, ce in zip(
                coordinates,
                np.zeros(coordinates.shape[0]),
                np.zeros([coordinates.shape[0], 2, 3]),
                [100.0 * np.identity(3) for _ in range(coordinates.shape[0])],
            )
        ]
        print(f"Calculating: {dataset_key}")
        calculate(estimator, data, args.batchsize, calc=True)
        all_distances = np.linalg.norm(coordinates[:, 0] - coordinates[:, 1], axis=-1)
        for d, distance in zip(data, all_distances.tolist()):
            d["distance"] = distance
        estimate_results[dataset_key] = data

    print(f"Writing: {args.outfilename}")
    pdf_pages = PdfPages(args.outfilename)
    figs = plt.figure(figsize=(6.0, 9.0), dpi=100)
    n_page = 0
    n_plot = 1

    x_divide = 2
    y_divide = 5

    # energy
    for dataset_key in ProgressBar()(dataset_key_list):
        estimate_result = estimate_results[dataset_key]
        dataset_result = dataset_results[dataset_key]

        sample_energy = [r["p_energies"] for r in estimate_result]
        sample_distance = [r["distance"] for r in estimate_result]
        sample_force_abs = np.linalg.norm(
            np.array([r["p_forces"] for r in estimate_result]), axis=2
        ).mean(axis=1)
        dft_energy = [r["energies"] for r in dataset_result]
        dft_distance = [r["distance"] for r in dataset_result]
        dft_force_abs = np.linalg.norm(
            np.array([r["forces"] for r in dataset_result]), axis=2
        ).mean(axis=1)

        title = dataset_key.decode()
        title_element_pos = title.rfind("/") + 1
        title = title[title_element_pos:]
        draw_potential_surface(
            [sample_distance, sample_energy, sample_force_abs],
            [dft_distance, dft_energy, dft_force_abs],
            figs,
            y_divide,
            x_divide,
            n_plot,
            title,
            "Distance [ang]",
            "Energy [eV]",
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
