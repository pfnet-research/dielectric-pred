import argparse
import json
import pathlib
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


def draw_force(
    gts: List[float],
    preds: List[float],
    figs: Any,
    n_plot: int,
    title: str = "",
    draw_abs: bool = True,
) -> None:
    gt = np.concatenate(gts, axis=0)
    pred = np.concatenate(preds, axis=0)

    gt_flat = gt.flatten()
    pred_flat = pred.flatten()
    diff_flat = gt_flat - pred_flat

    mae_flat = round(np.mean(np.absolute(diff_flat)), 5)
    mse_flat = round(np.mean(np.square(diff_flat)), 5)
    rmse_flat = round(np.sqrt(np.mean(np.square(diff_flat))), 5)

    ax = figs.add_subplot(2, 1, n_plot)
    H = ax.hist2d(gt_flat, pred_flat, bins=200, cmap=cm.jet, norm=LogNorm())
    ax.set_xlabel("gt [eV/ang]")
    ax.set_ylabel("prediction [eV/ang]")
    ax.set_title(title + " xyz")
    figs.colorbar(H[3], ax=ax)
    ax.grid()
    ax.text(
        0.02,
        0.98,
        f"""
        RMSE: {rmse_flat}
        MSE: {mse_flat}
        MAE: {mae_flat}
        n data: {len(gt_flat)}
        """,
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
    )


def main(args: argparse.Namespace) -> None:
    filepath = pathlib.Path(args.data_path)
    estimator = build_estimator(device=args.gpu)

    shift_energy_dict = json.load(open("vasp_energies.json"))
    shift_energy_list = [0.0 for _ in range(120)]
    for k, v in shift_energy_dict.items():
        shift_energy_list[int(k)] = float(v)
    shift_energy = np.array(shift_energy_list, dtype=np.float64)

    dataset = h5py.File(str(filepath), "r")

    estimate_results = {}

    data = []
    dataset_keys_list = dataset["keys"][()]

    n_sample = len(dataset_keys_list)

    print("Fetching data...")
    for i, (sample_key, sample_idx) in enumerate(ProgressBar()(dataset_keys_list)):
        if i >= n_sample:
            break
        sample_idx = int(sample_idx)
        sample = dataset[sample_key]
        data.append(
            {
                # "name": sample_key.decode()+", "+str(sample_idx),
                "atomic_numbers": sample["species"],
                "coordinates": sample["coordinates"][sample_idx],
                "energies": sample["energies"][sample_idx],
                "forces": sample["forces"][sample_idx],
                "cell": sample["cell"][sample_idx] if "cell" in sample else None,
            }
        )

    print(f"Calculating: {str(filepath)}")
    calculate(estimator, data, args.batchsize, calc=True)
    estimate_results["single"] = data

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
    # n_page = 0
    n_plot = 1
    draw_energy(
        estimate_results_total["energies_per_atom"],
        estimate_results_total["p_energies_per_atom"],
        figs,
        n_plot,
    )
    n_plot += 1
    draw_force(
        [r["forces"] for r in estimate_results["single"]],
        [r["p_forces"] for r in estimate_results["single"]],
        figs,
        n_plot,
        title="",
    )
    plt.tight_layout()
    pdf_pages.savefig(figs)
    figs.clear()

    pdf_pages.close()
    plt.close("all")


if __name__ == "__main__":
    main(parse())
