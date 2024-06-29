import argparse
from typing import Any, Dict, List, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from progressbar import ProgressBar
from qsu.dataset.h5container import H5Reader

from batch_calculate import calculate


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="training model")
    parser.add_argument(
        "--batchsize", "-b", type=int, default=32, help="The number of mini-batch",
    )
    parser.add_argument("--gpu", "-g", type=int, default=0, help="GPU ID (=-1 means CPU)")
    parser.add_argument(
        "--data_path", "-d", type=str, default="data", help="path to data file",
    )
    parser.add_argument(
        "--max_size", "-m", type=int, default=-1, help="Max number of data points (-1: all)",
    )
    parser.add_argument("--model", type=str, default="pfp", help="model selection")
    parser.add_argument("--output", type=str, default="out.pdf", help="output pdf name")

    return parser.parse_args()


def draw_energy(
    gts: List[float],
    preds: List[float],
    figs: Any,
    n_plot: int,
    title: str = "",
    mae_fit: bool = False,
) -> None:
    gt = np.array(gts)
    pred = np.array(preds)

    mae = round(np.mean(np.absolute(gt - pred)), 5)
    mse = round(np.mean(np.square(gt - pred)), 5)
    rmse = round(np.sqrt(np.mean(np.square(gt - pred))), 5)
    n_data = len(gt)
    if mae_fit:

        def leastsq_fun(param: float) -> float:
            return float(np.mean(np.square(gt - param * pred)).item())

        coeff, _ = scipy.optimize.leastsq(leastsq_fun, 1.0)
        mae_regress = round(np.mean(np.absolute(gt - coeff * pred)), 5)
        plot_text = (
            f"RMSE: {rmse}\nMSE: {mse}\nMAE: {mae}\nReg MAE: {mae_regress}\nn data: {n_data}"
        )
    else:
        plot_text = f"RMSE: {rmse}\nMSE: {mse}\nMAE: {mae}\nn data: {n_data}"

    ax = figs.add_subplot(3, 3, n_plot)
    H = ax.hist2d(gt.flatten(), pred.flatten(), bins=200, cmap=cm.jet, norm=LogNorm())
    ax.set_xlabel("gt [eV]")
    ax.set_ylabel("prediction [eV]")
    ax.set_title(title)
    figs.colorbar(H[3], ax=ax)
    ax.grid()
    ax.text(
        0.02,
        0.98,
        plot_text,
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

    gt_abs = np.sqrt(np.sum(np.square(gt), axis=1))
    pred_abs = np.sqrt(np.sum(np.square(pred), axis=1))
    diff_abs = gt_abs - pred_abs

    mae_abs = round(np.mean(np.absolute(diff_abs)), 5)
    mse_abs = round(np.mean(np.square(diff_abs)), 5)
    rmse_abs = round(np.sqrt(np.mean(np.square(diff_abs))), 5)

    ax = figs.add_subplot(3, 3, n_plot)
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

    n_plot += 1

    ax = figs.add_subplot(3, 3, n_plot)
    H = ax.hist2d(gt_abs, pred_abs, bins=200, cmap=cm.jet, norm=LogNorm())
    ax.set_xlabel("gt [eV/ang]")
    ax.set_ylabel("prediction [eV/ang]")
    ax.set_title(title + " abs")
    figs.colorbar(H[3], ax=ax)
    ax.grid()
    ax.text(
        0.02,
        0.98,
        f"""
        RMSE: {rmse_abs}
        MSE: {mse_abs}
        MAE: {mae_abs}
        n data: {len(gt_abs)}
        """,
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
    )


def main(args: argparse.Namespace) -> None:
    if args.model == "pfp":
        from pfp.nn.estimator_base import BaseEstimator, EstimatorCalcMode
        from pfp.nn.models.crystal.model_builder import build_estimator

        estimator: BaseEstimator = build_estimator(device=args.gpu)
    elif args.model == "pekoe":
        from pekoe.nn.estimator_base import EstimatorCalcMode
        from pekoe.nn.models import DEFAULT_MODEL
        from pekoe.nn.models.model_builder import build_estimator as build_estimator_pekoe

        estimator = build_estimator_pekoe(DEFAULT_MODEL, device=args.gpu)
    else:
        raise ValueError(args.model)

    estimator.set_calc_mode(EstimatorCalcMode.OC20)
    # load data
    data_name_list: List[Tuple[List[str], str]] = [
        (["is2res_train_last.h5", "is2res_train_begin.h5",], "OC20 train",),
        (["is2res_val_id_last.h5", "is2res_val_id_begin.h5",], "OC20 validation",),
    ]
    # data_name_list = data_name_list[:1]

    estimate_results: Dict[str, Any] = {}
    for dataset_keys, plot_name in data_name_list:
        estimate_results[plot_name] = []
        dataset_ref = dataset_keys[0]
        dataset_target = dataset_keys[1]
        h5_ref = H5Reader(
            args.data_path + "/" + dataset_ref, labels=["cell", "energies", "forces"]
        )
        h5_target = H5Reader(
            args.data_path + "/" + dataset_target, labels=["cell", "energies", "forces"]
        )
        ref_dict: Dict[str, List[int]] = {}
        for path, kid in h5_target.keys():
            ref_dict.setdefault(path, [])
            ref_dict[path].append(kid)
        n_max = min(len(ref_dict), args.max_size)

        for i, (path, kid_arr) in enumerate(ProgressBar()(ref_dict.items(), max_value=n_max)):
            data = []
            sample_ref = h5_ref.get((path, 0))
            data.append(
                {
                    "atomic_numbers": sample_ref["species"],
                    "coordinates": sample_ref["coordinates"],
                    "energies": sample_ref["energies"],
                    "forces": sample_ref["forces"],
                    "cell": sample_ref["cell"],
                }
            )
            for kid in kid_arr:
                sample_target = h5_target.get((path, kid))
                force_abs = np.sqrt(np.sum(sample_target["forces"] ** 2, axis=-1))
                force_max = force_abs.max()
                if force_max >= 20.0:
                    continue
                data.append(
                    {
                        "atomic_numbers": sample_target["species"],
                        "coordinates": sample_target["coordinates"],
                        "energies": sample_target["energies"],
                        "forces": sample_target["forces"],
                        "cell": sample_target["cell"],
                    }
                )

            calculate(estimator, data, args.batchsize, calc=True, show_progress=False)

            for d in data:
                d["relative_energies"] = d["energies"] - data[0]["energies"]
                d["p_relative_energies"] = d["p_energies"] - data[0]["p_energies"]

            estimate_results[plot_name].extend(data)

            if i == args.max_size:
                break

    pdf_pages = PdfPages(args.output)
    figs = plt.figure(figsize=(18.0, 15.0), dpi=100)
    n_page = 0
    n_plot = 1

    for _, plot_name in data_name_list:
        results = estimate_results[plot_name]
        # energy
        draw_energy(
            [r["energies"] for r in results],
            [r["p_energies"] for r in results],
            figs,
            n_plot,
            title=f"{plot_name} energy",
        )
        n_plot += 1
        # energy/atom
        draw_energy(
            [r["energies"] / len(r["atomic_numbers"]) for r in results],
            [r["p_energies"] / len(r["atomic_numbers"]) for r in results],
            figs,
            n_plot,
            title=f"{plot_name} energy/atom",
        )
        n_plot += 1
        # relative energy
        draw_energy(
            [r["relative_energies"] for r in results if float(r["relative_energies"]) != 0.0],
            [r["p_relative_energies"] for r in results if float(r["relative_energies"]) != 0.0],
            figs,
            n_plot,
            title=f"{plot_name} relative energy",
            mae_fit=True,
        )
        n_plot += 1

        # force
        draw_force(
            [r["forces"] for r in results],
            [r["p_forces"] for r in results],
            figs,
            n_plot,
            title=f"{plot_name} force",
        )
        n_plot += 2

        n_plot += 1
        n_plot += 1

        plt.tight_layout()
        pdf_pages.savefig(figs)
        n_page += 1
        figs.clear()
        n_plot = 1

    pdf_pages.close()
    plt.close("all")


if __name__ == "__main__":
    main(parse())
