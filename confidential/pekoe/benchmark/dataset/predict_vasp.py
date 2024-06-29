import argparse
import pickle
import zipfile
from typing import Any, Dict, List, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from progressbar import ProgressBar
from qsu.dataset.h5container import H5Reader

from batch_calculate import calculate


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="training model")
    parser.add_argument("--keyfile", "-k", type=str, required=True, help="key file")
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
    gts: List[float], preds: List[float], figs: Any, n_plot: int, title: str = ""
) -> None:
    gt = np.array(gts)
    pred = np.array(preds)

    mae = round(np.mean(np.absolute(gt - pred)), 5)
    mse = round(np.mean(np.square(gt - pred)), 5)
    rmse = round(np.sqrt(np.mean(np.square(gt - pred))), 5)
    n_data = len(gt)

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


def draw_charge(
    gts: List[float],
    preds: List[float],
    figs: Any,
    n_plot: int,
    title: str = "",
    draw_abs: bool = True,
) -> None:
    gt = np.concatenate(gts, axis=0).flatten()
    pred = np.concatenate(preds, axis=0).flatten()

    diff = gt - pred

    mae = round(np.mean(np.absolute(diff)), 5)
    mse = round(np.mean(np.square(diff)), 5)
    rmse = round(np.sqrt(np.mean(np.square(diff))), 5)

    ax = figs.add_subplot(3, 3, n_plot)
    H = ax.hist2d(gt, pred, bins=200, cmap=cm.jet, norm=LogNorm())
    ax.set_xlabel("gt [charge]")
    ax.set_ylabel("prediction [charge]")
    ax.set_title(title)
    figs.colorbar(H[3], ax=ax)
    ax.grid()
    ax.text(
        0.02,
        0.98,
        f"""
        RMSE: {rmse}
        MSE: {mse}
        MAE: {mae}
        n data: {len(gt)}
        """,
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
    )


def main(args: argparse.Namespace) -> None:
    if args.model == "pfp":
        from pfp.nn.estimator_base import BaseEstimator
        from pfp.nn.models.crystal.model_builder import build_estimator

        estimator: BaseEstimator = build_estimator(device=args.gpu)
    elif args.model == "pekoe":
        from pekoe.nn.models import DEFAULT_MODEL
        from pekoe.nn.models.model_builder import build_estimator as build_estimator_pekoe

        estimator = build_estimator_pekoe(DEFAULT_MODEL, device=args.gpu)
    else:
        raise ValueError(args.model)

    # load data
    data_name_list: List[Tuple[List[str], str]] = [
        (
            [
                "vasp_U_teanet_amorphous_1.pkl",
                "vasp_U_teanet_amorphous_2.pkl",
                "vasp_U_teanet_amorphous_3.pkl",
                "vasp_U_teanet_amorphous_7.pkl",
                "vasp_U_teanet_amorphous_9.pkl",
            ],
            "NNP amorphous (1st iter)",
        ),
        (
            [
                "vasp_U_teanet_amorphous_12.pkl",
                "vasp_U_teanet_amorphous_13.pkl",
                "vasp_U_teanet_amorphous_14.pkl",
                "vasp_U_teanet_amorphous_15.pkl",
                "vasp_U_teanet_amorphous_36.pkl",
                "vasp_U_teanet_amorphous_37.pkl",
            ],
            "NNP amorphous (2nd iter, 10000 K)",
        ),
        (
            [
                "vasp_U_teanet_amorphous_16.pkl",
                "vasp_U_teanet_amorphous_18.pkl",
                "vasp_U_teanet_amorphous_19.pkl",
                "vasp_U_teanet_amorphous_20.pkl",
                "vasp_U_teanet_amorphous_21.pkl",
                "vasp_U_teanet_amorphous_22.pkl",
                "vasp_U_teanet_amorphous_23.pkl",
                "vasp_U_teanet_amorphous_24.pkl",
                "vasp_U_teanet_amorphous_25.pkl",
                "vasp_U_teanet_amorphous_26.pkl",
            ],
            "NNP amorphous (2nd iter, 2000 K)",
        ),
        (
            [
                "vasp_U_teanet_amorphous_27.pkl",
                "vasp_U_teanet_amorphous_28.pkl",
                "vasp_U_teanet_amorphous_29.pkl",
                "vasp_U_teanet_amorphous_30.pkl",
                "vasp_U_teanet_amorphous_31.pkl",
                "vasp_U_teanet_amorphous_32.pkl",
                "vasp_U_teanet_amorphous_33.pkl",
                "vasp_U_teanet_amorphous_34.pkl",
                "vasp_U_teanet_amorphous_35.pkl",
            ],
            "NNP amorphous (2nd iter, 500 K)",
        ),
        (
            [
                "vasp_U_teanet_amorphous_43.pkl",
                "vasp_U_teanet_amorphous_44.pkl",
                "vasp_U_teanet_amorphous_45.pkl",
                "vasp_U_teanet_amorphous_46.pkl",
                "vasp_U_teanet_amorphous_47.pkl",
                "vasp_U_teanet_amorphous_48.pkl",
                "vasp_U_teanet_amorphous_49.pkl",
                "vasp_U_teanet_amorphous_50.pkl",
            ],
            "NNP amorphous (v0.6.0)",
        ),
        # (
        #     [
        #         "vasp_U_teanet_paper_40elem.pkl",
        #         "vasp_U_teanet_paper_others.pkl",
        #     ],
        #     "Amorphous H-Ar"
        # ),
        # (["twobody_pot_amorphous_HCOFePt.h5",], "Twobody amorphous HCOFePt"),
        # (
        #     ["twobody_pot_amorphous_40elem", "twobody_pot_amorphous_40elem_2",],
        #     "Twobody amorphous 40 elem",
        # ),
        # (["teanet_bulk_opt_1", "teanet_bulk_opt_2",], "Simple cubic-like 40 elem"),
    ]
    # data_name_list = data_name_list[:1]

    data_path = args.keyfile
    with zipfile.ZipFile(data_path) as f_zip:
        estimate_results: Dict[str, Any] = {}
        for dataset_keys, plot_name in data_name_list:
            estimate_results[plot_name] = []
            for dataset_key in dataset_keys:
                single_dataset_dict = pickle.load(f_zip.open(dataset_key))
                single_dataset_keys = single_dataset_dict["keys"]
                h5 = H5Reader(
                    args.data_path + "/" + single_dataset_dict["source"],
                    labels=["cell", "energies", "forces", "bader_charge"],
                )
                if len(single_dataset_keys) > args.max_size:
                    single_dataset_keys = single_dataset_keys[: args.max_size]

                print(f"Fetching: {dataset_key}")
                data = []
                for i, key_set in enumerate(ProgressBar()(single_dataset_keys)):
                    key, kid, energy = key_set
                    sample = h5.get((key, kid))
                    coordinates = sample["coordinates"]
                    charges = sample["bader_charge"]
                    # if charges is None or np.any(np.isnan(charges)):
                    #     charges = np.zeros((len(coordinates)), dtype=np.float64)
                    data.append(
                        {
                            "atomic_numbers": sample["species"],
                            "coordinates": coordinates,
                            "energies": energy,
                            "forces": sample["forces"],
                            "charges": charges,
                            "cell": sample["cell"],
                        }
                    )

                print(f"Calculating: {dataset_key}")
                calculate(estimator, data, args.batchsize, calc=True)
                estimate_results[plot_name].extend(data)

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
        # blank
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
        # charge
        # charge_no_index = [i for i, c in enumerate(results) if c is None or np.any(np.isnan(c))]
        draw_charge(
            [
                r["charges"]
                for r in results
                if not (r["charges"] is None or np.any(np.isnan(r["charges"])))
            ],
            [
                r["p_charges"]
                for r in results
                if not (r["charges"] is None or np.any(np.isnan(r["charges"])))
            ],
            figs,
            n_plot,
            title=f"{plot_name} charge",
        )
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
