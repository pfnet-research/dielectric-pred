import argparse
import json
import pathlib
from typing import Any, List

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
        "--batchsize", "-b", type=int, default=64, help="The number of mini-batch"
    )
    parser.add_argument(
        "--gpu", "-g", type=int, default=0, help="GPU ID (=-1 means PU)"
    )
    parser.add_argument(
        "--data_path", "-d", type=str, default="pfn_data.h5", help="path to data file"
    )
    parser.add_argument(
        "--max_size",
        "-m",
        type=int,
        default=-1,
        help="Max number of data points (-1: all)",
    )
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

    if not draw_abs:
        return

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
    estimator = build_estimator(device=args.gpu)

    shift_energy_dict = json.load(open("vasp_energies.json"))
    shift_energy_list = [0.0 for _ in range(120)]
    for k, v in shift_energy_dict.items():
        shift_energy_list[int(k)] = float(v)
    shift_energy = np.array(shift_energy_list, dtype=np.float64)

    # load data
    data_name_list = [
        # [
        #     "pubchem_CNOPS1-6_opt",
        #     "pubchem_CNOPS1-6_nms",
        #     "pubchem_CNOPS1-6_rts",
        #     "pubchem_CNOPS1-6_rad_opt",
        #     "pubchem_CNOPS1-6_rad_nms",
        #     "pubchem_CNOPS1-6_FClBr_opt",
        #     "pubchem_CNOPS1-6_FClBr_nms",
        # ],
        [
            # "pubchem_CNOPS7_opt",
            "pubchem_CNOPS7_nms_mwv",
            "pubchem_CNOPS7_rts_mwv",
            # "pubchem_CNOPS7_rad_opt",
            "pubchem_CNOPS7_rad_nms_mwv",
            # "pubchem_CNOPS7_FClBr_opt",
            "pubchem_CNOPS7_FClBr_nms_mwv",
        ],
        # [
        #     "pubchem_CNOPS8_opt",
        #     "pubchem_CNOPS8_nms",
        #     "pubchem_CNOPS8_rts",
        #     "pubchem_CNOPS8_rad_opt",
        #     "pubchem_CNOPS8_rad_nms",
        #     "pubchem_CNOPS8_FClBr_opt",
        #     # "pubchem_CNOPS8_FClBr_nms",
        # ],
        # [
        #     "gdb11_s00-06_opt",
        #     "gdb11_s00-06_nms",
        #     "gdb11_s00-06_rts",
        #     "gdb11_s00-06_rad_opt",
        #     "gdb11_s00-06_rad_nms",
        #     "gdb11_s00-06_FClBr_opt",
        #     "gdb11_s00-06_FClBr_nms",
        # ],
        # [
        #     "gdb11_s07_opt",
        #     "gdb11_s07_nms",
        #     "gdb11_s07_rts",
        #     "gdb11_s07_rad_opt",
        #     "gdb11_s07_rad_nms",
        #     "gdb11_s07_FClBr_opt",
        #     "gdb11_s07_FClBr_nms",
        # ],
        # [
        #     "gdb11_s08_opt",
        #     "gdb11_s08_nms",
        #     "gdb11_s08_rts",
        #     "gdb11_s08_rad_opt",
        #     "gdb11_s08_rad_nms",
        #     "gdb11_s08_FClBr_opt",
        #     # "gdb11_s08_FClBr_nms",
        # ],
    ]
    # dts_dataset_name = "gdb11_pubchem_CNOPS_0-6_dts"

    filepath = pathlib.Path(args.data_path)

    pdf_pages = PdfPages("test_molecule_vasp_energy_force.pdf")
    figs = plt.figure(figsize=(18.0, 15.0), dpi=100)
    n_page = 0
    n_plot = 1

    for i_list, data_names in enumerate(data_name_list):
        dataset_name_arr = [
            # ("opt", data_names[0]),
            ("nms", data_names[0]),
            ("rts", data_names[1]),
            # ("rad_opt", data_names[3]),
            ("rad_nms", data_names[2]),
            # ("halide_opt", data_names[5]),
        ]
        if len(data_names) >= 4:
            dataset_name_arr.append(("halide_nms", data_names[3]))

        # if i_list == len(data_name_list) - 1:
        #     dataset_name_arr.append(("dts", dts_dataset_name))

        estimate_results = {}
        for dataset_name, dataset_key in dataset_name_arr:
            dataset = h5py.File(filepath / (dataset_key + ".h5"))

            data = []
            dataset_keys_list = dataset["keys"][()]
            np.random.shuffle(dataset_keys_list)

            n_sample = len(dataset_keys_list)
            if args.max_size != -1:
                n_sample = min(n_sample, args.max_size)

            print("Fetching data...")
            for i, (sample_key, sample_idx) in enumerate(
                ProgressBar()(dataset_keys_list)
            ):
                if i >= n_sample:
                    break
                sample_idx = int(sample_idx)
                sample = dataset[sample_key]

                shift = np.sum(shift_energy[sample["species"][()]])

                data.append(
                    {
                        # "name": sample_key.decode()+", "+str(sample_idx),
                        "atomic_numbers": sample["species"],
                        "coordinates": sample["coordinates"][sample_idx],
                        "energies": sample["energies"][sample_idx] - shift,
                        "forces": sample["forces"][sample_idx],
                        "cell": sample["cell"][sample_idx]
                        if "cell" in sample
                        else None,
                    }
                )

            print(f"Calculating: {dataset_key}")
            calculate(estimator, data, args.batchsize, calc=True)
            estimate_results[dataset_name] = data

        for dataset_name, dataset_key in dataset_name_arr:
            # energy
            print("drawing energy")
            draw_energy(
                [r["energies"] for r in estimate_results[dataset_name]],
                [r["p_energies"] for r in estimate_results[dataset_name]],
                figs,
                n_plot,
                title=f"{dataset_key} energy",
            )
            n_plot += 1

            # force
            print("drawing force")
            draw_force(
                [r["forces"] for r in estimate_results[dataset_name]],
                [r["p_forces"] for r in estimate_results[dataset_name]],
                figs,
                n_plot,
                title=f"{dataset_key} force",
            )
            n_plot += 2

            if n_plot >= 3 * 3:
                plt.tight_layout()
                pdf_pages.savefig(figs)
                n_page += 1
                figs.clear()
                n_plot = 1

        atomic_number_total_arr = []
        forces_total = []
        p_forces_total = []
        for dataset_name, _ in dataset_name_arr:
            atomic_number_total_arr.extend(
                [r["atomic_numbers"] for r in estimate_results[dataset_name]]
            )
            forces_total.extend([r["forces"] for r in estimate_results[dataset_name]])
            p_forces_total.extend(
                [r["p_forces"] for r in estimate_results[dataset_name]]
            )
        atomic_number_total = np.concatenate(atomic_number_total_arr, axis=0)
        forces_total = np.concatenate(forces_total, axis=0)
        p_forces_total = np.concatenate(p_forces_total, axis=0)

        for atomic_number in [1, 6, 7, 8, 9, 15, 16, 17, 35]:
            forces_element = forces_total[atomic_number_total == atomic_number]
            p_forces_element = p_forces_total[atomic_number_total == atomic_number]
            draw_force(
                [forces_element],
                [p_forces_element],
                figs,
                n_plot,
                title=f"Element {atomic_number} force",
                draw_abs=False,
            )
            n_plot += 1

            if n_plot >= 3 * 3:
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

    plt.tight_layout()
    pdf_pages.savefig(figs)
    n_page += 1
    figs.clear()
    n_plot = 1

    pdf_pages.close()
    plt.close("all")


if __name__ == "__main__":
    main(parse())
