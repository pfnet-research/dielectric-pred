import argparse
import json
import os
import pathlib
import shutil
import time

import numpy as np
from ase import Atoms, units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet
from ase.optimize import BFGS

from pekoe.calculators.ase_calculator import ASECalculator
from pekoe.nn.estimator_base import EstimatorCalcMode
from pekoe.nn.models import DEFAULT_MODEL
from pekoe.nn.models.model_builder import build_estimator
from pekoe.nn.models.teanet.codegen_options import CodeGenOptions, MNCoreOptions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_MODEL),
        help="path to configuration yaml",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='device (default: "auto" which uses cuda device is available)',
    )
    parser.add_argument("--book-keeping", action="store_true", help="Enable book-keeping feature")
    parser.add_argument(
        "--calc-mode",
        type=EstimatorCalcMode,
        default=EstimatorCalcMode.CRYSTAL,
        help="Calculation mode",
    )
    parser.add_argument(
        "--pbc", action="store_true", help="Calculate under periodic-boundary condition"
    )
    parser.add_argument("--large-md", type=int, help="Multiply system for MD calculation")
    parser.add_argument(
        "--output-json",
        action="store_true",
        help="Output json files (can be used for tests)",
    )
    parser.add_argument(
        "--output-onnx",
        help="Outputs onnx with test data to directory",
    )
    parser.add_argument(
        "--output-onnx-only",
        action="store_true",
        help="Output onnx and exits immediately",
    )
    parser.add_argument(
        "--use_recomp_min_num_nodes",
        type=int,
        default=-1,
        help="minimum number of atoms to use recomputation mode",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Steps to run MD",
    )
    parser.add_argument(
        "--opt_steps",
        type=int,
        default=1000,
        help="Steps to run optimizer",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=0.001,
        help="fmax for optimizer",
    )
    args = parser.parse_args()
    device = args.device

    # create estimator and calculator:
    config_path = pathlib.Path(args.config)
    codegen_options = None
    if device.startswith("pfvm:"):
        load_outdir = os.environ.get("PFVM_BINARY_DIR")
        if args.use_recomp_min_num_nodes >= 0:  # NOTE: avoiding an error at the binary version
            codegen_options = CodeGenOptions(
                use_recomp_min_num_nodes=args.use_recomp_min_num_nodes, load_outdir=load_outdir
            )
        else:
            codegen_options = CodeGenOptions(load_outdir=load_outdir)
    if device.startswith("mncore:"):
        codegen_args = {
            "option_json": os.path.join(os.path.dirname(__file__), "test_mncore_config.json"),
        }
        codegen_options = CodeGenOptions(
            skip_precompile=True,
            skip_precompile_recomp=True,
            codegen_args=codegen_args,
            mncore_options=MNCoreOptions(pad_node=True, allow_missing_teanet_conv_helper=True),
        )
    estimator = build_estimator(
        config_path,
        device=device,
        output_onnx=args.output_onnx,
        codegen_options=codegen_options,
    )
    if args.book_keeping:
        estimator.set_book_keeping(True, 2.0)
    estimator.set_calc_mode(args.calc_mode)
    calculator = ASECalculator(estimator)

    # set up a crystal
    atoms = Atoms(
        "SON2C9H8",
        [
            [5.481891e00, -2.889420e-01, -4.510000e-04],
            [-3.706765e00, -1.605052e00, 2.049000e-03],
            [-2.905558e00, 6.122530e-01, -4.510000e-04],
            [3.784449e00, -1.067110e-01, 4.900000e-05],
            [-1.518376e00, 4.630520e-01, -5.510000e-04],
            [-9.540000e-01, -8.124600e-01, -1.051000e-03],
            [-6.959750e-01, 1.589554e00, 5.490000e-04],
            [4.329800e-01, -9.615650e-01, -3.510000e-04],
            [6.909990e-01, 1.440649e00, 1.049000e-03],
            [1.255378e00, 1.650370e-01, 5.490000e-04],
            [-3.883689e00, -3.911680e-01, -7.510000e-04],
            [-5.277011e00, 1.860060e-01, -4.510000e-04],
            [2.676251e00, 1.232600e-02, 2.490000e-04],
            [-1.524821e00, -1.730502e00, -2.151000e-03],
            [-1.122155e00, 2.589732e00, 1.049000e-03],
            [-3.238813e00, 1.574100e00, -2.510000e-04],
            [8.594640e-01, -1.961936e00, -8.510000e-04],
            [1.319151e00, 2.328228e00, 1.649000e-03],
            [-5.432002e00, 7.763330e-01, 9.069490e-01],
            [-6.013055e00, -6.221830e-01, -2.915100e-02],
            [-5.415477e00, 8.223330e-01, -8.790510e-01],
        ],
        pbc=args.pbc,
        cell=[100, 100, 100],
    )
    print(len(atoms), "atoms in the cell")
    atoms.calc = calculator
    if args.output_json:
        init_results = {
            "energy": atoms.get_total_energy(),
            "charges": atoms.get_charges().tolist(),
            "forces": atoms.get_forces().tolist(),
            "stress": atoms.get_stress().tolist(),
            "calc_stats": atoms.calc.results["calc_stats"],
        }
        json.dump(
            init_results, open("mol_properties_{}.json".format(str(args.calc_mode)), "w"), indent=2
        )

    print("Begin minimizing...")
    opt = BFGS(atoms)
    start_time = time.time()

    if args.output_onnx:
        print("Exporting ONNX to", args.output_onnx)
        shutil.rmtree(args.output_onnx, ignore_errors=True)

    if args.output_onnx_only:
        assert args.output_onnx is not None
        opt.run(steps=1)
        return

    opt_result = opt.run(fmax=args.fmax, steps=args.opt_steps)
    print("OPT elapsed : {}".format(time.time() - start_time))
    if not device.startswith("mncore"):
        assert opt_result, "OPT did not converge"

    if args.output_json:
        opt_results = {
            "energy": atoms.get_total_energy(),
            "charges": atoms.get_charges().tolist(),
            "forces": atoms.get_forces().tolist(),
            "stress": atoms.get_stress().tolist(),
            "calc_stats": atoms.calc.results["calc_stats"],
        }
        json.dump(
            opt_results, open("opt_results_{}.json".format(str(args.calc_mode)), "w"), indent=2
        )

    if args.large_md is not None:
        atoms = atoms.repeat(args.large_md)
        # Reset to redump ONNX
        estimator.output_onnx_count = 0
        atoms.calc = calculator

    def printenergy(a: Atoms = atoms) -> None:
        epot = a.get_potential_energy() / len(a)
        ekin = a.get_kinetic_energy() / len(a)
        with open("energy.out", "a") as fw:
            fw.write("{:.8f}\n".format(float(epot + ekin)))
        print(
            "Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  "
            "Etot = %.8feV" % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin)
        )

    np.random.seed(123456)
    MaxwellBoltzmannDistribution(atoms, 500.0 * units.kB)
    Stationary(atoms)
    dyn = VelocityVerlet(atoms, 0.2 * units.fs)
    dyn.attach(printenergy, interval=5)

    print("Beginning dynamics...")
    printenergy(atoms)
    start_time = time.time()
    dyn.run(args.steps)
    print("MD elapsed : {}".format(time.time() - start_time))


if __name__ == "__main__":
    main()
