import argparse
import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

from pekoe.nn.models.model_builder import build_estimator
from pekoe.nn.models.teanet.codegen_options import CodeGenOptions, MNCoreOptions
from pekoe.nn.models.teanet.codegen_teanet import SUPPORTED_MODEL_VERSIONS

logger = logging.getLogger(__name__)


# NOTE: any large enough values are enough
DUMMY_PFVM_USE_RECOMP_MIN_NUM_EDGES = 14000
DUMMY_PFVM_USE_RECOMP_MIN_NUM_NODES = 140000


def build_pfvm_estimator(
    version_or_config_path: Union[Path, str],
    pfvm_binary_dir: Path,
    codegen_args: Dict[str, Union[bool, int, float, str]],
) -> None:
    codegen_options = CodeGenOptions(
        codegen_args=codegen_args,
        use_recomp_min_num_nodes=DUMMY_PFVM_USE_RECOMP_MIN_NUM_NODES,
        use_recomp_min_num_edges=DUMMY_PFVM_USE_RECOMP_MIN_NUM_EDGES,
        outdir=pfvm_binary_dir,
    )
    build_estimator(
        version_or_config_path=version_or_config_path,
        device="pfvm:cuda",
        max_atoms=None,
        max_neighbors=None,
        codegen_options=codegen_options,
    )


# Only for testing because only node<64 and edge<2048 is supported.
def build_mncore_estimator_for_testing(
    version_or_config_path: Union[Path, str],
    mncore_binary_dir: Path,
    codegen_args: Dict[str, Union[bool, int, float, str]],
) -> None:
    # TODO(hamaji): Use "fast_double" instead of the vanila double.
    codegen_args["float_dtype"] = "double"
    mncore_options = MNCoreOptions(
        pad_edge=True,
        pad_node=True,
        force_num_edges=2048,
        force_num_nodes=64,
    )
    codegen_options = CodeGenOptions(
        codegen_args=codegen_args,
        outdir=mncore_binary_dir,
        skip_precompile_recomp=True,
        mncore_options=mncore_options,
    )
    build_estimator(
        version_or_config_path=version_or_config_path,
        device="mncore:auto",
        max_atoms=None,
        max_neighbors=None,
        codegen_options=codegen_options,
    )


def main(
    output_dir: Path,
    max_workers: int,
    pekoe_model_versions: Sequence[object],
    mncore: bool,
    option_json: Optional[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Compiling %s models: %s", len(pekoe_model_versions), pekoe_model_versions)
    start_time = time.time()

    codegen_args: Dict[str, Union[bool, int, float, str]] = dict()
    if option_json is not None:
        codegen_args["option_json"] = option_json

    if mncore:
        _build_estimator = partial(
            build_mncore_estimator_for_testing,
            mncore_binary_dir=output_dir,
            codegen_args=codegen_args,
        )
    else:
        _build_estimator = partial(
            build_pfvm_estimator,
            pfvm_binary_dir=output_dir,
            codegen_args=codegen_args,
        )
    # NOTE: pfvm compilation during the initalization of estimator
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_build_estimator, pekoe_model_version)
            for pekoe_model_version in pekoe_model_versions
        ]
        [future.result() for future in futures]

    elapsed_time = time.time() - start_time
    logger.info("Compilation ends in %s s", elapsed_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", help="a directory to store pfvm binaries", type=Path)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--mncore", action="store_true", default=False)
    parser.add_argument("--option_json")
    args = parser.parse_args()
    # NOTE: required for ProcessPoolExecutor
    multiprocessing.set_start_method("spawn")

    logging.basicConfig(
        level=logging.INFO,
    )
    output_dir = args.output_dir
    # TODO(hamaji): Compile more models for MN-Core.
    pekoe_model_versions = ["v1.4.1"] if args.mncore else SUPPORTED_MODEL_VERSIONS
    main(
        output_dir=output_dir,
        max_workers=args.max_workers,
        pekoe_model_versions=pekoe_model_versions,
        mncore=args.mncore,
        option_json=args.option_json,
    )

    num_pekoe_model_versions = len(pekoe_model_versions)
    if args.mncore:
        assert len(list(output_dir.iterdir())) == num_pekoe_model_versions
    else:
        # NOTE: There should be two output dir (standard and recomp) per model.
        assert len(list(output_dir.iterdir())) == num_pekoe_model_versions * 2
    for model_dir in output_dir.iterdir():
        app_file = model_dir / "model.app.zst"
        assert app_file.exists()
        onnx_file = model_dir / "model.onnx"
        assert onnx_file.exists()
        output_signature = model_dir / "output_signature.pkl"
        assert output_signature.exists()
