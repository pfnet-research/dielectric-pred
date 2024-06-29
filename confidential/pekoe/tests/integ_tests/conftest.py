import os

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "pekoe: mark test to run with pekoe (pytorch) model")
    config.addinivalue_line("markers", "pfp: mark test to run with pfp (binary) model")


@pytest.fixture(
    params=[
        pytest.param(
            ("pekoe", "cpu", "na"), marks=[pytest.mark.pekoe, pytest.mark.cpu], id="pekoe-cpu"
        ),
        pytest.param(
            ("pekoe", "cuda:0", "na"),
            marks=[pytest.mark.pekoe, pytest.mark.gpu],
            id="pekoe-gpu",
        ),
        pytest.param(
            ("pekoe", "pfvm:cuda:0", "standard"),
            marks=[pytest.mark.pekoe, pytest.mark.pfvm],
            id="pekoe-pfvm-standard",
        ),
        pytest.param(
            ("pekoe", "pfvm:cuda:0", "recomp"),
            marks=[pytest.mark.pekoe, pytest.mark.pfvm_recomp],
            id="pekoe-pfvm-recomp",
        ),
        pytest.param(
            ("pekoe", "mncore:auto", "na"),
            marks=[pytest.mark.pekoe, pytest.mark.mncore],
            id="pekoe-mncore",
        ),
        pytest.param(("pfp", 0, "na"), marks=[pytest.mark.pfp, pytest.mark.gpu], id="pfp-gpu"),
    ]
)
def module_config(request):
    return request.param


def build_estimator(module, device, pytestconfig, codegen_options: str):
    if module == "pfp":
        from pfp.nn.models.crystal.model_builder import build_estimator

        return build_estimator(device)

    elif module == "pekoe":
        from pekoe.nn.models import DEFAULT_MODEL
        from pekoe.nn.models.model_builder import build_estimator

        if device.startswith("pfvm"):
            load_outdir = os.environ.get("PFVM_BINARY_DIR")
            assert load_outdir is not None

            from pekoe.nn.models.teanet.codegen_options import CodeGenOptions

            if codegen_options == "recomp":
                return build_estimator(
                    DEFAULT_MODEL,
                    device=device,
                    codegen_options=CodeGenOptions(
                        use_recomp_min_num_nodes=0, load_outdir=load_outdir
                    ),
                )
            else:
                return build_estimator(
                    DEFAULT_MODEL,
                    device=device,
                    codegen_options=CodeGenOptions(load_outdir=load_outdir),
                )
        elif device.startswith("mncore"):
            load_outdir = os.environ.get("MNCORE_BINARY_DIR")
            assert load_outdir is not None

            from pekoe.nn.models.teanet.codegen_options import CodeGenOptions, MNCoreOptions

            codegen_options = CodeGenOptions(
                load_outdir=load_outdir,
                skip_precompile=True,
                skip_precompile_recomp=True,
                mncore_options=MNCoreOptions(
                    pad_edge=True, pad_node=True, allow_missing_teanet_conv_helper=True
                ),
            )
            return build_estimator(
                DEFAULT_MODEL,
                device=device,
                codegen_options=codegen_options,
            )
        else:
            return build_estimator(DEFAULT_MODEL, device=device)

    else:
        raise ValueError("Module {module} is not supported.")


@pytest.fixture
def estimator(module_config, pytestconfig):
    module, device, codegen_options = module_config
    estimator = build_estimator(module, device, pytestconfig, codegen_options)
    return estimator


@pytest.fixture
def calculator(module_config, estimator, calc_mode):
    module, _, _ = module_config
    estimator.set_calc_mode(calc_mode)
    if module == "pfp":
        from pfp.calculators.ase_calculator import ASECalculator

        return ASECalculator(estimator)
    elif module == "pekoe":
        from pekoe.calculators.ase_calculator import ASECalculator

        return ASECalculator(estimator)
    else:
        raise ValueError("Module {module} is not supported.")
