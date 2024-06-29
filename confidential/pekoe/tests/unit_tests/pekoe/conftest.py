import os
from typing import Optional, Tuple

import pytest

from pekoe.nn.models.teanet.codegen_options import CodeGenOptions, MNCoreOptions


@pytest.fixture(
    params=[
        pytest.param(["cpu", "na"], marks=[pytest.mark.cpu], id="cpu"),
        pytest.param(["gpu", "na"], marks=[pytest.mark.gpu], id="gpu"),
        pytest.param(["pfvm", "standard"], marks=[pytest.mark.pfvm], id="pfvm-standard"),
        pytest.param(["pfvm", "recomp"], marks=[pytest.mark.pfvm_recomp], id="pfvm-recomp"),
        pytest.param(["mncore", "na"], marks=[pytest.mark.mncore], id="mncore"),
    ]
)
def device_config(request, gpuid) -> Tuple[str, Optional[CodeGenOptions]]:
    device, codegen_options = request.param
    if device == "cpu":
        return "cpu", None
    elif device == "pfvm":
        load_outdir = os.environ.get("PFVM_BINARY_DIR")
        assert load_outdir is not None
        if codegen_options == "recomp":
            return f"pfvm:cuda:{gpuid}", CodeGenOptions(
                use_recomp_min_num_nodes=0, load_outdir=load_outdir
            )
        else:
            return f"pfvm:cuda:{gpuid}", CodeGenOptions(load_outdir=load_outdir)
    elif device == "mncore":
        load_outdir = os.environ.get("MNCORE_BINARY_DIR")
        assert load_outdir is not None
        return "mncore:auto", CodeGenOptions(
            load_outdir=load_outdir,
            skip_precompile=True,
            skip_precompile_recomp=True,
            mncore_options=MNCoreOptions(
                pad_edge=True, pad_node=True, allow_missing_teanet_conv_helper=True
            ),
        )
    else:
        return f"cuda:{gpuid}", None
