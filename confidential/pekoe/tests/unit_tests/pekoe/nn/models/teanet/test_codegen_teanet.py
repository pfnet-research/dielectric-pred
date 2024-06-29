import logging
import pathlib
import time

import numpy as np
import pytest
import torch
from pytest import LogCaptureFixture

from pekoe.nn.models.model_builder import build_estimator
from pekoe.nn.models.teanet.codegen_options import (
    _CODEGEN_ARGS_RECOMPUTATION,
    CODEGEN_ARGS,
    CodeGenOptions,
)
from pekoe.utils.dummy_inputs import create_dummy_input


def _contain_recomp_codegen_args(codegen_args: CODEGEN_ARGS) -> None:
    for key, value in _CODEGEN_ARGS_RECOMPUTATION.items():
        if codegen_args.get(key, None) != value:
            assert value is not None
            return False
    return True


@pytest.mark.pfvm
@pytest.mark.pfvm_recomp
def test_pfvm_load_outdir(caplog: LogCaptureFixture, gpuid: str) -> None:
    from pekoe.nn.models.teanet.codegen_teanet import clear_cache_for_testing

    device = f"pfvm:cuda:{gpuid}"
    caplog.set_level(logging.INFO)

    precompile_dir = pathlib.Path("/tmp/codegen_teanet_precompile")
    use_dir = pathlib.Path("/tmp/codegen_teanet_use")

    clear_cache_for_testing()

    codegen_options = CodeGenOptions()
    codegen_options.outdir = precompile_dir
    # For faster recomp.
    codegen_options.codegen_args["recomp_flags"] = ",--target=pekoe"

    estimator_input = create_dummy_input()

    start_time = time.time()
    estimator = build_estimator("v1.4.1", device, codegen_options=codegen_options)
    elapsed_precompile = time.time() - start_time

    expected_output = estimator.estimate(estimator_input)

    apps = list(precompile_dir.glob("teanet_v1_4_1-*/model.app.zst"))
    assert len(apps) == 2

    clear_cache_for_testing()

    codegen_options.outdir = use_dir
    codegen_options.load_outdir = precompile_dir

    start_time = time.time()
    estimator = build_estimator("v1.4.1", device, codegen_options=codegen_options)
    elapsed_use = time.time() - start_time

    # Loading precompiled binaries should achieve significantly faster load.
    assert elapsed_precompile > elapsed_use * 2.0

    actual_output = estimator.estimate(estimator_input)

    np.testing.assert_allclose(expected_output["energy"], actual_output["energy"])

    apps = list(use_dir.glob("teanet_v1_4_1-*/model.app.zst"))
    assert len(apps) == 2


@pytest.mark.pfvm
def test_pad_input() -> None:
    orig_num_batches = 3
    orig_num_nodes = 31
    orig_num_edges = 103
    inputs = {
        "coordinates": torch.zeros(orig_num_nodes, 3),
        "atomic_numbers": torch.zeros(orig_num_nodes, dtype=torch.int64),
        "atom_index1": torch.zeros(orig_num_edges, dtype=torch.int64),
        "atom_index2": torch.zeros(orig_num_edges, dtype=torch.int64),
        "shift": torch.zeros(orig_num_edges, 3, dtype=torch.int64),
        "cell": torch.zeros(orig_num_batches, 3, 3),
        "num_graphs": torch.zeros(orig_num_batches),
        "batch": torch.zeros(orig_num_nodes, dtype=torch.int64),
        "batch_edge": torch.zeros(orig_num_edges, dtype=torch.int64),
        "x_add": torch.zeros(orig_num_nodes, 8),
        "calc_mode_type": torch.zeros(orig_num_nodes, dtype=torch.int64),
    }
    _test_pad_shift_impl(inputs, orig_num_batches, orig_num_nodes, orig_num_edges)

    # 1D `shift` means it is already packed.
    inputs["shift"] = torch.zeros(orig_num_edges, dtype=torch.int64)
    _test_pad_shift_impl(inputs, orig_num_batches, orig_num_nodes, orig_num_edges)


def _test_pad_shift_impl(inputs, orig_num_batches, orig_num_nodes, orig_num_edges):
    from pekoe.nn.models.teanet.codegen_teanet import _pad_inputs

    for padded_num_batches, padded_num_nodes, padded_num_edges in [
        (128, 64, 256),
        (5, 32, 200),
        (4, 32, 103),
    ]:
        outputs = _pad_inputs(
            dict(inputs),
            orig_num_batches,
            padded_num_batches,
            orig_num_nodes,
            padded_num_nodes,
            orig_num_edges,
            padded_num_edges,
        )

        assert outputs["coordinates"].shape == (padded_num_nodes, 3)
        assert outputs["atomic_numbers"].shape == (padded_num_nodes,)
        assert outputs["atom_index1"].shape == (padded_num_edges,)
        assert outputs["atom_index2"].shape == (padded_num_edges,)
        # All padded edges must point padded nodes.
        assert torch.all(outputs["atom_index1"][orig_num_edges:] >= orig_num_nodes)
        assert torch.all(outputs["atom_index2"][orig_num_edges:] >= orig_num_nodes)
        # Note `shift` will be packed to a 1D tensor.
        assert outputs["shift"].shape == (padded_num_edges,)
        assert outputs["cell"].shape == (padded_num_batches, 3, 3)
        assert outputs["num_graphs"].shape == (padded_num_batches,)
        assert outputs["batch"].shape == (padded_num_nodes,)
        assert outputs["batch_edge"].shape == (padded_num_edges,)
        # All padded batch indices must point padded batches.
        assert torch.all(outputs["batch"][orig_num_nodes:] >= orig_num_batches)
        assert torch.all(outputs["batch_edge"][orig_num_edges:] >= orig_num_batches)
        assert outputs["x_add"].shape == (padded_num_nodes, 8)
        assert outputs["calc_mode_type"].shape == (padded_num_nodes,)
