import pytest

from pekoe.nn.models.teanet.codegen_options import (
    _CODEGEN_ARGS_RECOMPUTATION,
    CODEGEN_ARGS,
    CodeGenOptions,
)

_USE_RECOMP_MIN_NUM_NODES = 5
_USE_RECOMP_MIN_NUM_EDGES = 50


@pytest.mark.pfvm
@pytest.mark.pfvm_recomp
def test_use_recomp() -> None:
    codegen_options = CodeGenOptions(
        codegen_args={},
        use_recomp_min_num_nodes=_USE_RECOMP_MIN_NUM_NODES,
        use_recomp_min_num_edges=_USE_RECOMP_MIN_NUM_EDGES,
    )
    assert codegen_options.use_recomp(
        num_nodes=_USE_RECOMP_MIN_NUM_NODES, num_edges=1, use_always_recomp=False
    )
    assert codegen_options.use_recomp(
        num_nodes=1, num_edges=_USE_RECOMP_MIN_NUM_EDGES, use_always_recomp=False
    )
    assert codegen_options.use_recomp(
        num_nodes=_USE_RECOMP_MIN_NUM_NODES,
        num_edges=_USE_RECOMP_MIN_NUM_EDGES,
        use_always_recomp=False,
    )
    assert codegen_options.use_recomp(
        num_nodes=_USE_RECOMP_MIN_NUM_NODES + 1, num_edges=1, use_always_recomp=False
    )
    assert codegen_options.use_recomp(
        num_nodes=1, num_edges=_USE_RECOMP_MIN_NUM_EDGES + 1, use_always_recomp=False
    )
    assert codegen_options.use_recomp(
        num_nodes=_USE_RECOMP_MIN_NUM_NODES + 1,
        num_edges=_USE_RECOMP_MIN_NUM_EDGES + 1,
        use_always_recomp=False,
    )
    # NOTE: Since both num_nodes and num_edges are less than the thresholds, recomp is disabled.
    assert not codegen_options.use_recomp(num_nodes=1, num_edges=1, use_always_recomp=False)

    assert codegen_options.use_recomp(
        num_nodes=_USE_RECOMP_MIN_NUM_NODES, num_edges=1, use_always_recomp=True
    )
    assert codegen_options.use_recomp(
        num_nodes=1, num_edges=_USE_RECOMP_MIN_NUM_EDGES, use_always_recomp=True
    )
    assert codegen_options.use_recomp(num_nodes=1, num_edges=1, use_always_recomp=True)


def _contain_recomp_codegen_args(codegen_args: CODEGEN_ARGS) -> None:
    for key, value in _CODEGEN_ARGS_RECOMPUTATION.items():
        if codegen_args.get(key, None) != value:
            assert value is not None
            return False
    return True


@pytest.mark.pfvm
@pytest.mark.pfvm_recomp
def test_get_codegen_args() -> None:
    codegen_options = CodeGenOptions(
        codegen_args={},
        use_recomp_min_num_nodes=_USE_RECOMP_MIN_NUM_NODES,
        use_recomp_min_num_edges=_USE_RECOMP_MIN_NUM_EDGES,
    )
    assert _contain_recomp_codegen_args(
        codegen_options.get_codegen_args(
            codegen_args={},
            num_nodes=_USE_RECOMP_MIN_NUM_NODES,
            num_edges=1,
            use_always_recomp=False,
        )
    )
    assert _contain_recomp_codegen_args(
        codegen_options.get_codegen_args(
            codegen_args={},
            num_nodes=1,
            num_edges=_USE_RECOMP_MIN_NUM_EDGES,
            use_always_recomp=False,
        )
    )
    assert _contain_recomp_codegen_args(
        codegen_options.get_codegen_args(
            codegen_args={},
            num_nodes=_USE_RECOMP_MIN_NUM_NODES,
            num_edges=_USE_RECOMP_MIN_NUM_EDGES,
            use_always_recomp=False,
        )
    )
    assert _contain_recomp_codegen_args(
        codegen_options.get_codegen_args(
            codegen_args={},
            num_nodes=_USE_RECOMP_MIN_NUM_NODES + 1,
            num_edges=1,
            use_always_recomp=False,
        )
    )
    assert _contain_recomp_codegen_args(
        codegen_options.get_codegen_args(
            codegen_args={},
            num_nodes=1,
            num_edges=_USE_RECOMP_MIN_NUM_EDGES + 1,
            use_always_recomp=False,
        )
    )
    assert _contain_recomp_codegen_args(
        codegen_options.get_codegen_args(
            codegen_args={},
            num_nodes=_USE_RECOMP_MIN_NUM_NODES + 1,
            num_edges=_USE_RECOMP_MIN_NUM_EDGES + 1,
            use_always_recomp=False,
        )
    )
    # NOTE: Since both num_nodes and num_edges are less than the thresholds, recomp is disabled.
    assert not _contain_recomp_codegen_args(
        codegen_options.get_codegen_args(
            codegen_args={}, num_nodes=1, num_edges=1, use_always_recomp=False
        )
    )

    assert _contain_recomp_codegen_args(
        codegen_options.get_codegen_args(
            codegen_args={},
            num_nodes=_USE_RECOMP_MIN_NUM_NODES,
            num_edges=1,
            use_always_recomp=True,
        )
    )
    assert _contain_recomp_codegen_args(
        codegen_options.get_codegen_args(
            codegen_args={},
            num_nodes=1,
            num_edges=_USE_RECOMP_MIN_NUM_EDGES,
            use_always_recomp=True,
        )
    )
    assert _contain_recomp_codegen_args(
        codegen_options.get_codegen_args(
            codegen_args={}, num_nodes=1, num_edges=1, use_always_recomp=True
        )
    )
