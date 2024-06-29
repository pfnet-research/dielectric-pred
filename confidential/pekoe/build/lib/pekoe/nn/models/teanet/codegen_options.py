import pathlib
from dataclasses import dataclass, field
from typing import Dict, Optional, Union

DEFAULT_USE_RECOMP_MIN_NUM_NODES = 10000000
DEFAULT_USE_RECOMP_MIN_NUM_EDGES = 10000000

CODEGEN_ARGS = Dict[str, Union[bool, int, float, str]]


@dataclass
class MNCoreOptions:
    pad_edge: bool = False
    pad_node: bool = False
    float_dtype: Optional[str] = None
    force_num_edges: int = 0
    force_num_nodes: int = 0
    allow_missing_teanet_conv_helper: bool = False


@dataclass
class CodeGenOptions:
    codegen_args: CODEGEN_ARGS = field(default_factory=dict)
    use_recomp_min_num_nodes: int = DEFAULT_USE_RECOMP_MIN_NUM_NODES
    use_recomp_min_num_edges: int = DEFAULT_USE_RECOMP_MIN_NUM_EDGES

    def get_codegen_args(
        self, codegen_args: CODEGEN_ARGS, num_nodes: int, num_edges: int, use_always_recomp: bool
    ) -> CODEGEN_ARGS:
        codegen_args = codegen_args.copy()
        if self.use_recomp(num_nodes, num_edges, use_always_recomp=use_always_recomp):
            for key, value in _CODEGEN_ARGS_RECOMPUTATION.items():
                if key not in codegen_args:
                    codegen_args[key] = value
        return codegen_args

    def use_recomp(self, num_nodes: int, num_edges: int, use_always_recomp: bool) -> bool:
        return use_always_recomp or (
            num_nodes >= self.use_recomp_min_num_nodes
            or num_edges >= self.use_recomp_min_num_edges
        )

    outdir: Optional[pathlib.Path] = None
    load_outdir: Optional[pathlib.Path] = None

    enable_debug: bool = False

    skip_precompile: bool = False
    skip_precompile_recomp: bool = False

    mncore_options: MNCoreOptions = field(default_factory=MNCoreOptions)


_CODEGEN_ARGS_RECOMPUTATION = {
    "use_recomp": True,
    "estimate_dim_params": "num_batches=1,num_nodes=10000,others=1000000",
    "recomp_flags": ",--target=pekoe",
}
