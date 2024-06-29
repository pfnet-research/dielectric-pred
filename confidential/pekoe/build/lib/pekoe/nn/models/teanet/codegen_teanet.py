import argparse
import glob
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from logging import getLogger
from typing import Dict, Optional, Tuple, Union

import torch
from codegen.utils import op_registry, storage
from mncore import runtime_core
from mncore.mndevice import get_device
from mncore.utils import compiler_flags, perfetto_trace

from pekoe.nn.estimator_base import AtomsTooManyError, NeighborsTooManyError
from pekoe.nn.models import EDGE_FULL_MODEL
from pekoe.nn.models.teanet.codegen_options import CodeGenOptions
from pekoe.nn.models.teanet.teanet_base import TeaNetBase
from pekoe.utils.dummy_inputs import DUMMY_INPUT

logger = getLogger(__name__)

SUPPORTED_MODEL_VERSIONS = (
    "v0.10.0",
    "v1.0.0",
    "v1.1.0",
    "v1.2.0",
    "v1.2.1",
    "v1.2.2",
    "v1.3.0",
    "v1.3.1",
    "v1.4.0",
    "v1.4.1",
    EDGE_FULL_MODEL,
)
_max_shift = 512


def _pack_shift(shift: torch.Tensor) -> torch.Tensor:
    if shift.ndim == 1:
        # Already packed by C++ preprocessor.
        return shift
    assert shift.ndim == 2
    assert shift.size(1) == 3
    assert shift.max() < _max_shift
    assert shift.min() >= -_max_shift
    s = shift + _max_shift
    r = _max_shift * 2
    return s[:, 0] + r * (s[:, 1] + r * s[:, 2])


def _unpack_shift(packed: torch.Tensor) -> torch.Tensor:
    assert packed.ndim == 1
    r = _max_shift * 2
    s0 = packed % r
    sr = packed // r
    s1 = sr % r
    s2 = sr // r
    return torch.stack((s0, s1, s2), 1) - _max_shift


@dataclass(order=True)
class AppInfo:
    # Note the order of the fields are important as we try to find the fastest
    # app by iterating the sorted list of apps.
    expected_msec: float
    num_nodes: int
    num_edges: int
    key: str


class _TeaNetWrapper(torch.nn.Module):
    """A wrapper for TeaNet models to be exported as an ONNX."""

    def __init__(
        self,
        model: TeaNetBase,
        model_dtype: torch.dtype = torch.float32,
        use_packed_shift=False,
    ):
        super(_TeaNetWrapper, self).__init__()
        self.model = model
        self.model_dtype = model_dtype
        self.use_packed_shift = use_packed_shift

    def forward(
        self,
        coordinates: torch.Tensor,
        atomic_numbers: torch.Tensor,
        atom_index1: torch.Tensor,
        atom_index2: torch.Tensor,
        shift: torch.Tensor,
        cell: torch.Tensor,
        num_graphs: torch.Tensor,
        batch: torch.Tensor,
        batch_edge: torch.Tensor,
        x_add: torch.Tensor,
        calc_mode_type: torch.Tensor,
        num_edges: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.use_packed_shift:
            shift = _unpack_shift(shift)

        if num_edges is not None:
            mask = torch.arange(atom_index1.shape[0]) < num_edges
            atom_index1.mask = mask
            atom_index2.mask = mask
            batch_edge.mask = mask

        edge_vector = (
            coordinates[atom_index1]
            - coordinates[atom_index2]
            - torch.bmm(shift.unsqueeze(1).to(coordinates.dtype), cell[batch_edge]).squeeze(1)
        )
        energy, charge = self.model(
            edge_vector.to(self.model_dtype),
            atomic_numbers,
            atom_index1,
            atom_index2,
            num_graphs,
            batch,
            batch_edge,
            x_add,
            calc_mode_type=calc_mode_type,
        )

        # We output `edge_vector` so that the compiler can find it for
        # computation of virial.
        results = {
            "energy": energy,
            "charge": charge,
            "edge_vector": edge_vector,
        }
        return results


def _get_expected_msec(report_json):
    if not os.path.exists(report_json):
        return -1

    with open(report_json) as f:
        report = json.load(f)
    if "emulated_result" not in report:
        return -1
    if "total_cycles" not in report["emulated_result"]:
        return -1
    if "core_freq" not in report:
        return -1

    # Note `core_freq` is in MHz.
    return report["emulated_result"]["total_cycles"] / report["core_freq"] / 1000


_is_op_registered = False

_padded_num_batches = 128

_compiled_models = {}

_context_cache = {}

_context_lock = threading.Lock()


def _get_codegen_context(codegen_device: str) -> Optional[runtime_core.Context]:
    if codegen_device is None:
        return None

    with _context_lock:
        if codegen_device in _context_cache:
            return _context_cache[codegen_device]

        context = runtime_core.Context(codegen_device)
        # To fetch grad_out@coordinates to compute forces.
        context._fetch_params = True
        exclude_outputs = {"edge_vector", "grad_out@edge_vector"}
        context.exclude_outputs(exclude_outputs)
        _context_cache[codegen_device] = context
        return context


def clear_cache_for_testing():
    _compiled_models.clear()
    _context_cache.clear()


def _add_codegen_args_from_env(
    codegen_args: Union[bool, int, float, str]
) -> Union[bool, int, float, str]:
    env_args = os.environ.get("CODEGEN_ARGS")
    if env_args is None:
        argv = []
    else:
        argv = env_args.split(" ")
    parser = argparse.ArgumentParser()
    compiler_flags.add_codegen_compiler_flags(parser)
    args = parser.parse_args(argv)
    env_codegen_args = compiler_flags.build_codegen_compiler_flags(args)
    env_codegen_args.update(codegen_args)
    return env_codegen_args


def _pad_the_first_dimension(
    tensor: torch.Tensor, padded_size: int, pad_value: int
) -> torch.Tensor:
    pad = [0] * (len(tensor.shape) * 2 - 1) + [padded_size - tensor.shape[0]]
    return torch.nn.functional.pad(tensor, pad, value=pad_value)


def _pad_inputs(
    inputs: Dict[str, torch.Tensor],
    orig_num_batches: int,
    padded_num_batches: int,
    orig_num_nodes: int,
    padded_num_nodes: int,
    orig_num_edges: int,
    padded_num_edges: int,
) -> Dict[str, torch.Tensor]:
    if orig_num_batches >= padded_num_batches:
        raise RuntimeError(f"Too many batches: {orig_num_batches}")
    batch_pad_size = padded_num_batches - orig_num_batches

    num_graphs = inputs["num_graphs"]
    num_graphs_pad = torch.ones(
        (batch_pad_size,), dtype=num_graphs.dtype, device=num_graphs.device
    )
    num_graphs = torch.cat((num_graphs, num_graphs_pad))
    inputs["num_graphs"] = num_graphs

    cell = inputs["cell"]
    cell_pad = (
        torch.eye(3, dtype=cell.dtype, device=cell.device)
        .unsqueeze(0)
        .expand(batch_pad_size, 3, 3)
    )
    cell = torch.cat((cell, cell_pad))
    inputs["cell"] = cell

    inputs["num_edges"] = torch.tensor(orig_num_edges)
    dummy_node_index = orig_num_nodes
    dummy_batch_index = orig_num_batches
    for key, pad_value in [
        ("atom_index1", dummy_node_index),
        ("atom_index2", dummy_node_index),
        ("shift", 0),
        ("batch_edge", dummy_batch_index),
    ]:
        inputs[key] = _pad_the_first_dimension(inputs[key], padded_num_edges, pad_value)

    inputs["shift"] = _pack_shift(inputs["shift"])

    for key, pad_value in [
        ("coordinates", 0),
        ("atomic_numbers", 1),
        ("batch", dummy_batch_index),
        ("x_add", 0),
        ("calc_mode_type", 0),
    ]:
        inputs[key] = _pad_the_first_dimension(inputs[key], padded_num_nodes, pad_value)

    return inputs


class CodeGenTeaNet(TeaNetBase):
    """Manages compiled TeaNet models."""

    def __init__(
        self,
        config_name: str,
        model: TeaNetBase,
        device: str,
        codegen_options: CodeGenOptions,
        model_dtype: torch.dtype = torch.float32,
    ):
        super(CodeGenTeaNet, self).__init__()
        self.is_codegen = True
        self.config_name = config_name
        self.pad_edge = codegen_options.mncore_options.pad_edge
        self.pad_node = codegen_options.mncore_options.pad_node
        self.float_dtype = codegen_options.mncore_options.float_dtype
        self.force_num_edges = codegen_options.mncore_options.force_num_edges
        self.force_num_nodes = codegen_options.mncore_options.force_num_nodes
        if self.pad_node:
            assert self.pad_edge, "`pad_node` without `pad_edge` does not make sense"
        self.model = _TeaNetWrapper(model, model_dtype, self.pad_edge)
        self.coord_dtype = model.coord_dtype
        if device.startswith("torch:"):
            self.codegen_device = None
            model.to(device[len("torch:") :])
        else:
            self.codegen_device = get_device(device)
        self.device = self.torch_device_from_str(device)
        self.compiled_models = _compiled_models
        self.codegen_options = codegen_options
        self.codegen_args = codegen_options.codegen_args
        self.codegen_args["backprop"] = True
        self.codegen_args["backprop_from"] = "energy"
        self.codegen_args["backprop_to"] = "coordinates,edge_vector"
        self.codegen_args["compute_virial"] = True
        self.codegen_args["onnx_scheduler"] = "naive"

        if "cuda" in device or "cuda" in os.getenv("CODEGEN_PFVM_DEVICE_OVERWRITE", ""):
            self.codegen_args["use_cuda"] = True
        self.outdir = codegen_options.outdir
        self.load_outdir = codegen_options.load_outdir
        self.cutoff_list = model.cutoff_list

        global _is_op_registered
        if not _is_op_registered:
            _is_op_registered = True
            op_registry.register_ops()

        if self.pad_edge:
            model.use_mncore()

        self.context = _get_codegen_context(self.codegen_device)

        if not codegen_options.skip_precompile:
            self._codegen_precompile()

        self._apps_info = None
        if self.load_outdir is not None and self.pad_edge:
            assert self.pad_node, "pad_edge + load_outdir is invalid without pad_node"
            self._load_apps_info()

            self._preload_apps()

    def _codegen_precompile(self) -> None:
        inputs = DUMMY_INPUT.get_codegen_input(self.coord_dtype)

        if self.pad_edge:
            assert self.force_num_edges > 0, "Precompilation for MN-Core needs force_num_edges"
            assert self.force_num_nodes > 0, "Precompilation for MN-Core needs force_num_nodes"
            inputs = _pad_inputs(
                inputs,
                inputs["num_graphs"].size(0),
                _padded_num_batches,
                inputs["coordinates"].size(0),
                self.force_num_nodes,
                inputs["atom_index1"].size(0),
                self.force_num_edges,
            )

        self._get_compiled_model(inputs)
        if not self.codegen_options.skip_precompile_recomp:
            self._get_compiled_model(inputs, use_always_recomp=True)
        logger.info(f"Precompile done for {self.codegen_device}.")

    def _load_apps_info(self) -> None:
        self._apps_info = []
        for app in glob.glob(
            os.path.join(self.load_outdir, f"teanet_{self.config_name}_*/model.app.zst")
        ):
            app_dir = os.path.dirname(app)
            expected_msec = _get_expected_msec(os.path.join(app_dir, "report.json"))
            key = os.path.basename(app_dir)
            float_dtype, num_nodes, num_edges = self._parse_cache_key(key)
            if float_dtype != self.float_dtype:
                continue
            logger.info(
                f"App: expected_msec={expected_msec} num_nodes={num_nodes} num_edge={num_edges} key={key}"
            )
            self._apps_info.append(AppInfo(expected_msec, num_nodes, num_edges, key))
        self._apps_info.sort()
        if not self._apps_info:
            raise RuntimeError("No apps info is loaded")

    def _preload_apps(self) -> None:
        rtols = {
            "energy": 1e-4,
            "charge": 1e-4,
            "forces": 2e-3,
            "virial": 2e-3,
        }

        start = time.time()
        inputs = DUMMY_INPUT.get_codegen_input(self.coord_dtype)
        executor = ThreadPoolExecutor(10)
        futures = []
        for app in self._apps_info:
            futures.append(
                (
                    app,
                    executor.submit(
                        self.forward,
                        force_num_nodes=app.num_nodes,
                        force_num_edges=app.num_edges,
                        **inputs,
                    ),
                )
            )

        first_outputs = None
        error_app_names = set()
        for app, future in futures:
            energy, forces, virial, charge = future.result()

            outputs = {
                "energy": energy,
                "forces": forces,
                "virial": virial,
                "charge": charge,
            }

            result_dir = os.path.join("/tmp/preload_results", app.key)
            os.makedirs(result_dir, exist_ok=True)
            for k, a in outputs.items():
                torch.save(a, os.path.join(result_dir, f"{k}.pt"))

            if first_outputs is None:
                first_outputs = outputs
                continue

            for k, a in outputs.items():
                e = first_outputs[k]
                if not torch.allclose(e, a, rtol=rtols[k], atol=1e-7):
                    logger.error(
                        f"Discrepancy detected between {self._apps_info[0].key} and {app.key}"
                    )
                    logger.error(f"  {self._apps_info[0].key} {k}: {e}")
                    logger.error(f"  {app.key} {k}: {a}")
                    error_app_names.add(app.key)

        elapsed = time.time() - start
        logger.info(f"Took {elapsed} sec to load {len(self._apps_info)} apps")

        if error_app_names:
            raise RuntimeError(
                f"{error_app_names} return different output from {self._apps_info[0].key}"
            )

    def to(self, device: torch.device, **kwargs) -> None:
        if device != self.device:
            raise RuntimeError("CodeGenTeaNet does not support changing device")
        if kwargs:
            raise RuntimeError("CodeGenTeaNet does not support changing model")
        self.model.to(device)

    def _get_fastest_num_nodes_and_edges(self, num_nodes: int, num_edges: int):
        for app in self._apps_info:
            if app.num_nodes >= num_nodes and app.num_edges >= num_edges:
                return app.num_nodes, app.num_edges

        if self._apps_info[-1].num_nodes < num_nodes:
            raise AtomsTooManyError(num_nodes, self._apps_info[-1].num_nodes)
        else:
            raise NeighborsTooManyError(num_edges, self._apps_info[-1].num_edges)

    def _get_padded_num_edges(self, num_nodes: int, num_edges: int):
        if self.force_num_edges:
            return self.force_num_edges
        if self._apps_info is not None:
            return self._get_fastest_num_nodes_and_edges(num_nodes, num_edges)[1]

        # This is a suitable but not optimal sequence.
        candidates = [2048, 4096, 8192, 12288, 14336]
        for i in range(1, 64 + 1):
            candidates.append(16384 * i)
        for cand in candidates:
            if num_edges <= cand:
                return cand
        raise RuntimeError(f"Too many edges: {num_edges}")

    def _get_padded_num_nodes(self, num_nodes: int, num_edges: int):
        if self.force_num_nodes:
            return self.force_num_nodes
        if self._apps_info is not None:
            return self._get_fastest_num_nodes_and_edges(num_nodes, num_edges)[0]

        # Increase bucket size exponentially. Increasing bucket_step
        # makes paddings larger while we may decrease the number of
        # compilations. This is a suitable but not optimal sequence.
        bucket_step = 1.3
        candidates = [int(bucket_step ** i) for i in range(100)]
        candidates = [(i + 15) // 16 * 16 if i > 32 else i for i in candidates]
        for cand in candidates:
            if num_nodes <= cand:
                return cand
        raise RuntimeError(f"Too many nodes: {num_nodes}")

    def _get_cache_key(self, num_nodes: int, num_edges: int, use_always_recomp: bool):
        keys = ["teanet"]
        keys.append(self.config_name)
        if self.float_dtype is not None:
            keys.append(self.float_dtype)
        # We specify `dynamic_axes` when `self.pad_edge` is False and the
        # compiled model works for arbitrary number of nodes/edges.
        # Note MN-Core backend does not work without padding edge.
        if self.pad_edge:
            keys.append("node%d" % num_nodes)
            keys.append("edge%d" % num_edges)
        if self.codegen_options.use_recomp(
            num_nodes=num_nodes, num_edges=num_edges, use_always_recomp=use_always_recomp
        ):
            keys.append("recomp")
        return "_".join(keys)

    def _parse_cache_key(self, key: str):
        prefix = f"teanet_{self.config_name}_"
        assert key.startswith(prefix), key
        key = key[len(prefix) :]

        float_dtype = None
        for cand in ["mixed", "float", "fast_double", "double"]:
            if key.startswith(cand + "_"):
                assert float_dtype is None, f"dtype is specified twice: {key}"
                float_dtype = cand
                key = key[len(cand) + 1 :]

        toks = key.split("_")
        num_nodes = num_edges = None
        for tok in toks:
            if tok.startswith("node"):
                num_nodes = int(tok[len("node") :])
            elif tok.startswith("edge"):
                num_edges = int(tok[len("edge") :])
            else:
                raise RuntimeError(f"Unknown model cache key: {key}")
        assert num_nodes is not None, key
        assert num_edges is not None, key
        return float_dtype, num_nodes, num_edges

    def _get_compiled_model(
        self,
        inputs: Dict[str, torch.Tensor],
        use_always_recomp: bool = False,
    ):
        if self.codegen_device is None:
            return self.model, ""

        runtime_core.Context.switch_context(self.context)

        num_nodes = inputs["coordinates"].shape[0]
        num_edges = inputs["atom_index1"].shape[0]

        key = self._get_cache_key(num_nodes, num_edges, use_always_recomp)

        if key in self.compiled_models:
            logger.info(f"Use compiled model {key}")
            compiled_model = self.compiled_models[key]
            return compiled_model, key

        export_kwargs = {}
        if not self.pad_edge:
            # See `dynamic_axes` in `help(torch.onnx.export)`.
            export_kwargs["dynamic_axes"] = {
                "coordinates": {0: "num_nodes"},
                "atomic_numbers": {0: "num_nodes"},
                "atom_index1": {0: "num_edges"},
                "atom_index2": {0: "num_edges"},
                "num_graphs": {0: "num_batches"},
                "shift": {0: "num_edges"},
                "cell": {0: "num_batches"},
                "batch": {0: "num_nodes"},
                "batch_edge": {0: "num_edges"},
                "x_add": {0: "num_nodes"},
                "calc_mode_type": {0: "num_nodes"},
                "edge_vector": {0: "num_edges"},
                "energy": {0: "num_batches"},
                "charge": {0: "num_nodes"},
            }

        if not self.codegen_options.enable_debug:
            # NOTE: options not to include python's backtrace into ONNX
            export_kwargs["strip_doc_string"] = True
            export_kwargs["verbose"] = False

        load_codegen_dir = None
        if self.load_outdir is None:
            logger.info(f"Compile model {key}")
        else:
            load_codegen_dir = os.path.join(self.load_outdir, key)
            logger.info(f"Load model from {load_codegen_dir}")
        with perfetto_trace.trace_event("compile"):
            logger.info("Export and compile model...")
            # Apparently, ONNX export shows warnings if we try exporting
            # this on CPU.
            self.model.cpu()
            cpu_inputs = {k: v.cpu() for k, v in inputs.items()}
            codegen_args = _add_codegen_args_from_env(
                self.codegen_options.get_codegen_args(
                    self.codegen_args, num_nodes, num_edges, use_always_recomp=use_always_recomp
                )
            )
            codegen_args["load_codegen_dir"] = load_codegen_dir

            self.context.registry.register(key, self.model)

            def f(kwargs):
                return self.model(**kwargs)

            func, _ = self.context.compile(
                key,
                f,
                [],
                storage.path(str(self.outdir)),
                cpu_inputs,
                codegen_args,
                export_kwargs=export_kwargs,
            )
            func.get_all_outputs(False)
            self.model.to(self.device)

            def compiled_model(**kwargs):
                return func(kwargs)

            logger.info("Send weights and constants...")
            logger.info("Compile finished!")

        self.compiled_models[key] = compiled_model
        return compiled_model, key

    def forward(
        self,
        coordinates: torch.Tensor,
        atomic_numbers: torch.Tensor,
        atom_index1: torch.Tensor,
        atom_index2: torch.Tensor,
        shift: torch.Tensor,
        cell: torch.Tensor,
        num_graphs: torch.Tensor,
        batch: torch.Tensor,
        batch_edge: torch.Tensor,
        x_add: torch.Tensor,
        calc_mode_type: torch.Tensor,
        force_num_nodes: int = 0,
        force_num_edges: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = {
            "coordinates": coordinates,
            "atomic_numbers": atomic_numbers,
            "atom_index1": atom_index1,
            "atom_index2": atom_index2,
            "shift": shift,
            "cell": cell,
            "num_graphs": num_graphs,
            "batch": batch,
            "batch_edge": batch_edge,
            "x_add": x_add,
            "calc_mode_type": calc_mode_type,
        }
        orig_num_nodes = coordinates.shape[0]
        orig_num_edges = atom_index1.shape[0]

        if self.pad_edge:
            # Add a dummy system for padded nodes.
            orig_num_batches = num_graphs.shape[0]
            padded_num_batches = _padded_num_batches

            # Pad at least 1 node for the dummy system.
            padded_num_nodes = orig_num_nodes + 1
            padded_num_edges = self._get_padded_num_edges(padded_num_nodes, orig_num_edges)
            if self.pad_node:
                padded_num_nodes = self._get_padded_num_nodes(padded_num_nodes, orig_num_edges)

            if force_num_nodes > 0:
                padded_num_nodes = force_num_nodes
            if force_num_edges > 0:
                padded_num_edges = force_num_edges

            inputs = _pad_inputs(
                inputs,
                orig_num_batches,
                padded_num_batches,
                orig_num_nodes,
                padded_num_nodes,
                orig_num_edges,
                padded_num_edges,
            )

        compiled_model, model_name = self._get_compiled_model(inputs)
        self.last_model_name = model_name
        with perfetto_trace.trace_event("teanet"):
            if self.codegen_device is None:
                with torch.enable_grad():
                    coordinates = inputs["coordinates"] = inputs["coordinates"].requires_grad_(
                        True
                    )
                    outputs = compiled_model(**inputs)
                    energy = outputs["energy"]
                    charge = outputs["charge"]
                    edge_vector = outputs["edge_vector"].requires_grad_(True)
                    neg_forces, forces_raw = torch.autograd.grad(
                        energy, (coordinates, edge_vector), torch.ones_like(energy)
                    )
                    forces = -neg_forces
                    virial_raw = (
                        forces_raw[:, [0, 1, 2, 1, 2, 0]] * edge_vector[:, [0, 1, 2, 2, 0, 1]]
                    )
                    virial = torch.zeros(
                        (num_graphs.shape[0], 6), dtype=virial_raw.dtype, device=virial_raw.device
                    )
                    virial.index_add_(0, batch_edge, virial_raw[:orig_num_edges])
            else:
                outputs = compiled_model(**inputs)
                outputs = {k: v.cpu() for k, v in outputs.items()}

                energy = outputs["energy"]
                forces = -outputs["grad_out@coordinates"]
                virial = outputs["virial"]
                charge = outputs["charge"]

        if self.pad_edge:
            energy = energy[:orig_num_batches]
            virial = virial[:orig_num_batches]
            charge = charge[:orig_num_nodes]
            forces = forces[:orig_num_nodes]

        return energy, forces, virial, charge

    def torch_device_from_str(self, device_str: str) -> torch.device:
        if device_str.startswith(("mncore", "emu")):
            device_str = "cpu"
        else:
            # E.g., pfvm:cuda:0 => cuda:0
            device_str = ":".join(device_str.split(":")[1:])
        return torch.device(device_str)
