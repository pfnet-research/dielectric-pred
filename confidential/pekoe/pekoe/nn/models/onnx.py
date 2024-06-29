from typing import Any

import torch.onnx
import torch.onnx.symbolic_helper as sym_help
from torch.onnx.symbolic_helper import _unimplemented, parse_args
from torch.onnx.utils import register_custom_op_symbolic

# Copies from https://github.pfidev.jp/takamoto/chicle/blob/teanet_pfp/examples/pfp/output_onnx.py


@parse_args("v", "i", "v", "v")
def scatter_add(g, self, dim, index, src):
    """Replace PyTorch original scatter_add ONN transformation function"""
    dtype = sym_help._try_get_scalar_type(self)
    if dtype is None:
        return _unimplemented("scatter_add", "input dtype not accessible")
    dtype = sym_help.scalar_type_to_onnx.index(sym_help.cast_pytorch_to_onnx[dtype])
    dtype = sym_help.scalar_type_to_pytorch_type[dtype]
    shape = g.op("Shape", self)
    to_add = g.op("ConstantOfShape", shape, value_t=torch.tensor([0], dtype=dtype))
    to_add = g.op("ChainerScatterAdd", to_add, index, src, axis_i=dim)
    # return add(g, self, to_add)
    return g.op("Add", self, to_add)


def expand(g, self, size, implicit):
    size = sym_help._maybe_get_const(size, "is")
    if not sym_help._is_value(size):
        size = g.op("Constant", value_t=torch.LongTensor(size))
    return g.op("Expand", self, size)


@parse_args("v", "t", "t")
def softplus(g, self, beta, threshold):
    if beta != 1:
        beta_v = g.op("Constant", value_t=beta)
        self = g.op("Mul", self, beta_v)
    y = g.op("Softplus", self)
    if threshold != 20:
        threshold_v = g.op("Constant", value_t=threshold)
        y = g.op("Where", g.op("LessOrEqual", self, threshold_v), y, self)
    if beta != 1:
        y = g.op("Div", y, beta_v)
    return y


def export_arguments(model: torch.nn.Module, opset_version: int = 9):
    return {
        "input_names": ["vc", "sp", "a1", "a2", "ob", "ba", "be", "xa", "cm"],
        "output_names": ["e", "c"],
        "dynamic_axes": {
            "sp": {0: "ns"},
            "vc": {0: "es"},
            "a1": {0: "es"},
            "a2": {0: "es"},
            "ob": {0: "og"},
            "ba": {0: "ns"},
            "be": {0: "es"},
            "xa": {0: "ns"},
            "cm": {0: "ns"},
        },
        "opset_version": opset_version,
        "enable_onnx_checker": False,
        "do_constant_folding": False,
    }


def export_with_testcase(
    model: torch.nn.Module,
    args: Any,
    dir: str,
    opset_version: int = 9,
) -> None:
    import pytorch_pfn_extras.onnx

    register_custom_op_symbolic("aten::expand", expand, opset_version)
    register_custom_op_symbolic("aten::scatter_add", scatter_add, opset_version)
    register_custom_op_symbolic("aten::softplus", softplus, opset_version)

    pytorch_pfn_extras.onnx.export_testcase(
        model,
        args,
        dir,
        model_overwrite=False,
        # output_grad=True,
        **export_arguments(model=model, opset_version=opset_version),
    )
