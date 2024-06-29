import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import dacite
import numpy as np
import torch
import torch.nn.init as Init

from ..teanet_base import TeaNetBase

try:
    from .select_activation import (
        TeaNetActivation,
        TeaNetCutoffFunction,
        select_cutoff_function,
        teanet_select_activation,
    )
except ImportError:
    from chicle.functions.select_activation import (
        TeaNetActivation,
        TeaNetCutoffFunction,
        select_cutoff_function,
        teanet_select_activation,
    )

from . import indexing
from .indexing import index_add, index_select

try:
    from .teanet_conv import TeaNetConv, extend_input
except ImportError:
    from chicle.nn.conv.teanet_conv import TeaNetConv, extend_input

try:
    from .twobody import Twobody
except ImportError:
    from chicle.nn.models.twobody import Twobody

try:
    from torch_geometric.data import Data

    class TeaNetDataConvertWrapper(torch.nn.Module):
        def __init__(self, model: torch.nn.Module):
            super(TeaNetDataConvertWrapper, self).__init__()
            self.model = model

        def forward(self, *data):
            if len(data) == 1 and isinstance(data[0], Data):
                data_tuple = convert_data_to_tuple(data[0])
            elif len(data) == 8:  # This is used when JIT trace (e.g. onnx export)
                data_tuple = data
            else:
                raise ValueError(
                    "Unexpected argument type in TeaNet: {:s}".format(str(type(data)))
                )
            return self.model(*data_tuple)


except ImportError:
    # print("Warning: no torch_geometric module found.")
    pass


def convert_data_to_tuple(data):
    num_graphs_arr = torch.zeros(
        (data.__num_graphs__,),
        dtype=data.edge_vector.dtype,
        device=data.edge_vector.device,
    )
    res = (
        data.edge_vector,
        data.x,
        data.edge_index[0, :],
        data.edge_index[1, :],
        num_graphs_arr,
        data.batch,
        data.batch_edge,
        data.x_add,
    )
    return res


class ElementArrayTeanetTorch(torch.nn.Module):
    def __init__(self, n_scalar_init):
        super(ElementArrayTeanetTorch, self).__init__()
        self.n_species_max = n_scalar_init
        self.n_scalar = n_scalar_init
        elementnum_to_vector_raw = []
        for i in range(n_scalar_init + 1):
            tmp = [0.0 for _ in range(n_scalar_init)]
            for j in range(i):
                tmp[j] += 1.0
            elementnum_to_vector_raw.append(tmp)
        tmp = [0.0 for _ in range(n_scalar_init)]
        elementnum_to_vector_raw.append(tmp)
        elementnum_to_vector = torch.tensor(elementnum_to_vector_raw)
        self.register_buffer("elementnum_to_vector", elementnum_to_vector)

    def forward(self, species):
        convert_tensor = self.elementnum_to_vector
        if torch.any(species > self.n_species_max):
            print("Unknown species type.")
            raise NotImplementedError
        res = convert_tensor[species]
        return res

    def atom_vec_length(self):
        return self.n_scalar


class ElementArrayTeanetOriginal(torch.nn.Module):
    def __init__(self, n_scalar_init):
        super(ElementArrayTeanetOriginal, self).__init__()
        self.n_species_max = n_scalar_init * 2
        self.n_scalar = n_scalar_init
        elementnum_to_vector_raw = []
        for i in range(2 * n_scalar_init + 1):
            tmp = [0.0 for _ in range(n_scalar_init)]
            for j in range(i):
                tmp[j // 2] += 0.5
            elementnum_to_vector_raw.append(tmp)
        elementnum_to_vector_raw.append(tmp)
        elementnum_to_vector = torch.tensor(elementnum_to_vector_raw)
        self.register_buffer("elementnum_to_vector", elementnum_to_vector)

    def forward(self, species):
        convert_tensor = self.elementnum_to_vector
        if torch.any(species > self.n_species_max):
            print("Unknown species type.")
            raise NotImplementedError
        res = convert_tensor[species]
        return res

    def atom_vec_length(self):
        return self.n_scalar


class AtomEmbedding(torch.nn.Module):
    def __init__(self, n_scalar_init, embedding_dict=None, filename=None):
        super(AtomEmbedding, self).__init__()
        # embedding_dict is intended to be provided during inference
        if embedding_dict is not None:
            n2a_dict = embedding_dict
        else:
            if filename is None:
                filename = "n2a_encoding.json"
            with open(filename) as f:
                n2a_dict = json.load(f)
        self.n_species_max = n_scalar_init + 1
        self.n_scalar = len(n2a_dict["1"])
        elementnum_to_vector_raw = [[0.0] * self.n_scalar for _ in range(self.n_species_max + 1)]
        for k, v in n2a_dict.items():
            if int(k) >= len(elementnum_to_vector_raw):
                continue
            elementnum_to_vector_raw[int(k)] = v
        elementnum_to_vector = torch.tensor(elementnum_to_vector_raw)
        self.register_buffer("elementnum_to_vector", elementnum_to_vector)

    def forward(self, species):
        convert_tensor = self.elementnum_to_vector
        if not torch._C._get_tracing_state():
            assert not torch.any(species > self.n_species_max), "Unknown species type."
        res = convert_tensor[species]
        return res

    def atom_vec_length(self):
        return self.n_scalar


class ElementArrayTeanetWithEmbedding(torch.nn.Module):
    def __init__(self, n_scalar_init, embedding_dict=None, filename=None):
        super(ElementArrayTeanetWithEmbedding, self).__init__()
        self.n_species_max = n_scalar_init
        elementnum_to_vector_raw = []
        for i in range(n_scalar_init + 1):
            tmp = [0.0 for _ in range(n_scalar_init)]
            for j in range(i):
                tmp[j] += 1.0
            elementnum_to_vector_raw.append(tmp)
        tmp = [0.0 for _ in range(n_scalar_init)]
        elementnum_to_vector_raw.append(tmp)
        elementnum_to_vector = torch.tensor(elementnum_to_vector_raw)

        # embedding_dict is intended to be provided during inference
        if embedding_dict is not None:
            n2a_dict = embedding_dict
        else:
            if filename is None:
                filename = "n2a_encoding.json"
            with open(filename) as f:
                n2a_dict = json.load(f)
        self.n_species_max = n_scalar_init + 1
        n_scalar_embedding = len(n2a_dict["1"])
        embedding_raw = [[0.0] * n_scalar_embedding for _ in range(self.n_species_max + 1)]
        for k, v in n2a_dict.items():
            if int(k) >= len(embedding_raw):
                continue
            embedding_raw[int(k)] = v
        embedding_vector = torch.tensor(embedding_raw)
        elementnum_to_vector = torch.cat((elementnum_to_vector, embedding_vector), dim=1)
        self.register_buffer("elementnum_to_vector", elementnum_to_vector)
        self.n_scalar = n_scalar_init + n_scalar_embedding

    def forward(self, species):
        convert_tensor = self.elementnum_to_vector
        if not torch._C._get_tracing_state():
            assert not torch.any(species > self.n_species_max), "Unknown species type."
        res = convert_tensor[species]
        return res

    def atom_vec_length(self):
        return self.n_scalar


@dataclass
class TeaNetParameters_v1_2:
    activation: TeaNetActivation
    in_channels: int
    in_channels_edge: int
    dim: Union[int, List[int]]
    dim_edge: Union[int, List[int]]
    dim_vector: Union[int, List[int]]
    dim_middle: Union[int, List[int]]
    dim_charge: Union[int, List[int]]
    out_dim: int
    n_conv_layers: int
    repeat: bool = False
    cutoff: Union[float, List[float]] = 6.0
    twobody_cutoff: float = 3.0
    atom_embedding: int = 0
    atom_embedding_dict: Optional[Dict[Any, Any]] = None
    cutoff_function: TeaNetCutoffFunction = TeaNetCutoffFunction.XINVNEGEXP
    use_init_layer: bool = False
    use_mlp_layer: bool = False
    atom_vec_append: int = 0
    use_bn: bool = False
    dropout_ratio: float = 0.0
    return_node_feature: bool = False
    shrink_edge: bool = True
    fix_twobody: bool = True
    output_every_layer: bool = False
    infer_n_layer: Optional[int] = None
    is_inference: bool = True

    @classmethod
    def from_dict(class_, d: Dict[Any, Any]) -> "TeaNetParameters_v1_2":
        return dacite.from_dict(
            data_class=class_,
            data=d,
            config=dacite.Config(cast=[TeaNetActivation, TeaNetCutoffFunction]),
        )


class TeaNet_v1_2(TeaNetBase):

    """TeaNet

    See So Takamoto, Satoshi Izumi, Ju Li, \
        TeaNet: universal neural network interatomic potential inspired \
        by iterative electronic relaxations. \
        `arXiv:1912.01398 <https://arxiv.org/abs/1912.01398>`_

    Args:
        activation_name (str): activation name.
            "ssp" is recommended to train with force now.
            "polylog" is used in original paper.
        in_channels (int): input dimension of `x`
        in_channels_edge (int): channels for edge feature, made from `rs_input`.
        dim (int): hidden dim for node scalar feature
        dim_edge (int): hidden dim for edge scalar feature
        dim_vector (int): hidden dim for node vector, node tensor and edge vector feature
        dim_middle (int): hidden dim used inside TeaNetConv layer.
        out_dim (int): dimension of output feature vector
        n_conv_layers (int): the number of conv layers
        repeat (bool): weight_tying. When True, use same layer for all intermediate conv layers
            for parameter sharing.
        cutoff (float): cutoff distance.
    """

    def __init__(
        self,
        parameters: TeaNetParameters_v1_2,
    ):
        super(TeaNet_v1_2, self).__init__()
        if isinstance(parameters.cutoff, float):
            self.cutoff_list = [parameters.cutoff] * parameters.n_conv_layers
        else:
            assert len(parameters.cutoff) == parameters.n_conv_layers
            self.cutoff_list = parameters.cutoff

        self.twobody_cutoff = parameters.twobody_cutoff

        self.out_dim = parameters.out_dim
        self.use_init_layer = parameters.use_init_layer
        self.use_mlp_layer = parameters.use_mlp_layer
        self.atom_vec_append = parameters.atom_vec_append
        self.return_node_feature = parameters.return_node_feature
        self.shrink_edge = parameters.shrink_edge
        self.output_every_layer = parameters.output_every_layer
        self.infer_n_layer = (
            parameters.infer_n_layer if parameters.infer_n_layer else parameters.n_conv_layers
        )
        self.is_inference = parameters.is_inference

        # self.n_ncharge = parameters.dim_charge
        self.n_layer = parameters.n_conv_layers
        n_ns_init = parameters.in_channels
        length_factor = 4.0

        atom_embedding = parameters.atom_embedding
        if atom_embedding == 0:
            self.atom_vec = ElementArrayTeanetTorch(n_ns_init)
        elif atom_embedding == 1:
            self.atom_vec = AtomEmbedding(n_ns_init)
            if n_ns_init != self.atom_vec.n_scalar:
                print(
                    "Error! Input scalar size {} is \
                    different from atom embedding size {}.".format(
                        n_ns_init, self.atom_vec.n_scalar
                    )
                )
                raise NotImplementedError
        elif atom_embedding == 2:
            self.atom_vec = ElementArrayTeanetOriginal(n_ns_init)
        elif atom_embedding == 3:
            self.atom_vec = ElementArrayTeanetWithEmbedding(
                n_ns_init, embedding_dict=parameters.atom_embedding_dict
            )

        n_atom_vec = self.atom_vec.atom_vec_length()
        n_atom_vec += self.atom_vec_append
        # Create channel size list

        n_dim = parameters.dim
        n_dim_vector = parameters.dim_vector
        n_dim_e_in = parameters.in_channels_edge
        n_dim_e = parameters.dim_edge
        n_dim_middle = parameters.dim_middle
        n_dim_charge = parameters.dim_charge

        n_x_init = n_atom_vec
        if self.use_init_layer:
            n_x_init = n_dim

        if isinstance(n_dim, list):
            n_x_list = n_dim
        else:
            n_x_list = [n_x_init] + [n_dim] * self.n_layer
        if isinstance(n_dim_vector, list):
            n_xv_list = [1] + n_dim_vector
            n_xt_list = [1] + n_dim_vector
            n_pv_list = [1] + n_dim_vector
        else:
            n_xv_list = [1] + [n_dim_vector] * self.n_layer
            n_xt_list = [1] + [n_dim_vector] * self.n_layer
            n_pv_list = [1] + [n_dim_vector] * self.n_layer
        if isinstance(n_dim_e, list):
            n_p_list = [n_dim_e_in] + n_dim_e
        else:
            n_p_list = [n_dim_e_in] + [n_dim_e] * self.n_layer
        if isinstance(n_dim_middle, list):
            n_middle_list = n_dim_middle
        else:
            n_middle_list = [n_dim_middle] * self.n_layer
        if isinstance(n_dim_charge, list):
            n_ncharge_list = [1] + n_dim_charge
        else:
            n_ncharge_list = [1] + [n_dim_charge] * self.n_layer

        # Create convolution layers
        self._forward = []
        for i in range(self.n_layer):
            name = "m{}".format(i)
            if (not parameters.repeat) or (i <= 1 or i == self.n_layer - 1):
                m = TeaNetConv(
                    n_x_list[i],
                    n_x_list[i + 1],
                    n_xv_list[i],
                    n_xv_list[i + 1],
                    n_xt_list[i],
                    n_xt_list[i + 1],
                    n_p_list[i],
                    n_p_list[i + 1],
                    n_pv_list[i],
                    n_pv_list[i + 1],
                    n_middle_list[i],
                    n_ncharge_list[i],
                    n_ncharge_list[i + 1],
                    cutoff=self.cutoff_list[i],
                    activation_id=parameters.activation,
                    cutoff_function=parameters.cutoff_function,
                    use_bn=parameters.use_bn,
                    dropout_ratio=parameters.dropout_ratio,
                )
            setattr(self, name, m)
            self._forward.append(name)

            lns = torch.nn.Linear(n_x_list[i + 1], self.out_dim)
            les = torch.nn.Linear(n_p_list[i + 1], self.out_dim, bias=False)
            lncharge = torch.nn.Linear(n_ncharge_list[i + 1], 1, bias=False)
            setattr(self, name + "_lns", lns)
            setattr(self, name + "_les", les)
            setattr(self, name + "_lncharge", lncharge)
            if self.use_mlp_layer:
                mlp_x = torch.nn.Linear(n_x_list[i + 1], n_x_list[i + 1])
                mlp_p = torch.nn.Linear(n_p_list[i + 1], n_p_list[i + 1])
                setattr(self, name + "_mlp_x", mlp_x)
                setattr(self, name + "_mlp_p", mlp_p)

        self.lcuts_init = torch.nn.Linear(1, n_p_list[0], bias=False)

        lc_coeff = torch.tensor([self.cutoff_list])
        self.register_buffer("lc_coeff", lc_coeff)

        if self.use_init_layer:
            self.init_layer = torch.nn.Linear(n_atom_vec, n_x_list[0])

        (
            self.activation,
            self.activation_shift,
            self.activation_grad,
        ) = teanet_select_activation(parameters.activation, False)

        self.cutoff_function = select_cutoff_function(parameters.cutoff_function)

        self.twobody = Twobody(
            n_ns_init,
            cutoff=self.twobody_cutoff,
            cutoff_function=TeaNetCutoffFunction.TRANSITION,
        )

        self.reset_parameters(length_factor)

        if parameters.fix_twobody:
            for param in self.twobody.parameters():
                param.requires_grad = False

    def reset_parameters(self, length_factor):
        Init.uniform_(self.lcuts_init.weight, -2.0, -0.2)

        for name in self._forward:
            lns = getattr(self, name + "_lns")
            les = getattr(self, name + "_les")
            lncharge = getattr(self, name + "_lncharge")
            Init.kaiming_normal_(lns.weight)
            Init.kaiming_normal_(les.weight)
            Init.kaiming_normal_(lncharge.weight)
            with torch.no_grad():
                lns.weight *= 0.1
                les.weight *= 0.1
                lncharge.weight *= 0.1

    def set_precision(self):
        for name in self._forward:
            m = getattr(self, name)
            m.set_precision()

    def use_mncore(self):
        indexing.use_mncore()
        self.shrink_edge = False

    def forward(self, *data, calc_mode_type: Optional[torch.Tensor] = None):
        (
            edge_vector,
            x,
            left_indices,
            right_indices,
            num_graphs_arr,
            batch,
            batch_edge,
            x_add,
        ) = data

        rv_input = torch.unsqueeze(edge_vector, 2)  # (n_edges, 3, 1)
        ns_input = self.atom_vec(x).to(rv_input.dtype)
        if self.atom_vec_append > 0:
            ns_input = torch.cat((ns_input, x_add), dim=1)

        n_nodes = ns_input.size()[0]
        n_edges = rv_input.size()[0]

        nv_input = torch.zeros((n_nodes, 3, 1), dtype=rv_input.dtype, device=ns_input.device)
        nt_input = torch.zeros((n_nodes, 3, 3, 1), dtype=rv_input.dtype, device=ns_input.device)
        ev_input = torch.zeros((n_edges, 3, 1), dtype=rv_input.dtype, device=ns_input.device)
        ncharge = torch.zeros((n_nodes, 1), dtype=rv_input.dtype, device=ns_input.device)
        rs_input = torch.sqrt(torch.sum(rv_input * rv_input, dim=1))
        rt_input_p1 = torch.unsqueeze(rv_input, 1).expand(n_edges, 3, 3, 1)
        rt_input_p2 = torch.unsqueeze(rv_input, 2).expand(n_edges, 3, 3, 1)
        rt_input = rt_input_p1 * rt_input_p2

        rs_input_flatten = rs_input.reshape((rs_input.size()[0],))

        # TODO: how about using `edge_attr * cutoff` as input?
        # pair_all = self.cutoff_function(rs_input - self.lc_coeff[0], self.lcuts_init)
        pair_all = torch.zeros((n_edges, 1), dtype=rv_input.dtype, device=ns_input.device)

        h_ns = ns_input
        h_nv = nv_input
        h_nt = nt_input
        h_es = pair_all
        h_ev = ev_input
        h_rs = rs_input
        h_rv = rv_input
        h_rt = rt_input
        h_ncharge = ncharge

        if self.use_init_layer:
            h_ns = self.init_layer(h_ns)

        layer_outputs_list = []

        for i, name in enumerate(self._forward):
            m = getattr(self, name)

            if self.shrink_edge:
                active_index_layer = (rs_input_flatten < self.lc_coeff[:, i]).nonzero().reshape(-1)
                layer_es = index_select(h_es, active_index_layer)
                layer_ev = index_select(h_ev, active_index_layer)
                layer_rs = index_select(h_rs, active_index_layer)
                layer_rv = index_select(h_rv, active_index_layer)
                layer_rt = index_select(h_rt, active_index_layer)
                layer_left = index_select(left_indices, active_index_layer)
                layer_right = index_select(right_indices, active_index_layer)
            else:
                layer_es = h_es
                layer_ev = h_ev
                layer_rs = h_rs
                layer_rv = h_rv
                layer_rt = h_rt
                layer_left = left_indices
                layer_right = right_indices

            h_ns, h_nv, h_nt, layer_es, layer_ev, h_ncharge = m(
                (
                    h_ns,
                    h_nv,
                    h_nt,
                    layer_es,
                    layer_ev,
                    layer_rv,
                    layer_rs - self.lc_coeff[:, i].to(layer_rs.dtype),
                    layer_rt,
                    layer_left,
                    layer_right,
                    h_ncharge,
                )
            )

            if self.shrink_edge:
                # Make extend arr: for residual skip connection
                if m.n_es_in < m.n_es_out:
                    h_es = extend_input(h_es, 1, m.n_es_in, m.n_es_out)
                if m.n_ev_in < m.n_ev_out:
                    h_ev = extend_input(h_ev, 2, m.n_es_in, m.n_ev_out)
                n_edges = layer_es.size(0)
                # scatter() should be used for training, but scatter is broken at torch==1.10.1,
                # device==cpu, and zero-length tensor.
                # in that case, scatter_ can be used (not broken).
                if self.is_inference:
                    h_es.scatter_(
                        0,
                        active_index_layer.view(n_edges, 1).expand(n_edges, m.n_es_out),
                        layer_es,
                    )
                    h_ev.scatter_(
                        0,
                        active_index_layer.view(n_edges, 1, 1).expand(n_edges, 3, m.n_ev_out),
                        layer_ev,
                    )
                else:
                    h_es = h_es.scatter(
                        0,
                        active_index_layer.view(n_edges, 1).expand(n_edges, m.n_es_out),
                        layer_es,
                    )
                    h_ev = h_ev.scatter(
                        0,
                        active_index_layer.view(n_edges, 1, 1).expand(n_edges, 3, m.n_ev_out),
                        layer_ev,
                    )
            else:
                h_es = layer_es
                h_ev = layer_ev

            if i == self.infer_n_layer - 1 or self.output_every_layer:
                if hasattr(self, name + "_mlp_x"):
                    mlp_layer_x = getattr(self, name + "_mlp_x")
                    mlp_layer_p = getattr(self, name + "_mlp_p")
                    h_ns_out = self.activation(mlp_layer_x(h_ns)).float()
                    h_es_out = self.activation(mlp_layer_p(h_es)).float()
                else:
                    h_ns_out = h_ns.float()
                    h_es_out = h_es.float()
                h_ncharge_out = h_ncharge.float()

                lns = getattr(self, name + "_lns")
                les = getattr(self, name + "_les")
                lncharge = getattr(self, name + "_lncharge")
                ns_sum = lns(h_ns_out).to(torch.float64)
                es_sum = les(h_es_out).to(torch.float64)

                ncharge_out = lncharge(h_ncharge_out)

                res = torch.zeros(
                    (num_graphs_arr.size(0), self.out_dim),
                    dtype=torch.float64,
                    device=ns_input.device,
                )
                res = index_add(res, batch, ns_sum)
                res = index_add(res, batch_edge, es_sum)
                layer_outputs_list.append(res)

                if i == self.infer_n_layer - 1:  # Last layer
                    break

        if self.shrink_edge:
            twobody_index = (rs_input_flatten < self.twobody_cutoff).nonzero().reshape(-1)
            rs_input_twobody = index_select(rs_input, twobody_index)
            left_indices_twobody = index_select(left_indices, twobody_index)
            right_indices_twobody = index_select(right_indices, twobody_index)
            batch_edge_twobody = index_select(batch_edge, twobody_index)
        else:
            rs_input_twobody = rs_input
            left_indices_twobody = left_indices
            right_indices_twobody = right_indices
            batch_edge_twobody = batch_edge

        twobody_part = self.twobody(
            x, left_indices_twobody, right_indices_twobody, rs_input_twobody
        ).to(torch.float64)
        twobody_res = torch.zeros(
            (num_graphs_arr.size(0), self.out_dim),
            dtype=torch.float64,
            device=ns_input.device,
        )
        twobody_res = index_add(twobody_res, batch_edge_twobody, twobody_part)
        for layer_output in layer_outputs_list:
            layer_output += twobody_res

        if self.output_every_layer:
            layer_outputs = torch.stack(layer_outputs_list, dim=1)
        else:
            layer_outputs = layer_outputs_list[0]

        if self.return_node_feature:
            return layer_outputs, h_ns
        return layer_outputs, ncharge_out


class TeaNetForceWrapper(torch.nn.Module):
    """Force and virial calculation module for TeaNet

    There is no learnable parameters in this module.

    Args:
        teanet_model (TeaNet): TeaNet model to be calculated
        force_index (int, optional): Energy index. Defaults to 0.
        calc_virial (bool, optional): Whether to calculate virial or not. Defaults to False.
    """

    def __init__(
        self,
        teanet_model: TeaNet_v1_2,
        force_index: int = 0,
        calc_virial: bool = False,
        train_random_force_layer: bool = False,
        force_layer_coeff_list: Optional[List[int]] = None,
    ):
        super(TeaNetForceWrapper, self).__init__()
        self.teanet_model = teanet_model
        self.calc_virial = calc_virial
        self.train_random_force_layer = train_random_force_layer
        self.force_layer_coeff_list = force_layer_coeff_list
        if train_random_force_layer and (force_layer_coeff_list is not None):
            print("Warning: train_random_force_layer setting will be ignored.")

    def forward(self, data):
        with torch.enable_grad():
            data.edge_vector.requires_grad_(True)
            pred_y, charge = self.teanet_model(data)
            # pred_y = pred_y[:, pred_y.size()[1]-1:pred_y.size()[1], :]

            if self.force_layer_coeff_list:
                force_layer_coeff_list = torch.tensor(
                    self.force_layer_coeff_list, dtype=torch.float32, device=pred_y.device
                )
            else:
                if self.teanet_model.training and self.train_random_force_layer:
                    force_layer_coeff_list_structure = []
                    for _ in range(pred_y.size()[0]):
                        while True:
                            # force_layer_coeff_list_np = np.random.rand(pred_y.size()[1])
                            force_layer_coeff_list_np = np.array([0.0, 0.0, 0.05, 0.05, 0.9])
                            force_layer_coeff_list_np = (
                                1.0 / np.sum(force_layer_coeff_list_np)
                            ) * force_layer_coeff_list_np
                            if force_layer_coeff_list_np[-1] > 0.59:
                                break
                        # force_layer_coeff_list_np = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
                        onehot_index = np.argmax(
                            np.cumsum(force_layer_coeff_list_np) > np.random.rand()
                        )
                        force_layer_coeff_list_onehot = np.zeros(pred_y.size()[1])
                        force_layer_coeff_list_onehot[onehot_index] = 1.0
                        # print(force_layer_coeff_list_np)
                        # print(force_layer_coeff_list_onehot)
                        force_layer_coeff_list_structure.append(force_layer_coeff_list_onehot)
                    force_layer_coeff_list_np = np.stack(force_layer_coeff_list_structure, axis=0)
                    force_layer_coeff_list = torch.tensor(
                        force_layer_coeff_list_np, dtype=pred_y.dtype, device=pred_y.device
                    ).unsqueeze(2)
                else:
                    force_layer_coeff_list = torch.tensor(
                        [0.0] * (pred_y.size()[1] - 1) + [1.0],
                        dtype=pred_y.dtype,
                        device=pred_y.device,
                    )
                    force_layer_coeff_list = force_layer_coeff_list.view((1, pred_y.size()[1], 1))

            # print(force_layer_coeff_list.size())
            # print(pred_y.size())
            # print(torch.sum(force_layer_coeff_list * pred_y))
            # print(torch.sum(pred_y[:, -1, ]))
            grad_target = torch.sum(force_layer_coeff_list * pred_y)
            # grad_target = torch.sum(pred_y[:, -1, ])

            pred_grad = torch.autograd.grad(
                grad_target, data.edge_vector, create_graph=self.training
            )[0].to(torch.float64)

        left_indices = data.edge_index[0, :]
        right_indices = data.edge_index[1, :]
        forces_zeros = torch.zeros(
            (data.num_nodes, 3), dtype=torch.float64, device=pred_grad.device
        )
        forces = -index_add(forces_zeros, left_indices, pred_grad) + index_add(
            forces_zeros, right_indices, pred_grad
        )

        if self.calc_virial:
            virial = torch.sum(
                pred_grad[:, [0, 1, 2, 1, 2, 0]]
                * data.edge_vector[:, [0, 1, 2, 2, 0, 1]].to(torch.float64)
            )
        else:
            virial = torch.zeros(
                (data.num_graphs, 6),
                dtype=pred_grad.dtype,
                device=pred_grad.device,
            )

            return pred_y[:, -1], forces, virial, charge, pred_y[:, :-1]
