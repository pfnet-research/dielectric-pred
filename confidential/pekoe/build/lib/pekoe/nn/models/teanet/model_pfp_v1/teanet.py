import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import dacite
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
        (data.num_graphs,),
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
class TeaNetParameters_v1:
    activation: TeaNetActivation
    in_channels: int
    in_channels_edge: int
    dim: int
    dim_edge: int
    dim_vector: int
    dim_middle: int
    dim_charge: int
    out_dim: int
    n_conv_layers: int
    repeat: bool = False
    cutoff: Union[float, List[float]] = 6.0
    twobody_cutoff: float = 3.0
    atom_embedding: int = 0
    atom_embedding_dict: Optional[Dict[Any, Any]] = None
    cutoff_function: TeaNetCutoffFunction = TeaNetCutoffFunction.XINVNEGEXP
    use_init_layer: bool = False
    use_last_layer: bool = False
    atom_vec_append: int = 0
    use_bn: bool = False
    dropout_ratio: float = 0.0
    return_node_feature: bool = False
    shrink_edge: bool = True

    @classmethod
    def from_dict(class_, d: Dict[Any, Any]) -> "TeaNetParameters_v1":
        return dacite.from_dict(
            data_class=class_,
            data=d,
            config=dacite.Config(cast=[TeaNetActivation, TeaNetCutoffFunction]),
        )


class TeaNet_v1(TeaNetBase):

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
        parameters: TeaNetParameters_v1,
    ):
        super(TeaNet_v1, self).__init__()
        if isinstance(parameters.cutoff, float):
            self.cutoff_list = [parameters.cutoff] * parameters.n_conv_layers
        else:
            assert len(parameters.cutoff) == parameters.n_conv_layers
            self.cutoff_list = parameters.cutoff

        self.twobody_cutoff = parameters.twobody_cutoff

        self.out_dim = parameters.out_dim
        self.use_init_layer = parameters.use_init_layer
        self.use_last_layer = parameters.use_last_layer
        self.atom_vec_append = parameters.atom_vec_append
        self.return_node_feature = parameters.return_node_feature
        self.shrink_edge = parameters.shrink_edge
        self.coord_dtype = torch.float32

        self.n_ncharge = parameters.dim_charge
        n_layer = parameters.n_conv_layers
        n_ns_init = parameters.in_channels
        n_x = parameters.dim
        n_xv = parameters.dim_vector
        n_xt = parameters.dim_vector
        n_es_init = parameters.in_channels_edge
        n_p = parameters.dim_edge
        n_pv = parameters.dim_vector
        n_middle = parameters.dim_middle
        n_last = parameters.dim
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
        n_x_init = n_atom_vec
        if self.use_init_layer:
            n_x_init = n_x
        n_x_list = [n_x_init]
        n_xv_list = [1]
        n_xt_list = [1]
        n_p_list = [n_es_init]
        n_pv_list = [1]
        n_middle_list = [n_middle]
        for _ in range(n_layer - 1):
            n_x_list.append(n_x)
            n_xv_list.append(n_xv)
            n_xt_list.append(n_xt)
            n_p_list.append(n_p)
            n_pv_list.append(n_pv)
            n_middle_list.append(n_middle)
        n_x_list.append(n_last)
        n_xv_list.append(n_xv)
        n_xt_list.append(n_xt)
        n_p_list.append(n_p)
        n_pv_list.append(n_pv)

        # Create convolution layers
        self._forward = []
        for i in range(n_layer):
            name = "m{}".format(i)
            if (not parameters.repeat) or (i <= 1 or i == n_layer - 1):
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
                    self.n_ncharge,
                    cutoff=self.cutoff_list[i],
                    activation_id=parameters.activation,
                    cutoff_function=parameters.cutoff_function,
                    use_bn=parameters.use_bn,
                    dropout_ratio=parameters.dropout_ratio,
                )
            setattr(self, name, m)
            self._forward.append(name)

        self.lcuts_init = torch.nn.Linear(1, n_es_init, bias=False)
        self.lns = torch.nn.Linear(n_last, self.out_dim)
        self.les = torch.nn.Linear(n_p, self.out_dim, bias=False)
        self.lncharge = torch.nn.Linear(self.n_ncharge, 1, bias=False)

        lc_coeff = torch.tensor([self.cutoff_list])
        self.register_buffer("lc_coeff", lc_coeff)

        if self.use_init_layer:
            self.init_layer = torch.nn.Linear(n_atom_vec, n_x)
        if self.use_last_layer:
            self.last_layer_x = torch.nn.Linear(n_x_list[-1], n_x_list[-1])
            self.last_layer_p = torch.nn.Linear(n_p_list[-1], n_p_list[-1])

        (
            self.activation,
            self.activation_shift,
            self.activation_grad,
        ) = teanet_select_activation(parameters.activation, False)

        self.cutoff_function = select_cutoff_function(parameters.cutoff_function)

        self.atom_vec_twobody = AtomEmbedding(
            n_ns_init, embedding_dict=parameters.atom_embedding_dict
        )
        self.twobody = Twobody(
            self.atom_vec_twobody.n_scalar + self.atom_vec_append,
            cutoff=self.twobody_cutoff,
            cutoff_function=TeaNetCutoffFunction.TRANSITION,
        )

        self.reset_parameters(length_factor, n_es_init)

        for param in self.twobody.parameters():
            param.requires_grad = False

    def reset_parameters(self, length_factor, n_es_init):
        Init.kaiming_normal_(self.lns.weight)
        Init.kaiming_normal_(self.les.weight)
        Init.uniform_(self.lcuts_init.weight, -2.0, -0.2)
        with torch.no_grad():
            self.lns.weight *= 0.1
            self.les.weight *= 0.1

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
        ncharge = torch.zeros(
            (n_nodes, self.n_ncharge), dtype=rv_input.dtype, device=ns_input.device
        )
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

        if self.use_init_layer:
            h_ns = self.init_layer(h_ns)

        for i, name in enumerate(self._forward):
            m = getattr(self, name)

            if self.shrink_edge:
                active_index_layer = torch.nonzero(
                    rs_input_flatten < self.lc_coeff[:, i], as_tuple=False
                )
                active_index_layer = active_index_layer.reshape((active_index_layer.size()[0],))
                layer_es = h_es[active_index_layer]
                layer_ev = h_ev[active_index_layer]
                layer_rs = rs_input[active_index_layer]
                layer_rv = rv_input[active_index_layer]
                layer_rt = rt_input[active_index_layer]
                layer_left = left_indices[active_index_layer]
                layer_right = right_indices[active_index_layer]
            else:
                layer_es = h_es
                layer_ev = h_ev
                layer_rs = rs_input
                layer_rv = rv_input
                layer_rt = rt_input
                layer_left = left_indices
                layer_right = right_indices

            h_ns, h_nv, h_nt, layer_es, layer_ev, ncharge = m(
                (
                    h_ns,
                    h_nv,
                    h_nt,
                    layer_es,
                    layer_ev,
                    layer_rv,
                    layer_rs - self.lc_coeff[:, i],
                    layer_rt,
                    layer_left,
                    layer_right,
                    ncharge,
                )
            )
            # Make extend arr: for residual skip connection
            h_es = extend_input(h_es, 1, m.n_es_in, m.n_es_out)
            h_ev = extend_input(h_ev, 2, m.n_ev_in, m.n_ev_out)

            n_edges = layer_es.size()[0]
            if self.shrink_edge:
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
                h_es = layer_es
                h_ev = layer_ev

        if self.use_last_layer:
            h_ns = self.activation(self.last_layer_x(h_ns))
            h_es = self.activation(self.last_layer_p(h_es))
        ns_sum = self.lns(h_ns).to(torch.float64)
        es_sum = self.les(h_es).to(torch.float64)

        ncharge_out = self.lncharge(ncharge)

        # batch_from_n = self.readout_batch(ns_sum, batch)
        # batch_from_e = self.readout_batch_edge(es_sum, batch_edge)
        out_zero = torch.zeros(
            (num_graphs_arr.size()[0], self.out_dim),
            dtype=torch.float64,
            device=ns_input.device,
        )
        batch_from_n = out_zero.scatter_add(
            0,
            batch.view(batch.size()[0], 1).expand(batch.size()[0], self.out_dim),
            ns_sum,
        )
        batch_from_e = out_zero.scatter_add(
            0,
            batch_edge.view(batch_edge.size()[0], 1).expand(batch_edge.size()[0], self.out_dim),
            es_sum,
        )
        res = batch_from_n + batch_from_e

        if self.shrink_edge:
            twobody_index = torch.nonzero(rs_input_flatten < self.twobody_cutoff, as_tuple=False)
            twobody_index = twobody_index.reshape((twobody_index.size()[0],))
            rs_input_twobody = rs_input[twobody_index]
            left_indices_twobody = left_indices[twobody_index]
            right_indices_twobody = right_indices[twobody_index]
            batch_edge_twobody = batch_edge[twobody_index]
        else:
            rs_input_twobody = rs_input
            left_indices_twobody = left_indices
            right_indices_twobody = right_indices
            batch_edge_twobody = batch_edge

        twobody_input = torch.cat(
            (self.atom_vec_twobody(x).to(rv_input.dtype), torch.zeros_like(x_add)), dim=1
        )
        twobody_part = self.twobody(
            twobody_input, left_indices_twobody, right_indices_twobody, rs_input_twobody
        ).to(torch.float64)
        twobody_res = out_zero.scatter_add(
            0,
            batch_edge_twobody.view(twobody_part.size()[0], 1).expand(
                twobody_part.size()[0], self.out_dim
            ),
            twobody_part,
        )
        res += twobody_res

        if self.return_node_feature:
            return res, h_ns
        return res, ncharge_out


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
        teanet_model: TeaNet_v1,
        force_index: int = 0,
        calc_virial: bool = False,
    ):
        super(TeaNetForceWrapper, self).__init__()
        self.teanet_model = teanet_model
        self.force_index = force_index
        self.calc_virial = calc_virial

    def forward(self, data):
        with torch.enable_grad():
            data.edge_vector.requires_grad_(True)
            pred_y, charge = self.teanet_model(data)

            if self.force_index == 0 and pred_y.size()[1] == 1:
                grad_target = torch.sum(pred_y)
            else:
                assert self.force_index < pred_y.size()[1]
                grad_target = torch.sum(pred_y[:, self.force_index])

            pred_grad = torch.autograd.grad(
                grad_target, data.edge_vector, create_graph=self.training
            )[0].to(torch.float64)

        n_edge = data.batch_edge.size()[0]
        left_indices = data.edge_index[0, :]
        right_indices = data.edge_index[1, :]
        forces_zeros = torch.zeros(
            (data.num_nodes, 3), dtype=torch.float64, device=pred_grad.device
        )
        forces = -forces_zeros.scatter_add(
            0, left_indices.view(n_edge, 1).expand(n_edge, 3), pred_grad
        ) + forces_zeros.scatter_add(0, right_indices.view(n_edge, 1).expand(n_edge, 3), pred_grad)

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

        return pred_y, forces, virial, charge
