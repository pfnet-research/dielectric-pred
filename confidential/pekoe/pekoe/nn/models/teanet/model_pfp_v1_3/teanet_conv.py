import math
import typing

import torch
import torch.nn.functional as F
import torch.nn.init as Init

try:
    from .select_activation import (
        TeaNetActivation,
        logcosh,
        select_cutoff_function,
        teanet_select_activation,
    )
except ImportError:
    from chicle.functions.select_activation import (
        TeaNetActivation,
        select_cutoff_function,
        teanet_select_activation,
    )

from .indexing import index_add, index_select

try:
    from teanet_conv_helper import gather_mul_sum, mul_index_add_add, mul_index_add_sub
except ImportError:

    def gather_mul_sum(x, i, f):
        return (index_select(x, i) * torch.unsqueeze(f, dim=1)).sum(dim=2)

    def mul_index_add_add(x, y, i, j, f):
        z = y * f
        return index_add(x, i, z) + index_add(x, j, z)

    def mul_index_add_sub(x, y, i, j, f):
        z = y * f
        return index_add(x, i, z) - index_add(x, j, z)


def extend_input(
    x: torch.Tensor, extend_axis: int, current_length: int, extend_length: int
) -> torch.Tensor:
    """Padding zero array into input `x`, to make the shape `extend_length` for `extend_axis`.

    Args:
        x (torch.Tensor): input tensor.
        extend_axis (torch.Tensor): extend axis.
        extend_length (torch.Tensor): extend length.

    Returns:
        output (torch.Tensor): zero-padded tensor, `output.shape[extend_axis] == extend_length`.
    """

    if current_length == extend_length:
        return x
    zero = torch.zeros(
        (1,) * (extend_axis + 1),
        dtype=x.dtype,
        device=x.device,
    )
    if extend_axis == 1:
        s0, s1 = x.size()
        append_zeros = zero.expand(s0, extend_length - s1)
    elif extend_axis == 2:
        s0, s1, s2 = x.size()
        append_zeros = zero.expand(s0, s1, extend_length - s2)
    elif extend_axis == 3:
        s0, s1, s2, s3 = x.size()
        append_zeros = zero.expand(s0, s1, s2, extend_length - s3)
    else:
        raise NotImplementedError(f"Not implemented for extend_axis={extend_axis}")
    return torch.cat([x, append_zeros], dim=extend_axis)


class TeaNetConv(torch.nn.Module):
    """TeaNet convolution layer

    Definition of (rough) naming rules:
    * Initial character
        n: node-related values
        e: edge-related values
        r: edge length/vector
        x: node-related values which were transferred into corresponding edges.

        Since two nodes correspond to one edge, there are two
        x- variables named xi and xj.

    * Second character
        s: scalar values
        v: vector values
        t: tensor values

        Note: Be careful to "r"+"v" set.
        "rv" looks very similar to "ev". Although they are both
        edge-related vectors, the sign of rv depends on the order of
        node counting of edge. (Vector i-to-j or j-to-i).
        On the other hand, all other *v values such as nv, ev, and xv
        do not depend on the order.
        Therefore, careful sign treatment is required.

    Each tensor has a shape like
    ([batchsize], [scalar/vector/tensor shape], [channels]).

    Examples:
       nx (num_nodes,       num_channels)
       nv (num_nodes, 3,    num_channels)
       nt (num_nodes, 3, 3, num_channels)
       ev (num_edges, 3,    num_channels)
       rv (num_edges, 3,    1)
       xv (num_edges, 3,    num_channels)

    Args:
        n_ns_in (int): node scalar input channel
        n_ns_out (int): node scalar output channel
        n_nv_in (int): node vector input channel
        n_nv_out (int): node vector output channel
        n_nt_in (int): node tensor input channel
        n_nt_out (int): node tensor output channel
        n_es_in (int): edge scalar input channel
        n_es_out (int): edge scalar output channel
        n_ev_in (int): edge vector input channel
        n_ev_out (int): edge vector output channel
        n_middle (int): hidden channel
        cutoff (float): cutoff length (angstrom)
        activation_id (TeaNetActivation): activation function id
        cutoff_function (Optional[str]):
            cutoff function name.
            If None, activation function is used
        use_bn (bool):
        dropout_ratio (float):
    """

    def __init__(
        self,
        n_ns_in: int,
        n_ns_out: int,
        n_nv_in: int,
        n_nv_out: int,
        n_nt_in: int,
        n_nt_out: int,
        n_es_in: int,
        n_es_out: int,
        n_ev_in: int,
        n_ev_out: int,
        n_middle: int,
        n_ncharge_in: int,
        n_ncharge_out: int,
        cutoff: float = 6.0,
        activation_id: TeaNetActivation = TeaNetActivation.POLYLOG,
        cutoff_function: typing.Optional[str] = None,
        use_bn: bool = False,
        dropout_ratio: float = 0.0,
    ):
        super(TeaNetConv, self).__init__()

        if n_ns_in > n_ns_out:
            raise ValueError(
                f"Currently n_ns_in {n_ns_in} must be smaller or equal to n_ns_out {n_ns_out}"
            )
        if n_nv_in > n_nv_out:
            raise ValueError(
                f"Currently n_nv_in {n_nv_in} must be smaller or equal to n_nv_out {n_nv_out}"
            )
        if n_nt_in > n_nt_out:
            raise ValueError(
                f"Currently n_nt_in {n_nt_in} must be smaller or equal to n_nt_out {n_nt_out}"
            )
        if n_es_in > n_es_out:
            raise ValueError(
                f"Currently n_es_in {n_es_in} must be smaller or equal to n_es_out {n_es_out}"
            )
        if n_ev_in > n_ev_out:
            raise ValueError(
                f"Currently n_ev_in {n_ev_in} must be smaller or equal to n_ev_out {n_ev_out}"
            )

        self.cutoff_function = cutoff_function

        self.l0ns = torch.nn.Linear(n_ns_in, n_ns_in)
        self.l0nv = torch.nn.Linear(n_nv_in, n_ns_in, bias=False)
        self.l0es = torch.nn.Linear(n_es_in, n_middle, bias=False)
        self.l0ev = torch.nn.Linear(n_ev_in, n_middle, bias=False)
        self.l0ncharge = torch.nn.Linear(n_ncharge_in, n_ns_in, bias=False)
        self.l1ns = torch.nn.Linear(n_ns_in, n_middle)
        self.l1nv = torch.nn.Linear(n_nv_in, n_middle, bias=False)
        self.l1nt = torch.nn.Linear(n_nt_in, n_middle, bias=False)

        self.l2 = torch.nn.Linear(5 * n_middle, n_middle, bias=False)
        self.l2d = torch.nn.Linear(3 * n_middle, n_middle, bias=False)

        self.l3 = torch.nn.Linear(
            n_middle,
            n_es_out + n_ns_out + n_nv_out + n_ev_out + n_nt_out + n_middle + n_ncharge_out,
            bias=False,
        )
        self.l3evrv_diff = torch.nn.Linear(n_middle, n_ev_out, bias=False)
        self.l3nv = torch.nn.Linear(n_middle, n_ev_out, bias=False)
        self.l3ncharge = torch.nn.Linear(n_middle, n_ncharge_out, bias=False)

        self.l3evrv = torch.nn.Linear(n_ev_in, n_nt_out, bias=False)
        self.l3ev = torch.nn.Linear(n_ev_in, n_nv_out, bias=False)

        self.l3ns_gate = torch.nn.Linear(n_ns_in, n_ns_out)
        self.l3nv_gate = torch.nn.Linear(n_ns_in, n_nv_out)
        self.l3nt_gate = torch.nn.Linear(n_ns_in, n_nt_out)
        self.l3ntc_gate = torch.nn.Linear(n_ns_in, n_nt_out)

        self.l4ns = torch.nn.Linear(n_ns_in, n_ns_out)
        self.l4v2 = torch.nn.Linear(n_nv_in, n_nv_out, bias=False)
        self.l4t = torch.nn.Linear(n_nt_in, n_nt_out, bias=False)
        self.l4es = torch.nn.Linear(n_middle, n_es_out, bias=False)
        self.l4ev = torch.nn.Linear(n_ev_in, n_ev_out, bias=False)

        self.lcut = torch.nn.Linear(1, 2 * n_middle, bias=False)

        lc_coeff = torch.tensor([[cutoff]])
        tensor_id = torch.tensor(
            [[[[1.0], [0.0], [0.0]], [[0.0], [1.0], [0.0]], [[0.0], [0.0], [1.0]]]]
        )
        self.register_buffer("lc_coeff", lc_coeff)
        self.register_buffer("tensor_id", tensor_id)

        self.use_bn = use_bn
        self.dropout_ratio = dropout_ratio
        if use_bn:
            self.bn_node = torch.nn.BatchNorm1d(n_ns_out)
            self.bn_edge = torch.nn.BatchNorm1d(n_es_out)
        else:
            self.bn_node = None
            self.bn_edge = None

        self.n_ns_in = n_ns_in
        self.n_nv_in = n_nv_in
        self.n_nt_in = n_nt_in
        self.n_es_in = n_es_in
        self.n_ev_in = n_ev_in
        self.n_middle = n_middle
        self.n_ns_out = n_ns_out
        self.n_nv_out = n_nv_out
        self.n_nt_out = n_nt_out
        self.n_es_out = n_es_out
        self.n_ev_out = n_ev_out
        self.n_ncharge_in = n_ncharge_in
        self.n_ncharge_out = n_ncharge_out
        self.cutoff = cutoff

        self.divide_m_l = [
            self.n_es_out,
            self.n_ns_out,
            self.n_nv_out,
            self.n_ev_out,
            self.n_nt_out,
            n_middle,
            self.n_ncharge_out,
        ]

        (
            self.activation,
            self.activation_shift,
            self.activation_grad,
        ) = teanet_select_activation(activation_id, True)

        self.cutoff_function = select_cutoff_function(cutoff_function)

        self.reset_parameters()

    def reset_parameters(self):
        # TODO(Takamoto): This initialization could be improved.
        Init.kaiming_normal_(self.l0ns.weight)
        Init.kaiming_normal_(self.l0nv.weight)
        Init.kaiming_normal_(self.l0es.weight)
        Init.kaiming_normal_(self.l0ev.weight)
        Init.kaiming_normal_(self.l0ncharge.weight)
        Init.kaiming_normal_(self.l1ns.weight)
        Init.kaiming_normal_(self.l1nv.weight)
        Init.kaiming_normal_(self.l1nt.weight)
        Init.kaiming_normal_(self.l2.weight)
        Init.kaiming_normal_(self.l2d.weight)
        Init.kaiming_normal_(self.l3.weight)
        Init.kaiming_normal_(self.l3evrv_diff.weight)
        Init.kaiming_normal_(self.l3nv.weight)
        Init.kaiming_normal_(self.l3evrv.weight)
        Init.kaiming_normal_(self.l3ev.weight)
        Init.kaiming_normal_(self.l3ncharge.weight)
        Init.kaiming_normal_(self.l3ns_gate.weight)
        Init.kaiming_normal_(self.l3nv_gate.weight)
        Init.kaiming_normal_(self.l3nt_gate.weight)
        Init.kaiming_normal_(self.l3ntc_gate.weight)
        Init.kaiming_normal_(self.l4ns.weight)
        Init.kaiming_normal_(self.l4v2.weight)
        Init.kaiming_normal_(self.l4t.weight)
        Init.kaiming_normal_(self.l4es.weight)
        Init.kaiming_normal_(self.l4ev.weight)

        Init.uniform_(self.lcut.weight, -2.0, -0.2)

        with torch.no_grad():
            # These values are manually examined.
            self.l0ns.weight *= 0.1
            self.l0nv.weight *= 0.1
            self.l0es.weight *= 0.1
            self.l0ev.weight *= 0.1
            self.l0ncharge.weight *= 0.01
            self.l1ns.weight *= 0.01
            self.l1nv.weight *= 0.01
            self.l1nt.weight *= 0.01
            self.l2.weight *= 0.01
            self.l2d.weight *= 0.01
            self.l3.weight *= 0.1
            self.l3evrv_diff.weight *= 0.1
            self.l3nv.weight *= 0.1
            self.l3evrv.weight *= 0.1
            self.l3ev.weight *= 0.1
            self.l3ncharge.weight *= 0.1
            self.l3ns_gate.weight *= 0.1
            self.l3nv_gate.weight *= 0.1
            self.l3nt_gate.weight *= 0.1
            self.l3ntc_gate.weight *= 0.1
            self.l4ns.weight *= 0.01
            self.l4v2.weight *= 0.01
            self.l4t.weight *= 0.01
            self.l4es.weight *= 0.01
            self.l4ev.weight *= 0.01

            # We manually set these bias values for 1.0 so that the gate
            # functions initially pass the values.
            self.l3ns_gate.bias.fill_(1.0)
            self.l3nv_gate.bias.fill_(1.0)
            self.l3nt_gate.bias.fill_(1.0)
            self.l3ntc_gate.bias.fill_(0.0)

    def forward(self, inputs):
        """Forward of TeaNet convolution

        Args:
            inputs (tuple): consists of following tensors.
                # --- Feature tensors ---
                ns_input (torch.Tensor): (n_nodes, ch)
                nv_input (torch.Tensor): (n_nodes, 3, ch)
                nt_input (torch.Tensor): (n_nodes, 3, 3, ch)
                es_input (torch.Tensor): (n_edges, ch)
                ev_input (torch.Tensor): (n_edges, 3, ch)
                # --- Constant tensors ---
                rv_input (torch.Tensor): (n_edges, 3, 1)
                rs_input_cutoff (torch.Tensor): (n_edges, 1) (=rs_input-cutoff)
                rt_input (torch.Tensor): (n_edges, 3, 3, 1)
                ni_to_x_indices (torch.Tensor): edge_index of src
                nj_to_x_indices (torch.Tensor): edge_index of dst

        Returns:
            ns_sum (torch.Tensor): node scalar feature (n_nodes, ch)
            nv_sum (torch.Tensor): node vector feature (n_nodes, 3, ch)
            nt_sum (torch.Tensor): node tensor feature (n_nodes, 3, 3, ch)
            es_sum (torch.Tensor): edge scalar feature (n_edges, ch)
            ev_sum (torch.Tensor): edge vector feature (n_edges, 3, ch)
        """
        (
            ns_input,
            nv_input,
            nt_input,
            es_input,
            ev_input,
            rv_input,
            rs_input_cutoff,
            rt_input,
            ni_to_x_indices,
            nj_to_x_indices,
            ncharge_input,
        ) = inputs

        n_nodes = ns_input.size()[0]
        n_edges = es_input.size()[0]

        # Make extend arr: for residual skip connection
        ns_input_extend = extend_input(ns_input, 1, self.n_ns_in, self.n_ns_out)
        nv_input_extend = extend_input(nv_input, 2, self.n_nv_in, self.n_nv_out)
        nt_input_extend = extend_input(nt_input, 3, self.n_nt_in, self.n_nt_out)
        es_input_extend = extend_input(es_input, 1, self.n_es_in, self.n_es_out)
        ev_input_extend = extend_input(ev_input, 2, self.n_ev_in, self.n_ev_out)
        ncharge_input_extend = extend_input(
            ncharge_input, 1, self.n_ncharge_in, self.n_ncharge_out
        )

        # --- Layer 0 ---
        # Just linear transformation + activation
        # grad of `sqrt` at x=0 is infinity, add `eps` to make backward stable.
        eps = 1.0e-4
        nv_scalar = torch.sqrt(torch.sum(nv_input * nv_input, dim=1) + eps) - math.sqrt(eps)
        ev_scalar = torch.sqrt(torch.sum(ev_input * ev_input, dim=1) + eps) - math.sqrt(eps)
        # nt_scalar = nt_input[:,0,0,:]+nt_input[:,1,1,:]+nt_input[:,2,2,:] # trace

        # Eq (7) of paper
        # Adding tensor values is optional.
        ns0 = (
            self.l0ns(ns_input) + self.l0ncharge(ncharge_input) + self.l0nv(nv_scalar)
        )  # + self.l0x3(nt_scalar)
        es0 = self.l0es(es_input) + self.l0ev(ev_scalar)

        ns1 = self.activation_shift(ns0)
        es1 = self.activation_shift(es0)

        # --- Layer 1 ---
        # Second linear transformation. Various nonlinear operation will be added in layer 2
        # so no activation here.
        ns1l = self.l1ns(ns1)
        nv1l = self.l1nv(nv_input)
        nt1l = self.l1nt(nt_input)

        # --- Layer 2 ---
        # Eq (8) of paper
        # Transfer node variables to corresponding edge.
        # There are two node-to-edge variables on each edge since there are two nodes.
        # From here, all x_i and x_j variables should be equally treated.
        xsi = index_select(ns1l, ni_to_x_indices)
        xsj = index_select(ns1l, nj_to_x_indices)
        xvi = index_select(nv1l, ni_to_x_indices).view((n_edges, 3, self.n_middle))
        xvj = index_select(nv1l, nj_to_x_indices).view((n_edges, 3, self.n_middle))

        # Create vector values by dot(tensor values, vector values)
        # The signs are inverted because only signs of rv_input depends on the order of i and j.
        # From here, all rv-related dot calculation affects the signs.
        xt = nt1l.view(n_nodes, 3, 3, self.n_middle)
        xti_rv = gather_mul_sum(xt, ni_to_x_indices, rv_input)
        xtj_rv = -gather_mul_sum(xt, nj_to_x_indices, rv_input)

        # Vector-origin values and tensor*position_vector values are summed.
        # Reason of just summation: Both vector and tensor values are transformed by
        # previous linear functions, so adding some linear transformation here do not
        # improve anything. Just gradient vanishing happen. (Takamoto)
        xvi_sum = xvi + xti_rv
        xvj_sum = xvj + xtj_rv

        del xvi, xti_rv, xvj, xtj_rv

        # At this point, various vector-style variables (xvi_sum,xvj_sum from nodes, ev_input
        # and rv_input from edge are calculated (on each edge).
        # The next step is to create some vector-vector inner product values.
        # Be careful about the signs.
        # Node-position (x2), node-node, node-edge (x2).
        xvi_rv = torch.sum(xvi_sum * rv_input, dim=1)
        xvj_rv = -torch.sum(xvj_sum * rv_input, dim=1)
        xvi_xvj = torch.sum(xvi_sum * xvj_sum, dim=1)
        # TODO: when `n_middle != n_es_in` it fails! Support this case?
        xvi_ev = torch.sum(xvi_sum * ev_input, dim=1)
        xvj_ev = torch.sum(xvj_sum * ev_input, dim=1)

        # Applying cutoff functions so that all x- variables are 0 at cutoff distance.
        # The parameters are also learnable (but they are manually initialized).
        # We use different parameters for xs and xv.
        # At least xsi and xsj (xvi and xvj, similarly) should use same cutoff functions.
        # xvi_xvj_len theoretically can have another parameters.
        (xs_cutoff, xv_cutoff) = torch.split(
            self.cutoff_function(rs_input_cutoff, self.lcut),
            [self.n_middle, self.n_middle],
            dim=1,
        )

        xsi_len = xsi * xs_cutoff
        xsj_len = xsj * xs_cutoff

        xvi_rv_len = xvi_rv * xv_cutoff
        xvj_rv_len = xvj_rv * xv_cutoff
        xvi_xvj_len = xvi_xvj * xv_cutoff

        del xsi, xsj, xvi_rv, xvj_rv, xvi_xvj

        # At this point, all x- variables are 0 at cutoff distance.
        # From now, all calculations should hold 0 to 0 so that all variables do not change by
        # inserting/eliminating an edge whose distance is cutoff.
        # It is noted all e- variables should also be 0 at cutoff distance.
        # This is achieved by carefully constructing whole operations so that all e- variables
        # holds 0 to 0.

        # Eq (9) of paper
        # Summation and subtraction. Here, *_sum values are order invariance
        # while signs of *_diff values depends on the order of i-j
        xs_sum = xsi_len + xsj_len
        xs_diff = xsi_len - xsj_len

        xv_rv_sum = xvi_rv_len + xvj_rv_len
        xv_rv_diff = xvi_rv_len - xvj_rv_len

        xv_ev_sum = xvi_ev + xvj_ev
        xv_ev_diff = xvi_ev - xvj_ev

        del xvi_rv_len, xvj_rv_len, xvi_ev, xvj_ev, xsi_len, xsj_len

        # At this point, all middle-layer variables are collected.
        # These variables are transformed by linear layer. It is noted that
        # [concat]->[linear] is equivalent to [linear]->[add].
        # Only readability and speed matters.
        # Here, only signs of xe_diff_concat depend on order of i and j.
        xe_concat_sum = torch.cat((xs_sum, es1, xv_rv_sum, xv_ev_sum, xvi_xvj_len), dim=1)
        xe_diff_concat = torch.cat((xs_diff, xv_rv_diff, xv_ev_diff), dim=1)

        del xs_sum, xs_diff, xv_rv_sum, xv_rv_diff, xv_ev_sum, xv_ev_diff, xvi_xvj_len

        # Usual activation function is applied for order-invariance values.
        # Square function is applied for *_diff values so that the invariance
        # holds for later calculations. (Also, 0^2 = 0.)
        m_sum = self.l2(xe_concat_sum)
        m_diff = self.l2d(xe_diff_concat)

        m = self.activation_shift(m_sum) + logcosh(m_diff)

        # --- Layer 3 ---
        # Eq (10) of paper
        # Various coefficients are calculated from m.
        # Then, values for various styles of outputs (n,e and s,v,t) are calculated.
        m_l = self.l3(m)
        (
            es_add_m,
            ns_add_m_x,
            nv_add_rv_coeff,
            ev_add_rv_coeff,
            nt_add_rt_coeff,
            ev_add_xv_coeff,
            ncharge_add_m_coeff,
        ) = torch.split(m_l, self.divide_m_l, dim=1)

        nv_add_ev_x = self.l3ev(ev_input)

        nt_add_evrv_pv_tile = self.l3evrv(ev_input).view(n_edges, 3, 1, self.n_nt_out)
        nt_add_evrv_rv_tile = rv_input.view(n_edges, 1, 3, 1)

        ncharge_add_m_x = ncharge_add_m_coeff * self.l3ncharge(m_diff)

        # Some variables for node outputs are transferred to corresponding nodes.
        # The readability should be polished. (Takamoto)
        # Again, signs should be carefully treated.
        ns_add_zero = torch.zeros_like(ns_input_extend)
        nv_add_zero = torch.zeros_like(nv_input_extend)
        nt_add_zero = torch.zeros_like(nt_input_extend)
        ncharge_add_zero = torch.zeros_like(ncharge_input_extend)

        ns_add_m = index_add(ns_add_zero, ni_to_x_indices, ns_add_m_x) + index_add(
            ns_add_zero, nj_to_x_indices, ns_add_m_x
        )

        nv_add_rv = mul_index_add_sub(
            nv_add_zero,
            rv_input,
            ni_to_x_indices,
            nj_to_x_indices,
            torch.unsqueeze(nv_add_rv_coeff, dim=1),
        )

        nv_add_ev = index_add(nv_add_zero, ni_to_x_indices, nv_add_ev_x) + index_add(
            nv_add_zero, nj_to_x_indices, nv_add_ev_x
        )

        nt_add_rt = mul_index_add_add(
            nt_add_zero,
            rt_input,
            ni_to_x_indices,
            nj_to_x_indices,
            nt_add_rt_coeff.view(n_edges, 1, 1, self.n_nt_out),
        )

        nt_add_evrv = mul_index_add_sub(
            nt_add_zero, nt_add_evrv_pv_tile, ni_to_x_indices, nj_to_x_indices, nt_add_evrv_rv_tile
        )

        ncharge_add_m = index_add(ncharge_add_zero, ni_to_x_indices, ncharge_add_m_x) - index_add(
            ncharge_add_zero, nj_to_x_indices, ncharge_add_m_x
        )

        ev_add_rv_ijdiff = self.l3evrv_diff(m_diff)
        ev_add_rv = rv_input * torch.unsqueeze(ev_add_rv_coeff * ev_add_rv_ijdiff, dim=1)

        ev_add_nv = (xvi_sum + xvj_sum) * ev_add_xv_coeff.view(n_edges, 1, self.n_middle)
        ev_add_nv_l = self.l3nv(ev_add_nv)

        ns_add_m_gate = self.l3ns_gate(ns1)
        nv_add_m_gate = self.l3nv_gate(ns1).view(n_nodes, 1, self.n_nv_out)
        nt_add_m_gate = self.l3nt_gate(ns1).view(n_nodes, 1, 1, self.n_nt_out)
        nt_add_const_m_gate = self.l3ntc_gate(ns1).view(n_nodes, 1, 1, self.n_nt_out)

        # --- Layer 4 ---
        # Eq (11), 2nd term of summation in the paper.
        # Just another linear transformation. Node to node and edge to edge
        # (without middle-layer mixing) are also created.
        ns_add_ns = self.l4ns(ns1)
        es_add_es = self.l4es(es1)

        nv_add_nv = self.l4v2(nv_input)
        nt_add_nt = self.l4t(nt_input)
        nt_add_nt_id = self.tensor_id.detach()
        nt_add_nt_const = nt_add_nt_id * nt_add_const_m_gate
        ev_add_ev = self.l4ev(ev_input)

        # --- Layer 5 ---
        # Eq (11) of paper.
        # Applying gate function and resnet-style skip connection.
        ns_sum = ns_input_extend + ns_add_ns + ns_add_m * ns_add_m_gate
        nv_sum = nv_input_extend + nv_add_nv + (nv_add_rv + nv_add_ev) * nv_add_m_gate
        nt_sum = (
            nt_input_extend
            + nt_add_nt
            + nt_add_nt_const
            + (nt_add_rt + nt_add_evrv) * nt_add_m_gate
        )

        es_sum = es_input_extend + es_add_es + es_add_m
        ev_sum = ev_input_extend + ev_add_nv_l + ev_add_rv + ev_add_ev

        if self.use_bn:
            ns_sum = self.bn_node(ns_sum)
            es_sum = self.bn_edge(es_sum)
        if self.dropout_ratio > 0.0:
            ns_sum = F.dropout(ns_sum, p=self.dropout_ratio, training=self.training)
            es_sum = F.dropout(es_sum, p=self.dropout_ratio, training=self.training)

        ncharge_out = ncharge_input_extend + ncharge_add_m

        return ns_sum, nv_sum, nt_sum, es_sum, ev_sum, ncharge_out
