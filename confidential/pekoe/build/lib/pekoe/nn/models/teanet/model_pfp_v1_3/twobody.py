import torch
import torch.nn.init as Init

from .select_activation import select_cutoff_function


class Twobody(torch.nn.Module):
    def __init__(self, n_x_init: int, cutoff: float = 6.0, cutoff_function: str = "invnegexp"):
        super(Twobody, self).__init__()
        n_calc_mode = 2

        self.n_x_init = n_x_init
        self.n_x2 = n_x_init * n_x_init
        self.cutoff = cutoff
        self.cutoff_id = cutoff_function

        lookup = torch.zeros((n_x_init * n_x_init * n_calc_mode, 3))

        self.register_buffer("lookup", lookup)

        lc_coeff = torch.tensor([[self.cutoff]])
        self.register_buffer("lc_coeff", lc_coeff)

        assert self.cutoff_id is not None
        self.lcuts = torch.nn.Linear(1, 1, bias=False)

        self.cutoff_function = select_cutoff_function(self.cutoff_id)

        self.reset_parameters()

    def reset_parameters(self):

        Init.constant_(self.lcuts.weight, -1.0)

    def forward(self, *data):
        (ns_input, calc_mode_type, left_indices, right_indices, rs_input) = data
        nl = ns_input[left_indices]
        nr = ns_input[right_indices]
        calc_mode_type_l = calc_mode_type[left_indices]

        n_index = (nl * self.n_x_init + nr) + self.n_x2 * calc_mode_type_l
        params = self.lookup[n_index]

        a = params[:, 0:1]
        re1 = params[:, 1:2]
        re2 = params[:, 2:3]

        edge_out = torch.exp(-2.0 * a * (rs_input - re1)) - 2.0 * torch.exp(
            -1.0 * a * (rs_input - re2)
        )
        edge_cutoff = self.cutoff_function(rs_input - self.lc_coeff, self.lcuts)
        return edge_out * edge_cutoff


class Twobody2(torch.nn.Module):
    def __init__(
        self,
        n_ns: int,
        n_nv: int,
        n_nt: int,
        cutoff: float = 15.0,
        cutoff_function: str = "invnegexp",
    ):
        super(Twobody2, self).__init__()
        self.n_ns = n_ns
        self.n_nv = n_nv
        self.n_nt = n_nt

        self.register_parameter("q_coeff", torch.nn.Parameter(torch.tensor([14.3996])))

        lc_coeff = torch.tensor([[cutoff]])
        self.register_buffer("lc_coeff", lc_coeff)

        self.lcuts = torch.nn.Linear(1, 1, bias=False)

        self.cutoff_function = select_cutoff_function(cutoff_function)

    def reset_parameters(self):
        Init.constant_(self.lcuts.weight, -1.0)

    def forward(self, *data):
        (nq, ns, nv, nt, left_indices, right_indices, rs_input) = data
        # nq_l = nq[left_indices]
        # nq_r = nq[right_indices]
        ns_l = ns[left_indices]
        ns_r = ns[right_indices]
        # nv_l = nv[left_indices]
        # nv_r = nv[right_indices]
        # nt_l = nt[left_indices]
        # nt_r = nt[right_indices]
        n_edges = left_indices.size(0)
        # rs_input_div = 1.0 / (rs_input + 2.0)
        # edge_out = self.q_coeff * nq_l * nq_r * rs_input_div
        # edge_out = ns_l * ns_r / (1.0 + (torch.sqrt(0.1 + ns_l * ns_l + ns_r * ns_r))) * rs_input_div
        edge_out = ns_l + ns_r  # * rs_input_div
        edge_cutoff = self.cutoff_function(rs_input - self.lc_coeff, self.lcuts)
        edge_out = edge_out * edge_cutoff
        node_out_zero = torch.zeros_like(ns)
        node_out = node_out_zero.scatter_add(
            0,
            left_indices.view(n_edges, 1).expand(n_edges, node_out_zero.size(1)),
            edge_out,
        ) + node_out_zero.scatter_add(
            0,
            right_indices.view(n_edges, 1).expand(n_edges, node_out_zero.size(1)),
            edge_out,
        )
        return node_out
