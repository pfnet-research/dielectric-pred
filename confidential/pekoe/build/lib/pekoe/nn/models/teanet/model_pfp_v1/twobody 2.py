import torch
import torch.nn.init as Init

from .select_activation import TeaNetCutoffFunction, select_cutoff_function


class Twobody(torch.nn.Module):
    def __init__(
        self,
        n_x_init: int,
        n_middle: int = 128,
        cutoff: float = 6.0,
        cutoff_function: TeaNetCutoffFunction = TeaNetCutoffFunction.INVNEGEXP,
    ):
        super(Twobody, self).__init__()
        self.n_x_init = n_x_init
        self.n_middle = n_middle
        self.cutoff = cutoff
        self.cutoff_id = cutoff_function
        length_factor = 4.0

        self.lin1 = torch.nn.Linear(n_x_init, n_middle)
        self.lin2 = torch.nn.Linear(n_x_init, n_middle, bias=False)
        self.lin3 = torch.nn.Linear(2 * n_middle, 2 * n_middle)
        self.params_arr = torch.nn.Linear(2 * n_middle, 3)

        lc_coeff = torch.tensor([[self.cutoff]])
        self.register_buffer("lc_coeff", lc_coeff)

        assert self.cutoff_id is not None
        self.lcuts = torch.nn.Linear(1, 1, bias=False)

        self.cutoff_function = select_cutoff_function(self.cutoff_id)

        self.reset_parameters(length_factor)

    def reset_parameters(self, length_factor):
        Init.constant_(self.lcuts.weight, -1.0)

    def forward(self, *data):
        (ns_input, left_indices, right_indices, rs_input) = data
        nl = ns_input[left_indices]
        nr = ns_input[right_indices]

        n1 = torch.nn.functional.relu(self.lin1(nl + nr))
        n2_d = self.lin2(nl - nr)
        n2 = n2_d * n2_d
        n3 = torch.cat((n1, n2), dim=1)
        n4 = torch.nn.functional.relu(self.lin3(n3))

        params = self.params_arr(n4)
        a = params[:, 0:1]
        re1 = params[:, 1:2]
        re2 = params[:, 2:3]

        edge_out = torch.exp(-2.0 * a * (rs_input - re1)) - 2.0 * torch.exp(
            -1.0 * a * (rs_input - re2)
        )
        edge_cutoff = self.cutoff_function(rs_input - self.lc_coeff, self.lcuts)
        return edge_out * edge_cutoff
