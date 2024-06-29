import torch
import torch.nn.init as Init

from .select_activation import select_cutoff_function


class Twobody(torch.nn.Module):
    def __init__(self, n_x_init: int, cutoff: float = 6.0, cutoff_function: str = "invnegexp"):
        super(Twobody, self).__init__()
        self.n_x_init = n_x_init
        self.cutoff = cutoff
        self.cutoff_id = cutoff_function

        lookup = torch.zeros((n_x_init * n_x_init, 3))

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
        (ns_input, left_indices, right_indices, rs_input) = data
        nl = ns_input[left_indices]
        nr = ns_input[right_indices]

        n_index = nl * self.n_x_init + nr
        params = self.lookup[n_index]

        a = params[:, 0:1]
        re1 = params[:, 1:2]
        re2 = params[:, 2:3]

        edge_out = torch.exp(-2.0 * a * (rs_input - re1)) - 2.0 * torch.exp(
            -1.0 * a * (rs_input - re2)
        )
        edge_cutoff = self.cutoff_function(rs_input - self.lc_coeff, self.lcuts)
        return edge_out * edge_cutoff
