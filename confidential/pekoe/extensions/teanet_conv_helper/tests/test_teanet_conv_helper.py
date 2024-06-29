import pytest

import torch

from teanet_conv_helper import (
    gather_mul_sum,
    mul_index_add_add,
    mul_index_add_sub,
)


devices = ["cpu"]
try:
    torch.zeros(1).to("cuda")
    devices.append("cuda")
except:
    pass


@pytest.mark.parametrize("device", devices)
def test_gather_mul_sum(device):
    def gather_mul_sum_python(x, i, f):
        return (x[i] * torch.unsqueeze(f, dim=1)).sum(dim=2)

    num_atoms = 7
    num_edges = 23

    with torch.enable_grad():
        x = torch.rand(num_atoms, 3, 3, 128).to(device).requires_grad_(True)
        i = torch.randint(num_atoms, (num_edges,)).to(device)
        f = torch.rand(num_edges, 3, 1).to(device).requires_grad_(True)
        gy = torch.rand(num_edges, 3, 128).to(device)

        e = gather_mul_sum_python(x, i, f)
        a = gather_mul_sum(x, i, f)
        assert torch.allclose(e, a)

        gx, gf = torch.autograd.grad(e, (x, f), gy)
        gxa, gfa = torch.autograd.grad(a, (x, f), gy)

        assert torch.allclose(gx, gxa)
        assert torch.allclose(gf, gfa)


@pytest.mark.parametrize("device", devices)
def test_mul_index_add_twice(device):
    def mul_index_add_add_python(x, y, i, j, f):
        z = y * f
        r = x.index_add(0, i, z)
        r = r.index_add(0, j, z)
        return r

    def mul_index_add_sub_python(x, y, i, j, f):
        z = y * f
        r = -x.index_add(0, j, z)
        r = r.index_add(0, i, z)
        return r

    num_atoms = 7
    num_edges = 23

    for fn, py_fn in [(mul_index_add_add, mul_index_add_add_python),
                      (mul_index_add_sub, mul_index_add_sub_python)]:
        with torch.enable_grad():
            x = torch.rand(num_atoms, 3, 3, 128).to(device)
            y = torch.rand(num_edges, 3, 3, 128).to(device).requires_grad_(True)
            i = torch.randint(num_atoms, (num_edges,)).to(device)
            j = torch.randint(num_atoms, (num_edges,)).to(device)
            f = torch.rand(num_edges, 3, 3, 1).to(device).requires_grad_(True)
            gx = torch.rand(num_atoms, 3, 3, 128).to(device)

            e = py_fn(x, y, i, j, f)
            a = fn(x, y, i, j, f)
            assert torch.allclose(e, a, atol=1e-6)

            gy, gf = torch.autograd.grad(e, (y, f), gx)
            gya, gfa = torch.autograd.grad(a, (y, f), gx)

            assert torch.allclose(gy, gya)
            assert torch.allclose(gf, gfa)
