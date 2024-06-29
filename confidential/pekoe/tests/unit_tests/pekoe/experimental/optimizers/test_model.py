import ase
import numpy as np
import pytest
import torch

from pekoe.nn.models import DEFAULT_MODEL
from pekoe.nn.models.model_builder import build_estimator
from pfp.calculators.ase_calculator import ASECalculator


def get_ase_mols():
    mol0 = ase.Atoms(
        "H2O",
        [
            [0, 0, 0],
            [1.2, 0.9, 0],
            [1, 0, 0],
        ],
    )
    mol1 = ase.Atoms(
        "HCl",
        [
            [0.1, -0.2, 0.3],
            [0.9, 0.7, 0.8],
        ],
    )
    return [mol0, mol1]


def asnumpy(tensor):
    return tensor.detach().cpu().numpy()


@pytest.mark.gpu
@pytest.mark.experimental
def test_compare_with_ase_calculator(cupy):
    from pekoe.experimental.optimizers.data import ase_get
    from pekoe.experimental.optimizers.model import MyTeaNet

    device = "cuda:0"
    cupy_device = cupy.cuda.Device(0)

    estimator = build_estimator(DEFAULT_MODEL, device=device)

    net = MyTeaNet(estimator.model)
    net.calc_mode = 0  # FIXME

    ase_mols = get_ase_mols()
    num_mols = len(ase_mols)

    with cupy_device:
        # input
        cupy_mols = [ase_get(mol) for mol in ase_mols]
        ase_mols = [mol.copy() for mol in ase_mols]
        for mol in ase_mols:
            mol.calc = ASECalculator(estimator)

        # forward (energy)
        y, xs = net.forward(cupy_mols)
        assert len(xs) == num_mols
        for i, mol in enumerate(ase_mols):
            np.testing.assert_allclose(
                asnumpy(xs[i]),
                mol.positions,
                rtol=1e-6,
                atol=1e-6,
            )

        y_numpy = asnumpy(y)
        assert y_numpy.shape == (num_mols,)
        for i, mol in enumerate(ase_mols):
            np.testing.assert_allclose(
                y_numpy[i],
                mol.get_potential_energy(),
                rtol=1e-5,
                atol=1e-5,
            )

        # grad (-force)
        gxs = torch.autograd.grad([y], xs, [torch.ones_like(y)], create_graph=True)
        del y
        for i, mol in enumerate(ase_mols):
            np.testing.assert_allclose(
                asnumpy(gxs[i]),
                -mol.get_forces(),
                rtol=1e-4,
                atol=1e-4,
            )

        # gradgrad (hessian)
        # torch.autograd.gradcheck(
        #     (
        #         lambda *xs: torch.autograd.grad(
        #             [y], xs, [torch.ones_like(y)],
        #             retain_graph=True, create_graph=True)
        #     ),
        #     xs,
        #     eps=1e-5, rtol=1e-4, atol=1e-4,
        # )

        rs = np.random.RandomState(seed=42)
        dxs = [rs.normal(size=(len(mol), 3)) for mol in ase_mols]
        hxdxs = torch.autograd.grad(
            gxs,
            xs,
            [torch.tensor(dx, dtype=torch.float32, device=device) for dx in dxs],
        )

        eps = 1e-4
        for i, mol in enumerate(ase_mols):
            grads = {}
            old_positions = mol.positions.copy()
            for delta in [-1, 1]:  # [-2, -1, 1, 2]:
                mol.positions += delta * eps * dxs[i]
                grads[delta] = -mol.get_forces()
                mol.positions[...] = old_positions

            numerical_hxdx = (grads[1] - grads[-1]) / (2 * eps)
            # numerical_hxdx = (
            #     grads[-2]
            #     - 8 * grads[-1]
            #     + 8 * grads[1]
            #     - grads[2]
            # ) / (12 * eps)
            np.testing.assert_allclose(
                asnumpy(hxdxs[i]),
                numerical_hxdx,
                rtol=5e-3,
                atol=1e-4,
            )
