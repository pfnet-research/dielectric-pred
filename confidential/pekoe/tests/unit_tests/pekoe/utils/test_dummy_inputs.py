from typing import cast

import torch

from pekoe.nn.models.model_builder import build_estimator
from pekoe.nn.models.teanet.model_pfp_v1_3.teanet import TeaNet_v1_3
from pekoe.nn.models.teanet_estimator import TeaNetEstimator
from pekoe.utils.dummy_inputs import DUMMY_INPUT, create_dummy_input


def test_dummy_input(device_config) -> None:
    device, codegen_options = device_config
    estimator = cast(
        TeaNetEstimator, build_estimator("v1.4.1", device=device, codegen_options=codegen_options)
    )
    expected_result = estimator.estimate(create_dummy_input())
    expected_energy = expected_result["energy"]
    if device.startswith("pfvm"):
        from pekoe.nn.models.teanet.codegen_teanet import CodeGenTeaNet

        assert isinstance(estimator.model, CodeGenTeaNet)
        energy, _, _, _ = estimator.model(
            DUMMY_INPUT.coordinates,
            DUMMY_INPUT.atomic_numbers,
            DUMMY_INPUT.left_indices,
            DUMMY_INPUT.right_indices,
            DUMMY_INPUT.shift,
            DUMMY_INPUT.cell,
            DUMMY_INPUT.num_graphs,
            DUMMY_INPUT.batch,
            DUMMY_INPUT.batch_edge,
            DUMMY_INPUT.x_add,
            DUMMY_INPUT.calc_mode_type,
        )
    else:
        assert isinstance(estimator.model, TeaNet_v1_3)
        with torch.enable_grad():
            energy, _ = estimator.model.forward(
                DUMMY_INPUT.vecs.requires_grad_(True).to(device=device, dtype=torch.float32),
                DUMMY_INPUT.atomic_numbers.to(device=device),
                DUMMY_INPUT.left_indices.to(device=device),
                DUMMY_INPUT.right_indices.to(device=device),
                DUMMY_INPUT.shift.to(device=device),
                DUMMY_INPUT.batch.to(device=device),
                DUMMY_INPUT.batch_edge.to(device=device),
                DUMMY_INPUT.x_add.to(device=device),
                DUMMY_INPUT.calc_mode_type.to(device=device),
            )
    predicted_energy = float(energy.item()) + estimator.shifted_energy(DUMMY_INPUT.atomic_numbers)
    assert expected_energy == predicted_energy
