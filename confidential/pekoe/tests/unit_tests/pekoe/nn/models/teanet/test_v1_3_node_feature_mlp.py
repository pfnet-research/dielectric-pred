import pytest

from pekoe.nn.models import DEFAULT_MODEL_DIRECTORY
from pekoe.nn.models.config import BaseConfig
from pekoe.nn.models.teanet.teanet_v1_3_node_feature_mlp import TeaNetNodeFeatureMLPParameters_v1_3


@pytest.mark.parametrize("version", ["0_0_1", "0_0_2", "0_0_3", "0_0_4"])
def test_from_yaml(version: str) -> None:
    base_config = BaseConfig.from_yaml(
        DEFAULT_MODEL_DIRECTORY / f"model_v1_3_1_ccsd_t_correction_{version}.yaml"
    )

    parameters = TeaNetNodeFeatureMLPParameters_v1_3.from_dict(base_config.parameters)
    assert len(parameters.n_hiddens) > 0
