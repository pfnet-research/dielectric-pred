from pekoe.nn.models import DEFAULT_MODEL_DIRECTORY
from pekoe.nn.models.config import BaseConfig
from pekoe.nn.models.teanet.model_pfp_v1_2.teanet import TeaNetParameters_v1_2


def test_from_yaml(pytestconfig):
    base_config = BaseConfig.from_yaml(DEFAULT_MODEL_DIRECTORY / "model_v1_2_0.yaml")

    parameters = TeaNetParameters_v1_2.from_dict(base_config.parameters)
    if isinstance(parameters.cutoff, float):
        assert parameters.cutoff >= 0.0
    else:
        assert min(parameters.cutoff) >= 0.0
