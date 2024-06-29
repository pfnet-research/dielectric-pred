from pekoe.nn.models.teanet.model_pfp_v1.teanet import TeaNet_v1, TeaNetParameters_v1
from pekoe.nn.models.teanet.model_pfp_v1_1.teanet import TeaNet_v1_1, TeaNetParameters_v1_1
from pekoe.nn.models.teanet.model_pfp_v1_2.teanet import TeaNet_v1_2, TeaNetParameters_v1_2
from pekoe.nn.models.teanet.model_pfp_v1_3.teanet import TeaNet_v1_3, TeaNetParameters_v1_3
from pekoe.nn.models.teanet.teanet_v1_3_node_feature_mlp import (
    TeaNetNodeFeatureMLP_v1_3,
    TeaNetNodeFeatureMLPParameters_v1_3,
)

TeaNet = TeaNet_v1_3
TeaNetParameters = TeaNetParameters_v1_3

__all__ = [
    "TeaNet",
    "TeaNet_v1",
    "TeaNet_v1_1",
    "TeaNet_v1_2",
    "TeaNet_v1_3",
    "TeaNetNodeFeatureMLP_v1_3",
    "TeaNetParameters",
    "TeaNetParameters_v1",
    "TeaNetParameters_v1_1",
    "TeaNetParameters_v1_2",
    "TeaNetParameters_v1_3",
    "TeaNetNodeFeatureMLPParameters_v1_3",
]
