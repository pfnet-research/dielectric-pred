import pathlib
from typing import Optional

from pekoe.calculators.ase_calculator import ASECalculator
from pekoe.nn.models import DEFAULT_MODEL
from pekoe.nn.models.model_builder import build_estimator
from pekoe.nn.models.teanet.codegen_options import CodeGenOptions

from .ase_calculator import ASECalculator  # NOQA


def build_ase_calculator(
    model_path: pathlib.Path = DEFAULT_MODEL,
    device: str = "auto",
    codegen_options: Optional[CodeGenOptions] = None,
) -> ASECalculator:
    return ASECalculator(
        build_estimator(model_path, device=device, codegen_options=codegen_options)
    )
