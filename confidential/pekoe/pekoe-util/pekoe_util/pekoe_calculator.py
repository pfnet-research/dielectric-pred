import os
from typing import Optional

from ase.calculators.calculator import Calculator

from pekoe.calculators.ase_calculator import ASECalculator
from pekoe.nn.estimator_base import EstimatorCalcMode
from pekoe.nn.models import DEFAULT_MODEL
from pekoe.nn.models.model_builder import build_estimator


def get_calculator(calc_mode: str = "CRYSTAL") -> Calculator:
    estimator = build_estimator(DEFAULT_MODEL, device="auto")
    calculator = ASECalculator(estimator)

    calc_mode = os.environ.get("MATLANTIS_PFP_CALC_MODE", calc_mode)
    calculator.estimator.calc_mode = getattr(
        EstimatorCalcMode, calc_mode, EstimatorCalcMode.CRYSTAL
    )
    return calculator
