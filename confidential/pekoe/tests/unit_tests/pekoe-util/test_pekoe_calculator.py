from pekoe_util.pekoe_calculator import get_calculator

from pekoe.calculators.ase_calculator import ASECalculator


def test_get_calculator() -> None:
    calc = get_calculator()
    assert isinstance(calc, ASECalculator)


def test_get_calculator_with_statement() -> None:
    with get_calculator() as calc:
        assert isinstance(calc, ASECalculator)
