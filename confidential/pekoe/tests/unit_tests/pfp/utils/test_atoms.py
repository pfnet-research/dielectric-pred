import pytest

from pfp.utils.atoms import atomic_number, atomic_symbol


@pytest.mark.parametrize(
    "symbol, number",
    [
        ("H", 1),
        ("C", 6),
        ("O", 8),
        ("Cl", 17),
    ],
)
def test_atoms(symbol, number):
    assert atomic_number(symbol) == number
    assert atomic_symbol(number) == symbol
