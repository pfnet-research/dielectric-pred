from enum import IntEnum
from typing import List

# (symbol, supported)
_ATOMIC_LIST: List[str] = [
    "_",
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
    "Uue",
    "Ubn",
    "Ubu",
    "Ubb",
    "Ubt",
    "Ubq",
    "Ubp",
    "Ubh",
    "Ubs",
]

_ATOMIC_NUMBERS = {v: i for i, v in enumerate(_ATOMIC_LIST) if i > 0}
_LEN_ATOMIC_LIST = len(_ATOMIC_LIST)


class ElementStatusEnum(IntEnum):
    Expected = 0
    Experimental = 1
    Unexpected = 2
    Illegal = 3


def max_atomic_number() -> int:
    return _LEN_ATOMIC_LIST


def atomic_number(symbol: str) -> int:
    if symbol not in _ATOMIC_NUMBERS:
        raise InvalidAtomicSymbolError
    return _ATOMIC_NUMBERS[symbol]


def atomic_symbol(number: int) -> str:
    if number < 1 or number >= max_atomic_number():
        raise InvalidAtomicNumberError
    return _ATOMIC_LIST[number]


class InvalidAtomicNumberError(Exception):
    """The atomic number is invalid"""

    pass


class InvalidAtomicSymbolError(Exception):
    """The atomic symbol is invalid"""

    pass
