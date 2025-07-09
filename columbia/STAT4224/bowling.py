from sympy import log
from sympy.core.mul import Mul
from sympy.core.numbers import Float
from kret_studies.type_checking import assert_type

Omega = list(range(11))  # 0, 1, ..., 10 but EXCLUDES 11


def Pr(x: int, n: int = 10):
    assert isinstance(x, int)
    assert isinstance(n, int)
    assert n >= 0 & n <= 10
    if x > n | x < 0:
        return 0
    ret = assert_type(Mul, log(1 + 1 / (n + 1 - x), n + 2))
    return float(assert_type(Float, ret.evalf()))
