from __future__ import annotations

import typing as t

import numpy as np

NPT = t.TypeVar('NPT', bound=t.Any)
T = t.TypeVar('T')


class SingleReturnArray(np.ndarray, t.Generic[NPT]):
    """
    A subclass of np.ndarray that returns a single value when indexed with a single index.
    """

    def __getitem__(self, item: int) -> NPT:  # type: ignore
        return super().__getitem__(item)

    @t.overload
    def ravel(self: SingleReturnArray[SingleReturnArray[SingleReturnArray[T]]]) -> SingleReturnArray[T]: ...
    @t.overload
    def ravel(self: SingleReturnArray[SingleReturnArray[T]]) -> SingleReturnArray[T]: ...
    @t.overload
    def ravel(self: SingleReturnArray[T]) -> SingleReturnArray[T]: ...

    def ravel(self) -> SingleReturnArray[T]:  # type: ignore
        return super().ravel()  # type: ignore


def solve_linear_system(A: np.ndarray, b: np.ndarray) -> SingleReturnArray[np.ndarray]:
    """
    Solve the linear system Ax = b using NumPy's linear algebra solver.

    Parameters:
        A (np.ndarray): Coefficient matrix.
        b (np.ndarray): Right-hand side vector.

    Returns:
        SingleReturnArray[np.ndarray]: Solution vector x.
    """
    try:

        x = np.linalg.solve(A, b)  # only works if A is square and invertible
    except np.linalg.LinAlgError as e:
        x = np.linalg.lstsq(A, b, rcond=None)[0]
    return x.view(SingleReturnArray)
