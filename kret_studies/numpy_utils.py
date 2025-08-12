from __future__ import annotations

import typing as t

import numpy as np

NPT = t.TypeVar("NPT", bound=t.Any)
T = t.TypeVar("T")


class SingleReturnArray(np.ndarray, t.Generic[NPT]):
    """
    A subclass of np.ndarray that returns a single value when indexed with a single index.
    """

    def __getitem__(self, key: int) -> NPT:  # type: ignore
        return super().__getitem__(key)

    def __getitem__(self: SingleReturnArray[SingleReturnArray[T]], key: tuple[int, int]) -> T:  # type: ignore
        return super().__getitem__(key)

    # def __getitem__(self: SingleReturnArray[SingleReturnArray[SingleReturnArray[T]]], key: tuple[int, int]) -> SingleReturnArray[T]:  # type: ignore
    #     return super().__getitem__(key)

    @t.overload
    def ravel(self: SingleReturnArray[SingleReturnArray[SingleReturnArray[T]]]) -> SingleReturnArray[T]: ...
    @t.overload
    def ravel(self: SingleReturnArray[SingleReturnArray[T]]) -> SingleReturnArray[T]: ...
    @t.overload
    def ravel(self: SingleReturnArray[T]) -> SingleReturnArray[T]: ...

    def ravel(self) -> SingleReturnArray[T]:  # type: ignore
        return super().ravel()  # type: ignore


def solve_linear_system(X: np.ndarray, y: np.ndarray):
    """
    Solve the linear system Ax = b using NumPy's linear algebra solver.

    Parameters:
        A (np.ndarray): Coefficient matrix.
        b (np.ndarray): Right-hand side vector.

    Returns:
        SingleReturnArray[np.ndarray]: Solution vector x.
    """

    x = np.linalg.lstsq(X, y, rcond=None)
    return x


def solve_linear_system_by_hand(X: np.ndarray, y: np.ndarray):
    """
    Solve the linear system Ax = b by hand using NumPy's matrix operations.

    Parameters:
        A (np.ndarray): Coefficient matrix.
        b (np.ndarray): Right-hand side vector.
    Returns:

        SingleReturnArray[np.ndarray]: Solution vector x.
    """
    w_hat = np.linalg.inv(X.T @ X) @ X.T @ y
    return w_hat.view(SingleReturnArray)


def solve_psuedo_inverse(X: np.ndarray):
    return np.linalg.pinv(X)
