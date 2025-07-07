from __future__ import annotations

import numpy as np


def apply_polynomial_features(X_input: np.ndarray):
    """
    Applies the polynomial feature transformation phi(x1, x2) = (x1^2, x1*x2, x2^2)
    to each row of the input matrix X_input.

    Args:
        X_input (np.ndarray): A 2D numpy array where each row is [x1, x2].

    Returns:
        np.ndarray: A 2D numpy array with transformed features.
    """
    if X_input.shape[1] != 2:
        raise ValueError("Input matrix X_input must have 2 columns (x1, x2).")

    X_phi = np.zeros((X_input.shape[0], 3))  # Initialize new feature matrix with 3 columns

    # x1^2
    X_phi[:, 0] = X_input[:, 0] ** 2
    # x1*x2
    X_phi[:, 1] = X_input[:, 0] * X_input[:, 1]
    # x2^2
    X_phi[:, 2] = X_input[:, 1] ** 2

    return X_phi
