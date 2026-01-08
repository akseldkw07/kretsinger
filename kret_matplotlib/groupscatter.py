import typing as t

import numpy as np

from kret_rosetta.to_pd_np import TO_NP_TYPE

REG_FUNC = t.Literal["OLS", "Huber"]


class GroupScatter:
    """
    Plot a group scatter with regression line. Helpful for visualizing model performance, especially when there are thousands or millions of points.

    Take in y and y_hat, optional categorical column, optional filter, # centroids=25, downsample=False, and regression funcion (OLS, Huber, etc)=OLS
    """

    y_true: np.ndarray
    y_pred: np.ndarray
    category: np.ndarray | None
    filter_mask: np.ndarray | None
    n_centroids: int
    downsample: bool
    regression_func: REG_FUNC

    def __init__(self, y: TO_NP_TYPE, y_hat: TO_NP_TYPE): ...
