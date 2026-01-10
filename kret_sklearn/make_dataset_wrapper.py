import typing as t

import numpy as np
import pandas as pd
import sklearn.datasets

from kret_rosetta.UTILS_rosetta import UTILS_rosetta

from .constants_sklearn import SklearnDefaults
from .typed_cls_sklean import (
    MakeClassification_Params_TypedDict,
    MakeMultilabelClassification_Params_TypedDict,
    MakeRegression_Params_TypedDict,
)


class RegressionTuple(t.NamedTuple):
    x: pd.DataFrame
    y: np.ndarray
    coef: np.ndarray


class ClassificationTuple(t.NamedTuple):
    x: pd.DataFrame
    y: np.ndarray


class MultiLabelClassificationTuple(t.NamedTuple):
    x: pd.DataFrame
    y: pd.DataFrame
    p_c: np.ndarray
    p_w_c: pd.DataFrame


class MakeSklearnDatasetsWrapper:
    @classmethod
    def make_regression(cls, **params: t.Unpack[MakeRegression_Params_TypedDict]):

        args = SklearnDefaults.MAKE_REGRESSION_PARAMS_SKLEARN | params

        out = sklearn.datasets.make_regression(**args)
        x_df = UTILS_rosetta.coerce_to_df(out[0])
        out = (x_df, out[1], out[2])

        ret = RegressionTuple(*out)
        return ret

    @classmethod
    def make_classification(cls, **params: t.Unpack[MakeClassification_Params_TypedDict]):

        args = SklearnDefaults.MAKE_CLASSIFICATION_PARAMS_SKLEARN | params

        out = sklearn.datasets.make_classification(**args)
        x_df = UTILS_rosetta.coerce_to_df(out[0])
        out = (x_df, out[1])

        ret = ClassificationTuple(*out)
        return ret

    @classmethod
    def make_multilabel_classification(cls, **params: t.Unpack[MakeMultilabelClassification_Params_TypedDict]):

        args = SklearnDefaults.MAKE_MULTILABEL_CLASSIFICATION_PARAMS_SKLEARN | params

        out = sklearn.datasets.make_multilabel_classification(**args)
        x_df = UTILS_rosetta.coerce_to_df(out[0])
        y_df = UTILS_rosetta.coerce_to_df(out[1])
        p_c = out[2]
        p_w_c = UTILS_rosetta.coerce_to_df(out[3])

        ret = MultiLabelClassificationTuple(x=x_df, y=y_df, p_c=p_c, p_w_c=p_w_c)
        return ret
