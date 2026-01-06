import typing as t

import numpy as np
import pandas as pd

from .translate_libraries import PD_NP_Torch_Translation


class Enriched_DF(pd.DataFrame):
    """
    TODO docstring
    """

    target_dtypes: t.ClassVar[dict[str, type | str]] = {}

    @classmethod
    def column_order(cls) -> list[str]:
        # TODO use get_type_hints to respect inheritance and typing.get_origin/args for generics
        return list(cls.__annotations__.keys())  # TODO modify pandas insert to update internal container in-place

    @property
    def f_valid(self) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement f_valid property.")

    def validate_dtypes(self):
        """
        Validate that the DataFrame's columns have the expected dtypes
        as specified in `target_dtypes`.
        """
        for col, target_dtype in self.target_dtypes.items():
            if col not in self.columns:
                raise KeyError(f"{self.__class__.__name__}.validate_dtypes: column '{col}' not found in DataFrame.")

            actual_dtype = self[col].dtype.type
            if actual_dtype != target_dtype:
                raise TypeError(
                    f"{self.__class__.__name__}.validate_dtypes: column '{col}' has dtype "
                    f"{actual_dtype}, expected {target_dtype}."
                )

    def to_obs_numpy(self):
        """Return a pure-numeric numpy array for model input (no datetime/timedelta/object)."""
        df: pd.DataFrame = self.copy()[self.column_order()]

        ret = PD_NP_Torch_Translation.df_to_np_safe(df)
        return ret


T = t.TypeVar("T", bound=pd.DataFrame)


class EnrichedDFUtils:
    @classmethod
    def validate_typed_df_keys(
        cls, df: pd.DataFrame | dict, df_type: type[T], action: t.Literal["warn", "raise"] = "raise"
    ) -> bool:
        """
        Validate that a pandas DataFrame conforms to the specified typed DataFrame structure.
        TODO convert down to warning instead of raising
        """
        base_class_attrs = set(vars(Enriched_DF).keys())
        exp_cols = df_type.target_dtypes.keys() if isinstance(df_type, Enriched_DF) else df_type.__annotations__.keys()
        expected_columns = set(exp_cols) - base_class_attrs
        actual_columns = set(df.keys()) - base_class_attrs

        missing = expected_columns - actual_columns
        extra = actual_columns - expected_columns

        if action == "raise":
            assert (
                not missing and not extra
            ), f"DataFrame columns do not match expected structure. Missing: {missing}, Extra: {extra}"

        if missing:
            print(f"WARNING: Missing columns in DataFrame: {missing}")
        if extra:
            print(f"WARNING: Extra columns in DataFrame: {extra}")

        return not missing and not extra
