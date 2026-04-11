import typing as t

import polars as pl


class Enriched_DF_PL(pl.DataFrame):
    """
    Polars equivalent of Enriched_DF (kret_np_pd.enriched_df).

    A typed pl.DataFrame subclass with:
    - column_order() derived from annotations
    - dtype validation
    - col_filter for substring-based column selection
    - print_th() for generating class stubs

    NOTE: Polars operations (filter, select, join, etc.) return a plain pl.DataFrame,
    losing subclass identity. Designed for "construct & inspect" workflows.
    """

    target_dtypes: t.ClassVar[dict[str, pl.DataType | type[pl.DataType]]] = {}

    @classmethod
    def column_order(cls) -> list[str]:
        # TODO use get_type_hints to respect inheritance and typing.get_origin/args for generics
        return list(cls.__annotations__.keys())

    @property
    def f_valid(self) -> pl.Series:
        raise NotImplementedError("Subclasses must implement f_valid property.")

    def validate_dtypes(self):
        """
        Validate that the DataFrame's columns have the expected dtypes
        as specified in `target_dtypes`.
        """
        schema = self.schema
        for col, target_dtype in self.target_dtypes.items():
            if col not in schema:
                raise KeyError(f"{self.__class__.__name__}.validate_dtypes: column '{col}' not found in DataFrame.")

            actual_dtype = schema[col]
            if actual_dtype != target_dtype:
                raise TypeError(
                    f"{self.__class__.__name__}.validate_dtypes: column '{col}' has dtype "
                    f"{actual_dtype}, expected {target_dtype}."
                )

    def to_numpy(self, *args, **kwargs):
        """Return a pure-numeric numpy array for model input (no datetime/timedelta/object)."""
        df = self.select(self.column_order())

        # Cast temporals to numeric (epoch seconds) before conversion
        casts = []
        for col in df.columns:
            dtype = df.schema[col]
            if dtype == pl.Datetime or isinstance(dtype, pl.Datetime):
                casts.append(pl.col(col).cast(pl.Int64) / 1_000_000)  # us -> seconds
            elif dtype == pl.Duration or isinstance(dtype, pl.Duration):
                casts.append(pl.col(col).cast(pl.Int64) / 1_000_000)  # us -> seconds
            else:
                casts.append(pl.col(col))

        return df.select(casts).to_numpy()

    def col_filter(self, include: list[str] | None = None, exclude: list[str] | None = None) -> pl.DataFrame:
        """
        Return a DataFrame with only the specified columns included and/or excluded.
        include/exclude are substring matches against column names.
        """
        include = include or []
        exclude = exclude or []

        cols = (
            [col for col in self.columns if any(substr in col for substr in include)] if include else list(self.columns)
        )
        cols = [col for col in cols if not any(substr in col for substr in exclude)]

        cols_gone = [col for col in self.columns if col not in cols]
        print(f"Returning df without {len(cols_gone)} columns: {cols_gone}")
        return self.select(cols)

    @classmethod
    def print_th(cls, df: pl.DataFrame) -> None:
        """
        Helper method to print out something like:

        class MyDf(Enriched_DF_PL):
            a: pl.Series  # Int64
            b: pl.Series  # Utf8
        """
        cls_name = cls.__name__
        print(f"class {cls_name}(Enriched_DF_PL):")
        for col in df.columns:
            dtype = df.schema[col]
            type_hint = "pl.Series"
            if dtype == pl.Categorical:
                type_hint = "pl.Series  # Categorical"
            print(f"    {col}: {type_hint}  # {dtype}")


T = t.TypeVar("T", bound=pl.DataFrame)


class EnrichedDFUtilsPL:
    @classmethod
    def validate_typed_df_keys(
        cls, df: pl.DataFrame | dict, df_type: type[T], action: t.Literal["warn", "raise"] = "raise"
    ) -> bool:
        """
        Validate that a Polars DataFrame conforms to the specified typed DataFrame structure.
        """
        base_class_attrs = set(vars(Enriched_DF_PL).keys())
        exp_cols = (
            df_type.target_dtypes.keys() if isinstance(df_type, Enriched_DF_PL) else df_type.__annotations__.keys()
        )
        expected_columns = set(exp_cols) - base_class_attrs
        actual_columns = set(df.columns if isinstance(df, pl.DataFrame) else df.keys()) - base_class_attrs

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
