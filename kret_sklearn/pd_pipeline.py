from sklearn.pipeline import Pipeline
import pandas as pd
from kret_sklearn.custom_transformers import PandasColumnOrderBase


class PipelinePD(Pipeline):
    def post_init(self):
        self.set_output(transform="pandas")

    @staticmethod
    def validate_contents(idx: int, name: str, transform: PandasColumnOrderBase):
        assert isinstance(name, str), f"Expected str for step name, got {type(name)!r} for step at index {idx!r}"
        assert len(name) > 0, f"Step name cannot be empty for step at index {idx!r}"
        assert isinstance(
            transform, PandasColumnOrderBase
        ), f"Expected PandasColumnOrderBase, got {type(transform)!r} for step {name!r} at index {idx!r}"
        err_template = "Estimator {} does not provide get_feature_names_out. Did you mean to call pipeline[:-1].get_feature_names_out()?"
        if not hasattr(transform, "get_feature_names_out"):
            raise AttributeError(err_template.format(name))

    def get_feature_names_out_df(self, *args, **kwargs):
        """Get feature names for each step in the pipeline."""
        feature_names_out: dict[str | int, list[str]] = {}
        for idx, name, transform in self._iter():  # type: ignore[attr-defined]
            idx: int
            name: str
            transform: PandasColumnOrderBase
            self.validate_contents(idx, name, transform)

            feature_names_out[name] = transform.get_feature_names_out_list()

        return feature_names_out

    def fit_transform_df(self, X, y=None, **params) -> pd.DataFrame:
        self.post_init()
        out = super().fit_transform(X, y, **params)
        # Stubs say ndarray, runtime says DataFrame; we assert/cast to make typing happy.
        error_msg = f"Expected DataFrame output; got {type(out)!r}. Did set_output(transform='pandas') stick?"
        assert isinstance(out, pd.DataFrame), error_msg
        return out

    def transform_df(self, X, **params) -> pd.DataFrame:
        self.post_init()
        out = super().transform(X, **params)
        # Stubs say ndarray, runtime says DataFrame; we assert/cast to make typing happy.
        error_msg = f"Expected DataFrame output; got {type(out)!r}. Did set_output(transform='pandas') stick?"
        assert isinstance(out, pd.DataFrame), error_msg
        return out
