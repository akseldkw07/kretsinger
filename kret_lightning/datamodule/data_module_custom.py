# from __future__ import annotations
import typing as t
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from sklearn.utils.validation import check_is_fitted
from torch.utils.data import DataLoader

from kret_lightning.utils_lightning import LightningDataModuleAssert
from kret_np_pd.np_pd_nb_imports import *
from kret_sklearn.custom_transformers import MissingValueRemover
from kret_sklearn.pd_pipeline import PipelinePD
from kret_torch_utils.torch_defaults import TorchDefaults

if t.TYPE_CHECKING:
    from kret_torch_utils.torch_typehints import DataLoader___init___TypedDict

STAGE_LITERAL = t.Literal["fit", "validate", "predict", "test"]


class DataModuleABC(L.LightningDataModule):
    data_dir: Path
    data_split: "SplitTuple"
    ignore_hparams: tuple[str, ...] = ("pipeline_pd",)

    _train: torch.utils.data.Dataset
    _val: torch.utils.data.Dataset
    _test: torch.utils.data.Dataset
    _predict: torch.utils.data.Dataset

    _dataloader_passed_kwargs: "DataLoader___init___TypedDict"
    dataloader_kwargs_default: "DataLoader___init___TypedDict" = TorchDefaults.DATA_LOADER_INIT

    _pipeline_pd_x: PipelinePD | None = None
    _pipeline_pd_y: PipelinePD | None = None


class SplitTuple(t.NamedTuple):
    train: float
    val: float
    test: float = 0.0
    predict: float = 0.0
    contiguous: bool = True  # If True, splits are contiguous (for time series); if False, fully random


class CustomDataModule(DataModuleABC):

    def __init__(
        self,
        data_dir: Path | str,
        split: SplitTuple | None = None,
        pipeline_pd: tuple[PipelinePD, PipelinePD] | None = None,
        **kwargs,  # save_hyperparameters
    ) -> None:
        super().__init__()

        self.data_dir = Path(data_dir)
        self.data_split = split if split is not None else SplitTuple(train=0.8, val=0.2)
        self._pipeline_pd_x, self._pipeline_pd_y = pipeline_pd if pipeline_pd is not None else (None, None)
        print(f"Saving hparams, ignoring {self.ignore_hparams}")
        self.save_hyperparameters(ignore=self.ignore_hparams)
        LightningDataModuleAssert.initialization_check(self)

    def post_init(self, **dataloader_kwargs: t.Unpack["DataLoader___init___TypedDict"]):
        self._dataloader_passed_kwargs = dataloader_kwargs

    @property
    def DataLoaderKwargs(self) -> "DataLoader___init___TypedDict":
        return self.dataloader_kwargs_default | self._dataloader_passed_kwargs

    def prepare_data(self) -> None:
        raise NotImplementedError("Implement in subclass")

    def setup(self, stage: STAGE_LITERAL) -> None:  # type: ignore[override]
        raise NotImplementedError("Implement in subclass")

    # region Dataloaders
    def train_dataloader(self) -> DataLoader:
        return self._train_dataloader()

    def _train_dataloader(self) -> DataLoader:
        return DataLoader(self._train, **self.DataLoaderKwargs)

    def val_dataloader(self) -> DataLoader:
        return self._val_dataloader()

    def _val_dataloader(self) -> DataLoader:
        return DataLoader(self._val, **self.DataLoaderKwargs | {"shuffle": False})

    def test_dataloader(self) -> DataLoader:
        return self._test_dataloader()

    def _test_dataloader(self) -> DataLoader:
        return DataLoader(self._test, **self.DataLoaderKwargs | {"shuffle": False})

    def predict_dataloader(self) -> DataLoader:
        return self._predict_dataloader()

    def _predict_dataloader(self) -> DataLoader:
        return DataLoader(self._predict, **self.DataLoaderKwargs | {"shuffle": False})

    # endregion


class LoadedDfTuple(t.NamedTuple):
    X: pd.DataFrame
    y: pd.Series | pd.DataFrame


class SplitIndexes(t.NamedTuple):
    """Indexes for train/val/test/predict splits."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray | None = None
    predict: np.ndarray | None = None


class PandasInputMixin(DataModuleABC):
    """
    Mixin to load pandas DataFrame inputs into a LightningDataModule.
    """

    col_order: dict[str, list[str]] = {"start": [], "end": []}
    x_y_raw: LoadedDfTuple
    x_y_no_nans: LoadedDfTuple
    x_y_processed: LoadedDfTuple
    SplitIdx: SplitIndexes

    @property
    def PipelineX(self) -> PipelinePD:
        if self._pipeline_pd_x is None:
            print("Warning: pipeline_pd is None, raising error.")
            raise ValueError("_pipeline_pd is not set.")
        return self._pipeline_pd_x

    @property
    def PipelineY(self) -> PipelinePD:
        if self._pipeline_pd_y is None:
            print("Warning: pipeline_pd is None, raising error.")
            raise ValueError("_pipeline_pd is not set.")
        return self._pipeline_pd_y

    @property
    def NanPipeline(self):
        missing_value_remover = MissingValueRemover(how="any")  # Remove rows with any NaN values
        remove_nans_pipeline = PipelinePD(steps=[("remove_nans", missing_value_remover)])
        return remove_nans_pipeline

    def data_preprocess(self):
        """
        Call once at the start of setup to load and preprocess the DataFrame inputs.

        1- load data
        2- remove NaNs
        3- generate split indexes
        4- fit pipelines on training data only

        """
        df_tuple = self.load_df()
        self.x_y_raw = df_tuple
        df_no_nans = self._df_remove_nans(df_tuple)
        self.x_y_no_nans = df_no_nans

        split_indexes = self._generate_split_indexes(len(df_no_nans.X))
        self.SplitIdx = split_indexes

        X_train_raw = df_no_nans.X.iloc[split_indexes.train]
        y_train_raw = df_no_nans.y.iloc[split_indexes.train]

        self.fit_pipelines_once(X_train_raw, y_train_raw)
        self.x_y_processed = LoadedDfTuple(
            X=UKS_NP_PD.move_columns(self.PipelineX.transform_df(df_no_nans.X), **self.col_order),
            y=self.PipelineY.transform_df(df_no_nans.y),
        )

        return self.x_y_processed

    def load_df(self) -> LoadedDfTuple:
        """
        Load pandas DataFrame inputs and return as LoadedDfTuple.

        Set self.x_y_raw with the loaded data.
        """
        raise NotImplementedError("Implement in subclass to load pandas DataFrame inputs.")

    def _df_remove_nans(self, df_tuple: LoadedDfTuple) -> LoadedDfTuple:
        """
        Remove NaN values from the DataFrame inputs using NanPipeline.
        """
        X_clean = self.NanPipeline.fit_transform_df(df_tuple.X, df_tuple.y)  # .reset_index(drop=True)
        y_clean = df_tuple.y.loc[X_clean.index]  # .reset_index(drop=True)

        x_y_no_nans = LoadedDfTuple(X=X_clean.reset_index(drop=True), y=y_clean.reset_index(drop=True))

        return x_y_no_nans

    def _generate_split_indexes(self, dataset_size: int) -> SplitIndexes:
        """
        Generate train/val/test/predict indexes based on the split ratios.

        Args:
            dataset_size: Total size of the dataset

        Returns:
            SplitIndexes with train/val/test/predict indexes
        """
        # Validate split ratios
        total_ratio = self.data_split.train + self.data_split.val + self.data_split.test + self.data_split.predict
        if not np.isclose(total_ratio, 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

        # Calculate split sizes
        train_size = int(self.data_split.train * dataset_size)
        val_size = int(self.data_split.val * dataset_size)
        test_size = int(self.data_split.test * dataset_size)
        predict_size = dataset_size - train_size - val_size - test_size  # Remainder goes to predict

        if self.data_split.contiguous:
            # Contiguous splits (for time series)
            train_idx = np.arange(0, train_size)
            val_idx = np.arange(train_size, train_size + val_size)
            test_idx = np.arange(train_size + val_size, train_size + val_size + test_size) if test_size > 0 else None
            predict_idx = np.arange(train_size + val_size + test_size, dataset_size) if predict_size > 0 else None
        else:
            # Random splits
            rng = np.random.default_rng(seed=42)  # Use a fixed seed for reproducibility
            shuffled_idx = rng.permutation(dataset_size)

            train_idx = shuffled_idx[:train_size]
            val_idx = shuffled_idx[train_size : train_size + val_size]
            test_idx = (
                shuffled_idx[train_size + val_size : train_size + val_size + test_size] if test_size > 0 else None
            )
            predict_idx = shuffled_idx[train_size + val_size + test_size :] if predict_size > 0 else None

        return SplitIndexes(train=train_idx, val=val_idx, test=test_idx, predict=predict_idx)

    def fit_pipelines_once(self, X_train: pd.DataFrame, y_train: pd.Series | pd.DataFrame):

        try:
            check_is_fitted(self.PipelineX)
            check_is_fitted(self.PipelineY)
        except:
            # If pipelines are not fitted, fit them
            self.PipelineX.fit(X_train, y_train)
            self.PipelineY.fit(y_train)
