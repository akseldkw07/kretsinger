# from __future__ import annotations
import typing as t
from pathlib import Path

import lightning as L
import pandas as pd
import torch
from torch.utils.data import DataLoader

from kret_lightning.utils_lightning import LightningDataModuleAssert
from kret_sklearn.custom_transformers import MissingValueRemover
from kret_sklearn.pd_pipeline import PipelinePD
from kret_torch_utils.torch_defaults import TorchDefaults

if t.TYPE_CHECKING:
    from kret_torch_utils.torch_typehints import DataLoader___init___TypedDict


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


class CustomDataModule(DataModuleABC):

    def __init__(
        self,
        data_dir: Path | str,
        split: SplitTuple | None = None,
        pipeline_pd: tuple[PipelinePD, PipelinePD] | None = None,
    ) -> None:
        super().__init__()

        self.data_dir = Path(data_dir)
        self.data_split = split if split is not None else SplitTuple(train=0.8, val=0.2)
        self._pipeline_pd_x, self._pipeline_pd_y = pipeline_pd if pipeline_pd is not None else (None, None)
        self.save_hyperparameters(ignore=self.ignore_hparams)
        LightningDataModuleAssert.initialization_check(self)

    def post_init(self, **dataloader_kwargs: t.Unpack["DataLoader___init___TypedDict"]):
        self._dataloader_passed_kwargs = dataloader_kwargs

    @property
    def DataLoaderKwargs(self) -> "DataLoader___init___TypedDict":
        return self.dataloader_kwargs_default | self._dataloader_passed_kwargs

    def prepare_data(self) -> None:
        raise NotImplementedError("Implement in subclass")

    def setup(self, stage: str) -> None:
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


class PandasInputMixin(DataModuleABC):
    """
    Mixin to load pandas DataFrame inputs into a LightningDataModule.
    """

    x_y_raw: LoadedDfTuple | None = None
    x_y_no_nans: LoadedDfTuple | None = None

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

    def load_df(self) -> LoadedDfTuple:
        """
        Load pandas DataFrame inputs and return as LoadedDfTuple.

        Set self.x_y_raw with the loaded data.
        """
        raise NotImplementedError("Implement in subclass to load pandas DataFrame inputs.")

    def df_remove_nans(self, df_tuple: LoadedDfTuple) -> LoadedDfTuple:
        """
        Remove NaN values from the DataFrame inputs using NanPipeline.
        """
        X_clean = self.NanPipeline.fit_transform_df(df_tuple.X, df_tuple.y)
        if isinstance(df_tuple.y, pd.DataFrame):
            y_clean = df_tuple.y.loc[X_clean.index]
        else:
            y_clean = df_tuple.y.loc[X_clean.index]

        self.x_y_no_nans = LoadedDfTuple(X=X_clean, y=y_clean)

        return LoadedDfTuple(X=X_clean, y=y_clean)

    def load_and_strip_nans(self) -> LoadedDfTuple:
        """
        Load DataFrame inputs and remove NaN values.

        Sets self.x_y_raw and self.x_y_no_nans.
        """
        df_tuple = self.load_df()
        self.x_y_raw = df_tuple
        df_no_nans = self.df_remove_nans(df_tuple)
        self.x_y_no_nans = df_no_nans

        return df_no_nans
