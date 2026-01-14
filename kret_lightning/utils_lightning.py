import typing as t
from typing import get_type_hints

from kret_lightning.abc_lightning import ABCLM, HPasKwargs, HPDict
from kret_utils.filename_utils import FilenameUtils

if t.TYPE_CHECKING:
    from .datamodule.data_module_custom import DataModuleABC  # avoid circular import


class SharedAssert:
    @classmethod
    def assert_hparams(cls, datamodule: "DataModuleABC | ABCLM"):
        overlap = set(datamodule.hparams_initial.keys()).intersection(set(datamodule.ignore_hparams))
        assert not overlap, (
            f"Datamodule hparams_initial keys {set(datamodule.hparams_initial.keys())} overlap with ignore_hparams "
            f"{set(datamodule.ignore_hparams)}. Overlap: {overlap}"
        )


class LightningModuleAssert(SharedAssert):
    @classmethod
    def initialization_check(cls, lm: ABCLM) -> None:
        cls.assert_version_fmt(lm.version)
        cls.assert_filename_safe(lm)
        cls.assert_hparams(lm)

    @classmethod
    def assert_version_fmt(cls, version: str) -> None:
        if not isinstance(version, str) or not version.startswith("v_") or not version[2:].isdigit():
            raise ValueError(f"Version '{version}' is not in the correct format 'v_XXX' where XXX are digits.")

    @classmethod
    def assert_filename_safe(cls, lm: ABCLM) -> None:
        assert FilenameUtils.is_safe_filename_basic(
            lm.ckpt_path
        ), f"Filename '{lm.ckpt_path}' is not safe for filesystems."

    @classmethod
    def assert_dict_keys_consistency(cls):
        assert get_type_hints(HPDict) == get_type_hints(HPasKwargs)


class LightningDataModuleAssert(SharedAssert):
    @classmethod
    def initialization_check(cls, datamodule: "DataModuleABC"):
        cls.assert_hparams(datamodule)
        cls.assert_split_distribution(datamodule)

    @classmethod
    def assert_split_distribution(cls, datamodule: "DataModuleABC"):
        split = datamodule.data_split
        total = sum([split for split in split if isinstance(split, float)])
        assert total == 1.0, f"Data split proportions must sum to 1.0; got total {total} from split {split}"
