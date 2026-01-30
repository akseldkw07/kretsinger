import typing as t
import warnings
from pathlib import Path

from kret_lightning._core.constants_lightning import LightningConstants  # type: ignore

from .abc_lightning import ABCLM


class OptunaMixin(ABCLM):
    @classmethod
    def create_model_saver(
        cls,
        n: int | None = None,
        save_dir: str | Path | None = None,
        direction: t.Literal["minimize", "maximize"] = "minimize",
        create_new_on_fail: bool = True,
    ):
        from kret_optuna.top_model_saver import TopNModelSaver

        n_models = n if n is not None else 3
        save_dir = save_dir if save_dir is not None else cls.ckpt_path
        try:

            return TopNModelSaver.from_existing(n=n_models, save_dir=save_dir, direction=direction)
        except Exception as e:
            if not create_new_on_fail:
                raise e
            warnings.warn(f"Could not load existing TopNModelSaver: {e}. Creating a new one.")
            return TopNModelSaver(n=n_models, save_dir=save_dir, direction=direction)
