import os
import typing as t
from datetime import datetime

import wandb

from ._core.constants_wandb import WandbConstants
from ._core.typed_cls_wandb import *

if t.TYPE_CHECKING:
    from kret_lightning.abc_lightning import ABCLM


class WandB_Utils:
    @classmethod
    def generate_wandb_args(cls, nn: "ABCLM", **kwargs: t.Unpack[Wandb_Init_TypedDict]):
        args_nn: Wandb_Init_TypedDict = {"group": nn.name, "dir": nn.root_dir / "wandb_logs"}
        args_defaults: Wandb_Init_TypedDict = {
            "entity": WandbConstants.WANDB_TEAM_NAME,
            "project": WandbConstants.WANDB_PROJECT_NAME,
            "mode": "online",
            "job_type": "train",
            "reinit": "finish_previous",
            # TODO config?? tags??
        }
        ret_args = args_defaults | args_nn | kwargs
        return ret_args

    @classmethod
    def start_wandb_run(
        cls,
        group: str,  # your nn.Module with BaseModel.name()
        entity: str = WandbConstants.WANDB_TEAM_NAME,  # your team/org slug
        project: str | None = WandbConstants.WANDB_PROJECT_NAME,  # the shared project
        mode: t.Literal["online", "offline", "disabled"] = "online",  # "online" | "offline" | "disabled"
        job_type: str = "train",  # "train" | "eval" | etc.
        tags: list[str] | None = None,
        config: dict[str, t.Any] | str | None = None,
    ):
        """
        TODO fix DATA_DIR
        """
        run_name = f"{group}__{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        wandb_dir = os.getenv("WANDB_OUTPUT_DIR") or WandbConstants.WANDB_LOG_DIR
        # config = {"model_summary": model._summary()} if config is None else config

        run = wandb.init(
            entity=entity,
            project=project,
            mode=mode,
            group=group,  # collapsible group in the UI
            name=run_name,  # readable run name
            job_type=job_type,
            tags=tags or [],
            config=config,
            reinit="finish_previous",
            dir=wandb_dir,
        )
        return run
