import os
import typing as t
from datetime import datetime

import wandb

from ._core.constants_wandb import WandbConstants


class WandB_Utils:
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
