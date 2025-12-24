# autoflake: skip_file
import time

start_time = time.time()

import wandb

from .wandb_utils import WandB_Utils as UKS_WANDB

start_time_end = time.time()
print(
    f"[kret_wandb.wandb_nb_imports] Imported kret_wandb.wandb_nb_imports in {start_time_end - start_time:.4f} seconds"
)
