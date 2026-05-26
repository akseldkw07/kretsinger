# Notes to self

Use `@classproperty` for lazy class properties that need to import heavy libraries (e.g., torch) only when first accessed.

In `typed_cls_*.py` files, use `if t.TYPE_CHECKING:` to avoid heavy unnecessary imports during runtime.

os.env values:
    RUN_SUPERVISED - optuna hyperparameter optimization
    DEBUG_MODE - more verbose logging
    USE_WANDB - toggle wandb logging
    USE_TQDM - toggle tqdm progress bars
    ERROR_ON_BAD_OUTCOME - when running applicable code, raise error on bad outcome (or just warn)
