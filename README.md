# Notes to self

Use `@classproperty` for lazy class properties that need to import heavy libraries (e.g., torch) only when first accessed.

In `typed_cls_*.py` files, use `if t.TYPE_CHECKING:` to avoid heavy unnecessary imports during runtime.



# TODOs


kretsinger/
    kret_torch/
        optuna/
            RUN_SUPERVISED (os.environ value - do I skip raising on bad outcome?)
        rl_utils/
        graph_neural_networks/
    kret_RL/
        gymnasium/
        Stable-Baselines/
        CleanRL/
    kret_wandb/
            look at christian's wandb example (NNDL project)
            programatically download run data (util to expand summary)
    kret_sandbox/
        *new experimental code*

os.env values:
    RUN_SUPERVISED - optuna hyperparameter optimization
    DEBUG_MODE - more verbose logging
    USE_WANDB - toggle wandb logging
    USE_TQDM - toggle tqdm progress bars
    ERROR_ON_BAD_OUTCOME - when running applicable code, raise error on bad outcome (or just warn)
