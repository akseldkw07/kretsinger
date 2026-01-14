# Winter break TODO


## (1) Port over PMML project to new kret_torch package

- FastAI integration / replacement

- Grid search hyperparameter optimization

## (2) Port over PMML utils to kret_studies

1 - sklearn data pipelines

2 - pandas visualizations

dtt - use torch.tensor.cols_pd?

3-

kretsinger/
    kret_torch/
        optuna/
            RUN_SUPERVISED (os.environ value - do I skip raising on bad outcome?)
            https://optuna.readthedocs.io/en/stable/tutorial/index.html
        fastai_integration/
        nn_weight_viz/
        lightning/
            *tutorials*
        rl_utils/
        graph_neural_networks/
    kret_mpl/
        mpl_utils/
    kret_RL/
        gymnasium/
        Stable-Baselines/
        CleanRL/
    kret_tqdm/
        tqdm patterns for notebooks and scripts. tqdm.auto, tqdm.notebook, etc.
        how to safely keyboard interrupt tqdm loops/ (GracefulInterrupt)
    kret_wandb/
            look at christian's wandb example
            programatically download run data (util to expand summary)
    kret_sandbox/
        *new experimental code*

os.env values:
    RUN_SUPERVISED - optuna hyperparameter optimization
    DEBUG_MODE - more verbose logging
    USE_WANDB - toggle wandb logging
    USE_TQDM - toggle tqdm progress bars
    ERROR_ON_BAD_OUTCOME - when running applicable code, raise error on bad outcome (or just warn)
