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
    example_notebooks/
        notebooks downloaded from (Kaggle, etc)
    kret_utils/
        **Every other submodule will import from here**
        stuff/
        ~dtt/~
        dir/ (print class attributes better)
        ~wandb/~
        typehint_utils/
            func_to_typehint/
            single-return-array/
    kret_sklearn/
        sklearn_pipelines/ (PMML examples)
        sklearn.datasets.make_* # for generating datasets
    kret_torch/
        models/
            mixin/ (PMML mixins) (use pytorch lightning modules)
        optuna/
            RUN_SUPERVISED (os.environ value - do I skip raising on bad outcome?)
            https://optuna.readthedocs.io/en/stable/tutorial/index.html
        fastai_integration/
        pytorch_dataloaders/ (custom datasets and dataloaders)
        nn_weight_viz/
        pytorch_lightning/
            *tutorials*
        rl_utils/
        graph_neural_networks/
        data_loader_utils/
            DataLoader next() typehinted to return two tensors
        torch_utils/
            def obs_numpy_to_torch (RL) - timedelta conversion
    kret_numpy_pd/
        numpy_utils/
        pandas_utils/
            pd.options.mode.copy_on_write
            dtt/
            pd_heatmap/
    kret_notebook/
        notebook_imports/
            torch_stuff
            numpy_stuff
            generic_stuff
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
