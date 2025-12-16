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
    kret_utils/
        **Every other submodule will import from here**
        stuff/
        dtt/
        wandb/
            look at christian's wandb example
        typehint_utils/
            func_to_typehint/
            single-return-array/
    kret_sklearn/
        sklearn_pipelines/ (PMML examples)
    kret_torch/
        models/
            mixin/ (PMML mixins)
        optuna/
        fastai_integration/
        pytorch_dataloaders/
        nn_weight_viz/
        data_loader_utils/
            DataLoader next() typehinted to return two tensors
        torch_utils/
            def obs_numpy_to_torch (RL) - timedelta conversion
        RUN_SUPERVISED (os.environ value - do I skip raising on bad outcome?)
    kret_numpy_pd/
        numpy_utils/
        pandas_utils/
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
    kret_sandbox/
        *new experimental code*
