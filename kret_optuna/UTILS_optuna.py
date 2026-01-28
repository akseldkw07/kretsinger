from ._core.constants_optuna import OptunaDefaults

# TODO write function wrapper that takes takes in optuna trial and model class, and saves best n models according to some metric
"""
NOTE objective func below
def objective(trial: optuna.trial.Trial) -> float:
    # import traceback

    # try:

    # preset = trial.suggest_categorical("preset", ["tiny", "small", "medium", "large", "xlarge"])
    num_blocks = trial.suggest_int("num_blocks", 2, 3)  # 1=tiny, 2=small, 3=medium, 4=large
    num_filters = trial.suggest_int("num_filters", 50, 150, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.28, 0.35)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    l1 = trial.suggest_float("l1", 1e-4, 1e-3, log=True)
    l2 = trial.suggest_float("l2", 1e-5, 1e-3, log=True)

    model = CIFAR10ResNet(
        num_blocks=num_blocks,
        num_filters=num_filters,
        dropout_rate=dropout_rate,
        lr=lr,
        l1_penalty=l1,
        l2_penalty=l2,
        patience=8,
    )
    model._sweep_mode = True  # NOTE important!
    dynamic_args = TrainerDynamicDefaults.trainer_dynamic_defaults(model, cifar_dm, logtype=None, trial=trial)
    trainer_args = static_args | dynamic_args

    trainer = L.Trainer(**trainer_args)  # New trainer per trial!
    assert trainer.logger is not None
    trainer.logger.log_hyperparams(model.hparams_initial)
    trainer.fit(model, datamodule=cifar_dm, **TrainerStaticDefaults.TRAINER_FIT)

    return trainer.callback_metrics["val_f1"].item()


# except Exception as e:
#     print(f"\n{'='*80}")
#     print(f"Exception in trial {trial.number}:")
#     print(f"{'='*80}")
#     traceback.print_exc()
#     print(f"{'='*80}\n")
#     raise  # Re-raise so Optuna knows the trial failed


NOTE THEN:
def func_wrapper(trial: optuna.trial.Trial, model_class: Type[BaseLightningNN], **model_kwargs) -> BaseLightningNN:


"""


class KRET_OPTUNA_UTILS(OptunaDefaults):
    pass
