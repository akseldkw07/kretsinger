import typing as t
from pathlib import Path

import lightning as L
import optuna
from lightning.pytorch import LightningModule
from lightning.pytorch.core.saving import save_hparams_to_yaml


class TopNModelSaver:
    n: int
    save_dir: Path
    direction: t.Literal["minimize", "maximize"]

    def __init__(self, n: int, save_dir: str | Path, direction: t.Literal["minimize", "maximize"] = "minimize"):
        self.n = n
        self.save_dir = Path(save_dir)
        self.direction = direction

        # list of (score, trial_number, checkpoint_path)
        self._leaderboard: list[tuple[float, int, Path]] = []
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _is_better(self, score: float, worst_score: float) -> bool:
        if self.direction == "maximize":
            return score > worst_score
        return score < worst_score

    @property
    def _worst_entry(self) -> tuple[float, int, Path]:
        if self.direction == "maximize":
            return min(self._leaderboard, key=lambda x: x[0])
        return max(self._leaderboard, key=lambda x: x[0])

    def maybe_save(self, trainer: L.Trainer, model: LightningModule, score: float, trial: optuna.trial.Trial) -> bool:
        """Returns True if the model was saved."""
        if len(self._leaderboard) < self.n:
            self._save(trainer, model, score, trial)
            return True

        worst_score, _, _ = self._worst_entry
        if self._is_better(score, worst_score):
            self._evict_worst()
            self._save(trainer, model, score, trial)
            return True
        return False

    def _save(self, trainer: L.Trainer, model: LightningModule, score: float, trial: optuna.trial.Trial):
        stem = f"trial_{trial.number:04d}_score_{score:.4f}"
        ckpt_path = self.save_dir / f"{stem}.ckpt"
        yaml_path = self.save_dir / f"{stem}.hparams.yaml"

        trainer.save_checkpoint(ckpt_path, weights_only=False)
        save_hparams_to_yaml(yaml_path, dict(model.hparams_initial))

        self._leaderboard.append((score, trial.number, ckpt_path))

    @staticmethod
    def _yaml_path_for(ckpt_path: Path) -> Path:
        """trial_0001_score_0.6989.ckpt -> trial_0001_score_0.6989.hparams.yaml"""
        # Can't use .with_suffix() because the score contains a dot
        return ckpt_path.parent / (ckpt_path.name.removesuffix(".ckpt") + ".hparams.yaml")

    def _evict_worst(self):
        worst = self._worst_entry
        self._leaderboard.remove(worst)
        worst[2].unlink(missing_ok=True)  # delete checkpoint file
        self._yaml_path_for(worst[2]).unlink(missing_ok=True)

    @property
    def best_checkpoints(self) -> list[tuple[float, int, Path]]:
        reverse = self.direction == "maximize"
        return sorted(self._leaderboard, key=lambda x: x[0], reverse=reverse)
