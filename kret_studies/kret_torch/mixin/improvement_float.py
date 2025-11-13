from abc_nn import ABCNN


class CheckImprovementFloat(ABCNN):
    def _improved(self, val_loss: float) -> bool:
        min_imporvement = self.hparams["improvement_tol"]
        return val_loss < (self.model_state["best_loss"] - min_imporvement)

    def _on_improvement(self, improved: bool, val_loss: float, epochs_no_improve: int) -> int:
        if not improved:
            return epochs_no_improve + 1

        self.model_state["best_loss"] = val_loss
        self.save_weights()

        if self._log:
            self.logger.info(f"New {val_loss=}- early-stop counter reset to 0.")
        return 0
