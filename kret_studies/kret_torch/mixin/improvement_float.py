from .abc_nn import ABCNN


class CheckImprovementFloatMixin(ABCNN):
    def _improved(self, eval: dict[str, float]):
        min_improvement = self.hparams["improvement_tol"]
        # for key, val in eval.items():

        val_improve = eval["eval_loss"] < (self.model_state["best_eval_loss"] - min_improvement)
        r2_improve = eval["eval_r2"] > (self.model_state["best_eval_r2"] + min_improvement)
        return {"eval_loss": val_improve, "eval_r2": r2_improve}

    def _on_improvement(self, improved: dict[str, bool], eval: dict[str, float], epochs_no_improve: int) -> int:
        if not any(improved.values()):
            return epochs_no_improve + 1

        for key, val in improved.items():
            if val:
                self.model_state[f"best_{key}"] = eval[key]

        self.save_weights()

        if self._log:
            self.logger.info(f"New {self.model_state=} - early-stop counter reset to 0.")
        return 0
