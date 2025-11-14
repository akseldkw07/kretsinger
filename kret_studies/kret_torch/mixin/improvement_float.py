from .abc_nn import ABCNN

# Dictionary mapping metric names to whether improvement is indicated by a smaller value
IMPROVE_IS_LT = {"eval_loss"}


class CheckImprovementFloatMixin(ABCNN):
    def _improved(self, eval: dict[str, float]):
        min_improvement = self.hparams["improvement_tol"]
        ret = {}

        for key, val in eval.items():
            if key in IMPROVE_IS_LT:
                ret[key] = val < (self.model_state[f"best_{key}"] - min_improvement)
            else:
                ret[key] = val > (self.model_state[f"best_{key}"] + min_improvement)

        return ret

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
