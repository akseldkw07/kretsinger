import torch
import torch.nn as nn

from .mixin.base_nn import BaseNN
from .mixin.eval_mixin import ClassificationEvalMixin, LinearEvalMixin
from .mixin.improvement_float import CheckImprovementFloatMixin
from .mixin.train_mixin import SingleVariateMixin


class LinearNN(SingleVariateMixin, LinearEvalMixin, CheckImprovementFloatMixin, BaseNN):
    _criterion = nn.MSELoss()


class ClassificationNN(SingleVariateMixin, ClassificationEvalMixin, CheckImprovementFloatMixin, BaseNN):
    _criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
