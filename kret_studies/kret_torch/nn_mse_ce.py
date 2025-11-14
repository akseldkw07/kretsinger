import torch.nn as nn

from .mixin.base_nn import BaseNN
from .mixin.improvement_float import CheckImprovementFloatMixin
from .mixin.single_variate import SingleVariateMixin


class LinearNN(SingleVariateMixin, CheckImprovementFloatMixin, BaseNN):
    _criterion = nn.MSELoss()


class ClassificationNN(SingleVariateMixin, CheckImprovementFloatMixin, BaseNN):
    _criterion = nn.CrossEntropyLoss()
