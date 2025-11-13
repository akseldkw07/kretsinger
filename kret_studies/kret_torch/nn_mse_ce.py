import torch.nn as nn
from .mixin.base_nn import BaseNN
from .mixin.single_variate import SingleVariateMixin


class LinearNN(SingleVariateMixin, BaseNN):
    _criterion = nn.MSELoss()


class ClassificationNN(SingleVariateMixin, BaseNN):
    _criterion = nn.CrossEntropyLoss()
