from .constants_torch import TorchDefaults
from .load_utils import TorchLoadUtils
from .priors import PriorLosses
from .tensor_manipulate import TensorManipulate


class KRET_TORCH_UTILS(TorchLoadUtils, PriorLosses, TorchDefaults, TensorManipulate):
    pass
