from .load_utils import TorchLoadUtils
from .priors import PriorLosses
from .torch_defaults import TorchDefaults
from .tensor_manipulate import TensorManipulate


class KRET_TORCH_UTILS(TorchLoadUtils, PriorLosses, TorchDefaults, TensorManipulate):
    pass
