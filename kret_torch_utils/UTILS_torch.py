from .load_utils import TorchLoadUtils
from .priors import PriorLosses
from .tensor_manipulate import TensorManipulate
from .torch_defaults import TorchDefaults


class KRET_TORCH_UTILS(TorchLoadUtils, PriorLosses, TorchDefaults, TensorManipulate):
    pass
