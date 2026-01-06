from .load_utils import TorchLoadUtils
from .priors import PriorLosses
from .torch_defaults import TorchDefaults


class KRET_TORCH_UTILS(TorchLoadUtils, PriorLosses, TorchDefaults):
    pass
