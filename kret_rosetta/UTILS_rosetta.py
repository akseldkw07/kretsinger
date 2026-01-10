from .to_lgbm_ds import ToLGBM
from .to_pd_np import To_NP_PD
from .to_tensor import To_Tensor


class UTILS_rosetta(To_NP_PD, ToLGBM, To_Tensor): ...
