from kret_np_pd.dataset_to_table import PD_Display_Utils
from kret_np_pd.exp_decay import NP_ExpDecay_Utils
from kret_np_pd.filters import FilterUtils
from .np_bool_utils import NP_Boolean_Utils
from .np_dtype_utils import NP_Dtype_Utils
from .pd_cleanup import PD_Cleanup
from .pd_convenience_utils import PD_Convenience_utils
from .categoricals import CategoricalUtils


class NP_PD_Utils(
    NP_Dtype_Utils,
    NP_Boolean_Utils,
    NP_ExpDecay_Utils,
    PD_Cleanup,
    PD_Convenience_utils,
    PD_Display_Utils,
    CategoricalUtils,
    FilterUtils,
):
    """
    TODO debug long import time
    """
