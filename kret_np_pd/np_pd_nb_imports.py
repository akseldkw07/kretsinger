# autoflake: skip_file
import time

start_time = time.time()
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_timedelta64_dtype
from pandas.io.formats.style import Styler

from .UTILS_np_pd import NP_PD_Utils as UKS_NP_PD
from .single_ret_ndarray import SingleReturnArray as UKS_Typed_NDArray

dtt = UKS_NP_PD.dtt

start_time_end = time.time()
print(f"[{__name__}] Imported {__name__} in {start_time_end - start_time:.4f} seconds")
