# autoflake: skip_file
import time

start_time = time.time()
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_timedelta64_dtype

from .UTILS_np_pd import NP_PD_Utils as UKS_NP_PD

start_time_end = time.time()
print(
    f"[kret_np_pd.np_pd_nb_imports] Imported kret_np_pd.np_pd_nb_imports in {start_time_end - start_time:.4f} seconds"
)
