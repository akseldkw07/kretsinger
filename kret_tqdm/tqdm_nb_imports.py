# autoflake: skip_file
import time

start_time = time.time()

from tqdm import tqdm as tqdm_orig
from tqdm.auto import tqdm, trange
from tqdm.notebook import tqdm_notebook

from .UTILS_tqdm import UTILS_tqdm as UKS_TQDM_UTILS

start_time_end = time.time()
print(f"[{__name__}] Imported {__name__} in {start_time_end - start_time:.4f} seconds")
