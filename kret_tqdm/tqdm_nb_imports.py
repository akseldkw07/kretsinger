# autoflake: skip_file
import time

start_time = time.time()

from tqdm.auto import tqdm as tqdm_auto
from tqdm.notebook import tqdm as tqdm_notebook

start_time_end = time.time()
print(f"[{__name__}] Imported {__name__} in {start_time_end - start_time:.4f} seconds")
