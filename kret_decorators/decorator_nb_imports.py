# autoflake: skip_file
import time

start_time = time.time()
import joblib

from .post_init import post_init, post_init_no_inheritance
from .protocols import PostInitCompatible

start_time_end = time.time()
print(f"[{__name__}] Imported {__name__} in {start_time_end - start_time:.4f} seconds")
