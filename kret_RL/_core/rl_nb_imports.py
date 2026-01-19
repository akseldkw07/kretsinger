# autoflake: skip_file
import time

start_time = time.time()

import gymnasium as gym

start_time_end = time.time()
print(f"[{__name__}] Imported {__name__} in {start_time_end - start_time:.4f} seconds")
