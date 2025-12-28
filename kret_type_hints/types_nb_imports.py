# autoflake: skip_file
import time

start_time = time.time()

import typing as t

from kret_type_hints.UTILS_kret_types import KretTypeHints

start_time_end = time.time()
print(f"[{__name__}] Imported {__name__} in {start_time_end - start_time:.4f} seconds")
