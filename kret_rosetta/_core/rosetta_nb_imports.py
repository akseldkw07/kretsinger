# autoflake: skip_file
import time

start_time = time.time()

from ..UTILS_rosetta import UTILS_rosetta as UKS_ROSETTA

start_time_end = time.time()
print(f"[{__name__}] Imported {__name__} in {start_time_end - start_time:.4f} seconds")
