# autoflake: skip_file
import time

start_time = time.time()

import typing as t

from kret_type_hints.kret_types import KretTypeHints

start_time_end = time.time()
print(
    f"[kret_type_hints.types_nb_imports] Imported kret_type_hints.types_nb_imports in {start_time_end - start_time:.4f} seconds"
)
