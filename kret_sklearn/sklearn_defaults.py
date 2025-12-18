from dataclasses import dataclass
from types import ModuleType
import typing as t
import sys

from kret_utils.dataclass import ResettableDataclassMixin


@dataclass
class SklearnDefaults(ResettableDataclassMixin):
    VERB_FEAT_NAMES_OUT: bool = False
    VERB_PIPELINE: bool = True


# ====== Global singleton instance ======

_state_mod: ModuleType = sys.modules.setdefault(
    "kret_sklearn._global_state",
    ModuleType("kret_sklearn._global_state"),
)

if not hasattr(_state_mod, "SKLEARN_DEFAULTS"):
    setattr(_state_mod, "SKLEARN_DEFAULTS", SklearnDefaults())

SKLEARN_DEFAULTS = t.cast(SklearnDefaults, getattr(_state_mod, "SKLEARN_DEFAULTS"))
