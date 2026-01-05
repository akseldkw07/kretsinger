from __future__ import annotations

import typing as t


class TypedFuncHelper:

    @classmethod
    def _resolve_import_location(cls, type_obj, name: str) -> tuple[str, str]:
        """
        Resolve the best import location for a type.

        Returns (module, name) where module is the highest-level import possible.
        E.g. (torch.nn, Module) instead of (torch.nn.modules.module, Module)
        Also tries intermediate levels: torch.nn before torch

        Prefers modules that explicitly export the name in __all__ (public API).
        """
        module: str = type_obj.__module__
        best_match = (module, name)

        # Start with the original module
        if "." in module:
            parts = module.split(".")
            # Try progressively shorter module paths (e.g., torch.nn.modules.module -> torch.nn.modules -> torch.nn -> torch)
            for i in range(len(parts) - 1, 0, -1):
                candidate_module = ".".join(parts[:i])
                try:
                    imported = __import__(candidate_module, fromlist=[name])
                    # Check if name is available at this module level

                    best_match = (candidate_module, name) if hasattr(imported, name) else best_match

                except (ImportError, AttributeError):
                    continue

        # Fall back to the original module or best match found
        return best_match

    @classmethod
    def resolve_dict_name(cls, func: t.Callable) -> str:
        qualname = getattr(func, "__qualname__", "")
        parts = qualname.split(".")
        if len(parts) > 1 and "<locals>" not in qualname:
            class_name = parts[-2]
            dict_name = f"{class_name}_{func.__name__.capitalize()}_TypedDict"
        else:
            dict_name = f"{func.__name__.capitalize()}_TypedDict"
        return dict_name
