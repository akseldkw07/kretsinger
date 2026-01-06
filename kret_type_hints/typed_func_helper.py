from __future__ import annotations

import inspect
import typing as t
from typing import get_args, get_origin

# Types from typing module that we convert to PEP 604 syntax (don't import these)
PEP604_REPLACEMENTS = {"Union", "Optional", "List", "Dict", "Set", "Tuple", "UnionType"}


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

    @classmethod
    def extract_imports_from_annotation(cls, imports: dict[str, set], annotation):
        """Recursively extract import statements from an annotation."""
        if annotation is inspect.Parameter.empty or annotation is type(None):
            return

        # Handle string annotations
        if isinstance(annotation, str):
            return

        origin = get_origin(annotation)
        args = get_args(annotation)

        # Process the origin type
        if origin is not None:
            type_to_process = origin
        else:
            type_to_process = annotation

        # Skip builtin types
        if isinstance(type_to_process, type) and type_to_process.__module__ == "builtins":
            pass
        elif hasattr(type_to_process, "__module__") and hasattr(type_to_process, "__name__"):
            module_path = type_to_process.__module__
            name = type_to_process.__name__

            # Skip types we convert to PEP 604 syntax
            skip_typing = module_path == "typing" and name in PEP604_REPLACEMENTS
            skip_uniontype = module_path == "types" and name == "UnionType"
            if skip_typing or skip_uniontype:
                pass
            else:
                # Resolve to highest-level import
                module, resolved_name = cls._resolve_import_location(type_to_process, name)

                imports[module].add(resolved_name)

        # Recursively process generic arguments (but skip Literal string values)
        for arg in args:
            # Don't try to extract imports from literal values (strings, ints, etc)
            if not isinstance(arg, (str, int, float, bool, type(None))):
                cls.extract_imports_from_annotation(imports, arg)
