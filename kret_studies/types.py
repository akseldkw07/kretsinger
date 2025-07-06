from __future__ import annotations

import json
import inspect
import typing as t
from kret_studies import kret_gpt
import inspect
import typing as t

import inspect
import importlib.util
import json


def func_to_typed_hint_llm(func: t.Callable, include_ret: bool = False) -> t.Any:
    """
    Use an LLM to generate a type-hint dictionary for a function signature, using the logic and examples from func_to_typed_dict.
    """

    func_src = inspect.getsource(func)
    # Load prompt examples from Python file
    try:
        spec = importlib.util.spec_from_file_location(
            "func_to_typed_hint_examples",
            __file__.replace("types.py", "func_to_typed_hint_examples.py"),
        )
        if spec is not None:
            examples_mod = importlib.util.module_from_spec(spec)
            if spec.loader is not None:
                spec.loader.exec_module(examples_mod)  # type: ignore
                examples = examples_mod.examples
            else:
                examples = []
        else:
            examples = []
    except Exception:
        examples = []

    prompt = (
        "You are an expert Python type inference assistant. "
        "Given a function definition, generate a Python dictionary mapping argument names to their type hints, "
        "using the same logic as the func_to_typed_dict utility. Take extra care for cases "
        "in which the signature is wrong but the docstring is correct. "
        "Here are some non-trivial examples:\n"
        + "\n".join([f"Input:\n{ex['input']}\nOutput:\n{ex['output']}" for ex in examples])
        + f"\nNow, for this function (include_ret={include_ret}):\n{func_src}\nOutput:\n"
    )
    # Query the LLM
    result = kret_gpt.query_llm(prompt)
    # Try to parse as JSON, fallback to string
    try:
        result_dict = json.loads(result)
    except Exception:
        result_dict = None
    # Print using shared printer
    print_typed_dict(func, result_dict, include_ret=include_ret)
    return result_dict if result_dict is not None else result


import_replace_dict = {
    "pandas.core.frame.DataFrame": "pd.DataFrame",
    "pandas.core.series.Series": "pd.Series",
    "numpy": "np",
    "pandas": "pd",
    "t.Union": "Union",
    "t.Optional": "Optional",
    "typing.Union": "Union",
    "typing.Optional": "Optional",
    "typing.": "t.",
    "numba": "nb",
}


# --- Shared import collector ---
def collect_imports(type_str: str, extra_imports: set):
    import re

    known_type_map = {
        "Session": "from requests import Session",
    }
    for match in re.findall(r"\b([A-Z][A-Za-z0-9_]*)\b", type_str):
        if match in known_type_map:
            extra_imports.add(known_type_map[match])
        elif match not in {
            "None",
            "Any",
            "Literal",
            "Optional",
            "Union",
            "Sequence",
            "Mapping",
            "Dict",
            "List",
            "Tuple",
            "Set",
            "Type",
            "Callable",
            "str",
            "int",
            "float",
            "bool",
            "object",
        }:
            extra_imports.add(f"from {match.lower()} import {match}")


# --- Shared printer ---
def print_typed_dict(func: t.Callable, typed_dict: dict | None, include_ret: bool = False):
    import inspect

    name = func.__name__
    sig = inspect.signature(func)
    extra_imports = set()
    # If typed_dict is None, fallback to signature-based printing (like func_to_typed_dict)
    if typed_dict is None:
        # Reuse func_to_typed_dict logic for printing
        for param_name, param in sig.parameters.items():
            if param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.VAR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            ):
                continue
            arg_type = param.annotation
            if arg_type is inspect.Parameter.empty:
                str_argtype = "t.Any"
            else:
                str_argtype = str(arg_type)
                if "<class" in str_argtype:
                    str_argtype = str_argtype.replace("<class '", "").replace(">", "").replace("'", "")
                for to_replace, replacement in import_replace_dict.items():
                    str_argtype = str_argtype.replace(to_replace, replacement)
            if param.default is None and "None" not in str_argtype:
                str_argtype = f"{str_argtype} | None"
            collect_imports(str_argtype, extra_imports)
        # Return type
        if include_ret and sig.return_annotation is not inspect.Parameter.empty:
            return_type = sig.return_annotation
            str_return_type = str(return_type)
            if "<class" in str_return_type:
                str_return_type = str_return_type.replace("<class '", "").replace(">", "").replace("'", "")
            for to_replace, replacement in import_replace_dict.items():
                str_return_type = str_return_type.replace(to_replace, replacement)
            collect_imports(str_return_type, extra_imports)
        for imp in sorted(extra_imports):
            print(imp)
        print(f"class {name.capitalize()}_TypedDict(t.TypedDict, total=False):")
        for param_name, param in sig.parameters.items():
            if param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.VAR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            ):
                continue
            arg_type = param.annotation
            if arg_type is inspect.Parameter.empty:
                str_argtype = "t.Any"
            else:
                str_argtype = str(arg_type)
                if "<class" in str_argtype:
                    str_argtype = str_argtype.replace("<class '", "").replace(">", "").replace("'", "")
                for to_replace, replacement in import_replace_dict.items():
                    str_argtype = str_argtype.replace(to_replace, replacement)
            if param.default is None and "None" not in str_argtype:
                str_argtype = f"{str_argtype} | None"
            print(f"    {param_name}: {str_argtype}")
        if include_ret and sig.return_annotation is not inspect.Parameter.empty:
            return_type = sig.return_annotation
            str_return_type = str(return_type)
            if "<class" in str_return_type:
                str_return_type = str_return_type.replace("<class '", "").replace(">", "").replace("'", "")
            for to_replace, replacement in import_replace_dict.items():
                str_return_type = str_return_type.replace(to_replace, replacement)
            print(f"    # return: {str_return_type}")
        return
    # If typed_dict is a dict (from LLM/JSON), print using its keys/values
    for v in typed_dict.values():
        collect_imports(str(v), extra_imports)
    for imp in sorted(extra_imports):
        print(imp)
    print(f"class {name.capitalize()}_TypedDict(t.TypedDict, total=False):")
    for k, v in typed_dict.items():
        print(f"    {k}: {v}")


def func_to_typed_dict(func: t.Callable, include_ret: bool = False):
    """Convert function signature / annotations to a typed dict of accepted types."""

    import re

    sig = inspect.signature(func)
    func.__name__
    doc = func.__doc__ or ""

    # Parse docstring for :Parameters: section
    param_types_from_doc = {}
    param_descs = {}
    param_literals = {}
    param_section = False
    doc_lines = doc.splitlines()
    i = 0
    while i < len(doc_lines):
        line = doc_lines[i].strip()
        if line.lower().startswith(":parameters:"):
            param_section = True
            i += 1
            continue
        if param_section:
            if not line or line.startswith(":"):
                break
            # Try to parse lines like: name : type
            m = re.match(r"([\w*]+)\s*:\s*([^#]+)", line)
            if m:
                pname, ptype = m.group(1).strip(), m.group(2).strip()
                desc = line
                # Look ahead for valid values/intervals/periods in next lines
                lookahead = 1
                while i + lookahead < len(doc_lines):
                    next_line = doc_lines[i + lookahead].strip()
                    if not next_line or next_line.startswith(":") or re.match(r"^[\w*]+\s*:\s*", next_line):
                        break
                    valid_match = re.search(r"Valid (?:values|periods|intervals): ([^\n]+)", next_line, re.IGNORECASE)
                    if valid_match:
                        vals = [v.strip() for v in valid_match.group(1).replace(" ", "").split(",") if v.strip()]
                        param_literals[pname] = vals
                        desc += " " + next_line
                    lookahead += 1
                param_types_from_doc[pname] = ptype
                param_descs[pname] = desc
                i += lookahead - 1
            else:
                # Try to parse lines like: name : type, description
                m = re.match(r"([\w*]+)\s*:\s*([^,]+),?\s*(.*)", line)
                if m:
                    pname, ptype, desc = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
                    # Look ahead for valid values/intervals/periods in next lines
                    lookahead = 1
                    while i + lookahead < len(doc_lines):
                        next_line = doc_lines[i + lookahead].strip()
                        if not next_line or next_line.startswith(":") or re.match(r"^[\w*]+\s*:\s*", next_line):
                            break
                        valid_match = re.search(
                            r"Valid (?:values|periods|intervals): ([^\n]+)",
                            next_line,
                            re.IGNORECASE,
                        )
                        if valid_match:
                            vals = [v.strip() for v in valid_match.group(1).replace(" ", "").split(",") if v.strip()]
                            param_literals[pname] = vals
                            desc += " " + next_line
                        lookahead += 1
                    param_types_from_doc[pname] = ptype
                    param_descs[pname] = desc
                    i += lookahead - 1
        i += 1

    def doc_type_to_hint(ptype: str, literals: t.Optional[list[str]] = None) -> str:
        # Try to convert docstring type to python type hint
        ptype = ptype.strip()
        if literals:
            return f"t.Literal[{', '.join([repr(v) for v in literals])}]"
        # Handle common cases
        if ptype in {"str", "string"}:
            return "str"
        if ptype in {"int", "integer"}:
            return "int"
        if ptype in {"float"}:
            return "float"
        if ptype in {"bool", "boolean"}:
            return "bool"
        if ptype in {"list", "array", "sequence"}:
            return "list"
        if ptype in {"dict", "mapping"}:
            return "dict"
        if ptype == "str, list":
            return "str | list[str]"
        # Try to extract from ptype itself
        if any(x in ptype for x in [" or ", ",", "/"]):
            # e.g. "str, list" or "bool / int"
            parts = re.split(r",|/| or ", ptype)
            parts = [p.strip() for p in parts if p.strip()]
            return " | ".join(parts)
        return ptype

    extra_imports = set()

    # First pass: collect all types
    for param_name, param in sig.parameters.items():
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        ):
            continue
        if param_name in param_types_from_doc:
            doc_type = param_types_from_doc[param_name]
            literals = param_literals.get(param_name)
            str_argtype = doc_type_to_hint(doc_type, literals)
        else:
            arg_type = param.annotation
            if arg_type is inspect.Parameter.empty:
                str_argtype = "t.Any"
            else:
                str_argtype = str(arg_type)
                if "<class" in str_argtype:
                    str_argtype = str_argtype.replace("<class '", "").replace(">", "").replace("'", "")
                for to_replace, replacement in import_replace_dict.items():
                    str_argtype = str_argtype.replace(to_replace, replacement)
        if param.default is None and "None" not in str_argtype:
            str_argtype = f"{str_argtype} | None"
        collect_imports(str_argtype, extra_imports)
    # Return type
    if include_ret and sig.return_annotation is not inspect.Parameter.empty:
        return_type = sig.return_annotation
        str_return_type = str(return_type)
        if "<class" in str_return_type:
            str_return_type = str_return_type.replace("<class '", "").replace(">", "").replace("'", "")
        for to_replace, replacement in import_replace_dict.items():
            str_return_type = str_return_type.replace(to_replace, replacement)
        collect_imports(str_return_type, extra_imports)

    # Print import statements if needed

    print_typed_dict(func, None, include_ret=include_ret)
    return sig.parameters


def func_to_typed_dict_depr(func: t.Callable, include_ret: bool = False):
    """Convert function signature / annotations to a typed dict of accepted types."""
    annot = func.__annotations__
    if not include_ret:
        annot.pop("return", None)
    name = func.__name__

    print(f"class {name.capitalize()}_TypedDict(t.TypedDict, total=False):")
    for arg, arg_type in annot.items():

        str_argtype = str(arg_type)
        if "<class" in str_argtype:
            str_argtype = str_argtype.replace("<class '", "").replace(">", "").replace("'", "")
        for to_replace, replacement in import_replace_dict.items():
            str_argtype = str_argtype.replace(to_replace, replacement)

        print(f"    {arg}: {str(str_argtype)}")
    return annot
