from __future__ import annotations
import re
import inspect
import typing as t
from kret_studies.low_prio import kret_gpt

import importlib.util
import ast
import json


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


def parse_llm_typed_dict_output(result: str | dict) -> dict | None:
    """
    Parse LLM output (possibly in a code block, possibly a dict literal or JSON) into a Python dict.
    Returns None if parsing fails.
    """

    if isinstance(result, dict):
        return result
    code = result.strip()
    # Remove code block markers if present (```python ... ``` or ``` ... ```)
    if code.startswith("```"):
        code = code[3:]
        if code.lstrip().startswith("python"):
            code = code.lstrip()[6:]
        code = code.strip()
        if code.endswith("```"):
            code = code[:-3].strip()
    # Try to find the first dict-like structure
    try:
        start = code.find("{")
        end = code.rfind("}")
        if start != -1 and end != -1 and end > start:
            code = code[start : end + 1]
        # Use ast.literal_eval for safety
        return ast.literal_eval(code)
    except Exception:
        # Try to parse as JSON
        try:
            return json.loads(code)
        except Exception:
            print("[parse_llm_typed_dict_output] Could not parse LLM output as dict. Raw output:")
            print(result)
            return None


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
    result_dict = parse_llm_typed_dict_output(result)
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
