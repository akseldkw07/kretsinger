from __future__ import annotations

import inspect
import re
import typing as t

from .typed_func_helper import TypedFuncHelper


class FuncToTypedDict(TypedFuncHelper):
    @classmethod
    def func_to_typed_dict(cls, func: t.Callable, include_ret: bool = False):
        """Convert function signature / annotations to a typed dict of accepted types."""

        sig = inspect.signature(func)
        doc = func.__doc__ or ""

        # Parse docstring for :Parameters: section
        param_types_from_doc: dict[str, str] = {}
        param_descs: dict[str, str] = {}
        param_literals: dict[str, list[str]] = {}
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
                        valid_match = re.search(
                            r"Valid (?:values|periods|intervals): ([^\n]+)", next_line, re.IGNORECASE
                        )
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
                                vals = [
                                    v.strip() for v in valid_match.group(1).replace(" ", "").split(",") if v.strip()
                                ]
                                param_literals[pname] = vals
                                desc += " " + next_line
                            lookahead += 1
                        param_types_from_doc[pname] = ptype
                        param_descs[pname] = desc
                        i += lookahead - 1
            i += 1

        def _split_unionish(s: str) -> list[str]:
            """
            Split a type-ish string into parts for unions.

            Handles:
              - 'A | B | None'
              - 'A, B'
              - 'A / B'
              - 'A or B'
              - 'Union[A, B]'
              - 'Optional[T]'
            """
            s = s.strip()

            # Normalize typing.Optional / typing.Union textual forms
            m_opt = re.match(r"^(?:typing\.)?Optional\[(.*)\]$", s)
            if m_opt:
                inner = m_opt.group(1).strip()
                # Optional[T] == T | None
                return _split_unionish(inner) + ["None"]

            m_union = re.match(r"^(?:typing\.)?Union\[(.*)\]$", s)
            if m_union:
                inner = m_union.group(1).strip()
                # Split Union[...] on commas at top-level (not inside brackets)
                parts = _split_top_level_commas(inner)
                return [p.strip() for p in parts if p.strip()]

            # If already pipe unions
            if " | " in s or "|" in s:
                parts = [p.strip() for p in s.split("|")]
                return [p for p in parts if p]

            # Other union-ish separators
            if any(x in s for x in [" or ", ",", "/"]):
                parts = re.split(r",|/|\s+or\s+", s)
                parts = [p.strip() for p in parts if p.strip()]
                return parts

            return [s]

        def _split_top_level_commas(s: str) -> list[str]:
            """
            Split a string on commas not nested inside [] or ().
            """
            out: list[str] = []
            buf: list[str] = []
            depth_sq = 0
            depth_par = 0
            for ch in s:
                if ch == "[":
                    depth_sq += 1
                elif ch == "]":
                    depth_sq = max(0, depth_sq - 1)
                elif ch == "(":
                    depth_par += 1
                elif ch == ")":
                    depth_par = max(0, depth_par - 1)

                if ch == "," and depth_sq == 0 and depth_par == 0:
                    out.append("".join(buf).strip())
                    buf = []
                else:
                    buf.append(ch)
            tail = "".join(buf).strip()
            if tail:
                out.append(tail)
            return out

        def _dedupe_keep_order(parts: list[str]) -> list[str]:
            seen: set[str] = set()
            out: list[str] = []
            for p in parts:
                if p not in seen:
                    out.append(p)
                    seen.add(p)
            return out

        def _normalize_none(parts: list[str]) -> list[str]:
            """
            Convert any 'NoneType' mentions to 'None' and remove imports for NoneType later.
            """
            norm: list[str] = []
            for p in parts:
                p2 = p.strip()
                # Handle both 'NoneType' and qualified versions
                if p2.endswith("NoneType") or p2 == "NoneType":
                    norm.append("None")
                else:
                    norm.append(p2)
            return norm

        def _normalize_optional_union_string(s: str) -> str:
            """
            Ensure we output unions only as 'A | B | None' (no Union/Optional),
            and never mention NoneType.
            """
            parts = _split_unionish(s)
            parts = _normalize_none(parts)
            parts = _dedupe_keep_order(parts)
            # Ensure 'None' is present only once; keep it at end if present
            has_none = "None" in parts
            parts_wo_none = [p for p in parts if p != "None"]
            if has_none:
                parts = parts_wo_none + ["None"]
            else:
                parts = parts_wo_none
            return " | ".join(parts) if parts else "t.Any"

        def doc_type_to_hint(ptype: str, literals: list[str] | None = None) -> str:
            # Try to convert docstring type to python type hint (pipe-unions only)
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

            # Handle some common union-ish doc patterns
            if ptype == "str, list":
                return "str | list[str]"

            # If ptype itself encodes unions with commas/slashes/or
            if any(x in ptype for x in [" or ", ",", "/"]):
                parts = _split_unionish(ptype)
                return _normalize_optional_union_string(" | ".join(parts))

            return ptype

        # We'll store imports as fully-qualified names (for full import tree printing)
        extra_imports: set[str] = set()

        def _collect_imports_full(type_str: str) -> None:
            """
            Collect imports but keep them fully-qualified for easier copy/paste.

            Also ensure:
              - never import NoneType
              - do not import Optional/Union
            """
            # First apply replacements to match your house style
            s = type_str
            for to_replace, replacement in TypedFuncHelper.import_replace_dict.items():
                s = s.replace(to_replace, replacement)

            # Normalize Union/Optional/NoneType into pipe unions and None
            s = _normalize_optional_union_string(s)

            # Now scan for tokens that look like fully qualified names
            # We'll keep fully qualified modules (e.g. lightning.pytorch.X.Y)
            # and generate imports as: `import lightning.pytorch.X.Y`
            # rather than local `from Y import X`.
            tokens = re.findall(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+", s)

            for tok in tokens:
                # Skip typing module artifacts we never want to import
                if tok.startswith("typing."):
                    continue
                # Skip NoneType anywhere
                if tok.endswith(".NoneType") or tok.endswith("NoneType"):
                    continue
                extra_imports.add(tok)

        # First pass: collect all types and resolve param types
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
                    for to_replace, replacement in TypedFuncHelper.import_replace_dict.items():
                        str_argtype = str_argtype.replace(to_replace, replacement)

            # If default None, ensure union includes None (not NoneType)
            if param.default is None and "None" not in str_argtype and "NoneType" not in str_argtype:
                str_argtype = f"{str_argtype} | None"

            # Enforce pipe union style + squash NoneType
            str_argtype = _normalize_optional_union_string(str_argtype)

            _collect_imports_full(str_argtype)
            param_types_from_doc[param_name] = str_argtype

        # Return type
        str_return_type = None
        if include_ret and sig.return_annotation is not inspect.Parameter.empty:
            return_type = sig.return_annotation
            str_return_type = str(return_type)
            if "<class" in str_return_type:
                str_return_type = str_return_type.replace("<class '", "").replace(">", "").replace("'", "")
            for to_replace, replacement in TypedFuncHelper.import_replace_dict.items():
                str_return_type = str_return_type.replace(to_replace, replacement)
            str_return_type = _normalize_optional_union_string(str_return_type)
            _collect_imports_full(str_return_type)

        def _print_full_import_tree(imports: set[str]) -> None:
            """
            Print imports as full module paths, one per line:
              import lightning.pytorch.callbacks.callback
            """
            for imp in sorted(imports):
                # Guard: never print NoneType imports
                if imp.endswith(".NoneType") or imp.endswith("NoneType"):
                    continue
                print(f"import {imp}")

        def print_with_resolved_types() -> None:
            name = func.__name__

            _print_full_import_tree(extra_imports)

            print(f"class {name.capitalize()}_TypedDict(t.TypedDict, total=False):")
            for param_name, param in sig.parameters.items():
                if param.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.VAR_KEYWORD,
                    inspect.Parameter.VAR_POSITIONAL,
                ):
                    continue
                str_argtype = param_types_from_doc.get(param_name, "t.Any")
                # Ensure we never show NoneType and never show Union/Optional
                str_argtype = _normalize_optional_union_string(str_argtype)
                print(f"    {param_name}: {str_argtype}")

            if include_ret and str_return_type is not None:
                print(f"    # return: {str_return_type}")

        print_with_resolved_types()
        # return sig.parameters
