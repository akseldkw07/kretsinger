#!/usr/bin/env python3
# mdtype.py - the hand-typeable half of repo_md.
#
# Only the *export* runs on the offline machine, so only this has to arrive
# through the keyboard. ~45 lines. The rebuild half (repo_md.from_markdown)
# runs on the Mac, where files can arrive normally.
#
# Output is format-compatible with repo_md.py: rebuild with
#     python3 repo_md.py rebuild transcribed.md ~/recovered
#
# Usage: python3 mdtype.py ROOT OUT.md [.py,.md,.toml] [N]
#        filetypes default to DEFAULT below; trailing N (any value) turns on
#        "NNNN| " line-number prefixes.
#
# .ipynb files are exported as input cells only - no outputs, no execution
# counts, no metadata - so what rebuilds is a cleared notebook, not a copy.
import ast, hashlib, json, os, sys

DEFAULT = (".py", ".ipynb", ".zsh", ".md", ".bash", ".json", ".sh", ".toml", ".code-workspace")

if len(sys.argv) < 3:
    sys.exit(
        "usage: python3 mdtype.py ROOT OUT.md [FILETYPES] [N]\n"
        "  python3 mdtype.py ~/myrepo repo.md            # default filetypes\n"
        "  python3 mdtype.py ~/myrepo repo.md .py,.md    # only these\n"
        "  python3 mdtype.py ~/myrepo repo.md .py,.md N  # + line numbers\n"
        f"default filetypes: {','.join(DEFAULT)}"
    )

root, out = sys.argv[1], sys.argv[2]
exts = tuple(sys.argv[3].split(",")) if len(sys.argv) > 3 else DEFAULT
num = len(sys.argv) > 4


def strip_nb(raw):
    """A notebook reduced to its input cells: no outputs, no counts, no metadata.

    Usually 90%+ of a notebook's bytes are stored output, and it is the one
    part you get back for free by re-running. outputs/execution_count survive
    as empty on code cells only because nbformat requires the keys.
    """
    try:
        nb = json.loads(raw)
    except ValueError:
        return raw
    cells = []
    for c in nb.get("cells", []):
        k = c.get("cell_type")
        if k not in ("code", "markdown", "raw"):
            continue
        s = c.get("source", "")
        cell = {"cell_type": k, "metadata": {}, "source": s.splitlines(True) if isinstance(s, str) else s}
        if k == "code":
            cell["execution_count"], cell["outputs"] = None, []
        cells.append(cell)
    # minor 4: 4.5 requires a per-cell "id" and we emit none
    d = {"cells": cells, "metadata": {}, "nbformat": nb.get("nbformat", 4), "nbformat_minor": 4}
    return (json.dumps(d, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


SKIP = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "build",
    "dist",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".pytest_cache",
    "vault",
}
F = "`" * 5

files = []
for dp, dn, fn in os.walk(root):
    dn[:] = sorted(d for d in dn if d not in SKIP)
    for n in sorted(fn):
        f = os.path.join(dp, n)
        if n.startswith(".env") or "secret" in n or "token" in n:
            continue
        if n.endswith((".pem", ".key", ".netrc")) or n.startswith("id_rsa"):
            continue
        if exts and not n.endswith(exts):
            continue
        if not os.path.isfile(f) or os.path.islink(f):
            continue
        d = open(f, "rb").read()
        if b"\0" in d[:8192]:
            continue
        if n.endswith(".ipynb"):  # size judged after stripping, not on disk
            d = strip_nb(d)
        if len(d) > 1 << 20:
            continue
        files.append((os.path.relpath(f, root).replace(os.sep, "/"), d))
files.sort()

# Nonce so the delimiter never occurs inside the content it delimits.
seed = nonce = hashlib.sha256(b"".join(d for _, d in files)).hexdigest()[:6]
for i in range(64):
    if not any(nonce.encode() in d for _, d in files):
        break
    nonce = hashlib.sha256(f"{seed}{i}".encode()).hexdigest()[:6]
B, E = f"<!--F{nonce}:", f"<!--E{nonce}:"

# Indent depth, tabs, trailing spaces and line ends become visible glyphs.
# These are the things a reader with judgement CANNOT recover: a misread
# identifier repairs itself from context, but 4 spaces vs 8 is two different
# valid programs. Codepoints, not literals, so exporting this file is safe.
IND, SPC, TAB, EOL = map(chr, (0x2192, 0x00B7, 0x21E5, 0x00B6))


def mark(t):
    ws = [w for x in t.split("\n") if x.strip() and (w := len(x) - len(x.lstrip(" ")))]
    u = max((8, 4, 3, 2), key=lambda k: (sum(w % k == 0 for w in ws), k)) if ws else 4
    o = []
    for x in t.split("\n"):
        b = x.lstrip(" \t")
        w = x[: len(x) - len(b)]
        o.append(
            ("".join(TAB if c == "\t" else SPC for c in w) if "\t" in w or len(w) % u else IND * (len(w) // u))
            + b
            + EOL
        )
    return "\n".join(o), u


rows, body = [], []
for i, (rel, d) in enumerate(files, 1):
    t = d.decode("utf-8", "replace")
    h = hashlib.sha256(d).hexdigest()
    ln = t.count("\n") + (0 if t.endswith("\n") or not t else 1)
    rows.append(f"| `{rel}` | {ln} | {len(d)} | `{h[:16]}` |")
    s = "\n".join(f"{j:>4}| {x}" for j, x in enumerate(t.split("\n"), 1)) if num else t
    s, u = mark(s)
    # Fingerprint of what the code MEANS - no line numbers, no comments. Lets
    # the rebuild say "only formatting drifted" instead of just "sha mismatch".
    try:
        fp = hashlib.sha256(ast.dump(ast.parse(t)).encode()).hexdigest()[:12] if rel.endswith(".py") else ""
    except (SyntaxError, ValueError, RecursionError):
        fp = ""
    body.append(
        f"\n{B}{rel}-->\n### `{rel}` &nbsp;&nbsp; file {i} of {len(files)}\n\n"
        f"`sha256:{h}` &nbsp; `bytes:{len(d)}` &nbsp; `lines:{ln}`"
        f" &nbsp; `indent:{u}` &nbsp; `marks:{IND} {SPC} {TAB} {EOL}`"
        f"{f' &nbsp; `ast:{fp}`' if fp else ''}\n\n"
        f"{F}\n{s}\n{F}\n{E}{rel}--> END file {i} of {len(files)}\n"
    )

doc = (
    # f"# Repository export\n\nSource: `{os.path.abspath(root)}`\n\n"
    # f"Block delimiter nonce: `{nonce}`\n\n{len(files)} files.\n\n"
    # f"Transcribing this? Repair misread identifiers and syntax with your own\n"
    # f"judgement — that is expected. Do NOT guess at the marked things:\n"
    # f"`{IND}` = one indent level (`indent:N` spaces each), `{SPC}` = one leading\n"
    # f"space, `{TAB}` = one tab, `{EOL}` = end of that source line, with the line's\n"
    # f"exact content, trailing spaces included, sitting just before it.\n"
    # f"Indentation comes only from the glyphs, never from apparent margin.\n\n"
    f"## Manifest\n\n| path | lines | bytes | sha256 |\n|---|---:|---:|---|\n"
    + "\n".join(rows)
    + "\n\n---\n"
    + "".join(body)
)
open(out, "w", encoding="utf-8").write(doc)
print(f"{len(files)} files -> {len(doc):,} chars, ~{-(-len(doc) // 4000)} screenfuls")
