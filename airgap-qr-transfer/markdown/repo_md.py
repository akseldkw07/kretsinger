#!/usr/bin/env python3
"""
repo_md - flatten a repo into one markdown file, and rebuild it afterwards.

    import repo_md

    # on the offline machine
    info = repo_md.to_markdown("~/kretsinger", "repo.md",
                               include=[".py", ".md", ".toml"])

    # on the receiving machine, after transcription
    result = repo_md.from_markdown("transcribed.md", "~/recovered")
    print(result["bad"])        # files whose checksum did not match

Every file block carries its own SHA-256, line count and byte count, and the
document opens with a manifest of all of them. That is the whole point: a
transcription round trip WILL introduce errors, and this turns them from silent
corruption into a named list of files to re-check.

Standard library only. Read-only with respect to `root`.
"""

import fnmatch, hashlib, os, re

SKIP_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "build",
    "dist",
    ".mypy_cache",
    ".pytest_cache",
    ".ipynb_checkpoints",
}

# Excluded before `include` is applied, so no include list can pull one in.
SECRETS = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "id_rsa*",
    ".netrc",
    "credentials*",
    "*secret*",
    "*.token",
    "*token*",
    "vault/*",
    "*-api.txt",
    "*password*",
)

FENCE = "`" * 5  # 5 backticks so 3-backtick blocks inside .md survive

# Delimiters carry a per-export nonce: this file (and any doc describing the
# format) contains the delimiter text literally, and a fixed delimiter would
# split its own block in half. The nonce is chosen so it appears nowhere in the
# content being exported.
_NONCE_RE = re.compile(r"<!--F([0-9a-f]{6}):")


def _delims(nonce):
    return f"<!--F{nonce}:", f"<!--E{nonce}:"


def _pick_nonce(blobs):
    seed = hashlib.sha256(b"".join(blobs)).hexdigest()
    for i in range(64):
        nonce = hashlib.sha256(f"{seed}{i}".encode()).hexdigest()[:6]
        if not any(nonce.encode() in b for b in blobs):
            return nonce
    raise RuntimeError("could not find an unused delimiter nonce")


LANG = {
    ".py": "python",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "zsh",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".md": "markdown",
    ".txt": "text",
    ".json": "json",
    ".cfg": "ini",
    ".ini": "ini",
}

# ---------------------------------------------------------------- markers
# The document is read back by something with judgement, so it does not need
# to protect the parts a language model reconstructs for free: identifiers,
# keywords and syntax are massively redundant, and "whi1e True:" repairs
# itself. It needs to protect what judgement CANNOT recover, because both
# readings are valid programs and nothing in the text says which was meant:
#
#   indent depth  - 4 spaces vs 8 is a different program, no syntax error
#   trailing ws   - invisible on screen, changes the SHA
#   line breaks   - a merged or dropped line reads as plausible code
#
# So each becomes a visible glyph. Non-ASCII on purpose: these must not be
# confusable with l/1/I or O/0, which is the entire failure mode of OCR'd
# code. Indent markers are per LEVEL, not per space, so a model counts 3
# arrows rather than measuring 12 columns of whitespace in a video frame.
# Chosen per export, like the delimiter nonce and for the same reason. Written
# as codepoints rather than literals on purpose: if the glyphs appeared in this
# file, every set would collide the moment you export a tree containing
# repo_md.py, and markers would silently switch themselves off.
# Order is indent / space / tab / end-of-line.
MARK_SETS = [
    tuple(map(chr, cps))
    for cps in (
        (0x2192, 0x00B7, 0x21E5, 0x00B6),  # arrow, middot, tab, pilcrow
        (0x25B6, 0x2219, 0x21C9, 0x23CE),
        (0x00BB, 0x2218, 0x226B, 0x25C4),
    )
]


def _pick_marks(blobs):
    """First marker set that cannot be confused with the content itself.

    Only two positions are ambiguous, so only those are checked: a line that
    *begins* with an indent/space/tab glyph, and a line that *ends* with the
    end-of-line glyph. A middot in the middle of a comment is harmless, and
    rejecting a set over one would disable markers far more often than needed.
    """
    lines = [ln for b in blobs for ln in b.decode("utf-8", "replace").split("\n")]
    for marks in MARK_SETS:
        lead, eol = marks[:3], marks[3]
        if not any(ln.lstrip(" \t").startswith(lead) or ln.endswith(eol) for ln in lines):
            return marks
    return None


def _unit(text):
    """Indent step for a file: the one that covers the most indented lines.

    Not the gcd of every indent. One paren-aligned continuation line sitting
    at 7 spaces would drag the gcd to 1 and force the entire file onto
    per-space dots. _mark() already falls back to dots for individual lines
    that are not a clean multiple, so the right choice is the step that fits
    the most lines and lets the rest be exceptions.
    """
    widths = [w for ln in text.split("\n") if ln.strip() and (w := len(ln) - len(ln.lstrip(" ")))]
    if not widths:
        return 4
    best = max((8, 4, 3, 2), key=lambda u: (sum(w % u == 0 for w in widths), u))
    return best if sum(w % best == 0 for w in widths) >= len(widths) / 2 else 1


def _mark(text, unit, marks):
    IND, SPC, TAB, EOL = marks
    out = []
    for ln in text.split("\n"):
        body = ln.lstrip(" \t")
        ws = ln[: len(ln) - len(body)]
        if "\t" in ws or unit == 1 or len(ws) % unit:
            pre = "".join(TAB if c == "\t" else SPC for c in ws)
        else:
            pre = IND * (len(ws) // unit)
        out.append(pre + body + EOL)
    return "\n".join(out)


def _unmark(text, unit, marks):
    """Inverse of _mark, tolerant of whitespace the transcription invented."""
    IND, SPC, TAB, EOL = marks
    out = []
    for ln in text.split("\n"):
        ln = ln.rstrip()  # noise after the end-of-line mark
        if ln.endswith(EOL):
            ln = ln[:-1]
        i = 0
        while i < len(ln) and ln[i] in (IND, SPC, TAB):
            i += 1
        ws = "".join("\t" if c == TAB else (" " * unit if c == IND else " ") for c in ln[:i])
        out.append(ws + ln[i:])
    return "\n".join(out)


def _ast_fp(text):
    """Fingerprint of what a Python file *means*, ignoring how it is written.

    ast.dump() omits line numbers and column offsets, and comments never reach
    the tree at all, so this is stable across whitespace, comment and layout
    differences but changes the moment an identifier, literal or structure
    does. It is recorded at export time because the receiving machine has no
    copy of the original to compare against — the fingerprint has to travel
    inside the document. See _classify() for what it buys.
    """
    try:
        import ast

        return hashlib.sha256(ast.dump(ast.parse(text)).encode()).hexdigest()[:12]
    except (SyntaxError, ValueError, RecursionError):
        return None


def _classify(rel, text, want_fp):
    """Why a file failed its SHA: 'cosmetic', 'semantic', or 'unparsable'.

    A decoder with judgement produces near-misses that a byte comparison
    cannot rank. A repaired comment and a wrong loop bound both just say
    "sha mismatch". This separates them: same AST means the code means the
    same thing and only its formatting drifted, which is safe to keep.
    """
    if not want_fp or not rel.endswith(".py"):
        return "semantic"
    got = _ast_fp(text)
    if got is None:
        return "unparsable"
    return "cosmetic" if got == want_fp else "semantic"


def _guide(marks):
    IND, SPC, TAB, EOL = marks
    return f"""\
## How to decode this document

You are reading a repository that was displayed on a screen, filmed, and
transcribed back into text. Transcription errors are expected. **Use your
judgement to repair them** — that is why this format exists. Do not copy an
obvious error through just because it is what the page appears to say.

Repair freely: misread identifiers, keywords and punctuation. `whi1e`, `pr1nt`,
`sel f`, a smart quote where a straight one belongs — fix all of it. The code
must parse and be internally consistent (a name defined in one file and used in
another must match).

**Do not guess at these**, they are marked explicitly for exactly that reason:

| glyph | means |
|---|---|
| `{IND}` | one indent level, repeated per level. `indent:N` in the file header gives N spaces per level. Three arrows = 3 levels, never "about 12 spaces". |
| `{SPC}` | one literal space of leading whitespace (used when the indent is not a clean multiple) |
| `{TAB}` | one literal tab character |
| `{EOL}` | end of that source line. Everything between the last glyph and `{EOL}` is the line's exact content, trailing spaces included. |

A line's indentation comes **only** from its leading glyphs. Ignore how far
the text appears to sit from the left margin.

Each file sits between `<!--F<nonce>:path-->` and `<!--E<nonce>:path-->` and
carries `sha256`, `bytes` and `lines`. Reproduce those header lines verbatim —
they are how the rebuild locates files and reports which ones need another
pass. If a block's line count disagrees with `lines:`, you dropped or merged a
line; find it before moving on. Do not reformat, do not "improve" the code, and
do not fill in a section you could not read — leave it and let the checksum
report it.
"""


def _globs(pats):
    if pats is None:
        return None
    if isinstance(pats, str):
        pats = [pats]
    return [("*" + p if p.startswith(".") and not any(c in p for c in "*?/") else p) for p in pats if p.strip()]


def _hit(rel, name, globs):
    return any(fnmatch.fnmatch(name.lower(), g.lower()) or fnmatch.fnmatch(rel.lower(), g.lower()) for g in globs)


def collect(root, include=None, exclude=(), max_size=1 << 20, secrets=False):
    """Relative paths that to_markdown() would write, in the order it writes them."""
    root = os.path.abspath(os.path.expanduser(root))
    inc, exc = _globs(include), _globs(exclude) or []
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            if not os.path.isfile(full) or os.path.islink(full):
                continue
            if not secrets and _hit(rel, name, SECRETS):
                continue
            if inc is not None and not _hit(rel, name, inc):
                continue
            if exc and _hit(rel, name, exc):
                continue
            if os.path.getsize(full) > max_size:
                continue
            with open(full, "rb") as fh:
                if b"\0" in fh.read(8192):
                    continue
            out.append(rel)
    return sorted(out)


def to_markdown(
    root,
    out="repo.md",
    include=None,
    exclude=(),
    max_size=1 << 20,
    secrets=False,
    number_lines=False,
    markers=True,
    quiet=False,
):
    """Write every matching file into one markdown document, in sorted order.

    markers=True (default) makes indentation, tabs, trailing whitespace and
    line ends visible as glyphs, and prepends a decoding guide. This is what
    makes the document survive a camera: a reader with judgement repairs
    misread identifiers for free, but cannot recover an indent level, since
    both depths are valid programs. Costs little — indent markers are one
    glyph per level, which usually buys back more than the line-end glyphs
    spend. from_markdown() reverses it automatically.

    number_lines=True additionally prefixes each line with "NNNN| ", ~16%
    more document. Mostly redundant once markers are on, since the per-file
    `lines:` count already catches a dropped line.

    Returns a summary dict.
    """
    root = os.path.abspath(os.path.expanduser(root))
    files = collect(root, include, exclude, max_size, secrets)
    if not files:
        raise ValueError(f"no files matched under {root}")

    blobs = []
    for rel in files:
        with open(os.path.join(root, rel), "rb") as fh:
            blobs.append(fh.read())
    nonce = _pick_nonce(blobs)
    BEGIN, END = _delims(nonce)

    marks = _pick_marks(blobs) if markers else None
    if markers and marks is None:
        markers = False
        if not quiet:
            print("note: source contains every marker set; markers disabled")

    entries, body = [], []
    for i, (rel, raw) in enumerate(zip(files, blobs), 1):
        text = raw.decode("utf-8", "replace")
        digest = hashlib.sha256(raw).hexdigest()
        lines = text.count("\n") + (0 if text.endswith("\n") or not text else 1)
        entries.append((rel, digest, len(raw), lines))

        shown, unit = text, _unit(text) if markers else 1
        fp = _ast_fp(text) if rel.endswith(".py") else None
        if number_lines:
            shown = "\n".join(f"{i:>4}| {ln}" for i, ln in enumerate(text.split("\n"), 1))
        if markers:
            shown = _mark(shown, unit, marks)
        lang = LANG.get(os.path.splitext(rel)[1].lower(), "")
        body.append(
            f"\n{BEGIN}{rel}-->\n"
            f"### `{rel}` &nbsp;&nbsp; file {i} of {len(files)}\n\n"
            f"`sha256:{digest}` &nbsp; `bytes:{len(raw)}` &nbsp; `lines:{lines}`"
            f"{f' &nbsp; `indent:{unit}` &nbsp; `marks:{chr(32).join(marks)}`' if marks else ''}"
            f"{f' &nbsp; `ast:{fp}`' if fp else ''}"
            f"{' &nbsp; `numbered`' if number_lines else ''}\n\n"
            f"{FENCE}{lang}\n{shown}\n{FENCE}\n"
            f"{END}{rel}--> END file {i} of {len(files)}, sha256 {digest[:16]}\n"
        )

    manifest = "\n".join(f"| `{rel}` | {n} | {b} | `{d[:16]}` |" for rel, d, b, n in entries)
    head = (
        f"# Repository export\n\n"
        f"Source: `{root}`\n\n"
        f"Block delimiter nonce: `{nonce}`\n\n"
        f"{len(entries)} files, {sum(e[2] for e in entries):,} bytes, "
        f"{sum(e[3] for e in entries):,} lines.\n\n"
        # NB: the delimiters are described here, never written literally, or
        # this sentence would parse as the start of a file block called "path".
        f"Rebuild with `repo_md.from_markdown(path, out_dir)`. Each block is "
        f"delimited by `<!--F<nonce>:path-->` and `<!--E<nonce>:path-->` and "
        f"carries a SHA-256, so any file that did not survive transcription is "
        f"reported by name rather than failing silently.\n\n"
        f"{_guide(marks) if marks else ''}\n"
        f"## Manifest\n\n"
        f"| path | lines | bytes | sha256 |\n|---|---:|---:|---|\n{manifest}\n\n"
        f"---\n"
    )

    doc = head + "".join(body)
    out = os.path.abspath(os.path.expanduser(out))
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(doc)

    chars = len(doc)
    info = {
        "path": out,
        "files": len(entries),
        "source_bytes": sum(e[2] for e in entries),
        "source_lines": sum(e[3] for e in entries),
        "markdown_chars": chars,
        # ~45 lines x ~90 cols is a comfortable, still-legible screenful
        "screens_at_4000_chars": -(-chars // 4000),
    }
    if not quiet:
        print(f"{info['files']} files, {info['source_bytes']:,} bytes -> " f"{chars:,} characters of markdown")
        print(f"~{info['screens_at_4000_chars']} screenfuls at ~4,000 chars each")
        print(f"wrote {out}")
    return info


_SHA = re.compile(r"`sha256:([0-9a-fA-F]{64})`")
_NUMBERED = re.compile(r"^\s*\d+\|\s?", re.MULTILINE)


_INDENT = re.compile(r"`indent:(\d+)`")
_MARKS = re.compile(r"`marks:(\S) (\S) (\S) (\S)`")
_AST = re.compile(r"`ast:([0-9a-f]{12})`")


def _blocks(doc):
    """Yield (path, sha, unit, marks, code) per file block.

    Deliberately a linear split rather than one big regex: a regex with `.*?`
    spanning hundreds of blocks in a half-megabyte document backtracks
    quadratically and effectively hangs.
    """
    m = _NONCE_RE.search(doc)
    if not m:
        return
    BEGIN, END = _delims(m.group(1))
    for seg in doc.split(BEGIN)[1:]:
        try:
            rel = seg[: seg.index("-->")].strip()
        except ValueError:
            continue
        stop = seg.find(END)
        block = seg if stop == -1 else seg[:stop]

        m = _SHA.search(block)
        open_at = block.find(FENCE)
        if not m or open_at == -1:
            continue
        code_at = block.find("\n", open_at)
        close_at = block.rfind("\n" + FENCE)
        if code_at == -1 or close_at <= code_at:
            continue
        head = block[:code_at]
        u, mk, fp = _INDENT.search(head), _MARKS.search(head), _AST.search(head)
        yield (
            rel,
            m.group(1).lower(),
            int(u.group(1)) if u else 0,
            mk.groups() if mk else MARK_SETS[0],
            fp.group(1) if fp else None,
            block[code_at + 1 : close_at],
        )


def from_markdown(md_path, out_dir, force=False, strip_numbers=None, quiet=False):
    """Rebuild a tree from a markdown export. Returns a summary dict.

    Refuses a non-empty out_dir unless force=True. Verifies every file against
    the SHA-256 recorded in its block and returns the failures in ["bad"],
    so you know exactly what to re-transcribe.
    """
    md_path = os.path.abspath(os.path.expanduser(md_path))
    out_dir = os.path.abspath(os.path.expanduser(out_dir))
    if os.path.exists(out_dir) and os.listdir(out_dir) and not force:
        raise FileExistsError(f"{out_dir} exists and is not empty; pass force=True")

    with open(md_path, encoding="utf-8") as fh:
        doc = fh.read()

    ok, bad, kinds = [], [], {}
    for rel, want, unit, marks, want_fp, code in _blocks(doc):
        if unit:  # markers=True export; undo the glyphs
            code = _unmark(code, unit, marks)
        if strip_numbers or (strip_numbers is None and _NUMBERED.search(code)):
            code = _NUMBERED.sub("", code)
        blob = code.encode("utf-8")
        got = hashlib.sha256(blob).hexdigest()
        if got != want:  # tolerate a lost or added trailing newline
            for alt in (code + "\n", code.rstrip("\n")):
                if hashlib.sha256(alt.encode("utf-8")).hexdigest() == want:
                    blob, got = alt.encode("utf-8"), want
                    break

        dest = os.path.join(out_dir, rel)
        if os.path.relpath(dest, out_dir).startswith(".."):
            raise ValueError(f"refusing path outside out_dir: {rel}")
        os.makedirs(os.path.dirname(dest) or out_dir, exist_ok=True)
        with open(dest, "wb") as fh:
            fh.write(blob)
        if got == want:
            ok.append(rel)
        else:
            bad.append(rel)
            kinds[rel] = _classify(rel, blob.decode("utf-8", "replace"), want_fp)

    # anything listed in the manifest but missing a block entirely
    listed = set(re.findall(r"^\| `(.+?)` \| \d+ \| \d+ \| `[0-9a-f]{16}` \|$", doc, re.MULTILINE))
    missing = sorted(listed - set(ok) - set(bad))

    cosmetic = sorted(r for r in bad if kinds.get(r) == "cosmetic")
    redo = sorted(r for r in bad if kinds.get(r) != "cosmetic")
    info = {
        "out_dir": out_dir,
        "recovered": len(ok) + len(bad),
        "ok": len(ok),
        "bad": sorted(bad),
        # `bad` split by whether the difference actually matters. Same AST
        # means only comments/formatting drifted and the file is safe to keep;
        # everything else has to go round again.
        "cosmetic": cosmetic,
        "redo": redo,
        "unparsable": sorted(r for r in bad if kinds.get(r) == "unparsable"),
        "missing": missing,
    }
    if not quiet:
        print(
            f"{info['recovered']} blocks parsed -> {info['ok']} verified, "
            f"{len(bad)} corrupt, {len(missing)} missing"
        )
        if cosmetic:
            print(f"   {len(cosmetic)} differ only in comments/formatting (same AST) - safe to keep:")
            for rel in cosmetic[:5]:
                print(f"      {rel}")
        for rel in (redo + missing)[:20]:
            note = " (does not parse)" if kinds.get(rel) == "unparsable" else ""
            print(f"   re-check {rel}{note}")
        if len(redo) + len(missing) > 20:
            print(f"   ... and {len(redo) + len(missing) - 20} more")
    return info


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "rebuild":
        from_markdown(sys.argv[2], sys.argv[3])
    else:
        inc = sys.argv[3].split(",") if len(sys.argv) > 3 else None
        to_markdown(sys.argv[1], sys.argv[2], include=inc)
